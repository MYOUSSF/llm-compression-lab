"""
mixed_precision.py
------------------
Mixed-precision quantization policy builder for transformer models.

Takes the output of SensitivityAnalyzer and produces a per-layer
qconfig map that assigns INT8 to sensitive layers and INT4 (or skip)
to insensitive ones. Also handles the full PTQ + mixed-precision
conversion pipeline.

Usage:
    from src.mixed_precision import MixedPrecisionPolicy

    # Build policy from sensitivity scores
    policy = MixedPrecisionPolicy(
        sensitivity_scores=scores,       # dict from SensitivityAnalyzer.run()
        high_sensitivity_threshold=0.3,  # delta above this → INT8
        use_int4_for_low=True,           # below threshold → INT4 via bitsandbytes
    )

    # Apply to a model
    quantized = policy.apply(fp32_model, calibration_loader, device)

    # Inspect the assignment
    policy.print_summary()
    df = policy.to_dataframe()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.quantization


# ---------------------------------------------------------------------------
# Precision tiers
# ---------------------------------------------------------------------------

class Precision(str, Enum):
    FP32 = "FP32"   # kept at full precision (most sensitive)
    INT8 = "INT8"   # standard symmetric INT8 quantization
    INT4 = "INT4"   # weight-only INT4 via bitsandbytes (least sensitive)


# ---------------------------------------------------------------------------
# Layer assignment dataclass
# ---------------------------------------------------------------------------

@dataclass
class LayerAssignment:
    name: str
    sensitivity_delta: float
    precision: Precision
    reason: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MixedPrecisionPolicy:
    """
    Assigns per-layer precision based on sensitivity scores.

    Precision tiers (configurable thresholds):
    ┌──────────────────────────────────────────────────────────┐
    │  delta > fp32_threshold   →  FP32  (protect accuracy)   │
    │  fp32_threshold ≥ delta > int8_threshold  →  INT8        │
    │  delta ≤ int8_threshold   →  INT4  (maximise compression)│
    └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    sensitivity_scores : dict {layer_name: accuracy_delta}
        Output of SensitivityAnalyzer.run().
    fp32_threshold : float
        Layers with delta above this are kept at FP32.
        Default 1.0% — conservative; tune based on your accuracy budget.
    int8_threshold : float
        Layers with delta above this (but below fp32_threshold) get INT8.
        Layers below this get INT4.
        Default 0.3%.
    force_fp32 : list[str]
        Layer name substrings that always stay FP32 regardless of score
        (e.g. ["embeddings", "classifier"] for safety).
    force_int8 : list[str]
        Layer name substrings that always get INT8 (skip INT4 promotion).
    """

    def __init__(
        self,
        sensitivity_scores: dict[str, float],
        fp32_threshold: float = 1.0,
        int8_threshold: float = 0.3,
        force_fp32: list[str] | None = None,
        force_int8: list[str] | None = None,
    ) -> None:
        self.sensitivity_scores = sensitivity_scores
        self.fp32_threshold  = fp32_threshold
        self.int8_threshold  = int8_threshold
        self.force_fp32      = force_fp32  or ["embeddings", "classifier", "pooler"]
        self.force_int8      = force_int8  or []

        self._assignments: list[LayerAssignment] = []
        self._build_assignments()

    # ------------------------------------------------------------------
    # Policy construction
    # ------------------------------------------------------------------

    def _build_assignments(self) -> None:
        for name, delta in self.sensitivity_scores.items():
            # Check forced overrides first
            if any(kw in name for kw in self.force_fp32):
                precision = Precision.FP32
                reason = "forced FP32 (safety override)"
            elif any(kw in name for kw in self.force_int8):
                precision = Precision.INT8
                reason = "forced INT8 (override)"
            # Then apply threshold logic
            elif delta > self.fp32_threshold:
                precision = Precision.FP32
                reason = f"high sensitivity (Δ={delta:.3f}% > {self.fp32_threshold}%)"
            elif delta > self.int8_threshold:
                precision = Precision.INT8
                reason = f"medium sensitivity (Δ={delta:.3f}%)"
            else:
                precision = Precision.INT4
                reason = f"low sensitivity (Δ={delta:.3f}% ≤ {self.int8_threshold}%)"

            self._assignments.append(
                LayerAssignment(name=name, sensitivity_delta=delta, precision=precision, reason=reason)
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_precision(self, layer_name: str) -> Precision:
        for a in self._assignments:
            if a.name == layer_name:
                return a.precision
        return Precision.INT8  # default for unknown layers

    def layers_by_precision(self, precision: Precision) -> list[str]:
        return [a.name for a in self._assignments if a.precision == precision]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "layer": a.name,
                "sensitivity_delta": a.sensitivity_delta,
                "precision": a.precision.value,
                "reason": a.reason,
            }
            for a in self._assignments
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("sensitivity_delta", ascending=False)
            .reset_index(drop=True)
        )

    def print_summary(self) -> None:
        df = self.to_dataframe()
        fp32_count = (df["precision"] == "FP32").sum()
        int8_count = (df["precision"] == "INT8").sum()
        int4_count = (df["precision"] == "INT4").sum()
        total      = len(df)

        print("\n" + "=" * 65)
        print("Mixed-Precision Policy Summary")
        print("=" * 65)
        print(f"  Total layers assigned : {total}")
        print(f"  FP32 (protected)      : {fp32_count}  ({100*fp32_count/total:.1f}%)")
        print(f"  INT8                  : {int8_count}  ({100*int8_count/total:.1f}%)")
        print(f"  INT4 (compressed)     : {int4_count}  ({100*int4_count/total:.1f}%)")
        print(f"\n  Thresholds: FP32 if Δ>{self.fp32_threshold}%, INT4 if Δ≤{self.int8_threshold}%")
        print("=" * 65)

        print("\n  Top 10 layer assignments:")
        print(f"  {'Layer':<55} {'Precision':<8} {'Delta':>8}")
        print("  " + "-" * 73)
        for _, row in df.head(10).iterrows():
            print(f"  {row['layer']:<55} {row['precision']:<8} {row['sensitivity_delta']:>+7.3f}%")

    # ------------------------------------------------------------------
    # Quantization config builder (torch.ao)
    # ------------------------------------------------------------------

    def build_qconfig_map(
        self, backend: str = "fbgemm"
    ) -> dict[str, Any]:
        """
        Build a {layer_name: qconfig | None} map for torch.ao quantization.

        FP32 layers get qconfig=None (skipped by torch.ao).
        INT8 layers get the default per-channel qconfig.
        INT4 layers also get None here — INT4 is handled separately
        via bitsandbytes weight-only quantization after the INT8 pass.

        Returns
        -------
        qconfig_map : dict suitable for passing to
            torch.ao.quantization.prepare / prepare_qat as the
            `qconfig_mapping` argument via QConfigMapping.
        """
        from torch.ao.quantization import QConfigMapping, get_default_qconfig

        default_qconfig = get_default_qconfig(backend)
        mapping = QConfigMapping()

        # Global default: quantize everything
        mapping.set_global(default_qconfig)

        # Override FP32-protected layers to None (skip quantization)
        for name in self.layers_by_precision(Precision.FP32):
            mapping.set_module_name(name, None)

        return mapping

    # ------------------------------------------------------------------
    # Apply policy (PTQ path)
    # ------------------------------------------------------------------

    def apply(
        self,
        fp32_model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: torch.device,
        backend: str = "fbgemm",
        num_calibration_batches: int = 100,
        apply_int4: bool = True,
    ) -> nn.Module:
        """
        Apply the mixed-precision policy to fp32_model using PTQ.

        Steps:
        1. Set per-layer qconfigs according to policy.
        2. Prepare model (insert observers).
        3. Run calibration batches to collect activation statistics.
        4. Convert to INT8 (FP32 layers are skipped automatically).
        5. Optionally apply INT4 weight-only quantization for INT4 layers.

        Parameters
        ----------
        fp32_model           : Original float model.
        calibration_loader   : DataLoader for calibration (no labels needed).
        device               : CPU required for static quantization.
        backend              : "fbgemm" (x86) or "qnnpack" (ARM).
        num_calibration_batches : How many batches to run for calibration.
        apply_int4           : If True, apply bitsandbytes INT4 to low-sensitivity layers.

        Returns
        -------
        quantized_model : Mixed-precision quantized model.
        """
        torch.backends.quantized.engine = backend
        model = copy.deepcopy(fp32_model).cpu().eval()

        # Assign per-layer qconfigs manually (torch.ao QConfigMapping approach)
        default_qconfig = torch.quantization.get_default_qconfig(backend)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                precision = self.get_precision(name)
                if precision == Precision.FP32:
                    module.qconfig = None
                else:
                    module.qconfig = default_qconfig

        # Exclude module types with no CPU-quantized kernel (Embedding,
        # LayerNorm, GELU, Dropout) — see _skip_embedding_qconfig docstring
        _skip_embedding_qconfig(model)

        # Prepare (inserts observers)
        torch.quantization.prepare(model, inplace=True)

        # Calibration
        print(f"\nCalibrating ({num_calibration_batches} batches)...")
        model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                _forward_batch(model, batch, device="cpu")
        print("Calibration complete.")

        # Convert to INT8
        torch.quantization.convert(model, inplace=True)

        # Optional INT4 weight-only (bitsandbytes)
        if apply_int4:
            int4_layers = self.layers_by_precision(Precision.INT4)
            if int4_layers:
                model = _apply_int4_weight_only(model, int4_layers)

        return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skip_embedding_qconfig(model: nn.Module) -> None:
    """
    Set qconfig=None on module types that have no CPU-quantized kernel.

    Affected types and their failure modes on plain CPU:
      nn.Embedding  -> AssertionError: needs float_qparams_weight_only_qconfig
      nn.LayerNorm  -> NotImplementedError: quantized::layer_norm not on CPU
      nn.GELU       -> no quantized kernel on CPU
      nn.Dropout    -> passthrough op, safe to skip

    Excluding these has negligible impact on compression ratio because
    they contain no learned weights (Dropout, GELU) or are small relative
    to the linear layers (Embedding, LayerNorm).
    """
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm,
                                nn.GELU, nn.Dropout)):
            module.qconfig = None


def _forward_batch(
    model: nn.Module, batch: Any, device: str = "cpu"
) -> None:
    """Run a forward pass for calibration (ignores output)."""
    if len(batch) == 3:
        input_ids, attention_mask, _ = batch
        model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )
    elif len(batch) == 2:
        inputs, _ = batch
        if isinstance(inputs, torch.Tensor):
            model(inputs.to(device))
        else:
            model(**{k: v.to(device) for k, v in inputs.items()})
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} elements.")


def _apply_int4_weight_only(
    model: nn.Module, target_layer_names: list[str]
) -> nn.Module:
    """
    Apply INT4 weight-only quantization to specified layers using bitsandbytes.
    Falls back gracefully to INT8 if bitsandbytes is not available.

    This uses linear_4bit from bitsandbytes, which stores weights in 4-bit
    NF4 format but performs computation in BF16/FP16. This is the same
    technique used in QLoRA.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit

        replaced = 0
        for name in target_layer_names:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]
            original = getattr(parent, child_name)

            if not isinstance(original, nn.Linear):
                continue

            int4_layer = Linear4bit(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                compute_dtype=torch.float16,
                quant_type="nf4",
            )
            int4_layer.weight = bnb.nn.Params4bit(
                original.weight.data,
                requires_grad=False,
                quant_type="nf4",
            )
            if original.bias is not None:
                int4_layer.bias = original.bias

            setattr(parent, child_name, int4_layer)
            replaced += 1

        print(f"INT4 (NF4) applied to {replaced} low-sensitivity layers via bitsandbytes.")
        return model

    except ImportError:
        print(
            "bitsandbytes not available — INT4 layers will remain INT8.\n"
            "Install with: pip install bitsandbytes"
        )
        return model

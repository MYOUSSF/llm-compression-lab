"""
sensitivity.py
--------------
Layerwise sensitivity scoring for transformer models.

Core idea: quantize the entire model, then for each linear layer,
temporarily restore it to FP32 and measure the accuracy delta.
The delta is that layer's "sensitivity score" — how much accuracy
was lost by quantizing it. High-sensitivity layers should stay FP32
or be assigned higher precision in a mixed-precision policy.

Usage:
    from src.sensitivity import SensitivityAnalyzer

    analyzer = SensitivityAnalyzer(fp32_model, quantized_model, eval_loader, device)
    scores = analyzer.run()           # dict: {layer_name: accuracy_delta}
    df     = analyzer.to_dataframe()  # sorted DataFrame
    fig    = analyzer.plot()          # matplotlib Figure, saveable
    policy = analyzer.recommend_policy(top_k=5)  # names of sensitive layers
"""

from __future__ import annotations

import copy
import time
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Return {fully_qualified_name: module} for every nn.Linear in model."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module
    return layers


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace the sub-module at dotted `name` with `new_module` in-place."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Retrieve the sub-module at dotted `name`."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Return top-1 accuracy (0–100) on `loader`.

    Args:
        model:       Any classification model returning logits.
        loader:      DataLoader yielding (input_ids, attention_mask, labels)
                     OR (input_ids, labels) — both formats handled.
        device:      Target device.
        max_batches: If set, stop after this many batches (faster approximation).
    """
    model.eval()
    correct = total = 0

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            # Handle two common batch formats
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels         = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels    = labels.to(device)
                outputs   = model(input_ids=input_ids)

            # HuggingFace models return a ModelOutput; plain models return tensors
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            preds    = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """
    Quantize-then-restore sensitivity analysis for transformer linear layers.

    Parameters
    ----------
    fp32_model : nn.Module
        The original, unquantized model (used as the FP32 restoration source).
    quantized_model : nn.Module
        A fully INT8-quantized copy of fp32_model (produced by PTQ or QAT).
        This model is modified in-place during analysis, then restored.
    eval_loader : DataLoader
        Validation loader. A subset (max_batches) is used for speed.
    device : torch.device
    baseline_accuracy : float | None
        If None, computed automatically from quantized_model on eval_loader.
    max_batches : int
        Number of batches used per layer evaluation (trade speed vs precision).
        Default 50 gives a solid approximation on most datasets.
    """

    def __init__(
        self,
        fp32_model: nn.Module,
        quantized_model: nn.Module,
        eval_loader: torch.utils.data.DataLoader,
        device: torch.device,
        baseline_accuracy: float | None = None,
        max_batches: int = 50,
    ) -> None:
        self.fp32_model      = fp32_model.to(device)
        self.quantized_model = quantized_model.to(device)
        self.eval_loader     = eval_loader
        self.device          = device
        self.max_batches     = max_batches

        # Identify all quantizable linear layers in the FP32 model
        self._linear_layers = _get_linear_layers(fp32_model)

        # Baseline = accuracy when EVERY layer is quantized
        if baseline_accuracy is not None:
            self.baseline_accuracy = baseline_accuracy
        else:
            print("Computing INT8 baseline accuracy...")
            self.baseline_accuracy = evaluate_accuracy(
                quantized_model, eval_loader, device, max_batches
            )
            print(f"  INT8 baseline: {self.baseline_accuracy:.2f}%")

        self._scores: dict[str, float] | None = None  # populated by run()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> dict[str, float]:
        """
        Run layerwise sensitivity analysis.

        For each linear layer:
          1. Replace the quantized version with the FP32 original.
          2. Evaluate accuracy.
          3. Compute delta = accuracy_with_fp32_layer - baseline.
          4. Restore the quantized layer.

        Returns
        -------
        scores : dict {layer_name: accuracy_delta}
            Higher delta → layer is more sensitive to quantization.
        """
        scores: dict[str, float] = {}
        n = len(self._linear_layers)

        if verbose:
            print(f"\nRunning sensitivity analysis across {n} linear layers...")
            print(f"Baseline INT8 accuracy: {self.baseline_accuracy:.2f}%\n")

        for idx, (name, fp32_layer) in enumerate(self._linear_layers.items()):
            t0 = time.time()

            # 1. Save the current quantized layer
            try:
                quantized_layer = copy.deepcopy(_get_module(self.quantized_model, name))
            except AttributeError:
                # Layer name exists in FP32 model but not in quantized model
                # (e.g. embedding layers that were skipped during quantization)
                if verbose:
                    print(f"  [{idx+1:3d}/{n}] {name:<55} SKIPPED (not in quantized model)")
                continue

            # 2. Temporarily restore FP32 layer
            _set_module(self.quantized_model, name, copy.deepcopy(fp32_layer).to(self.device))

            # 3. Evaluate
            acc = evaluate_accuracy(
                self.quantized_model, self.eval_loader, self.device, self.max_batches
            )
            delta = acc - self.baseline_accuracy

            # 4. Restore quantized layer
            _set_module(self.quantized_model, name, quantized_layer)

            scores[name] = delta
            elapsed = time.time() - t0

            if verbose:
                bar = "█" * int(delta / 0.2) if delta > 0 else ""
                print(
                    f"  [{idx+1:3d}/{n}] {name:<55} Δ={delta:+.3f}%  {bar}  ({elapsed:.1f}s)"
                )

        self._scores = scores
        return scores

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def scores(self) -> dict[str, float]:
        if self._scores is None:
            raise RuntimeError("Call .run() first.")
        return self._scores

    def to_dataframe(self) -> pd.DataFrame:
        """Return scores as a DataFrame sorted by sensitivity (highest first)."""
        df = pd.DataFrame(
            [{"layer": k, "sensitivity_delta": v} for k, v in self.scores.items()]
        )
        return df.sort_values("sensitivity_delta", ascending=False).reset_index(drop=True)

    def recommend_policy(self, top_k: int = 5) -> list[str]:
        """
        Return the top_k most sensitive layer names.
        These are the layers that should remain in FP32 (or INT8 if everything
        else is INT4) in a mixed-precision policy.
        """
        df = self.to_dataframe()
        return df.head(top_k)["layer"].tolist()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        top_k: int | None = None,
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot a horizontal bar chart of per-layer sensitivity scores.

        Parameters
        ----------
        top_k     : Show only the top-k most sensitive layers. None = all.
        save_path : If provided, saves the figure to this path (PNG).
        figsize   : Figure dimensions in inches.

        Returns
        -------
        fig : matplotlib Figure
        """
        df = self.to_dataframe()
        if top_k is not None:
            df = df.head(top_k)

        fig, ax = plt.subplots(figsize=figsize)

        colors = [
            "#E85D24" if v > 1.0 else   # high sensitivity → coral/red
            "#F2A623" if v > 0.3 else   # medium → amber
            "#3B8BD4"                    # low → blue
            for v in df["sensitivity_delta"]
        ]

        bars = ax.barh(df["layer"], df["sensitivity_delta"], color=colors, height=0.65)

        # Annotate bars
        for bar, val in zip(bars, df["sensitivity_delta"]):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"+{val:.3f}%",
                va="center",
                fontsize=9,
                color="#444",
            )

        ax.set_xlabel("Accuracy delta when layer restored to FP32 (%)", fontsize=11)
        ax.set_title(
            f"Layerwise quantization sensitivity\n"
            f"(baseline INT8 accuracy: {self.baseline_accuracy:.2f}%)",
            fontsize=12,
        )
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlim(left=-0.1)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E85D24", label="High (>1.0%)"),
            Patch(facecolor="#F2A623", label="Medium (0.3–1.0%)"),
            Patch(facecolor="#3B8BD4", label="Low (<0.3%)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Sensitivity chart saved to: {save_path}")

        return fig

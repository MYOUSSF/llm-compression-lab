"""
benchmark.py
------------
Latency, throughput, and memory benchmarking for compressed models.

Produces the central results table that goes in the README:
  | Method | Accuracy | Size (MB) | Latency @bs1 (ms) | Throughput (samples/s) |

Also generates activation distribution histograms — required by the JD's
"analyze activation distributions" qualification.

Usage:
    from src.benchmark import ModelBenchmark, plot_activation_histograms

    bench = ModelBenchmark(device=device)

    bench.register("FP32 baseline",    fp32_model,    accuracy=fp32_acc)
    bench.register("PTQ INT8",         ptq_model,     accuracy=ptq_acc)
    bench.register("Mixed-precision",  mixed_model,   accuracy=mixed_acc)
    bench.register("Sparse + INT8",    sparse_model,  accuracy=sparse_acc)

    df  = bench.run(sample_input)         # runs latency + size for all registered models
    fig = bench.plot_comparison()          # grouped bar chart
    bench.print_table()                   # ASCII results table

    # Activation histograms
    fig = plot_activation_histograms(fp32_model, int8_model, sample_input, n_layers=6)
"""

from __future__ import annotations

import gc
import os
import timeit
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    accuracy: float                  # top-1 accuracy (%)
    size_mb: float                   # state_dict file size
    latency_bs1_ms: float            # mean inference latency, batch=1 (ms)
    latency_bs32_ms: float           # mean inference latency, batch=32 (ms)
    latency_bs1_std: float           # std dev
    latency_bs32_std: float
    throughput_bs1: float            # samples / second at batch=1
    throughput_bs32: float           # samples / second at batch=32
    n_params: int = 0                # total parameter count
    extra: dict = field(default_factory=dict)   # any extra metrics

    @property
    def speedup_vs_fp32(self) -> float | None:
        return self.extra.get("speedup_vs_fp32")


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class ModelBenchmark:
    """
    Benchmarks multiple model variants and produces a comparison table.

    Parameters
    ----------
    device : torch.device
        Device to run benchmarks on. For quantized models this must be CPU.
    warmup_runs : int
        Number of warm-up forward passes before timing starts.
    timed_runs : int
        Number of timed runs per configuration (mean ± std reported).
    """

    def __init__(
        self,
        device: torch.device,
        warmup_runs: int = 10,
        timed_runs: int = 100,
    ) -> None:
        self.device      = device
        self.warmup_runs = warmup_runs
        self.timed_runs  = timed_runs

        self._registry: list[tuple[str, nn.Module, float]] = []  # (name, model, accuracy)
        self._results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, model: nn.Module, accuracy: float) -> None:
        """Add a model variant to benchmark."""
        self._registry.append((name, model, accuracy))

    # ------------------------------------------------------------------
    # Run all benchmarks
    # ------------------------------------------------------------------

    def run(
        self,
        sample_input: dict[str, torch.Tensor] | torch.Tensor,
        batch_sizes: tuple[int, ...] = (1, 32),
    ) -> pd.DataFrame:
        """
        Benchmark all registered models.

        Parameters
        ----------
        sample_input : dict or Tensor
            A single-sample input (batch size 1). Will be tiled to larger batches.
            For HuggingFace models: {"input_ids": ..., "attention_mask": ...}
            For plain models: a single tensor.

        Returns
        -------
        df : DataFrame with all benchmark metrics, sorted by latency @bs=1.
        """
        self._results = []
        fp32_latency_bs1 = None

        print(f"\n{'='*65}")
        print("Running benchmarks...")
        print(f"  Warmup: {self.warmup_runs} runs | Timed: {self.timed_runs} runs")
        print(f"  Device: {self.device}")
        print(f"{'='*65}\n")

        for name, model, accuracy in self._registry:
            print(f"  Benchmarking: {name}")
            model = model.to(self.device).eval()

            size_mb  = _get_model_size_mb(model)
            n_params = sum(p.numel() for p in model.parameters())

            lat_bs1,  std_bs1  = self._measure_latency(model, sample_input, batch_size=1)
            lat_bs32, std_bs32 = self._measure_latency(model, sample_input, batch_size=32)

            tput_bs1  = 1000.0 / lat_bs1  if lat_bs1  > 0 else 0.0
            tput_bs32 = 32000.0 / lat_bs32 if lat_bs32 > 0 else 0.0

            extra = {}
            if fp32_latency_bs1 is None:
                fp32_latency_bs1 = lat_bs1   # first model is assumed to be FP32
            else:
                extra["speedup_vs_fp32"] = fp32_latency_bs1 / lat_bs1

            result = BenchmarkResult(
                name=name,
                accuracy=accuracy,
                size_mb=size_mb,
                latency_bs1_ms=lat_bs1,
                latency_bs32_ms=lat_bs32,
                latency_bs1_std=std_bs1,
                latency_bs32_std=std_bs32,
                throughput_bs1=tput_bs1,
                throughput_bs32=tput_bs32,
                n_params=n_params,
                extra=extra,
            )
            self._results.append(result)

            print(f"    Size:          {size_mb:.2f} MB")
            print(f"    Accuracy:      {accuracy:.2f}%")
            print(f"    Latency @bs=1: {lat_bs1:.2f} ± {std_bs1:.2f} ms")
            print(f"    Latency @bs=32:{lat_bs32:.2f} ± {std_bs32:.2f} ms")
            print(f"    Throughput:    {tput_bs1:.1f} / {tput_bs32:.1f} samples/s")
            if "speedup_vs_fp32" in extra:
                print(f"    Speedup vs FP32: {extra['speedup_vs_fp32']:.2f}x")
            print()

            torch.cuda.empty_cache()
            gc.collect()

        return self.to_dataframe()

    # ------------------------------------------------------------------
    # Latency measurement
    # ------------------------------------------------------------------

    def _measure_latency(
        self,
        model: nn.Module,
        sample_input: dict[str, torch.Tensor] | torch.Tensor,
        batch_size: int,
    ) -> tuple[float, float]:
        """Return (mean_ms, std_ms) over timed_runs forward passes."""
        batched = _tile_input(sample_input, batch_size, self.device)

        # Warm-up
        with torch.inference_mode():
            for _ in range(self.warmup_runs):
                _forward(model, batched)

        # Timed runs
        times_ms = []
        with torch.inference_mode():
            for _ in range(self.timed_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = timeit.default_timer()
                _forward(model, batched)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times_ms.append((timeit.default_timer() - t0) * 1000)

        return float(np.mean(times_ms)), float(np.std(times_ms))

    # ------------------------------------------------------------------
    # Results output
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._results:
            rows.append({
                "Method":                r.name,
                "Accuracy (%)":          round(r.accuracy, 2),
                "Size (MB)":             round(r.size_mb, 2),
                "Latency @bs=1 (ms)":    f"{r.latency_bs1_ms:.1f} ± {r.latency_bs1_std:.1f}",
                "Latency @bs=32 (ms)":   f"{r.latency_bs32_ms:.1f} ± {r.latency_bs32_std:.1f}",
                "Throughput bs=1 (s/s)": round(r.throughput_bs1, 1),
                "Throughput bs=32 (s/s)":round(r.throughput_bs32, 1),
                "Speedup":               f"{r.speedup_vs_fp32:.2f}x" if r.speedup_vs_fp32 else "—",
            })
        return pd.DataFrame(rows)

    def print_table(self) -> None:
        """Print a clean ASCII benchmark comparison table."""
        df = self.to_dataframe()
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

    def save_csv(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Results saved to: {path}")

    def plot_comparison(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Four-panel comparison: accuracy, size, latency @bs=1, throughput @bs=32.
        """
        if not self._results:
            raise RuntimeError("Run .run() first.")

        names    = [r.name for r in self._results]
        accuracy = [r.accuracy for r in self._results]
        sizes    = [r.size_mb for r in self._results]
        lat_bs1  = [r.latency_bs1_ms for r in self._results]
        tput_bs32= [r.throughput_bs32 for r in self._results]

        colors = ["#3B8BD4", "#E85D24", "#1D9E75", "#BA7517", "#7F77DD"][:len(names)]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Model compression comparison", fontsize=14, fontweight="bold", y=1.01)

        panels = [
            (axes[0, 0], accuracy, "Accuracy (%)", True),
            (axes[0, 1], sizes,    "Model size (MB)", False),
            (axes[1, 0], lat_bs1,  "Latency @bs=1 (ms)", False),
            (axes[1, 1], tput_bs32,"Throughput @bs=32 (samples/s)", True),
        ]

        for ax, values, ylabel, higher_is_better in panels:
            bars = ax.bar(range(len(names)), values, color=colors, width=0.6)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.1f}",
                    ha="center",
                    fontsize=8,
                )

            indicator = "↑ higher is better" if higher_is_better else "↓ lower is better"
            ax.set_title(indicator, fontsize=8, color="gray", pad=4)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comparison chart saved to: {save_path}")

        return fig


# ---------------------------------------------------------------------------
# Activation distribution analysis
# ---------------------------------------------------------------------------

def plot_activation_histograms(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    sample_input: dict[str, torch.Tensor] | torch.Tensor,
    device: torch.device,
    n_layers: int = 6,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Compare pre- and post-quantization activation distributions.

    Hooks into the first `n_layers` linear layers of each model and
    captures the output activations on `sample_input`. Plots side-by-side
    histograms highlighting distribution shift caused by quantization.

    This addresses the JD requirement: "analyze activation distributions
    and numerical stability issues."

    Parameters
    ----------
    fp32_model   : Original float model.
    int8_model   : Quantized model (INT8 or mixed-precision).
    sample_input : Single-sample input dict or tensor.
    device       : Device to run on.
    n_layers     : Number of linear layers to compare.
    save_path    : If given, saves the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    fp32_activations = _collect_activations(fp32_model, sample_input, device, n_layers)
    int8_activations = _collect_activations(int8_model, sample_input, device, n_layers)

    # Use the layer names common to both models
    common_layers = [k for k in fp32_activations if k in int8_activations][:n_layers]
    n_plot = len(common_layers)

    if n_plot == 0:
        raise RuntimeError("No matching layers found between fp32 and int8 models.")

    ncols = 3
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plot > 1 else [axes]

    for i, layer_name in enumerate(common_layers):
        ax = axes[i]
        fp32_vals = fp32_activations[layer_name]
        int8_vals = int8_activations[layer_name]

        ax.hist(fp32_vals, bins=60, alpha=0.55, color="#3B8BD4", label="FP32", density=True)
        ax.hist(int8_vals, bins=60, alpha=0.55, color="#E85D24", label="INT8", density=True)

        # Overlay key statistics
        fp32_std = np.std(fp32_vals)
        int8_std = np.std(int8_vals)
        kl_div = _approx_kl_divergence(fp32_vals, int8_vals)

        short_name = layer_name.split(".")[-3:] if "." in layer_name else [layer_name]
        ax.set_title(".".join(short_name), fontsize=8)
        ax.set_xlabel(
            f"FP32 σ={fp32_std:.3f}  INT8 σ={int8_std:.3f}  KL≈{kl_div:.4f}",
            fontsize=7,
        )
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Activation distributions: FP32 vs INT8 (per linear layer)\n"
        "High KL divergence or std shift indicates numerical instability risk",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Activation histograms saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model_size_mb(model: nn.Module) -> float:
    """Measure model size by saving state_dict to disk."""
    path = "/tmp/_benchmark_model_size.pt"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size


def _tile_input(
    sample_input: dict[str, torch.Tensor] | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """Repeat a single-sample input to fill a batch."""
    if isinstance(sample_input, dict):
        return {
            k: v.repeat(batch_size, *([1] * (v.dim() - 1))).to(device)
            for k, v in sample_input.items()
        }
    return sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1))).to(device)


def _forward(
    model: nn.Module,
    batched_input: dict[str, torch.Tensor] | torch.Tensor,
) -> Any:
    """Run a forward pass; handles both dict and tensor inputs."""
    if isinstance(batched_input, dict):
        return model(**batched_input)
    return model(batched_input)


def _collect_activations(
    model: nn.Module,
    sample_input: dict[str, torch.Tensor] | torch.Tensor,
    device: torch.device,
    n_layers: int,
) -> dict[str, np.ndarray]:
    """
    Register forward hooks on the first n_layers linear layers,
    run a forward pass, and return captured output activations.
    """
    model = model.to(device).eval()
    activations: dict[str, np.ndarray] = {}
    hooks = []

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ][:n_layers]

    def make_hook(layer_name: str):
        def hook(module, input, output):
            # output may be a tensor or tuple (e.g. from some quantized wrappers)
            if isinstance(output, torch.Tensor):
                activations[layer_name] = output.detach().cpu().float().numpy().flatten()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                activations[layer_name] = output[0].detach().cpu().float().numpy().flatten()
        return hook

    for name, module in linear_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    batched = _tile_input(sample_input, batch_size=1, device=device)
    with torch.inference_mode():
        _forward(model, batched)

    for h in hooks:
        h.remove()

    return activations


def _approx_kl_divergence(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Approximate KL divergence KL(P || Q) between two sets of samples
    using histogram binning.

    High KL (>0.01) indicates the quantized layer has a meaningfully
    different activation distribution — a signal of potential accuracy loss.
    """
    eps = 1e-10
    all_vals = np.concatenate([p_vals, q_vals])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)

    p_hist = p_hist + eps
    q_hist = q_hist + eps

    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))

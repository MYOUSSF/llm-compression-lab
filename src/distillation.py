"""
distillation.py
---------------
Knowledge distillation: FP32 teacher → compressed student.

Implements soft-label distillation (Hinton et al., 2015) with a
temperature sweep, plus a LoRA-aware training loop that can fine-tune
a PEFT-wrapped student.

The distillation loss is a weighted combination of:
  - Hard loss:  CrossEntropy(student_logits, true_labels)
  - Soft loss:  KLDiv(softmax(student/T), softmax(teacher/T)) × T²

Increasing T "softens" the teacher distribution, forcing the student
to learn richer inter-class relationships rather than just the argmax.

Usage:
    from src.distillation import DistillationTrainer, temperature_sweep

    trainer = DistillationTrainer(
        teacher=fp32_model,
        student=int8_model,          # or LoRA-wrapped model
        device=device,
        temperature=4.0,
        alpha=0.7,                   # weight on soft loss (1-alpha = hard loss)
        learning_rate=2e-5,
    )
    history = trainer.train(train_loader, eval_loader, epochs=3)
    trainer.plot_history(save_path="results/distillation_history.png")

    # Sweep temperatures and compare recovery
    results = temperature_sweep(
        teacher, student_factory_fn, train_loader, eval_loader,
        temperatures=[2, 4, 8], device=device
    )
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrainingStep:
    epoch: int
    train_loss: float
    train_hard_loss: float
    train_soft_loss: float
    eval_accuracy: float
    elapsed_sec: float


@dataclass
class DistillationHistory:
    steps: list[TrainingStep] = field(default_factory=list)
    temperature: float = 4.0
    alpha: float = 0.7
    baseline_accuracy: float | None = None   # student accuracy before distillation

    @property
    def best_accuracy(self) -> float:
        return max((s.eval_accuracy for s in self.steps), default=0.0)

    @property
    def accuracy_recovery(self) -> float | None:
        if self.baseline_accuracy is None:
            return None
        return self.best_accuracy - self.baseline_accuracy

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "epoch":           s.epoch,
                "train_loss":      s.train_loss,
                "train_hard_loss": s.train_hard_loss,
                "train_soft_loss": s.train_soft_loss,
                "eval_accuracy":   s.eval_accuracy,
                "elapsed_sec":     s.elapsed_sec,
            }
            for s in self.steps
        ])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """
    Teacher-student knowledge distillation trainer.

    Parameters
    ----------
    teacher : nn.Module
        FP32 model (frozen). Provides soft target distributions.
    student : nn.Module
        Compressed model (INT8, mixed-precision, or LoRA-wrapped).
        Only student parameters are updated.
    device : torch.device
    temperature : float
        Softmax temperature T. Higher T → softer distributions → richer
        signal. Typical range: 2–8. Use temperature_sweep() to find best.
    alpha : float
        Weight on soft distillation loss. (1 - alpha) weights hard loss.
        alpha=1.0 → pure distillation; alpha=0.0 → pure cross-entropy.
        Typical values: 0.5–0.8.
    learning_rate : float
        AdamW learning rate. QAT recovery usually needs 1e-5 to 5e-5.
    weight_decay : float
        AdamW weight decay.
    max_grad_norm : float
        Gradient clipping norm. Prevents instability in quantized models.
    label_smoothing : float
        Applied to the hard CE loss. 0.1 often helps generalization.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: torch.device,
        temperature: float = 4.0,
        alpha: float = 0.7,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.1,
    ) -> None:
        self.device       = device
        self.temperature  = temperature
        self.alpha        = alpha
        self.max_grad_norm= max_grad_norm

        # Teacher is frozen — no gradients needed
        self.teacher = teacher.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = student.to(device)

        # Only optimize parameters that require grad
        # (LoRA adapters set requires_grad=True only on adapter weights)
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay
        )

        self.hard_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._history: DistillationHistory | None = None

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined distillation loss.

        Returns
        -------
        total_loss, hard_loss, soft_loss
            All are scalar tensors. Call total_loss.backward().
        """
        T = self.temperature

        # Hard loss: standard cross-entropy with true labels
        hard_loss = self.hard_criterion(student_logits, labels)

        # Soft loss: KL divergence between softened distributions
        # Multiply by T² to preserve gradient magnitude (Hinton et al.)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits   / T, dim=-1)
        soft_loss    = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss, hard_loss, soft_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        epochs: int = 3,
        baseline_accuracy: float | None = None,
        scheduler_type: str = "cosine",
        eval_every_n_steps: int | None = None,
    ) -> DistillationHistory:
        """
        Run distillation training.

        Parameters
        ----------
        train_loader : DataLoader
        eval_loader  : DataLoader
        epochs       : Number of full passes over the training data.
        baseline_accuracy : Student accuracy before distillation (for reporting).
        scheduler_type    : "cosine" or "linear" or None.
        eval_every_n_steps: If set, evaluate mid-epoch at this step interval.

        Returns
        -------
        history : DistillationHistory
        """
        history = DistillationHistory(
            temperature=self.temperature,
            alpha=self.alpha,
            baseline_accuracy=baseline_accuracy,
        )

        # Learning rate scheduler
        total_steps = epochs * len(train_loader)
        scheduler   = _build_scheduler(self.optimizer, scheduler_type, total_steps)

        n_trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        print(f"\n{'='*65}")
        print(f"Distillation Training  |  T={self.temperature}  α={self.alpha}")
        print(f"Trainable parameters   : {n_trainable:,}")
        print(f"Epochs                 : {epochs}")
        print(f"Scheduler              : {scheduler_type}")
        print(f"{'='*65}\n")

        if baseline_accuracy is not None:
            print(f"Student baseline accuracy: {baseline_accuracy:.2f}%\n")

        global_step = 0

        for epoch in range(1, epochs + 1):
            self.student.train()
            epoch_loss = epoch_hard = epoch_soft = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = _unpack_batch(batch, self.device)

                # Teacher forward (no grad)
                with torch.inference_mode():
                    teacher_out = self.teacher(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    teacher_logits = (
                        teacher_out.logits
                        if hasattr(teacher_out, "logits")
                        else teacher_out
                    )

                # Student forward
                student_out = self.student(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logits = (
                    student_out.logits
                    if hasattr(student_out, "logits")
                    else student_out
                )

                loss, hard_loss, soft_loss = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                if scheduler:
                    scheduler.step()

                epoch_loss += loss.item()
                epoch_hard += hard_loss.item()
                epoch_soft += soft_loss.item()
                global_step += 1

                # Mid-epoch evaluation
                if (
                    eval_every_n_steps is not None
                    and global_step % eval_every_n_steps == 0
                ):
                    acc = self._evaluate(eval_loader)
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  step {global_step:5d} | acc={acc:.2f}% | "
                        f"loss={loss.item():.4f} | lr={lr:.2e}"
                    )
                    self.student.train()

            # Epoch-end evaluation
            n_batches   = len(train_loader)
            avg_loss    = epoch_loss / n_batches
            avg_hard    = epoch_hard / n_batches
            avg_soft    = epoch_soft / n_batches
            elapsed     = time.time() - t0
            eval_acc    = self._evaluate(eval_loader)

            step_record = TrainingStep(
                epoch=epoch,
                train_loss=avg_loss,
                train_hard_loss=avg_hard,
                train_soft_loss=avg_soft,
                eval_accuracy=eval_acc,
                elapsed_sec=elapsed,
            )
            history.steps.append(step_record)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"loss={avg_loss:.4f} (hard={avg_hard:.4f}, soft={avg_soft:.4f}) | "
                f"eval_acc={eval_acc:.2f}% | "
                f"time={elapsed:.1f}s"
            )

        self._history = history
        print(f"\nBest accuracy: {history.best_accuracy:.2f}%")
        if history.accuracy_recovery is not None:
            print(f"Accuracy recovery vs baseline: +{history.accuracy_recovery:.2f}%")

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, eval_loader: DataLoader) -> float:
        self.student.eval()
        correct = total = 0
        with torch.inference_mode():
            for batch in eval_loader:
                input_ids, attention_mask, labels = _unpack_batch(batch, self.device)
                out = self.student(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, "logits") else out
                preds    = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_history(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """Plot training loss and eval accuracy curves."""
        if self._history is None:
            raise RuntimeError("Call .train() first.")

        df = self._history.to_dataframe()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Loss components
        ax1.plot(df["epoch"], df["train_loss"],      label="Total loss",  color="#E85D24", linewidth=2)
        ax1.plot(df["epoch"], df["train_hard_loss"], label="Hard (CE)",   color="#3B8BD4", linestyle="--")
        ax1.plot(df["epoch"], df["train_soft_loss"], label="Soft (KL×T²)",color="#1D9E75", linestyle="--")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Distillation loss")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(df["epoch"], df["eval_accuracy"], color="#7F77DD", linewidth=2, marker="o")
        if self._history.baseline_accuracy is not None:
            ax2.axhline(
                self._history.baseline_accuracy,
                color="gray", linestyle="--", linewidth=1,
                label=f"Baseline {self._history.baseline_accuracy:.2f}%",
            )
            ax2.legend()
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Eval accuracy  (T={self._history.temperature}, α={self._history.alpha})")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Knowledge distillation training", fontsize=12, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"History plot saved to: {save_path}")

        return fig


# ---------------------------------------------------------------------------
# Temperature sweep
# ---------------------------------------------------------------------------

def temperature_sweep(
    teacher: nn.Module,
    student_factory: callable,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    temperatures: list[float] | None = None,
    alpha: float = 0.7,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    baseline_accuracy: float | None = None,
) -> pd.DataFrame:
    """
    Train with multiple temperature values and compare recovery.

    This is the experiment that shows hiring managers you understand
    *why* temperature matters — not just that you used a library.

    Parameters
    ----------
    teacher          : Frozen FP32 teacher model.
    student_factory  : Callable() → fresh student model (called per temperature).
                       Must return a new, untrained copy each time.
    temperatures     : List of T values to sweep. Default [2, 4, 8].
    alpha            : Soft-loss weight (fixed across sweep).
    epochs           : Training epochs per temperature.
    baseline_accuracy: Student accuracy before any distillation.

    Returns
    -------
    df : DataFrame with temperature, best_accuracy, accuracy_recovery per run.
    """
    if temperatures is None:
        temperatures = [2.0, 4.0, 8.0]

    rows = []
    print(f"\nTemperature sweep: T ∈ {temperatures}")
    print(f"{'='*65}\n")

    for T in temperatures:
        print(f"\n--- Temperature T={T} ---")
        student = student_factory()

        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            device=device,
            temperature=T,
            alpha=alpha,
            learning_rate=learning_rate,
        )
        history = trainer.train(
            train_loader, eval_loader, epochs=epochs,
            baseline_accuracy=baseline_accuracy,
        )

        rows.append({
            "temperature": T,
            "best_accuracy": history.best_accuracy,
            "accuracy_recovery": history.accuracy_recovery,
            "final_loss": history.steps[-1].train_loss if history.steps else None,
        })

    df = pd.DataFrame(rows).sort_values("best_accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "="*50)
    print("Temperature sweep results:")
    print(df.to_string(index=False))
    print("="*50)
    print(f"\nBest temperature: T={df.iloc[0]['temperature']}  →  {df.iloc[0]['best_accuracy']:.2f}%")

    return df


def plot_temperature_sweep(
    sweep_df: pd.DataFrame,
    baseline_accuracy: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing accuracy recovery across temperature values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#3B8BD4" if v != sweep_df["best_accuracy"].max() else "#E85D24"
              for v in sweep_df["best_accuracy"]]

    bars = ax.bar(
        [f"T={t}" for t in sweep_df["temperature"]],
        sweep_df["best_accuracy"],
        color=colors,
        width=0.5,
    )

    for bar, val in zip(bars, sweep_df["best_accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}%",
            ha="center", fontsize=9,
        )

    if baseline_accuracy is not None:
        ax.axhline(
            baseline_accuracy, color="gray", linestyle="--", linewidth=1.2,
            label=f"Baseline (no distillation): {baseline_accuracy:.2f}%",
        )
        ax.legend(fontsize=9)

    ax.set_ylabel("Eval accuracy (%)", fontsize=11)
    ax.set_title("Effect of distillation temperature on accuracy recovery", fontsize=11)
    ax.set_ylim(bottom=max(0, (baseline_accuracy or 80) - 2))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Temperature sweep chart saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_batch(
    batch: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack a training batch into (input_ids, attention_mask, labels).
    Handles both 3-element (input_ids, mask, labels) and 2-element (input_ids, labels).
    """
    if len(batch) == 3:
        input_ids, attention_mask, labels = batch
    elif len(batch) == 2:
        input_ids, labels = batch
        attention_mask = torch.ones_like(input_ids)
    else:
        raise ValueError(f"Expected batch of 2 or 3 elements, got {len(batch)}")

    return (
        input_ids.to(device),
        attention_mask.to(device),
        labels.to(device),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str | None,
    total_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if scheduler_type is None:
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}. Use 'cosine', 'linear', or None.")

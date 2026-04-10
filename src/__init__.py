"""
llm-compression-lab/src
-----------------------
Reusable modules for the LLM compression and fine-tuning pipeline.

    from src.sensitivity     import SensitivityAnalyzer
    from src.mixed_precision import MixedPrecisionPolicy, Precision
    from src.benchmark       import ModelBenchmark, plot_activation_histograms
    from src.distillation    import DistillationTrainer, temperature_sweep
"""

from src.sensitivity      import SensitivityAnalyzer, evaluate_accuracy
from src.mixed_precision  import MixedPrecisionPolicy, Precision, LayerAssignment
from src.benchmark        import ModelBenchmark, plot_activation_histograms
from src.distillation     import DistillationTrainer, temperature_sweep, plot_temperature_sweep

__all__ = [
    "SensitivityAnalyzer",
    "evaluate_accuracy",
    "MixedPrecisionPolicy",
    "Precision",
    "LayerAssignment",
    "ModelBenchmark",
    "plot_activation_histograms",
    "DistillationTrainer",
    "temperature_sweep",
    "plot_temperature_sweep",
]

from wakeUp.eval.metrics import (
    evaluate_scores,
    per_attack_metrics,
    fpr_at_recall,
)
from wakeUp.eval.plots import (
    plot_pr_curves,
    plot_robustness_curves,
    plot_score_hist,
    plot_window_example,
)
from wakeUp.eval.splits import (
    train_test_split_by_vessel,
    window_labels,
)
from wakeUp.eval.robustness import (
    DEFAULT_SWEEPS,
    SWEEP_PARAM,
    run_robustness_sweeps,
    sweep_attack_severity,
)
from wakeUp.eval.ablation import (
    ablation_cells,
    run_ablation_sweeps,
    run_ablation_table,
)
from wakeUp.eval.latency import (
    clean_fpr_threshold,
    detection_latency,
    run_detection_latency,
    summarize_latency,
)
from wakeUp.eval.dashboard import (
    build_payload,
    render_dashboard,
    write_dashboard,
)

__all__ = [
    "evaluate_scores",
    "per_attack_metrics",
    "fpr_at_recall",
    "plot_pr_curves",
    "plot_robustness_curves",
    "plot_score_hist",
    "plot_window_example",
    "DEFAULT_SWEEPS",
    "SWEEP_PARAM",
    "run_robustness_sweeps",
    "sweep_attack_severity",
    "train_test_split_by_vessel",
    "window_labels",
    "ablation_cells",
    "run_ablation_sweeps",
    "run_ablation_table",
    "clean_fpr_threshold",
    "detection_latency",
    "run_detection_latency",
    "summarize_latency",
    "build_payload",
    "render_dashboard",
    "write_dashboard",
]

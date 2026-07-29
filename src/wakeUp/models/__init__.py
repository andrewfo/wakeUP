from wakeUp.models.baselines import (
    KinematicRuleDetector,
    IsolationForestDetector,
    LogisticFeatureDetector,
)

__all__ = [
    "KinematicRuleDetector",
    "IsolationForestDetector",
    "LogisticFeatureDetector",
    "LSTMAutoencoderDetector",
    "TransformerDetector",
    "ReconTransformerDetector",
    "HybridDetector",
]


def __getattr__(name):
    """Import the torch-backed detectors lazily.

    Keeps ``import wakeUp.models`` working (and the milestone slice runnable)
    when the optional ``learned`` extra is not installed; asking for the
    detector by name raises the actionable ImportError from its module.
    """
    if name == "LSTMAutoencoderDetector":
        from wakeUp.models.sequence_ae import LSTMAutoencoderDetector

        return LSTMAutoencoderDetector
    if name == "TransformerDetector":
        from wakeUp.models.transformer import TransformerDetector

        return TransformerDetector
    if name == "ReconTransformerDetector":
        from wakeUp.models.transformer import ReconTransformerDetector

        return ReconTransformerDetector
    if name == "HybridDetector":
        from wakeUp.models.transformer import HybridDetector

        return HybridDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

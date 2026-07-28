from wakeUp.models.baselines import (
    KinematicRuleDetector,
    IsolationForestDetector,
)

__all__ = [
    "KinematicRuleDetector",
    "IsolationForestDetector",
    "LSTMAutoencoderDetector",
    "TransformerDetector",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

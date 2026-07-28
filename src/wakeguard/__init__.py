"""wakeguard — AIS spoofing / anomaly detection benchmark.

Package layout:
    data/      ingest, clean, resample, window synthetic + real AIS
    attacks/   labeled synthetic attack injectors (eval backbone)
    features/  kinematic-consistency feature extraction
    models/    baselines (rules, IsolationForest, LSTM-AE) + main models
    eval/      metrics + figure generation
"""

__version__ = "0.1.0"

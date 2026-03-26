"""
blade.core.deviation
====================
Deviation vector assembly and anomaly classification for BLADE.

DeviationVector  : assembles 5-stream MAD scores into a labeled vector
AnomalyClassifier: lightweight decision-tree-based classifier
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Stream names (canonical ordering)
# ---------------------------------------------------------------------------

STREAM_NAMES = ["fp", "mpi", "io", "power", "checkpoint"]
ANOMALY_TYPES = [
    "normal",
    "sdc",
    "network_attack",
    "thermal_fault",
    "filesystem_corruption",
    "rank_imbalance",
    "checkpoint_corruption",
    "rogue_process",
]


# ---------------------------------------------------------------------------
# Deviation vector
# ---------------------------------------------------------------------------

@dataclass
class DeviationVector:
    """
    5-element vector of per-stream Benford MAD scores.

    Each element in [0, 1]:
      0 = perfect Benford conformance
      1 = maximum deviation
    """
    fp:         float = 0.0
    mpi:        float = 0.0
    io:         float = 0.0
    power:      float = 0.0
    checkpoint: float = 0.0
    timestamp:  Optional[float] = None
    label:      Optional[str] = None  # ground truth (for evaluation)
    job_id:     Optional[str] = None
    workload:   Optional[str] = None

    def as_array(self) -> np.ndarray:
        return np.array([self.fp, self.mpi, self.io, self.power, self.checkpoint])

    def to_dict(self) -> dict:
        return {
            "fp": self.fp, "mpi": self.mpi, "io": self.io,
            "power": self.power, "checkpoint": self.checkpoint,
            "timestamp": self.timestamp, "label": self.label,
            "job_id": self.job_id, "workload": self.workload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeviationVector":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_array(cls, arr: np.ndarray, **kwargs) -> "DeviationVector":
        return cls(fp=arr[0], mpi=arr[1], io=arr[2], power=arr[3],
                   checkpoint=arr[4], **kwargs)

    def l2_norm(self) -> float:
        return float(np.linalg.norm(self.as_array()))

    def max_component(self) -> tuple[str, float]:
        arr = self.as_array()
        idx = np.argmax(arr)
        return (STREAM_NAMES[idx], float(arr[idx]))


# ---------------------------------------------------------------------------
# Rule-based classifier (decision tree logic)
# ---------------------------------------------------------------------------

# Signature matrix: expected deviation level per (anomaly, stream)
# 0=none, 1=moderate (MAD 0.01-0.02), 2=strong (MAD >0.02)
SIGNATURE_MATRIX = {
    #              fp   mpi   io   pwr   ckpt
    "sdc":                  [2,   0,    0,   0,    2],
    "network_attack":       [0,   2,    1,   1,    0],
    "thermal_fault":        [1,   0,    0,   2,    0],
    "filesystem_corruption":[0,   0,    2,   0,    1],
    "rank_imbalance":       [1,   2,    1,   1,    0],
    "checkpoint_corruption":[0,   0,    1,   0,    2],
    "rogue_process":        [1,   1,    1,   2,    0],
}

# MAD thresholds for level classification
LEVEL_NONE     = 0.010   # < this = no deviation
LEVEL_MODERATE = 0.020   # < this = moderate
# >= LEVEL_MODERATE = strong


class AnomalyClassifier:
    """
    Lightweight anomaly classifier operating on DeviationVector inputs.

    Two modes:
      1. Rule-based (default): matches observed deviation pattern against
         the signature matrix using cosine similarity.
      2. ML-trained: fits a decision tree or logistic regression on labeled
         training data (call .fit() with labeled DeviationVectors).

    Parameters
    ----------
    mode : str
        "rule" or "ml"
    threshold_none : float
        MAD below this = no deviation (per stream).
    threshold_moderate : float
        MAD in [threshold_none, threshold_moderate) = moderate.
    anomaly_threshold : float
        L2 norm of vector below which we declare "normal".
    """

    def __init__(self, mode: str = "rule",
                 threshold_none: float = LEVEL_NONE,
                 threshold_moderate: float = LEVEL_MODERATE,
                 anomaly_threshold: float = 0.060):
        self.mode = mode
        self.threshold_none = threshold_none
        self.threshold_moderate = threshold_moderate
        self.anomaly_threshold = anomaly_threshold
        self._ml_model = None
        self._training_data: list[DeviationVector] = []

    def _discretize(self, arr: np.ndarray) -> np.ndarray:
        """Convert continuous MAD values to 0/1/2 levels."""
        levels = np.zeros(len(arr), dtype=int)
        levels[arr >= self.threshold_none] = 1
        levels[arr >= self.threshold_moderate] = 2
        return levels

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def predict(self, dv: DeviationVector) -> dict:
        """
        Classify a deviation vector.

        Returns
        -------
        dict with keys: predicted_class, confidence, scores, normal
        """
        arr = dv.as_array()

        # Quick normal check
        if dv.l2_norm() < self.anomaly_threshold:
            return {
                "predicted_class": "normal",
                "confidence": 1.0 - dv.l2_norm() / self.anomaly_threshold,
                "scores": {a: 0.0 for a in ANOMALY_TYPES},
                "normal": True,
                "vector": arr.tolist(),
            }

        if self.mode == "rule":
            return self._rule_predict(arr)
        elif self.mode == "ml" and self._ml_model is not None:
            return self._ml_predict(arr)
        else:
            return self._rule_predict(arr)

    def _rule_predict(self, arr: np.ndarray) -> dict:
        levels = self._discretize(arr)
        scores = {}
        for anomaly, sig in SIGNATURE_MATRIX.items():
            sig_arr = np.array(sig, dtype=float)
            # Cosine similarity between observed levels and signature
            scores[anomaly] = self._cosine_similarity(levels.astype(float), sig_arr)

        scores["normal"] = 0.0
        best = max(scores, key=scores.get)
        confidence = scores[best]

        return {
            "predicted_class": best,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "normal": False,
            "vector": arr.tolist(),
            "levels": levels.tolist(),
        }

    def fit(self, training_vectors: list[DeviationVector]):
        """
        Fit an sklearn decision tree classifier on labeled training data.
        Falls back to rule-based if sklearn is unavailable.
        """
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            print("sklearn not available — staying in rule-based mode")
            return

        X = np.array([dv.as_array() for dv in training_vectors])
        y_raw = [dv.label or "normal" for dv in training_vectors]
        self._le = LabelEncoder().fit(ANOMALY_TYPES)
        y = self._le.transform(y_raw)
        self._ml_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3)
        self._ml_model.fit(X, y)
        self.mode = "ml"
        self._training_data = training_vectors
        print(f"Classifier trained on {len(X)} samples, {len(set(y_raw))} classes.")

    def _ml_predict(self, arr: np.ndarray) -> dict:
        proba = self._ml_model.predict_proba(arr.reshape(1, -1))[0]
        classes = self._le.classes_
        best_idx = np.argmax(proba)
        scores = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        return {
            "predicted_class": classes[best_idx],
            "confidence": round(float(proba[best_idx]), 4),
            "scores": scores,
            "normal": classes[best_idx] == "normal",
            "vector": arr.tolist(),
        }

    def evaluate(self, test_vectors: list[DeviationVector]) -> dict:
        """
        Compute accuracy metrics on a labeled test set.

        Returns
        -------
        dict with overall accuracy, per-class precision/recall, confusion matrix
        """
        y_true = []
        y_pred = []
        for dv in test_vectors:
            result = self.predict(dv)
            y_true.append(dv.label or "normal")
            y_pred.append(result["predicted_class"])

        classes = sorted(set(y_true + y_pred))
        n = len(y_true)
        correct = sum(t == p for t, p in zip(y_true, y_pred))

        # Per-class metrics
        per_class = {}
        for cls in classes:
            tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
            fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
            fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[cls] = {"tp": tp, "fp": fp, "fn": fn,
                               "precision": round(precision, 4),
                               "recall": round(recall, 4),
                               "f1": round(f1, 4)}

        # Confusion matrix
        confusion = {t: {p: 0 for p in classes} for t in classes}
        for t, p in zip(y_true, y_pred):
            confusion[t][p] += 1

        return {
            "n_samples": n,
            "overall_accuracy": round(correct / n, 4),
            "per_class": per_class,
            "confusion_matrix": confusion,
            "classes": classes,
        }

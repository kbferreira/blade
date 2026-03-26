"""
blade.core.benford
==================
Core Benford's Law statistical primitives for the BLADE framework.

Provides:
  - expected_distribution()  : theoretical Benford probabilities
  - leading_digit()          : vectorized leading-digit extraction
  - observed_distribution()  : observed digit frequencies from a sample
  - mad_score()              : Mean Absolute Deviation conformance score
  - ks_score()               : Kolmogorov-Smirnov statistic
  - chi2_score()             : chi-squared goodness-of-fit
  - conformance_report()     : full report dict for a sample
  - RollingMAD               : online O(1) streaming MAD estimator
"""

import numpy as np
from scipy import stats
import json


# ---------------------------------------------------------------------------
# Theoretical distribution
# ---------------------------------------------------------------------------

DIGITS = np.arange(1, 10)  # digits 1..9

BENFORD_EXPECTED = np.log10(1.0 + 1.0 / DIGITS)  # P(d) = log10(1 + 1/d)


def expected_distribution() -> np.ndarray:
    """Return the 9-element Benford expected probability vector (digits 1–9)."""
    return BENFORD_EXPECTED.copy()


# ---------------------------------------------------------------------------
# Leading digit extraction
# ---------------------------------------------------------------------------

def leading_digit(values: np.ndarray) -> np.ndarray:
    """
    Extract the leading (most significant) decimal digit from each value.

    Parameters
    ----------
    values : array-like of float
        Input values. Zeros, NaNs, and infs are filtered out.

    Returns
    -------
    digits : np.ndarray of int, shape (N,)
        Leading digits in range [1, 9].
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v) & (v != 0.0)]
    v = np.abs(v)
    # Shift to [1, 10) range via log10
    exponents = np.floor(np.log10(v))
    normalized = v / (10.0 ** exponents)
    digits = np.floor(normalized).astype(int)
    # Clamp edge cases
    digits = np.clip(digits, 1, 9)
    return digits


# ---------------------------------------------------------------------------
# Observed distribution
# ---------------------------------------------------------------------------

def observed_distribution(values: np.ndarray) -> np.ndarray:
    """
    Compute observed leading-digit frequency vector from sample values.

    Returns
    -------
    freqs : np.ndarray, shape (9,)
        Relative frequencies for digits 1–9. Sums to 1.0.
        Returns uniform distribution if sample is empty.
    """
    digits = leading_digit(values)
    if len(digits) == 0:
        return np.ones(9) / 9.0
    counts = np.array([(digits == d).sum() for d in DIGITS], dtype=float)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Conformance statistics
# ---------------------------------------------------------------------------

def mad_score(values: np.ndarray, baseline: np.ndarray = None) -> float:
    """
    Mean Absolute Deviation between observed and expected (or baseline) Benford distribution.

    MAD = (1/9) * sum_d |observed(d) - expected(d)|

    Interpretation thresholds (Nigrini 2012):
      0.000 – 0.006  : close conformance
      0.006 – 0.012  : acceptable conformance
      0.012 – 0.015  : marginally acceptable
      > 0.015        : non-conformance

    Parameters
    ----------
    values   : sample data
    baseline : optional workload-specific expected distribution (9-element).
               If None, uses universal Benford distribution.

    Returns
    -------
    mad : float in [0, 1]
    """
    obs = observed_distribution(values)
    exp = baseline if baseline is not None else BENFORD_EXPECTED
    return float(np.mean(np.abs(obs - exp)))


def ks_score(values: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test against Benford CDF.

    Returns
    -------
    (statistic, p_value)
    """
    digits = leading_digit(values)
    if len(digits) == 0:
        return (1.0, 0.0)
    observed_cdf = np.array([(digits <= d).mean() for d in DIGITS])
    expected_cdf = np.cumsum(BENFORD_EXPECTED)
    ks_stat = float(np.max(np.abs(observed_cdf - expected_cdf)))
    # Approximate p-value via scipy
    result = stats.ks_1samp(digits, lambda x: np.interp(x, DIGITS, expected_cdf))
    return (float(result.statistic), float(result.pvalue))


def chi2_score(values: np.ndarray) -> tuple[float, float]:
    """
    Chi-squared goodness-of-fit test against Benford distribution.

    Returns
    -------
    (statistic, p_value)
    """
    digits = leading_digit(values)
    n = len(digits)
    if n == 0:
        return (np.inf, 0.0)
    observed_counts = np.array([(digits == d).sum() for d in DIGITS], dtype=float)
    expected_counts = BENFORD_EXPECTED * n
    # Avoid zero expected cells
    mask = expected_counts > 0
    chi2 = float(np.sum((observed_counts[mask] - expected_counts[mask])**2
                        / expected_counts[mask]))
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=mask.sum() - 1))
    return (chi2, p_value)


# ---------------------------------------------------------------------------
# Full conformance report
# ---------------------------------------------------------------------------

def conformance_report(values: np.ndarray,
                       label: str = "unnamed",
                       baseline: np.ndarray = None) -> dict:
    """
    Generate a complete Benford conformance report for a sample.

    Returns
    -------
    dict with keys:
      label, n_samples, n_valid, mad, ks_stat, ks_pvalue,
      chi2_stat, chi2_pvalue, observed_freq, expected_freq,
      conformant (bool), deviation_level (str)
    """
    all_values = np.asarray(values, dtype=float).ravel()
    valid = all_values[np.isfinite(all_values) & (all_values != 0.0)]
    n_valid = len(valid)

    mad = mad_score(valid, baseline)
    ks_stat, ks_p = ks_score(valid)
    chi2_stat, chi2_p = chi2_score(valid)
    obs = observed_distribution(valid).tolist()
    exp = (baseline if baseline is not None else BENFORD_EXPECTED).tolist()

    if mad <= 0.006:
        level = "close"
    elif mad <= 0.012:
        level = "acceptable"
    elif mad <= 0.015:
        level = "marginal"
    else:
        level = "non-conformant"

    return {
        "label": label,
        "n_samples": len(all_values),
        "n_valid": n_valid,
        "mad": round(mad, 6),
        "ks_stat": round(ks_stat, 6),
        "ks_pvalue": round(ks_p, 6),
        "chi2_stat": round(chi2_stat, 4),
        "chi2_pvalue": round(chi2_p, 6),
        "observed_freq": [round(x, 6) for x in obs],
        "expected_freq": [round(x, 6) for x in exp],
        "conformant": mad <= 0.015,
        "deviation_level": level,
    }


# ---------------------------------------------------------------------------
# Online rolling MAD estimator
# ---------------------------------------------------------------------------

class RollingMAD:
    """
    O(1) per-sample streaming Benford MAD estimator.

    Maintains a leading-digit histogram in a fixed 9-element array.
    At each flush interval, computes MAD against the expected (or baseline)
    distribution and resets (or slides, if window_size is set).

    Parameters
    ----------
    window_size : int or None
        Number of samples per MAD computation window.
        If None, accumulates indefinitely until reset() is called.
    baseline : np.ndarray or None
        Workload-specific expected distribution. Defaults to Benford.
    """

    def __init__(self, window_size: int = None, baseline: np.ndarray = None):
        self.window_size = window_size
        self.baseline = baseline if baseline is not None else BENFORD_EXPECTED.copy()
        self._counts = np.zeros(9, dtype=np.int64)  # counts[i] = count of digit (i+1)
        self._total = 0
        self._history: list[dict] = []  # timestamped MAD scores

    def update(self, value: float) -> float | None:
        """
        Add a single value and return MAD if window is complete, else None.

        Parameters
        ----------
        value : float

        Returns
        -------
        mad : float or None
        """
        if not np.isfinite(value) or value == 0.0:
            return None
        v = abs(value)
        exp = np.floor(np.log10(v))
        d = int(np.clip(np.floor(v / 10.0**exp), 1, 9))
        self._counts[d - 1] += 1
        self._total += 1

        if self.window_size and self._total >= self.window_size:
            mad = self._compute_mad()
            self._history.append({"n": self._total, "mad": mad})
            self._counts[:] = 0
            self._total = 0
            return mad
        return None

    def update_batch(self, values: np.ndarray) -> list[float]:
        """
        Add a batch of values. Returns list of MAD scores (one per completed window).
        """
        mads = []
        digits = leading_digit(np.asarray(values, dtype=float))
        for d in digits:
            self._counts[d - 1] += 1
            self._total += 1
            if self.window_size and self._total >= self.window_size:
                mad = self._compute_mad()
                self._history.append({"n": self._total, "mad": mad})
                mads.append(mad)
                self._counts[:] = 0
                self._total = 0
        return mads

    def flush(self) -> float:
        """Force a MAD computation from accumulated samples. Does not reset.
        Returns 0.0 if no samples have been accumulated."""
        if self._total == 0:
            return 0.0
        return self._compute_mad()

    def reset(self):
        """Clear accumulated state."""
        self._counts[:] = 0
        self._total = 0

    def _compute_mad(self) -> float:
        if self._total == 0:
            return 0.0
        obs = self._counts / self._total
        return float(np.mean(np.abs(obs - self.baseline)))

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def current_count(self) -> int:
        return self._total

    def snapshot(self) -> dict:
        return {
            "total": self._total,
            "counts": self._counts.tolist(),
            "current_mad": self.flush(),
            "history_len": len(self._history),
        }

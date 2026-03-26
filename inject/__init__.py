"""
blade.inject
============
Fault injection library for the BLADE test framework.

Strategy: All injectors replace a `severity` fraction of stream values
with digit-9 concentrated values. This GUARANTEES a positive MAD delta
for all five streams regardless of their baseline Benford distribution,
because digit-9 (Benford expected: 4.6%) is universally under-represented
and injecting it always shifts observed frequencies away from Benford.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core anti-Benford generator (digit-9 concentration)
# ---------------------------------------------------------------------------

def _d9_values(n: int, rng: np.random.Generator,
               min_exp: int = -8, max_exp: int = 8) -> np.ndarray:
    """
    Generate n values all starting with digit 9, spanning many orders of
    magnitude. MAD-delta is guaranteed positive for any stream baseline.
    """
    exponents = rng.integers(min_exp, max_exp + 1, size=n)
    frac = rng.uniform(0.0, 0.999, size=n)
    return (9.0 + frac) * (10.0 ** exponents.astype(float))


def _d9_pos(n: int, rng: np.random.Generator) -> np.ndarray:
    """Digit-9 values all >= 9 (for MPI/IO streams requiring positive integers)."""
    return _d9_values(n, rng, min_exp=0, max_exp=7)


def _replace_fraction(arr: np.ndarray, fraction: float,
                      replacements: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Replace `fraction` of arr values with replacements in-place copy."""
    out = arr.copy()
    n = max(1, int(len(out) * fraction))
    n = min(n, len(replacements))
    idx = rng.choice(len(out), size=n, replace=False)
    out[idx] = replacements[:n]
    return out


# ---------------------------------------------------------------------------
# SDC  (FP primary, checkpoint secondary)
# ---------------------------------------------------------------------------

def inject_sdc(fp_values, checkpoint_values, severity=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_fp   = max(1, int(len(fp_values)   * severity))
    n_ck   = max(1, int(len(checkpoint_values) * severity))
    fp_out   = _replace_fraction(fp_values,   severity, _d9_values(n_fp, rng), rng)
    ckpt_out = _replace_fraction(checkpoint_values, severity, _d9_values(n_ck, rng), rng)
    return fp_out, ckpt_out


# ---------------------------------------------------------------------------
# Network attack  (MPI primary, IO secondary, power tertiary)
# ---------------------------------------------------------------------------

def inject_network_attack(mpi_values, io_values, power_values,
                          severity=0.15, seed=1):
    rng = np.random.default_rng(seed)
    n_mpi = max(1, int(len(mpi_values) * severity))
    n_io  = max(1, int(len(io_values)  * severity * 0.5))
    n_pwr = max(1, int(len(power_values) * severity * 0.25))
    mpi_out = _replace_fraction(mpi_values,   severity,       _d9_pos(n_mpi, rng), rng)
    io_out  = _replace_fraction(io_values,    severity * 0.5, _d9_pos(n_io,  rng), rng)
    pwr_out = _replace_fraction(power_values, severity * 0.25, _d9_pos(n_pwr, rng), rng)
    return mpi_out, io_out, pwr_out


# ---------------------------------------------------------------------------
# Thermal fault  (power primary, FP secondary)
# ---------------------------------------------------------------------------

def inject_thermal_fault(power_values, fp_values, severity=0.20, seed=2):
    rng = np.random.default_rng(seed)
    n_pwr = max(1, int(len(power_values) * severity))
    n_fp  = max(1, int(len(fp_values)    * severity * 0.15))
    pwr_out = _replace_fraction(power_values, severity,       _d9_pos(n_pwr, rng), rng)
    fp_out  = _replace_fraction(fp_values,    severity * 0.15, _d9_values(n_fp, rng), rng)
    return pwr_out, fp_out


# ---------------------------------------------------------------------------
# Filesystem corruption  (IO primary, checkpoint secondary)
# ---------------------------------------------------------------------------

def inject_filesystem_corruption(io_values, checkpoint_values,
                                 severity=0.10, seed=3):
    rng = np.random.default_rng(seed)
    n_io   = max(1, int(len(io_values)         * severity))
    n_ck   = max(1, int(len(checkpoint_values) * severity * 0.5))
    io_out   = _replace_fraction(io_values,         severity,       _d9_pos(n_io, rng), rng)
    ckpt_out = _replace_fraction(checkpoint_values, severity * 0.5, _d9_values(n_ck, rng), rng)
    return io_out, ckpt_out


# ---------------------------------------------------------------------------
# Rank imbalance  (MPI primary, IO/FP/power secondary)
# ---------------------------------------------------------------------------

def inject_rank_imbalance(mpi_values, io_values, fp_values, power_values,
                          severity=0.25, n_overloaded_ranks=4, seed=4):
    rng = np.random.default_rng(seed)
    n_mpi = max(1, int(len(mpi_values)   * severity))
    n_io  = max(1, int(len(io_values)    * severity * 0.5))
    n_fp  = max(1, int(len(fp_values)    * severity * 0.2))
    n_pwr = max(1, int(len(power_values) * severity * 0.4))
    mpi_out = _replace_fraction(mpi_values,   severity,       _d9_pos(n_mpi, rng), rng)
    io_out  = _replace_fraction(io_values,    severity * 0.5, _d9_pos(n_io,  rng), rng)
    fp_out  = _replace_fraction(fp_values,    severity * 0.2, _d9_values(n_fp, rng), rng)
    pwr_out = _replace_fraction(power_values, severity * 0.4, _d9_pos(n_pwr, rng), rng)
    return mpi_out, io_out, fp_out, pwr_out


# ---------------------------------------------------------------------------
# Checkpoint corruption  (checkpoint primary, IO secondary)
# ---------------------------------------------------------------------------

def inject_checkpoint_corruption(checkpoint_values, io_values,
                                 severity=0.08, seed=5):
    rng = np.random.default_rng(seed)
    n_ck = max(1, int(len(checkpoint_values) * severity))
    n_io = max(1, int(len(io_values)         * severity * 0.4))
    ckpt_out = _replace_fraction(checkpoint_values, severity,       _d9_values(n_ck, rng), rng)
    io_out   = _replace_fraction(io_values,         severity * 0.4, _d9_pos(n_io,  rng), rng)
    return ckpt_out, io_out


# ---------------------------------------------------------------------------
# Rogue process  (power primary, all others secondary)
# ---------------------------------------------------------------------------

def inject_rogue_process(fp_values, mpi_values, io_values, power_values,
                         severity=0.15, seed=6):
    rng = np.random.default_rng(seed)
    n_pwr = max(1, int(len(power_values) * severity))
    n_fp  = max(1, int(len(fp_values)    * severity * 0.5))
    n_mpi = max(1, int(len(mpi_values)   * severity * 0.5))
    n_io  = max(1, int(len(io_values)    * severity * 0.5))
    fp_out  = _replace_fraction(fp_values,    severity * 0.5, _d9_values(n_fp,  rng), rng)
    mpi_out = _replace_fraction(mpi_values,   severity * 0.5, _d9_pos(n_mpi,    rng), rng)
    io_out  = _replace_fraction(io_values,    severity * 0.5, _d9_pos(n_io,     rng), rng)
    pwr_out = _replace_fraction(power_values, severity,       _d9_pos(n_pwr,    rng), rng)
    return fp_out, mpi_out, io_out, pwr_out


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def inject_fault(anomaly_type: str, stream_data: dict,
                 severity: float = 0.10, seed: int = 0) -> dict:
    """
    Dispatch fault injection for a given anomaly type.

    Parameters
    ----------
    anomaly_type : str
        One of: sdc, network_attack, thermal_fault, filesystem_corruption,
                rank_imbalance, checkpoint_corruption, rogue_process
    stream_data : dict  keys: fp, mpi, io, power, checkpoint
    severity    : float  fraction of values to corrupt
    seed        : int

    Returns
    -------
    dict with same keys, corrupted arrays where applicable
    """
    fp   = stream_data.get("fp",         np.array([1.0]))
    mpi  = stream_data.get("mpi",        np.array([1.0]))
    io   = stream_data.get("io",         np.array([1.0]))
    pwr  = stream_data.get("power",      np.array([1.0]))
    ckpt = stream_data.get("checkpoint", np.array([1.0]))

    result = {k: v.copy() for k, v in stream_data.items()}

    if anomaly_type == "sdc":
        result["fp"], result["checkpoint"] = inject_sdc(fp, ckpt, severity, seed)
    elif anomaly_type == "network_attack":
        result["mpi"], result["io"], result["power"] = \
            inject_network_attack(mpi, io, pwr, severity, seed)
    elif anomaly_type == "thermal_fault":
        result["power"], result["fp"] = inject_thermal_fault(pwr, fp, severity, seed)
    elif anomaly_type == "filesystem_corruption":
        result["io"], result["checkpoint"] = \
            inject_filesystem_corruption(io, ckpt, severity, seed)
    elif anomaly_type == "rank_imbalance":
        result["mpi"], result["io"], result["fp"], result["power"] = \
            inject_rank_imbalance(mpi, io, fp, pwr, severity, seed=seed)
    elif anomaly_type == "checkpoint_corruption":
        result["checkpoint"], result["io"] = \
            inject_checkpoint_corruption(ckpt, io, severity, seed)
    elif anomaly_type == "rogue_process":
        result["fp"], result["mpi"], result["io"], result["power"] = \
            inject_rogue_process(fp, mpi, io, pwr, severity, seed)
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type!r}")

    return result

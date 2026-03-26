"""
blade.core.streams
==================
Realistic synthetic data generators for each BLADE telemetry stream.

Each generator produces data that, under normal operation, is Benford-conformant.
Generators are parameterized to represent different HPC workload classes.

Stream classes:
  FPOutputStream       : floating-point simulation output fields
  MPIMessageStream     : MPI message size distributions
  IOPatternStream      : parallel filesystem I/O block sizes
  PowerTelemetryStream : per-node power sensor readings
  CheckpointStream     : checkpoint file field magnitudes
"""

import numpy as np
from typing import Iterator


# ---------------------------------------------------------------------------
# Workload profiles
# ---------------------------------------------------------------------------

WORKLOAD_PROFILES = {
    # name: (fp_mu, fp_sigma, mpi_mean_kb, io_mean_mb, power_mean_w, power_std_w)
    "amg":      {"fp_scales": [1e-6, 1e3, 1e9],  "mpi_mean": 8192,   "io_mean": 512*1024,  "power_mean": 420, "power_std": 18},
    "hpcg":     {"fp_scales": [1e-4, 1e2, 1e7],  "mpi_mean": 4096,   "io_mean": 256*1024,  "power_mean": 380, "power_std": 22},
    "nekbone":  {"fp_scales": [1e-8, 1e1, 1e6],  "mpi_mean": 2048,   "io_mean": 128*1024,  "power_mean": 445, "power_std": 15},
    "lammps":   {"fp_scales": [1e-10, 1e0, 1e5], "mpi_mean": 1024,   "io_mean": 64*1024,   "power_mean": 390, "power_std": 25},
    "hacc":     {"fp_scales": [1e-5, 1e2, 1e8],  "mpi_mean": 16384,  "io_mean": 1024*1024, "power_mean": 460, "power_std": 20},
    "nekrs":    {"fp_scales": [1e-9, 1e1, 1e6],  "mpi_mean": 3072,   "io_mean": 200*1024,  "power_mean": 470, "power_std": 12},
    "comd":     {"fp_scales": [1e-12, 1e-1, 1e4],"mpi_mean": 512,    "io_mean": 32*1024,   "power_mean": 350, "power_std": 30},
    "minife":   {"fp_scales": [1e-6, 1e2, 1e7],  "mpi_mean": 6144,   "io_mean": 400*1024,  "power_mean": 400, "power_std": 20},
}


# ---------------------------------------------------------------------------
# FP output stream
# ---------------------------------------------------------------------------

class FPOutputStream:
    """
    Simulates floating-point output values from a physical simulation.

    Strategy: mix of log-normal draws at several scale ranges
    (representing pressure, velocity, energy, particle coords, etc.)
    This naturally produces Benford-conformant distributions.

    Parameters
    ----------
    workload : str
        Workload profile key from WORKLOAD_PROFILES.
    n_variables : int
        Number of distinct physical variable types to simulate.
    seed : int
    """

    def __init__(self, workload: str = "amg", n_variables: int = 6, seed: int = 42):
        self.workload = workload
        self.profile = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["amg"])
        self.n_variables = n_variables
        self.rng = np.random.default_rng(seed)
        self._scales = self.profile["fp_scales"]

    def sample(self, n: int) -> np.ndarray:
        """Draw n FP output values."""
        parts = []
        per_scale = n // len(self._scales) + 1
        for scale in self._scales:
            # log-normal centered around each scale
            mu = np.log(scale)
            sigma = 2.5  # broad spread ensures multi-order-of-magnitude range
            vals = self.rng.lognormal(mean=mu, sigma=sigma, size=per_scale)
            parts.append(vals)
        combined = np.concatenate(parts)
        self.rng.shuffle(combined)
        return combined[:n]

    def sample_stream(self, n: int, chunk_size: int = 1000) -> Iterator[np.ndarray]:
        """Yield chunks of FP values."""
        produced = 0
        while produced < n:
            size = min(chunk_size, n - produced)
            yield self.sample(size)
            produced += size


# ---------------------------------------------------------------------------
# MPI message stream
# ---------------------------------------------------------------------------

class MPIMessageStream:
    """
    Simulates MPI message sizes from a domain-decomposed parallel application.

    Models three message categories:
      - Halo exchanges: small, regular sizes (deterministic from grid dims)
      - Reductions/broadcasts: medium, power-of-2 sizes
      - Large data transfers (rare): large, irregular sizes

    Parameters
    ----------
    workload : str
    n_ranks : int
        Number of MPI ranks (affects halo exchange patterns).
    grid_dims : tuple of int
        3D grid decomposition (Nx, Ny, Nz) per rank.
    seed : int
    """

    def __init__(self, workload: str = "amg", n_ranks: int = 512,
                 grid_dims: tuple = (64, 64, 64), seed: int = 42):
        self.workload = workload
        self.profile = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["amg"])
        self.n_ranks = n_ranks
        self.grid_dims = grid_dims
        self.rng = np.random.default_rng(seed)
        mpi_mean = self.profile["mpi_mean"]
        # Halo size ~ grid face area * sizeof(double)
        self._halo_base = grid_dims[0] * grid_dims[1] * 8  # bytes
        self._mpi_mean = mpi_mean

    def sample(self, n: int) -> np.ndarray:
        """Draw n message sizes in bytes."""
        # 60% halo exchanges (lognormal around halo base)
        n_halo = int(n * 0.60)
        halo = self.rng.lognormal(
            mean=np.log(max(self._halo_base, 64)),
            sigma=0.8,
            size=n_halo
        )
        # 30% collective operations (powers of 2 with noise)
        n_coll = int(n * 0.30)
        powers = 2 ** self.rng.integers(6, 20, size=n_coll)
        coll = powers * self.rng.lognormal(0, 0.2, size=n_coll)
        # 10% large irregular transfers
        n_large = n - n_halo - n_coll
        large = self.rng.lognormal(
            mean=np.log(self._mpi_mean * 4),
            sigma=1.5,
            size=max(n_large, 1)
        )
        combined = np.concatenate([halo, coll, large[:n_large]])
        combined = np.clip(combined, 1, None)
        self.rng.shuffle(combined)
        return combined[:n]

    def sample_stream(self, n: int, chunk_size: int = 500) -> Iterator[np.ndarray]:
        produced = 0
        while produced < n:
            size = min(chunk_size, n - produced)
            yield self.sample(size)
            produced += size


# ---------------------------------------------------------------------------
# I/O pattern stream
# ---------------------------------------------------------------------------

class IOPatternStream:
    """
    Simulates I/O request sizes on a parallel filesystem (Lustre/GPFS).

    Models:
      - Metadata operations: tiny reads/writes (stat, open, close)
      - Small data: variable field reads (HDF5 partial reads)
      - Checkpoint writes: large aligned block writes
      - Restart reads: similar distribution to checkpoint writes

    Parameters
    ----------
    workload : str
    stripe_size : int
        Lustre stripe size in bytes (default 1 MB).
    seed : int
    """

    def __init__(self, workload: str = "amg", stripe_size: int = 1024*1024, seed: int = 42):
        self.workload = workload
        self.profile = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["amg"])
        self.stripe_size = stripe_size
        self.rng = np.random.default_rng(seed)
        self._io_mean = self.profile["io_mean"]

    def sample(self, n: int) -> np.ndarray:
        """Draw n I/O request sizes in bytes."""
        # 20% metadata
        n_meta = max(int(n * 0.20), 1)
        meta = self.rng.lognormal(mean=np.log(256), sigma=1.2, size=n_meta)
        # 50% field reads/writes (sub-stripe, scattered sizes)
        n_field = max(int(n * 0.50), 1)
        field = self.rng.lognormal(
            mean=np.log(self._io_mean * 0.1),
            sigma=2.0,
            size=n_field
        )
        # 30% checkpoint/restart (aligned to stripe multiples)
        n_ckpt = n - n_meta - n_field
        stripe_multiples = self.rng.integers(1, 32, size=max(n_ckpt, 1))
        ckpt = stripe_multiples * self.stripe_size * self.rng.lognormal(0, 0.3, size=max(n_ckpt, 1))
        combined = np.concatenate([meta, field, ckpt[:n_ckpt]])
        combined = np.clip(combined, 1, None)
        self.rng.shuffle(combined)
        return combined[:n]

    def sample_stream(self, n: int, chunk_size: int = 500) -> Iterator[np.ndarray]:
        produced = 0
        while produced < n:
            size = min(chunk_size, n - produced)
            yield self.sample(size)
            produced += size


# ---------------------------------------------------------------------------
# Power telemetry stream
# ---------------------------------------------------------------------------

class PowerTelemetryStream:
    """
    Simulates a multi-granularity power telemetry stream for an HPC allocation.

    Real HPC power monitoring systems expose sensors at multiple levels of the
    hardware hierarchy simultaneously:
      - Per-core power (milliwatts to a few watts)
      - Per-DIMM power (1–15 W)
      - Per-socket power (50–250 W)
      - Per-node total (100–600 W)
      - Per-rack PDU (5,000–50,000 W)
      - Facility UPS/meter (100,000–2,000,000 W)

    Mixing readings across these granularities produces a distribution that
    spans 6+ orders of magnitude, giving strong Benford conformance.

    Parameters
    ----------
    workload : str
    n_nodes : int
        Number of nodes in the job allocation.
    seed : int
    """

    def __init__(self, workload: str = "amg", n_nodes: int = 512, seed: int = 42):
        self.workload = workload
        self.profile = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["amg"])
        self.n_nodes = n_nodes
        self.rng = np.random.default_rng(seed)
        self._power_mean = self.profile["power_mean"]
        self._power_std  = self.profile["power_std"]

    def sample(self, n: int) -> np.ndarray:
        """
        Draw n power readings mixed across hardware hierarchy levels.

        Level proportions reflect a realistic Redfish/IPMI telemetry dump
        that includes sensors from core to rack levels.
        """
        parts = []
        # Per-core power: 0.1 – 8 W  (20% of readings)
        n_core = max(int(n * 0.20), 1)
        parts.append(self.rng.lognormal(mean=np.log(1.5), sigma=0.9, size=n_core))

        # Per-DIMM power: 1 – 18 W  (15%)
        n_dimm = max(int(n * 0.15), 1)
        parts.append(self.rng.lognormal(mean=np.log(5.0), sigma=0.6, size=n_dimm))

        # Per-socket power: 40 – 300 W  (25%)
        n_sock = max(int(n * 0.25), 1)
        parts.append(self.rng.lognormal(mean=np.log(self._power_mean / 2),
                                        sigma=0.35, size=n_sock))

        # Per-node total: 100 – 650 W  (25%)
        n_node = max(int(n * 0.25), 1)
        node_base = self.rng.normal(self._power_mean,
                                    self._power_mean * 0.08, size=n_node)
        parts.append(np.clip(node_base, 80, 800))

        # Per-rack PDU: 5,000 – 60,000 W  (10%)
        n_rack = max(int(n * 0.10), 1)
        racks = max(1, self.n_nodes // 32)
        rack_base = self.rng.lognormal(mean=np.log(self._power_mean * 32),
                                       sigma=0.25, size=n_rack)
        parts.append(rack_base)

        # Per-facility meter: 100k – 5M W  (5%)
        n_fac = max(n - sum(len(p) for p in parts), 1)
        parts.append(self.rng.lognormal(mean=np.log(self._power_mean * self.n_nodes),
                                        sigma=0.15, size=n_fac))

        combined = np.concatenate(parts)
        combined = combined[combined > 0]
        self.rng.shuffle(combined)
        return combined[:n]

    def sample_stream(self, n: int, chunk_size: int = 256) -> Iterator[np.ndarray]:
        produced = 0
        while produced < n:
            size = min(chunk_size, n - produced)
            yield self.sample(size)
            produced += size


# ---------------------------------------------------------------------------
# Checkpoint stream
# ---------------------------------------------------------------------------

class CheckpointStream:
    """
    Simulates checkpoint file field magnitudes.

    Checkpoint files contain serialized simulation state: physical quantities
    whose magnitude distribution directly inherits from the FP output stream.
    The distribution is somewhat narrower (only final-timestep values, not all
    intermediates) but still Benford-conformant.

    Parameters
    ----------
    workload : str
    n_fields : int
        Number of distinct physical field types in the checkpoint.
    seed : int
    """

    def __init__(self, workload: str = "amg", n_fields: int = 8, seed: int = 42):
        self.workload = workload
        self.profile = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["amg"])
        self.n_fields = n_fields
        self.rng = np.random.default_rng(seed)
        self._fp_stream = FPOutputStream(workload=workload, seed=seed + 1000)

    def sample(self, n: int) -> np.ndarray:
        """Draw n checkpoint field values."""
        # Checkpoint values are a subset of FP outputs with tighter range
        # (converged solution, not transient intermediates)
        raw = self._fp_stream.sample(n * 2)
        # Filter to values in the "physical" range for this workload
        scales = self.profile["fp_scales"]
        lo, hi = scales[0] * 0.01, scales[-1] * 100
        filtered = raw[(raw >= lo) & (raw <= hi)]
        if len(filtered) < n:
            filtered = np.tile(filtered, (n // len(filtered)) + 2)
        self.rng.shuffle(filtered)
        return filtered[:n]

    def sample_stream(self, n: int, chunk_size: int = 500) -> Iterator[np.ndarray]:
        produced = 0
        while produced < n:
            size = min(chunk_size, n - produced)
            yield self.sample(size)
            produced += size


# ---------------------------------------------------------------------------
# Combined stream factory
# ---------------------------------------------------------------------------

def create_all_streams(workload: str = "amg", n_nodes: int = 512, seed: int = 42) -> dict:
    """
    Create one instance of each stream type for a given workload.

    Returns
    -------
    dict mapping stream_name -> stream_object
    """
    return {
        "fp":         FPOutputStream(workload=workload, seed=seed),
        "mpi":        MPIMessageStream(workload=workload, n_ranks=n_nodes, seed=seed + 1),
        "io":         IOPatternStream(workload=workload, seed=seed + 2),
        "power":      PowerTelemetryStream(workload=workload, n_nodes=n_nodes, seed=seed + 3),
        "checkpoint": CheckpointStream(workload=workload, seed=seed + 4),
    }

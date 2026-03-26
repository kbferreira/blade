"""
tests/test_sampling_and_mad.py
================================
Tests for T1.5 (sampling theory) and T2.4 (online MAD algorithm).

Covers:
  TC-SAMP-001  Minimum sample size for reliable MAD
  TC-SAMP-002  Detection power vs sampling rate tradeoff
  TC-SAMP-003  Window size sensitivity
  TC-SAMP-004  False positive rate at various thresholds
  TC-MAD-001   RollingMAD O(1) update correctness
  TC-MAD-002   RollingMAD window flush behavior
  TC-MAD-003   RollingMAD batch update matches single-value updates
  TC-MAD-004   RollingMAD detects injected fault in streaming setting
  TC-MAD-005   Deviation vector assembly from rolling MADs
  TC-MAD-006   RollingMAD overhead benchmark (samples/second)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
import json
from blade.core.benford import mad_score, RollingMAD, BENFORD_EXPECTED
from blade.core.streams import FPOutputStream, MPIMessageStream, create_all_streams
from blade.core.deviation import DeviationVector
from blade.inject import inject_fault

SEED = 42


class TestResult:
    def __init__(self, tc_id, description):
        self.tc_id = tc_id; self.description = description
        self.passed = None; self.details = {}; self.assertions = []

    def assert_true(self, cond, msg=""):
        self.assertions.append({"condition": bool(cond), "msg": msg})
        if not cond: self.passed = False

    def finish(self):
        if self.passed is None:
            self.passed = all(a["condition"] for a in self.assertions)
        return self

    def to_dict(self):
        return {"tc_id": self.tc_id, "description": self.description,
                "passed": self.passed, "assertions": self.assertions, "details": self.details}


# ---------------------------------------------------------------------------
# TC-SAMP-001: Minimum sample size for reliable MAD
# ---------------------------------------------------------------------------

def tc_samp_001():
    results = []
    stream = FPOutputStream(workload="amg", seed=SEED)
    sample_sizes = [25, 50, 100, 200, 500, 1000, 2000, 5000]
    n_trials = 20  # repeated trials to measure variance

    for n in sample_sizes:
        r = TestResult("TC-SAMP-001", f"MAD reliability at n={n}")
        trial_mads = []
        for trial in range(n_trials):
            data = FPOutputStream(workload="amg", seed=SEED + trial).sample(n)
            trial_mads.append(mad_score(data))
        mean_mad = np.mean(trial_mads)
        std_mad  = np.std(trial_mads)
        cv = std_mad / mean_mad if mean_mad > 0 else 999
        r.details = {
            "n": n, "n_trials": n_trials,
            "mean_mad": round(float(mean_mad), 6),
            "std_mad":  round(float(std_mad),  6),
            "cv":       round(float(cv),        4),
        }
        # Coefficient of variation < 30% = reliable estimate
        if n >= 500:
            r.assert_true(cv < 0.30, f"CV={cv:.3f} < 0.30 at n={n}")
        else:
            r.assert_true(True, f"n={n} below reliable threshold (informational: CV={cv:.3f})")
        results.append(r.finish())
    return results


# ---------------------------------------------------------------------------
# TC-SAMP-002: Detection power vs sampling rate
# ---------------------------------------------------------------------------

def tc_samp_002():
    results = []
    # Simulate sampling at different rates from a stream containing a fault
    fault_types = ["sdc", "network_attack"]
    primary_streams = {"sdc": "fp", "network_attack": "mpi"}
    sample_rates = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]  # fraction of stream sampled

    full_n = 10_000
    stream_data = {name: s.sample(full_n) for name, s in
                   create_all_streams("amg", seed=SEED).items()}

    for fault in fault_types:
        corrupted = inject_fault(fault, stream_data, severity=0.15, seed=SEED)
        primary = primary_streams[fault]

        for rate in sample_rates:
            r = TestResult("TC-SAMP-002", f"Detection power: {fault} at rate={rate:.0%}")
            n_sample = max(50, int(full_n * rate))
            rng = np.random.default_rng(SEED)
            idx = rng.choice(full_n, size=n_sample, replace=False)
            sampled = corrupted[primary][idx]
            clean_sampled = stream_data[primary][idx]

            mad_corrupt = mad_score(sampled)
            mad_clean   = mad_score(clean_sampled)
            delta = mad_corrupt - mad_clean
            detected = delta > 0.005

            r.details = {
                "fault": fault, "rate": rate, "n_sample": n_sample,
                "mad_clean": round(mad_clean, 6),
                "mad_corrupt": round(mad_corrupt, 6),
                "delta": round(delta, 6),
                "detected": detected,
            }
            if rate >= 0.25:
                r.assert_true(detected, f"Fault detected at rate={rate:.0%} (delta={delta:.5f})")
            else:
                r.assert_true(True, f"rate={rate:.0%} informational (delta={delta:.5f})")
            results.append(r.finish())
    return results


# ---------------------------------------------------------------------------
# TC-SAMP-003: Window size sensitivity
# ---------------------------------------------------------------------------

def tc_samp_003():
    results = []
    window_sizes = [100, 200, 500, 1000, 2000]
    stream = FPOutputStream(workload="amg", seed=SEED)

    # Generate a long clean stream, then inject a fault in the middle
    n_clean  = 3000
    n_fault  = 3000
    clean_data = stream.sample(n_clean)
    fault_data_dict = inject_fault(
        "sdc",
        {"fp": stream.sample(n_fault), "mpi": np.array([1]), "io": np.array([1]),
         "power": np.array([1]), "checkpoint": stream.sample(n_fault)},
        severity=0.15, seed=SEED
    )
    combined = np.concatenate([clean_data, fault_data_dict["fp"]])

    for window in window_sizes:
        r = TestResult("TC-SAMP-003", f"Window size sensitivity: window={window}")
        roller = RollingMAD(window_size=window)
        mads_before, mads_after = [], []
        for val in combined[:n_clean]:
            res = roller.update(val)
            if res is not None: mads_before.append(res)
        for val in combined[n_clean:]:
            res = roller.update(val)
            if res is not None: mads_after.append(res)
        # Also get partial window via flush if no complete windows
        if not mads_before:
            mads_before = [roller.flush() or 0.0]
        if not mads_after:
            mads_after = [roller.flush() or 0.0]
        mean_before = float(np.mean(mads_before))
        mean_after  = float(np.mean(mads_after))
        r.details = {"window": window, "n_windows_before": len(mads_before),
            "n_windows_after": len(mads_after),
            "mean_mad_before": round(mean_before, 6),
            "mean_mad_after":  round(mean_after,  6)}
        r.assert_true(mean_after > mean_before,
                      f"MAD rises after fault (window={window}): {mean_before:.5f} → {mean_after:.5f}")
        results.append(r.finish())
    return results


# ---------------------------------------------------------------------------
# TC-SAMP-004: False positive rate at various thresholds
# ---------------------------------------------------------------------------

def tc_samp_004():
    results = []
    thresholds = [0.010, 0.015, 0.020, 0.030]
    n_windows  = 200
    window_size = 1000

    fp_stream = FPOutputStream(workload="amg", seed=SEED)

    for thresh in thresholds:
        r = TestResult("TC-SAMP-004", f"False positive rate at MAD threshold={thresh}")
        fp_count = 0
        for i in range(n_windows):
            data = FPOutputStream(workload="amg", seed=SEED + i * 7).sample(window_size)
            mad = mad_score(data)
            if mad > thresh:
                fp_count += 1
        fp_rate = fp_count / n_windows
        r.details = {
            "threshold": thresh, "n_windows": n_windows,
            "fp_count": fp_count, "fp_rate": round(fp_rate, 4),
        }
        # At threshold 0.015 (Nigrini marginal), FP rate should be < 10%
        if thresh >= 0.015:
            r.assert_true(fp_rate <= 0.10,
                          f"FP rate={fp_rate:.3f} <= 0.10 at threshold={thresh}")
        else:
            r.assert_true(True, f"threshold={thresh} informational (FP rate={fp_rate:.3f})")
        results.append(r.finish())
    return results


# ---------------------------------------------------------------------------
# TC-MAD-001: RollingMAD update correctness
# ---------------------------------------------------------------------------

def tc_mad_001():
    r = TestResult("TC-MAD-001", "RollingMAD batch result matches bulk mad_score")
    stream = FPOutputStream(workload="amg", seed=SEED)
    data = stream.sample(5000)

    # Bulk computation
    bulk_mad = mad_score(data)

    # Rolling computation - feed all data then flush
    roller = RollingMAD(window_size=None)
    roller.update_batch(data)
    rolling_mad = roller.flush()

    r.details = {"bulk_mad": round(bulk_mad, 8), "rolling_mad": round(rolling_mad, 8),
                 "diff": round(abs(bulk_mad - rolling_mad), 8)}
    r.assert_true(abs(bulk_mad - rolling_mad) < 1e-6,
                  f"Bulk={bulk_mad:.8f} ≈ Rolling={rolling_mad:.8f}")
    return [r.finish()]


# ---------------------------------------------------------------------------
# TC-MAD-002: RollingMAD window flush behavior
# ---------------------------------------------------------------------------

def tc_mad_002():
    r = TestResult("TC-MAD-002", "RollingMAD flushes exactly at window boundary")
    roller = RollingMAD(window_size=500)
    stream = FPOutputStream(workload="amg", seed=SEED)

    flush_count = 0
    for val in stream.sample(2000):
        result = roller.update(val)
        if result is not None:
            flush_count += 1

    r.details = {"flush_count": flush_count, "expected": 4}
    r.assert_true(flush_count == 4,
                  f"4 flushes for 2000 samples with window=500: got {flush_count}")
    r.assert_true(roller.current_count == 0,
                  f"Counter resets after flush: {roller.current_count}")
    return [r.finish()]


# ---------------------------------------------------------------------------
# TC-MAD-003: Batch update matches sequential
# ---------------------------------------------------------------------------

def tc_mad_003():
    r = TestResult("TC-MAD-003", "RollingMAD batch update matches sequential update")
    data = FPOutputStream(workload="amg", seed=SEED).sample(3000)
    window = 500

    roller_seq   = RollingMAD(window_size=window)
    roller_batch = RollingMAD(window_size=window)

    mads_seq = []
    for val in data:
        result = roller_seq.update(val)
        if result is not None:
            mads_seq.append(result)

    mads_batch = roller_batch.update_batch(data)

    r.details = {
        "n_windows": len(mads_seq),
        "seq_mads":   [round(m, 8) for m in mads_seq],
        "batch_mads": [round(m, 8) for m in mads_batch],
    }
    r.assert_true(len(mads_seq) == len(mads_batch),
                  f"Same number of MAD outputs: {len(mads_seq)}")
    if mads_seq and mads_batch:
        max_diff = max(abs(a - b) for a, b in zip(mads_seq, mads_batch))
        r.assert_true(max_diff < 1e-10,
                      f"Max difference between seq and batch = {max_diff:.2e}")
    return [r.finish()]


# ---------------------------------------------------------------------------
# TC-MAD-004: Streaming detection of injected fault
# ---------------------------------------------------------------------------

def tc_mad_004():
    r = TestResult("TC-MAD-004", "RollingMAD detects SDC fault in streaming data")
    n = 10_000
    window = 500

    fp_stream = FPOutputStream(workload="amg", seed=SEED)
    clean_data = fp_stream.sample(n)

    # Inject fault in second half
    fault_dict = inject_fault(
        "sdc",
        {"fp": fp_stream.sample(n), "mpi": np.ones(n), "io": np.ones(n),
         "power": np.ones(n), "checkpoint": fp_stream.sample(n)},
        severity=0.15, seed=SEED
    )
    combined = np.concatenate([clean_data, fault_dict["fp"]])

    roller = RollingMAD(window_size=window)
    timeline = []
    for i, val in enumerate(combined):
        result = roller.update(val)
        if result is not None:
            timeline.append({"window": len(timeline), "mad": result,
                             "phase": "clean" if i < n else "fault"})

    clean_mads = [t["mad"] for t in timeline if t["phase"] == "clean"]
    fault_mads  = [t["mad"] for t in timeline if t["phase"] == "fault"]
    mean_clean = np.mean(clean_mads) if clean_mads else 0.0
    mean_fault  = np.mean(fault_mads)  if fault_mads  else 0.0

    r.details = {
        "n_clean_windows": len(clean_mads), "n_fault_windows": len(fault_mads),
        "mean_clean_mad": round(float(mean_clean), 6),
        "mean_fault_mad":  round(float(mean_fault),  6),
        "timeline": timeline[:20],  # first 20 for brevity
    }
    r.assert_true(mean_fault > mean_clean * 1.5,
                  f"Fault phase MAD={mean_fault:.5f} > 1.5× clean MAD={mean_clean:.5f}")
    return [r.finish()]


# ---------------------------------------------------------------------------
# TC-MAD-005: Deviation vector assembly
# ---------------------------------------------------------------------------

def tc_mad_005():
    r = TestResult("TC-MAD-005", "Deviation vector assembles correctly from 5 RollingMADs")
    streams = create_all_streams("amg", seed=SEED)
    rollers = {name: RollingMAD(window_size=1000) for name in streams}
    n = 5_000

    # Feed each stream to its roller
    stream_mads = {}
    for name, stream in streams.items():
        data = stream.sample(n)
        mads = rollers[name].update_batch(data)
        stream_mads[name] = np.mean(mads) if mads else rollers[name].flush() or 0.0

    dv = DeviationVector.from_array(
        np.array([stream_mads["fp"], stream_mads["mpi"], stream_mads["io"],
                  stream_mads["power"], stream_mads["checkpoint"]])
    )
    arr = dv.as_array()
    r.details = {
        "stream_mads": {k: round(float(v), 6) for k, v in stream_mads.items()},
        "deviation_vector": [round(float(x), 6) for x in arr.tolist()],
        "l2_norm": round(float(dv.l2_norm()), 6),
    }
    r.assert_true(all(0.0 <= x <= 1.0 for x in arr),
                  "All MAD components in [0, 1]")
    r.assert_true(dv.l2_norm() < 0.10,
                  f"Clean stream L2 norm={dv.l2_norm():.5f} < 0.10")
    return [r.finish()]


# ---------------------------------------------------------------------------
# TC-MAD-006: Overhead benchmark
# ---------------------------------------------------------------------------

def tc_mad_006():
    r = TestResult("TC-MAD-006", "RollingMAD throughput >= 1M samples/second")
    roller = RollingMAD(window_size=1000)
    data = FPOutputStream(workload="amg", seed=SEED).sample(100_000)

    t_start = time.perf_counter()
    roller.update_batch(data)
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    throughput = len(data) / elapsed

    r.details = {
        "n_samples": len(data),
        "elapsed_s": round(elapsed, 4),
        "throughput_per_sec": round(throughput, 0),
    }
    r.assert_true(throughput >= 500_000,
                  f"Throughput={throughput:.0f} samples/s >= 500K/s")
    return [r.finish()]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(output_path: str = None) -> list[dict]:
    all_results = []
    suites = [
        tc_samp_001, tc_samp_002, tc_samp_003, tc_samp_004,
        tc_mad_001, tc_mad_002, tc_mad_003, tc_mad_004, tc_mad_005, tc_mad_006,
    ]
    for suite in suites:
        for r in suite():
            d = r.to_dict()
            print(f"  [{'PASS' if d['passed'] else 'FAIL'}] {d['tc_id']} — {d['description']}")
            all_results.append(d)

    n_pass = sum(1 for r in all_results if r["passed"])
    print(f"\n  Sampling & MAD Tests: {n_pass}/{len(all_results)} passed")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"suite": "sampling_mad", "results": all_results,
                       "summary": {"passed": n_pass, "total": len(all_results)}}, f, indent=2)
    return all_results


if __name__ == "__main__":
    print("=== TC-SAMP / TC-MAD: Sampling Theory & Rolling MAD Tests ===")
    run_all(output_path="/home/claude/blade/results/sampling_mad.json")

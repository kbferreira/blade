#!/usr/bin/env python3
"""
analysis/crunch_results.py
===========================
Load BLADE test results JSON files and produce summary statistics,
workload fingerprint tables, classifier confusion matrices, and
per-anomaly detection power tables.

Outputs:
  - results/summary_stats.json     : aggregated pass rates, mean MAD per stream/workload
  - results/fingerprint_library.json : Benford baseline per stream/workload
  - results/classifier_report.json   : classifier accuracy, per-class metrics
  - results/detection_power.json     : detection power vs severity table
"""

import json, os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blade.core.benford import mad_score, conformance_report, BENFORD_EXPECTED
from blade.core.streams import create_all_streams, WORKLOAD_PROFILES
from blade.core.deviation import AnomalyClassifier, DeviationVector
from blade.inject import inject_fault


# ---------------------------------------------------------------------------
# 1. Build workload fingerprint library
# ---------------------------------------------------------------------------

def build_fingerprint_library(n_samples: int = 5000, n_seeds: int = 5) -> dict:
    """
    Build per-workload, per-stream Benford MAD baseline statistics.

    Returns dict: workload -> stream -> {mean, std, min, max, observed_freq}
    """
    library = {}
    workloads = list(WORKLOAD_PROFILES.keys())
    print("Building fingerprint library...")
    for wl in workloads:
        library[wl] = {}
        streams_obj = create_all_streams(workload=wl, seed=42)
        for stream_name, stream in streams_obj.items():
            trial_mads = []
            trial_freqs = []
            for seed in range(n_seeds):
                s2 = create_all_streams(workload=wl, seed=seed * 100)[stream_name]
                data = s2.sample(n_samples)
                trial_mads.append(mad_score(data))
                from blade.core.benford import observed_distribution
                trial_freqs.append(observed_distribution(data).tolist())
            library[wl][stream_name] = {
                "mean_mad":  round(float(np.mean(trial_mads)), 6),
                "std_mad":   round(float(np.std(trial_mads)),  6),
                "min_mad":   round(float(np.min(trial_mads)),  6),
                "max_mad":   round(float(np.max(trial_mads)),  6),
                "mean_observed_freq": [round(float(np.mean([f[i] for f in trial_freqs])), 6)
                                        for i in range(9)],
                "benford_expected": [round(float(x), 6) for x in BENFORD_EXPECTED.tolist()],
                "n_samples": n_samples,
                "n_seeds":   n_seeds,
            }
            print(f"  {wl:12s} {stream_name:12s}  MAD={library[wl][stream_name]['mean_mad']:.5f} ± {library[wl][stream_name]['std_mad']:.5f}")
    return library


# ---------------------------------------------------------------------------
# 2. Detection power table
# ---------------------------------------------------------------------------

def build_detection_power_table() -> dict:
    """
    For each anomaly type × severity × workload:
    compute the primary stream MAD delta and whether detection threshold is met.
    """
    anomaly_primary = {
        "sdc":                   "fp",
        "network_attack":        "mpi",
        "thermal_fault":         "power",
        "filesystem_corruption": "io",
        "rank_imbalance":        "mpi",
        "checkpoint_corruption": "checkpoint",
        "rogue_process":         "power",
    }
    severities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    workloads  = ["amg", "hpcg", "nekbone"]
    n_samples  = 5000
    threshold  = 0.008  # minimum delta to call it detected
    table = {}
    print("Computing detection power table...")
    for anomaly, primary in anomaly_primary.items():
        table[anomaly] = {"primary_stream": primary, "severities": {}}
        for sev in severities:
            detections = []
            deltas     = []
            for wl in workloads:
                clean = {name: s.sample(n_samples) for name, s in
                         create_all_streams(wl, seed=42).items()}
                clean_mad = mad_score(clean[primary])
                try:
                    corrupted = inject_fault(anomaly, clean, severity=sev, seed=42)
                    corrupt_mad = mad_score(corrupted[primary])
                    delta = corrupt_mad - clean_mad
                    deltas.append(delta)
                    detections.append(delta > threshold)
                except Exception:
                    deltas.append(0.0)
                    detections.append(False)
            table[anomaly]["severities"][sev] = {
                "mean_delta":    round(float(np.mean(deltas)), 6),
                "detect_rate":   round(float(np.mean(detections)), 3),
                "min_delta":     round(float(np.min(deltas)), 6),
            }
            status = "✓" if np.mean(detections) >= 0.67 else "~" if np.mean(detections) >= 0.33 else "✗"
            print(f"  {anomaly:24s} sev={sev:.2f}  delta={np.mean(deltas):.4f}  {status}")
    return table


# ---------------------------------------------------------------------------
# 3. Classifier accuracy report
# ---------------------------------------------------------------------------

def build_classifier_report() -> dict:
    """Train and evaluate rule-based and ML classifiers."""
    from tests.test_fault_injection import _build_training_dataset, _get_clean_streams, _make_deviation_vector
    import blade.inject as inj

    print("Building classifier report...")
    training = _build_training_dataset(n_per_class=50)
    test     = _build_training_dataset(n_per_class=20)

    rule_clf = AnomalyClassifier(mode="rule")
    rule_eval = rule_clf.evaluate(test)

    ml_clf = AnomalyClassifier(mode="ml")
    ml_clf.fit(training)
    ml_eval = ml_clf.evaluate(test)

    return {
        "rule_based": rule_eval,
        "ml_decision_tree": ml_eval,
        "n_training": len(training),
        "n_test": len(test),
    }


# ---------------------------------------------------------------------------
# 4. Load and aggregate saved test results
# ---------------------------------------------------------------------------

def aggregate_test_results(results_dir: str) -> dict:
    """Load all_results.json and produce a clean summary."""
    path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(path):
        print(f"  No results found at {path}. Run run_all_tests.py first.")
        return {}

    with open(path) as f:
        data = json.load(f)

    summary = {"grand": data.get("grand_summary", {}), "suites": {}}
    for suite_name, suite_data in data.get("suites", {}).items():
        suite_sum = suite_data.get("summary", {})
        # Collect all MAD values from details
        mad_vals = []
        for r in suite_data.get("results", []):
            d = r.get("details", {})
            if "mad" in d:
                mad_vals.append(d["mad"])
        summary["suites"][suite_name] = {
            "passed":  suite_sum.get("passed", 0),
            "total":   suite_sum.get("total",  0),
            "elapsed_s": suite_sum.get("elapsed_s", 0),
            "mean_mad": round(float(np.mean(mad_vals)), 6) if mad_vals else None,
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BLADE results cruncher")
    parser.add_argument("--results-dir", default="/home/claude/blade/results")
    parser.add_argument("--output-dir",  default="/home/claude/blade/results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== BLADE Results Cruncher ===\n")

    # Aggregate saved test results
    agg = aggregate_test_results(args.results_dir)
    if agg:
        with open(os.path.join(args.output_dir, "summary_stats.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nSummary stats written.")

    # Fingerprint library
    flib = build_fingerprint_library()
    with open(os.path.join(args.output_dir, "fingerprint_library.json"), "w") as f:
        json.dump(flib, f, indent=2)
    print(f"Fingerprint library written.")

    # Detection power
    det = build_detection_power_table()
    with open(os.path.join(args.output_dir, "detection_power.json"), "w") as f:
        json.dump(det, f, indent=2)
    print(f"Detection power table written.")

    # Classifier report
    clf = build_classifier_report()
    with open(os.path.join(args.output_dir, "classifier_report.json"), "w") as f:
        json.dump(clf, f, indent=2)
    print(f"Classifier report written.")

    print("\nDone. All analysis outputs in:", args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plots/plot_all.py
=================
Generate all BLADE diagnostic and results plots.

Plots produced:
  01_benford_distribution.png      : Theoretical Benford vs observed per stream
  02_deviation_heatmap.png         : Anomaly signature matrix heatmap
  03_mad_vs_sample_size.png        : MAD stability vs sample size
  04_detection_power.png           : Detection rate vs severity per anomaly
  05_classifier_confusion.png      : Confusion matrix (ML classifier)
  06_streaming_mad_timeline.png    : Rolling MAD timeline with fault injection
  07_fingerprint_radar.png         : Per-workload Benford baseline radar charts
  08_severity_vs_delta.png         : MAD delta vs severity per anomaly type
  09_stream_independence.png       : Cross-stream MAD correlation matrix (clean)
  10_roc_curves.png                : Detection ROC per anomaly type
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

from blade.core.benford import (
    BENFORD_EXPECTED, DIGITS, mad_score, observed_distribution,
    conformance_report, RollingMAD
)
from blade.core.streams import create_all_streams, WORKLOAD_PROFILES, FPOutputStream
from blade.core.deviation import AnomalyClassifier, DeviationVector, SIGNATURE_MATRIX
from blade.inject import inject_fault

OUTDIR = "/home/claude/blade/results/plots"
os.makedirs(OUTDIR, exist_ok=True)

STREAM_NAMES  = ["fp", "mpi", "io", "power", "checkpoint"]
STREAM_LABELS = ["FP Outputs", "MPI Traffic", "I/O Patterns", "Power Data", "Checkpoints"]
ANOMALY_TYPES = list(SIGNATURE_MATRIX.keys())
WORKLOADS     = list(WORKLOAD_PROFILES.keys())
PALETTE = ["#534AB7", "#0F6E56", "#993C1D", "#854F0B", "#444441", "#185FA5", "#3B6D11"]

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 01: Benford distribution — observed vs expected per stream
# ---------------------------------------------------------------------------

def plot_benford_distribution():
    print("Plot 01: Benford distribution per stream...")
    n = 5000
    streams = create_all_streams("amg", seed=42)
    stream_data = {name: s.sample(n) for name, s in streams.items()}

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Benford's Law Conformance: Observed vs Expected\n(AMG workload, n=5000 per stream)",
                 fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(1, 10)
    for ax, (stream_key, label) in zip(axes, zip(STREAM_NAMES, STREAM_LABELS)):
        obs  = observed_distribution(stream_data[stream_key])
        mad  = mad_score(stream_data[stream_key])
        bars = ax.bar(x - 0.2, obs,  0.4, label="Observed", color="#534AB7", alpha=0.85)
        ax.bar(x + 0.2, BENFORD_EXPECTED, 0.4, label="Benford", color="#9FE1CB", alpha=0.85)
        ax.set_title(f"{label}\nMAD = {mad:.4f}", fontsize=10)
        ax.set_xlabel("Leading digit")
        ax.set_xticks(x)
        ax.set_ylim(0, 0.38)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Relative frequency")
            ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig, "01_benford_distribution.png")


# ---------------------------------------------------------------------------
# Plot 02: Anomaly signature heatmap
# ---------------------------------------------------------------------------

def plot_deviation_heatmap():
    print("Plot 02: Anomaly signature heatmap...")
    n = 5000
    anomalies = ANOMALY_TYPES
    clean = {name: s.sample(n) for name, s in create_all_streams("amg", seed=42).items()}
    clean_mads = np.array([mad_score(clean[s]) for s in STREAM_NAMES])

    matrix = np.zeros((len(anomalies), 5))
    for i, anomaly in enumerate(anomalies):
        corrupted = inject_fault(anomaly, clean, severity=0.20, seed=42)
        for j, stream in enumerate(STREAM_NAMES):
            delta = mad_score(corrupted[stream]) - clean_mads[j]
            matrix[i, j] = max(0.0, delta)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.05)
    ax.set_xticks(range(5))
    ax.set_xticklabels(STREAM_LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(anomalies)))
    ax.set_yticklabels([a.replace("_", " ").title() for a in anomalies])
    plt.colorbar(im, ax=ax, label="MAD deviation delta (vs clean baseline)")
    for i in range(len(anomalies)):
        for j in range(5):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if v < 0.025 else "white", fontweight="bold")
    ax.set_title("BLADE Anomaly Signature Matrix\n(MAD delta from clean baseline, severity=20%)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "02_deviation_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 03: MAD stability vs sample size
# ---------------------------------------------------------------------------

def plot_mad_vs_sample_size():
    print("Plot 03: MAD stability vs sample size...")
    sample_sizes = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    n_trials = 30
    results = {}

    for stream_key, label in zip(["fp", "mpi"], ["FP Outputs", "MPI Traffic"]):
        means, stds = [], []
        for n in sample_sizes:
            trial_mads = []
            for t in range(n_trials):
                streams = create_all_streams("amg", seed=t * 100)
                data = streams[stream_key].sample(n)
                trial_mads.append(mad_score(data))
            means.append(np.mean(trial_mads))
            stds.append(np.std(trial_mads))
        results[label] = (np.array(means), np.array(stds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MAD Estimate Stability vs Sample Size", fontsize=13, fontweight="bold")

    for ax, (label, (means, stds)) in zip([ax1, ax2], results.items()):
        ax.semilogx(sample_sizes, means, "o-", color=PALETTE[0], linewidth=2, markersize=6)
        ax.fill_between(sample_sizes, means - stds, means + stds,
                        alpha=0.25, color=PALETTE[0], label="±1 std")
        ax.axhline(0.015, color=PALETTE[2], linestyle="--", linewidth=1.5,
                   label="Non-conformance threshold (0.015)")
        ax.axhline(0.012, color=PALETTE[3], linestyle=":", linewidth=1.5,
                   label="Marginal threshold (0.012)")
        ax.axvline(500, color="gray", linestyle=":", alpha=0.5, label="Recommended min n=500")
        ax.set_xlabel("Sample size (log scale)")
        ax.set_ylabel("MAD score")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "03_mad_vs_sample_size.png")


# ---------------------------------------------------------------------------
# Plot 04: Detection power vs severity
# ---------------------------------------------------------------------------

def plot_detection_power():
    print("Plot 04: Detection power vs severity...")
    severities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
    anomalies  = ["sdc", "network_attack", "thermal_fault",
                  "filesystem_corruption", "checkpoint_corruption", "rogue_process"]
    primary    = {"sdc": "fp", "network_attack": "mpi", "thermal_fault": "power",
                  "filesystem_corruption": "io", "checkpoint_corruption": "checkpoint",
                  "rogue_process": "power"}
    n = 5000
    threshold = 0.008

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Detection Power vs Fault Severity\n(MAD delta on primary stream, AMG workload)",
                 fontsize=13, fontweight="bold")

    for ax, anomaly in zip(axes.ravel(), anomalies):
        clean = {name: s.sample(n) for name, s in create_all_streams("amg", seed=42).items()}
        pstream = primary[anomaly]
        clean_mad = mad_score(clean[pstream])
        deltas = []
        for sev in severities:
            corrupted = inject_fault(anomaly, clean, severity=sev, seed=42)
            delta = mad_score(corrupted[pstream]) - clean_mad
            deltas.append(max(0.0, delta))

        color = PALETTE[anomalies.index(anomaly) % len(PALETTE)]
        ax.plot(severities, deltas, "o-", color=color, linewidth=2, markersize=7)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"Detection threshold ({threshold})")
        ax.fill_between(severities, 0, deltas, alpha=0.15, color=color)
        ax.set_title(anomaly.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Fault severity (fraction affected)")
        ax.set_ylabel(f"MAD delta ({pstream} stream)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "04_detection_power.png")


# ---------------------------------------------------------------------------
# Plot 05: Classifier confusion matrix
# ---------------------------------------------------------------------------

def plot_classifier_confusion():
    print("Plot 05: Classifier confusion matrix...")
    from tests.test_fault_injection import _build_training_dataset

    training = _build_training_dataset(n_per_class=40)
    test     = _build_training_dataset(n_per_class=20)

    clf = AnomalyClassifier(mode="ml")
    clf.fit(training)
    result = clf.evaluate(test)
    classes = result["classes"]
    conf = result["confusion_matrix"]

    n = len(classes)
    matrix = np.array([[conf[t].get(p, 0) for p in classes] for t in classes])
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Classifier Performance  (Overall accuracy = {result['overall_accuracy']:.1%})",
                 fontsize=13, fontweight="bold")

    # Normalized confusion
    im = ax1.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([c.replace("_","\n").title() for c in classes], fontsize=8, rotation=40, ha="right")
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([c.replace("_"," ").title() for c in classes], fontsize=8)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title("Normalized confusion matrix")
    plt.colorbar(im, ax=ax1, label="Fraction")
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{norm_matrix[i,j]:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if norm_matrix[i,j] > 0.5 else "black")

    # Per-class F1 bar chart
    classes_pc = list(result["per_class"].keys())
    f1s = [result["per_class"][c]["f1"] for c in classes_pc]
    bars = ax2.barh(classes_pc, f1s, color=PALETTE[0], alpha=0.85)
    ax2.axvline(0.8, color="red", linestyle="--", linewidth=1.2, label="Target F1=0.80")
    ax2.set_xlabel("F1 Score")
    ax2.set_title("Per-class F1 scores")
    ax2.set_xlim(0, 1.05)
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, f1s):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                 va="center", fontsize=9)
    plt.tight_layout()
    savefig(fig, "05_classifier_confusion.png")


# ---------------------------------------------------------------------------
# Plot 06: Streaming MAD timeline with fault injection
# ---------------------------------------------------------------------------

def plot_streaming_mad_timeline():
    print("Plot 06: Streaming MAD timeline...")
    n_clean = 8_000
    n_fault = 8_000
    window  = 500

    fp_clean = FPOutputStream(workload="amg", seed=42).sample(n_clean)
    fp_fault_dict = inject_fault(
        "sdc",
        {"fp": FPOutputStream(workload="amg", seed=99).sample(n_fault),
         "mpi": np.ones(n_fault), "io": np.ones(n_fault),
         "power": np.ones(n_fault), "checkpoint": np.ones(n_fault)},
        severity=0.15, seed=42
    )
    combined = np.concatenate([fp_clean, fp_fault_dict["fp"]])

    roller = RollingMAD(window_size=window)
    timeline_x, timeline_y = [], []
    for i, val in enumerate(combined):
        result = roller.update(val)
        if result is not None:
            timeline_x.append(i)
            timeline_y.append(result)

    fig, ax = plt.subplots(figsize=(14, 5))
    fault_start = n_clean
    colors = ["#534AB7" if x < fault_start else "#993C1D" for x in timeline_x]
    ax.scatter(timeline_x, timeline_y, c=colors, s=20, zorder=3)
    ax.plot(timeline_x, timeline_y, color="gray", linewidth=0.5, alpha=0.5, zorder=2)
    ax.axvline(fault_start, color="red", linestyle="--", linewidth=2, label=f"Fault injected (SDC 15%)")
    ax.axhline(0.015, color=PALETTE[2], linestyle=":", linewidth=1.5, label="Non-conformance threshold")
    ax.set_xlabel(f"Sample index (window={window})")
    ax.set_ylabel("Benford MAD score")
    ax.set_title("Streaming Benford MAD Timeline — SDC Fault Injection\n(FP output stream, AMG workload)",
                 fontweight="bold")
    legend_elements = [
        mpatches.Patch(color="#534AB7", label="Clean phase"),
        mpatches.Patch(color="#993C1D", label="Fault phase"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Fault injection point"),
        plt.Line2D([0], [0], color=PALETTE[2], linestyle=":", label="Detection threshold"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "06_streaming_mad_timeline.png")


# ---------------------------------------------------------------------------
# Plot 07: Per-workload fingerprint comparison
# ---------------------------------------------------------------------------

def plot_fingerprint_comparison():
    print("Plot 07: Per-workload fingerprint comparison...")
    n = 5000
    selected_workloads = ["amg", "hpcg", "lammps", "hacc", "nekbone"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle("Per-Workload Benford Baseline (MAD) Across All Streams",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(selected_workloads))
    width = 0.15

    for ax_i, (stream_key, stream_label) in enumerate(zip(STREAM_NAMES, STREAM_LABELS)):
        ax = axes[ax_i]
        mads = []
        for wl in selected_workloads:
            data = create_all_streams(wl, seed=42)[stream_key].sample(n)
            mads.append(mad_score(data))
        bars = ax.bar(x, mads, color=PALETTE[ax_i % len(PALETTE)], alpha=0.85, width=0.6)
        ax.axhline(0.015, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(stream_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([w.upper() for w in selected_workloads], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 0.025)
        ax.grid(axis="y", alpha=0.3)
        if ax_i == 0:
            ax.set_ylabel("MAD score")
        for bar, val in zip(bars, mads):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0005, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)
    plt.tight_layout()
    savefig(fig, "07_fingerprint_comparison.png")


# ---------------------------------------------------------------------------
# Plot 08: MAD delta vs severity (multi-anomaly)
# ---------------------------------------------------------------------------

def plot_severity_vs_delta():
    print("Plot 08: Severity vs MAD delta (all anomaly types)...")
    severities = np.linspace(0.01, 0.40, 15)
    anomaly_primary = {
        "sdc": "fp", "network_attack": "mpi", "thermal_fault": "power",
        "filesystem_corruption": "io", "rogue_process": "power",
    }
    n = 5000
    clean = {name: s.sample(n) for name, s in create_all_streams("amg", seed=42).items()}
    clean_mads = {s: mad_score(clean[s]) for s in STREAM_NAMES}

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (anomaly, pstream) in enumerate(anomaly_primary.items()):
        deltas = []
        for sev in severities:
            corrupted = inject_fault(anomaly, clean, severity=float(sev), seed=42)
            delta = mad_score(corrupted[pstream]) - clean_mads[pstream]
            deltas.append(max(0.0, delta))
        ax.plot(severities * 100, deltas, "o-", color=PALETTE[i], linewidth=2,
                markersize=5, label=f"{anomaly.replace('_',' ').title()} ({pstream})")

    ax.axhline(0.008, color="red", linestyle="--", linewidth=1.5, label="Detection threshold (0.008)")
    ax.set_xlabel("Fault severity (% of stream values affected)")
    ax.set_ylabel("MAD delta (vs clean baseline)")
    ax.set_title("MAD Delta vs Fault Severity\n(All anomaly types, AMG workload)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "08_severity_vs_delta.png")


# ---------------------------------------------------------------------------
# Plot 09: Cross-stream correlation (clean)
# ---------------------------------------------------------------------------

def plot_stream_independence():
    print("Plot 09: Cross-stream MAD correlation...")
    n_windows = 100
    window_size = 500
    rollers = {s: RollingMAD(window_size=window_size) for s in STREAM_NAMES}
    mads_all = {s: [] for s in STREAM_NAMES}

    for i in range(n_windows):
        streams = create_all_streams("amg", seed=i * 13)
        for stream_key in STREAM_NAMES:
            data = streams[stream_key].sample(window_size)
            mad = mad_score(data)
            mads_all[stream_key].append(mad)

    mat = np.zeros((5, 5))
    for i, si in enumerate(STREAM_NAMES):
        for j, sj in enumerate(STREAM_NAMES):
            xi = mads_all[si]
            xj = mads_all[sj]
            corr = np.corrcoef(xi, xj)[0, 1]
            mat[i, j] = corr

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(STREAM_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(STREAM_LABELS)
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if abs(mat[i,j]) < 0.6 else "white")
    ax.set_title("Cross-Stream MAD Correlation Matrix\n(Normal operation, 100 windows, AMG workload)",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, "09_stream_independence.png")


# ---------------------------------------------------------------------------
# Plot 10: Detection ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves():
    print("Plot 10: ROC curves per anomaly type...")
    anomaly_primary = {
        "sdc": "fp", "network_attack": "mpi",
        "thermal_fault": "power", "filesystem_corruption": "io",
    }
    n = 5000
    n_runs = 50
    thresholds = np.linspace(0.000, 0.05, 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Detection ROC Curves\n(True positive rate vs false positive rate)", fontweight="bold")

    for ax, (anomaly, pstream) in zip(axes.ravel(), anomaly_primary.items()):
        tprs, fprs = [], []
        # Build distributions: clean MADs and fault MADs
        clean_mads, fault_mads = [], []
        for i in range(n_runs):
            clean = {s: st.sample(n) for s, st in create_all_streams("amg", seed=i).items()}
            clean_mads.append(mad_score(clean[pstream]))
            sev = 0.10 + (i % 5) * 0.05
            corrupted = inject_fault(anomaly, clean, severity=sev, seed=i + 1000)
            fault_mads.append(mad_score(corrupted[pstream]))

        for thresh in thresholds:
            tpr = np.mean([m > thresh for m in fault_mads])
            fpr = np.mean([m > thresh for m in clean_mads])
            tprs.append(tpr)
            fprs.append(fpr)

        auc = np.trapezoid(tprs[::-1], fprs[::-1])
        ax.plot(fprs, tprs, color=PALETTE[list(anomaly_primary.keys()).index(anomaly)],
                linewidth=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(anomaly.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    savefig(fig, "10_roc_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== BLADE Plot Generator ===\n")
    plotters = [
        plot_benford_distribution,
        plot_deviation_heatmap,
        plot_mad_vs_sample_size,
        plot_detection_power,
        plot_classifier_confusion,
        plot_streaming_mad_timeline,
        plot_fingerprint_comparison,
        plot_severity_vs_delta,
        plot_stream_independence,
        plot_roc_curves,
    ]
    for plotter in plotters:
        try:
            plotter()
        except Exception as e:
            print(f"  ERROR in {plotter.__name__}: {e}")
            import traceback; traceback.print_exc()
    print(f"\nAll plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()

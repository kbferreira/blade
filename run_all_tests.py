#!/usr/bin/env python3
"""
run_all_tests.py
================
Master BLADE test runner. Executes all test suites and writes a
combined results file for the analysis and plotting pipelines.

Usage:
    python3 run_all_tests.py [--output-dir RESULTS_DIR]
"""

import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("/home/claude/blade/results", exist_ok=True)

from tests.test_benford_conformance import run_all as run_conformance
from tests.test_fault_injection     import run_all as run_fault
from tests.test_sampling_and_mad    import run_all as run_sampling


def main():
    parser = argparse.ArgumentParser(description="BLADE master test runner")
    parser.add_argument("--output-dir", default="/home/claude/blade/results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    all_suites = {}

    print("\n" + "="*60)
    print("  BLADE Test Suite")
    print("="*60)

    suites = [
        ("conformance",  run_conformance,
         os.path.join(args.output_dir, "conformance.json")),
        ("fault_injection", run_fault,
         os.path.join(args.output_dir, "fault_injection.json")),
        ("sampling_mad", run_sampling,
         os.path.join(args.output_dir, "sampling_mad.json")),
    ]

    grand_pass = 0
    grand_total = 0

    for suite_name, runner, output_path in suites:
        print(f"\n--- {suite_name.upper().replace('_',' ')} ---")
        t_suite = time.time()
        results = runner(output_path=output_path)
        elapsed = time.time() - t_suite
        n_pass  = sum(1 for r in results if r.get("passed"))
        n_total = len(results)
        grand_pass  += n_pass
        grand_total += n_total
        all_suites[suite_name] = {
            "results": results,
            "summary": {"passed": n_pass, "total": n_total, "elapsed_s": round(elapsed, 2)}
        }

    total_elapsed = time.time() - t0

    print("\n" + "="*60)
    print(f"  GRAND TOTAL: {grand_pass}/{grand_total} tests passed")
    print(f"  Total time : {total_elapsed:.1f}s")
    print("="*60)

    # Write combined results
    combined_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump({
            "blade_version": "0.1.0",
            "total_elapsed_s": round(total_elapsed, 2),
            "grand_summary": {"passed": grand_pass, "total": grand_total,
                              "pass_rate": round(grand_pass / grand_total, 4)},
            "suites": all_suites,
        }, f, indent=2)

    print(f"\n  Results written to: {combined_path}")

    # Print failures
    failures = []
    for suite_name, data in all_suites.items():
        for r in data["results"]:
            if not r.get("passed"):
                failures.append(f"  [{suite_name}] {r['tc_id']} — {r['description']}")
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f)
    else:
        print("\n  All tests passed!")

    return 0 if grand_pass == grand_total else 1


if __name__ == "__main__":
    sys.exit(main())

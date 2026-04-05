#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List


C1_COL = "n_C1_think_loop"
C2_COL = "n_C2_tasktrack_loop"
A1_COL = "n_A1_invented_file_content"
A2_COL = "n_A2_invented_search_target"
A3_COL = "n_A3_invalid_edit_path"
B1_COL = "n_B1_state_amnesia"
B2_COL = "n_B2_wrong_path"
B3_COL = "n_B3_failed_command"


def load_instance_summary(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_int(x: Any) -> int:
    if x is None or x == "":
        return 0
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def safe_float(x: Any) -> float:
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def safe_bool(x: Any) -> bool:
    if x is None or x == "":
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "t")


def compute_group_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "avg_hallucination_score": 0.0,
            "pct_with_A1": 0.0, "pct_with_A2": 0.0, "pct_with_A3": 0.0,
            "pct_with_B1": 0.0, "pct_with_B2": 0.0, "pct_with_B3": 0.0,
            "pct_with_C1": 0.0, "pct_with_C2": 0.0,
        }
    scores = [safe_float(r.get("hallucination_score")) for r in rows]
    def pct(col: str) -> float:
        return sum(1 for r in rows if safe_int(r.get(col)) > 0) / n * 100
    return {
        "n": n,
        "avg_hallucination_score": sum(scores) / len(scores),
        "pct_with_A1": pct(A1_COL),
        "pct_with_A2": pct(A2_COL),
        "pct_with_A3": pct(A3_COL),
        "pct_with_B1": pct(B1_COL),
        "pct_with_B2": pct(B2_COL),
        "pct_with_B3": pct(B3_COL),
        "pct_with_C1": pct(C1_COL),
        "pct_with_C2": pct(C2_COL),
    }


def run_comparison(rows: List[Dict[str, Any]], source_label: str = "") -> str:
    lines: List[str] = []

    title = "Success vs failure comparison"
    if source_label:
        title += f" — {source_label}"
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

    resolved_rows = [r for r in rows if safe_bool(r.get("is_resolved"))]
    failed_rows = [r for r in rows if not safe_bool(r.get("is_resolved"))]

    n_total = len(rows)
    n_resolved = len(resolved_rows)
    n_failed = len(failed_rows)

    if n_total == 0:
        lines.append("\nNo data (0 instances).")
        return "\n".join(lines)

    lines.append(f"\nSplit: resolved = {n_resolved}, failed = {n_failed} (total = {n_total})")

    m_resolved = compute_group_metrics(resolved_rows)
    m_failed = compute_group_metrics(failed_rows)

    lines.append("\n[Comparison]")
    lines.append("")
    lines.append(f"{'Metric':<30} | {'Resolved':^20} | {'Failed':^10}")
    lines.append("-" * 66)
    lines.append(f"  {'N instances':<28} | {m_resolved['n']:<20} | {m_failed['n']}")
    lines.append(f"  {'Avg hallucination_score':<28} | {m_resolved['avg_hallucination_score']:<20.2f} | {m_failed['avg_hallucination_score']:.2f}")
    lines.append(f"  {'% with A1 invented file content':<28} | {m_resolved['pct_with_A1']:<20.1f} | {m_failed['pct_with_A1']:.1f}%")
    lines.append(f"  {'% with A2 invented search target':<28} | {m_resolved['pct_with_A2']:<20.1f} | {m_failed['pct_with_A2']:.1f}%")
    lines.append(f"  {'% with A3 invalid edit path':<28} | {m_resolved['pct_with_A3']:<20.1f} | {m_failed['pct_with_A3']:.1f}%")
    lines.append(f"  {'% with B1 state amnesia':<28} | {m_resolved['pct_with_B1']:<20.1f} | {m_failed['pct_with_B1']:.1f}%")
    lines.append(f"  {'% with B2 wrong path':<28} | {m_resolved['pct_with_B2']:<20.1f} | {m_failed['pct_with_B2']:.1f}%")
    lines.append(f"  {'% with B3 failed command':<28} | {m_resolved['pct_with_B3']:<20.1f} | {m_failed['pct_with_B3']:.1f}%")
    lines.append(f"  {'% with C1 think loop':<28} | {m_resolved['pct_with_C1']:<20.1f} | {m_failed['pct_with_C1']:.1f}%")
    lines.append(f"  {'% with C2 tasktrack loop':<28} | {m_resolved['pct_with_C2']:<20.1f} | {m_failed['pct_with_C2']:.1f}%")
    lines.append("")

    c1_low_success = m_resolved["n"] and m_resolved["pct_with_C1"] < 30.0
    c2_low_success = m_resolved["n"] and m_resolved["pct_with_C2"] < 30.0
    c1_high_fail = m_failed["n"] and m_failed["pct_with_C1"] >= 40.0
    c2_high_fail = m_failed["n"] and m_failed["pct_with_C2"] >= 25.0
    gap_c1 = m_failed["pct_with_C1"] - m_resolved["pct_with_C1"]
    gap_c2 = m_failed["pct_with_C2"] - m_resolved["pct_with_C2"]
    gap_a2 = m_failed["pct_with_A2"] - m_resolved["pct_with_A2"]
    gap_b3 = m_failed["pct_with_B3"] - m_resolved["pct_with_B3"]
    behavioral_gap = gap_c1 >= 20.0 or gap_c2 >= 20.0
    content_gap = gap_a2 >= 20.0
    env_gap = gap_b3 >= 15.0

    lines.append("[Conclusion]")
    if n_resolved == 0:
        lines.append("  No resolved instances; cannot compare.")
    elif n_failed == 0:
        lines.append("  No failed instances; cannot compare.")
    else:
        if (c1_low_success and c2_low_success) and (c1_high_fail or c2_high_fail):
            lines.append("  Behavioral hallucinations (C1 think loop, C2 task_tracking loop) are")
            lines.append("  strongly correlated with task failure: success samples have low C1/C2,")
            lines.append("  failure samples show high C1/C2 prevalence.")
        elif behavioral_gap:
            lines.append("  C1/C2 rates are notably higher in failed instances than in resolved")
            lines.append("  instances. Behavioral hallucinations are associated with task failure.")
        else:
            lines.append("  C1/C2 gap between resolved and failed is below the strong-correlation")
            lines.append("  threshold. Consider larger samples.")
        if content_gap:
            lines.append(f"  A2 gap = {gap_a2:.1f}pp: failed instances searched for nonexistent content more often.")
        if env_gap:
            lines.append(f"  B3 gap = {gap_b3:.1f}pp: failed instances made more bad environment assumptions.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_summary", default="", help="Path to instance_summary.csv")
    ap.add_argument("--input_dir", default="", help="Directory to search for instance_summary.csv")
    ap.add_argument("--out_report", default="", help="Output report path")
    args = ap.parse_args()

    if args.instance_summary:
        path = os.path.abspath(args.instance_summary)
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")
        rows = load_instance_summary(path)
        label = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        report = run_comparison(rows, label)
        print(report)
        if args.out_report:
            out_path = os.path.abspath(args.out_report)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Written: {out_path}")

    elif args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_dir):
            raise SystemExit(f"Directory not found: {input_dir}")
        found = []
        for root, _dirs, files in os.walk(input_dir):
            if "instance_summary.csv" in files:
                found.append(os.path.join(root, "instance_summary.csv"))
        if not found:
            raise SystemExit(f"No instance_summary.csv under {input_dir}")
        for csv_path in sorted(found):
            rows = load_instance_summary(csv_path)
            label = os.path.basename(os.path.dirname(csv_path)) or "instance_summary"
            report = run_comparison(rows, label)
            print(report)
            out_path = os.path.join(os.path.dirname(csv_path), "success_failure_report.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Written: {out_path}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(script_dir, "out")
        if os.path.isdir(default_dir):
            input_dir = os.path.abspath(default_dir)
            found = []
            for root, _dirs, files in os.walk(input_dir):
                if "instance_summary.csv" in files:
                    found.append(os.path.join(root, "instance_summary.csv"))
            if not found:
                raise SystemExit(f"No instance_summary.csv under {default_dir}. Use --instance_summary or --input_dir.")
            for csv_path in sorted(found):
                rows = load_instance_summary(csv_path)
                label = os.path.basename(os.path.dirname(csv_path)) or "instance_summary"
                report = run_comparison(rows, label)
                print(report)
                out_path = os.path.join(os.path.dirname(csv_path), "success_failure_report.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Written: {out_path}")
        else:
            raise SystemExit("Specify --instance_summary <path> or --input_dir <dir>.")


if __name__ == "__main__":
    main()

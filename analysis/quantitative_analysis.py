#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from typing import Any, Dict, List


HALLUC_TYPES = [
    "A1_invented_file_content",
    "A2_invented_search_target",
    "A3_invalid_edit_path",
    "B1_state_amnesia",
    "B2_wrong_path",
    "B3_failed_command",
    "C1_think_loop",
    "C2_tasktrack_loop",
]

HALLUC_LABELS = {
    "A1_invented_file_content": "A1 invented file content (edit failed)",
    "A2_invented_search_target": "A2 invented search target (grep empty)",
    "A3_invalid_edit_path": "A3 invalid edit path (wrong/missing path)",
    "B1_state_amnesia": "B1 state amnesia (repeated same command)",
    "B2_wrong_path": "B2 wrong path (nonexistent file/path via shell)",
    "B3_failed_command": "B3 failed command (bad env assumption)",
    "C1_think_loop": "C1 think loop",
    "C2_tasktrack_loop": "C2 task_tracking loop",
}


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


def run_quantitative_analysis(rows: List[Dict[str, Any]], source_label: str = "") -> str:
    lines: List[str] = []

    title = "Quantitative analysis"
    if source_label:
        title += f" — {source_label}"
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

    n_total = len(rows)
    if n_total == 0:
        lines.append("\nNo data (0 instances).")
        return "\n".join(lines)

    lines.append("\n[Basic stats]")
    lines.append(f"  Total instances: {n_total}")

    resolved_count = sum(1 for r in rows if safe_bool(r.get("is_resolved")))
    resolved_pct = (resolved_count / n_total * 100) if n_total else 0.0
    lines.append(f"  Resolved %: {resolved_count} / {n_total} = {resolved_pct:.2f}%")

    scores = [safe_float(r.get("hallucination_score")) for r in rows]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    lines.append(f"  Avg hallucination_score: {avg_score:.2f}")

    lines.append("\n[Hallucination type frequency]")
    lines.append("  (1) Instances with >=1 occurrence / total steps per type")

    for ht in HALLUC_TYPES:
        col = f"n_{ht}"
        inst_with_type = sum(1 for r in rows if safe_int(r.get(col)) > 0)
        total_steps = sum(safe_int(r.get(col)) for r in rows)
        label = HALLUC_LABELS.get(ht, ht)
        lines.append(f"    {label}: {inst_with_type} instances, {total_steps} steps")

    lines.append("\n  (2) Dominant hallucination type per instance")
    dom_counter: Counter = Counter()
    for r in rows:
        dom = (r.get("dominant_halluc_type") or "none").strip() or "none"
        dom_counter[dom] += 1
    for dom, cnt in dom_counter.most_common():
        label = HALLUC_LABELS.get(dom, dom) if dom != "none" else "none"
        pct = (cnt / n_total * 100) if n_total else 0.0
        lines.append(f"    {label}: {cnt} instances ({pct:.1f}%)")

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
        report = run_quantitative_analysis(rows, label)
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
            report = run_quantitative_analysis(rows, label)
            print(report)
            out_path = os.path.join(os.path.dirname(csv_path), "quantitative_report.txt")
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
                report = run_quantitative_analysis(rows, label)
                print(report)
                out_path = os.path.join(os.path.dirname(csv_path), "quantitative_report.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Written: {out_path}")
        else:
            raise SystemExit("Specify --instance_summary <path> or --input_dir <dir>.")


if __name__ == "__main__":
    main()

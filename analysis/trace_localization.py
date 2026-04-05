#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

HALLUC_TYPES = ["A1_invented_file_content", "A2_invented_search_target", "A3_invalid_edit_path",
                "B1_state_amnesia", "B2_wrong_path", "B3_failed_command",
                "C1_think_loop", "C2_tasktrack_loop"]

FIRST_HALLUC_BUCKETS = [(0, 2), (3, 5), (6, 10), (11, 15), (16, 20), (21, 999)]


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_int(x: Any) -> int:
    if x is None or x == "":
        return 0
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def safe_bool(x: Any) -> bool:
    if x is None or x == "":
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "t")


def steps_by_instance(step_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in step_rows:
        out[r["instance_id"]].append(r)
    for k in out:
        out[k].sort(key=lambda x: safe_int(x.get("step_idx", 0)))
    return dict(out)


def first_halluc_step(steps: List[Dict[str, Any]]) -> Optional[int]:
    for s in steps:
        ht = (s.get("hallucination_type") or "").strip()
        if ht and ht != "none":
            return safe_int(s.get("step_idx", -1))
    return None


def first_halluc_step_by_type(steps: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    result: Dict[str, Optional[int]] = {t: None for t in HALLUC_TYPES}
    for s in steps:
        ht = (s.get("hallucination_type") or "").strip()
        if ht in result and result[ht] is None:
            result[ht] = safe_int(s.get("step_idx", -1))
    return result


def trajectory_signature(steps: List[Dict[str, Any]], max_len: int = 12) -> str:
    parts = []
    for s in steps[:max_len]:
        cat = (s.get("action_category") or "?").strip()[:4]
        ht = (s.get("hallucination_type") or "none").strip()
        if ht and ht != "none":
            parts.append(ht.split("_")[0])
        else:
            parts.append(cat)
    return " -> ".join(parts)


def bucket_label(lo: int, hi: int) -> str:
    if hi >= 999:
        return f"step {lo}+"
    return f"step {lo}-{hi}"


def run_trace_localization(
    step_rows: List[Dict[str, Any]],
    instance_summary_rows: Optional[List[Dict[str, Any]]] = None,
    source_label: str = "",
) -> str:
    lines: List[str] = []
    title = "Trace localization (time-dimension analysis)"
    if source_label:
        title += f" — {source_label}"
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

    by_inst = steps_by_instance(step_rows)
    inst_ids = sorted(by_inst.keys())
    n_inst = len(inst_ids)

    if n_inst == 0:
        lines.append("\nNo step data.")
        return "\n".join(lines)

    resolved_set = set()
    if instance_summary_rows:
        for r in instance_summary_rows:
            if safe_bool(r.get("is_resolved")):
                resolved_set.add(r.get("instance_id", ""))

    first_any: List[Optional[int]] = []
    first_by_type: Dict[str, List[Optional[int]]] = {t: [] for t in HALLUC_TYPES}
    traj_len: List[int] = []
    signatures: List[Tuple[str, str]] = []

    for iid in inst_ids:
        steps = by_inst[iid]
        traj_len.append(len(steps))
        first_any.append(first_halluc_step(steps))
        for t in HALLUC_TYPES:
            by_type = first_halluc_step_by_type(steps)
            first_by_type[t].append(by_type[t])
        sig = trajectory_signature(steps)
        res = "resolved" if iid in resolved_set else "failed"
        signatures.append((sig, res))

    n_with_any = sum(1 for x in first_any if x is not None)
    lines.append(f"\nInstances: {n_inst}  |  With at least one hallucination step: {n_with_any} ({100*n_with_any/max(1,n_inst):.1f}%)")

    lines.append("\n[1] First hallucination step (when does hallucination first appear?)")
    lines.append("    Distribution (all instances that have any hallucination):")
    bucket_counts = defaultdict(int)
    for s in first_any:
        if s is None:
            continue
        for (lo, hi) in FIRST_HALLUC_BUCKETS:
            if lo <= s <= hi:
                bucket_counts[(lo, hi)] += 1
                break
    for (lo, hi) in FIRST_HALLUC_BUCKETS:
        c = bucket_counts[(lo, hi)]
        pct = 100 * c / max(1, n_with_any)
        lines.append(f"      {bucket_label(lo, hi):12} : {c:4} instances ({pct:5.1f}%)")
    if n_with_any == 0:
        lines.append("      (none)")

    lines.append("\n    First occurrence by type (median step index, among instances that have that type):")
    for ht in ["C1_think_loop", "C2_tasktrack_loop", "A2_invented_search_target",
               "A1_invented_file_content", "A3_invalid_edit_path",
               "B1_state_amnesia", "B2_wrong_path", "B3_failed_command"]:
        vals = [x for x in first_by_type[ht] if x is not None]
        if not vals:
            lines.append(f"      {ht}: no occurrences")
            continue
        vals.sort()
        med = vals[len(vals) // 2]
        lines.append(f"      {ht}: median first at step {med}  (n={len(vals)})")

    lines.append("\n[2] First hallucination step — resolved vs failed")
    res_ids = [iid for iid in inst_ids if iid in resolved_set]
    fail_ids = [iid for iid in inst_ids if iid not in resolved_set]
    res_first = [first_halluc_step(by_inst[iid]) for iid in res_ids]
    fail_first = [first_halluc_step(by_inst[iid]) for iid in fail_ids]
    res_with = [x for x in res_first if x is not None]
    fail_with = [x for x in fail_first if x is not None]
    if res_with:
        res_with.sort()
        med_res = res_with[len(res_with) // 2]
        lines.append(f"    Resolved: {len(res_with)}/{len(res_ids)} with any hallucination; median first step = {med_res}")
    else:
        lines.append(f"    Resolved: {len(res_with)}/{len(res_ids)} with any hallucination")
    if fail_with:
        fail_with.sort()
        med_fail = fail_with[len(fail_with) // 2]
        lines.append(f"    Failed:   {len(fail_with)}/{len(fail_ids)} with any hallucination; median first step = {med_fail}")
    else:
        lines.append(f"    Failed:   {len(fail_with)}/{len(fail_ids)} with any hallucination")

    lines.append("\n[3] Error path evolution (cumulative % of instances that have seen >=1 hallucination by step N)")
    max_step = max(len(by_inst[iid]) for iid in inst_ids)
    check_steps = [1, 3, 5, 8, 10, 15, 20]
    check_steps = [s for s in check_steps if s <= max_step]
    if max_step <= 20:
        check_steps = sorted(set(check_steps) | set([max_step]))
    for step_n in check_steps:
        count_by_n = sum(1 for s in first_any if s is not None and s <= step_n)
        pct = 100 * count_by_n / n_inst
        lines.append(f"    By step {step_n:2}: {count_by_n:4} / {n_inst} ({pct:5.1f}%) have had at least one hallucination")
    lines.append("")

    lines.append("[4] Typical trajectory structure")
    traj_len.sort()
    med_len = traj_len[len(traj_len) // 2] if traj_len else 0
    lines.append(f"    Median trajectory length (steps): {med_len}")
    if res_ids:
        res_lens = [len(by_inst[iid]) for iid in res_ids]
        res_lens.sort()
        lines.append(f"    Median length (resolved): {res_lens[len(res_lens)//2]}")
    if fail_ids:
        fail_lens = [len(by_inst[iid]) for iid in fail_ids]
        fail_lens.sort()
        lines.append(f"    Median length (failed):  {fail_lens[len(fail_lens)//2]}")
    lines.append("")
    lines.append("    Example trajectory signatures (first 12 steps: action/halluc type):")
    seen: Dict[str, int] = defaultdict(int)
    for sig, res in signatures:
        key = (sig, res)
        seen[key] += 1
    by_freq = sorted(seen.items(), key=lambda x: -x[1])[:8]
    for (sig, res), cnt in by_freq:
        lines.append(f"      [{res}] (n={cnt})  {sig[:70]}{'...' if len(sig)>70 else ''}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_actions", default="", help="Path to step_actions.csv")
    ap.add_argument("--instance_summary", default="", help="Path to instance_summary.csv (optional, for resolved/failed)")
    ap.add_argument("--input_dir", default="", help="Directory: look for step_actions.csv (+ instance_summary.csv) in each subdir")
    ap.add_argument("--out_report", default="", help="Output report path")
    args = ap.parse_args()

    def do_one(step_path: str, summary_path: Optional[str], label: str) -> str:
        step_rows = load_csv(step_path)
        summary_rows = load_csv(summary_path) if summary_path and os.path.isfile(summary_path) else None
        return run_trace_localization(step_rows, summary_rows, label)

    if args.step_actions:
        path = os.path.abspath(args.step_actions)
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")
        summary_path = args.instance_summary or path.replace("step_actions.csv", "instance_summary.csv")
        if not os.path.isfile(summary_path):
            summary_path = None
        label = os.path.basename(os.path.dirname(path)) or "step_actions"
        report = do_one(path, summary_path, label)
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
            if "step_actions.csv" in files:
                found.append(os.path.join(root, "step_actions.csv"))
        if not found:
            raise SystemExit(f"No step_actions.csv under {input_dir}")
        for step_path in sorted(found):
            d = os.path.dirname(step_path)
            summary_path = os.path.join(d, "instance_summary.csv")
            label = os.path.basename(d) or "run"
            report = do_one(step_path, summary_path, label)
            print(report)
            out_path = os.path.join(d, "trace_localization_report.txt")
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
                if "step_actions.csv" in files:
                    found.append(os.path.join(root, "step_actions.csv"))
            if not found:
                raise SystemExit(f"No step_actions.csv under {default_dir}. Use --step_actions or --input_dir.")
            for step_path in sorted(found):
                d = os.path.dirname(step_path)
                summary_path = os.path.join(d, "instance_summary.csv")
                label = os.path.basename(d) or "run"
                report = do_one(step_path, summary_path, label)
                print(report)
                out_path = os.path.join(d, "trace_localization_report.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Written: {out_path}")
        else:
            raise SystemExit("Specify --step_actions <path> or --input_dir <dir>.")


if __name__ == "__main__":
    main()

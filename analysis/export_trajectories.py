#!/usr/bin/env python3
"""
Trajectory export script for OpenHands style logs.

Inputs
- output.jsonl: instance-level records with history, error, test_result.git_patch, metrics, etc.
- llm_completions/*.json (optional): per-call completion logs, used to aggregate token stats.

Outputs
1) instance_summary.csv (one row per instance)
2) step_actions.csv (one row per tool action step)

How to run (example)
  python3 export_trajectories.py \
    --output_jsonl /mnt/data/output.jsonl \
    --completions_glob "/mnt/data/openai__*.json" \
    --out_dir ./analysis_out

Notes
- This script does NOT require exact step-level alignment between completions and history.
  It aggregates tokens per instance when it can infer instance_id from completion content.
  If it cannot infer instance_id, it will still produce the two CSVs from output.jsonl alone.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict


# -----------------------------
# Helpers
# -----------------------------

def safe_read_text(path: str, max_bytes: int = 2_000_000) -> str:
    with open(path, "rb") as f:
        b = f.read(max_bytes)
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return b.decode(errors="replace")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clip(s: str, n: int) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    return s[:n]


def stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


# -----------------------------
# Action categorization
# -----------------------------

SEARCH_RE = re.compile(r"\b(rg|grep|ripgrep|find|locate|fd|ls)\b")
READ_RE = re.compile(r"\b(cat|sed|head|tail|less|more|awk)\b")
TEST_RE = re.compile(r"\b(pytest|tox|nox|runtests\.py|python\s+-m\s+pytest)\b")
BUILD_RE = re.compile(r"\b(pip|pip3|poetry|uv|conda|mamba|apt-get|apt|brew)\b")
GIT_RE = re.compile(r"\bgit\b")
EDIT_RE = re.compile(r"\b(apply_patch)\b")

MULTI_CMD_ERR_NEEDLE = "Cannot execute multiple commands at once"
MAXITER_NEEDLE = "reached maximum iteration"
STUCK_IN_LOOP_NEEDLE = "AgentStuckInLoopError"
TIMEOUT_NEEDLES = ["litellm.Timeout", "APITimeoutError", "Request timed out"]
LLM_SERVER_ERROR_NEEDLE = "ERROR_LLM_INTERNAL_SERVER_ERROR"
MODEL_NOT_FOUND_NEEDLE = "NotFoundError"

# Hallucination detection
# Failed edits: agent hallucinated file content that doesn't exist verbatim
FAILED_EDIT_NEEDLES = [
    "No replacement was performed",
    "did not appear verbatim",
    "ERROR: No replacement",
]
# Invalid path in edit: agent used a wrong / nonexistent / already-existing path
INVALID_PATH_EDIT_NEEDLES = [
    "Invalid `path` parameter",
    "File already exists at",
    "path should be an absolute path",
    "is a directory and only",
]
# Actions that don't change environment state (pure cognition / bookkeeping)
# condensation_request = context window overflow compaction, not a hallucination
# but signals context loss that often precedes B1/C1 hallucinations
STALL_ACTIONS = frozenset({"think", "task_tracking", "recall", "system", "message", "condensation_request"})

# Hallucination taxonomy
# Type A — Content Hallucination: model invented content that doesn't exist in the repo
#   A1_invented_file_content  edit failed: old_str not found verbatim
#   A2_invented_search_target grep/rg searched for a string that isn't in the codebase
#   A3_invalid_edit_path      edit used a path that is wrong/already-exists/relative
# Type B — State Hallucination: model has wrong belief about current environment state
#   B1_state_amnesia   exact same command repeated (forgot it already ran)
#   B2_wrong_path      tried to access a nonexistent file / directory via shell
#   B3_failed_command  non-search run command failed (wrong tool usage / bad assumptions)
# Type C — Goal Drift / Behavioral Loop: model lost track of task and spun
#   C1_think_loop      run of >= N consecutive "think" steps (no real action)
#   C2_tasktrack_loop  run of >= N task_tracking plan/view cycles with no progress
CONSEC_STALL_THRESHOLD = 3   # min consecutive stall steps to qualify as a C-type loop

GREP_CMD_RE = re.compile(r"\b(grep|rg|ripgrep)\b")
LS_FIND_RE = re.compile(r"\b(ls|find|fd|locate)\b")
B2_NEEDLES = [
    "No such file or directory",
    "cannot access",
    "does not exist",
    "No such file",
]

HALLUC_TYPES = ["A1_invented_file_content", "A2_invented_search_target", "A3_invalid_edit_path",
                "B1_state_amnesia", "B2_wrong_path", "B3_failed_command",
                "C1_think_loop", "C2_tasktrack_loop", "none"]


def categorize_command(cmd: str) -> str:
    c = cmd.strip()

    # Very common Git subcommands
    if GIT_RE.search(c):
        return "Git/Inspect"

    if TEST_RE.search(c):
        return "Test"

    if BUILD_RE.search(c):
        return "Build/Install"

    if EDIT_RE.search(c):
        return "Edit"

    if READ_RE.search(c):
        return "Read"

    if SEARCH_RE.search(c):
        return "Search"

    # Fallbacks
    if c == "":
        return "Other"
    return "Other"


def detect_tool_misuse(history_text: str) -> int:
    return history_text.count(MULTI_CMD_ERR_NEEDLE)


def detect_maxiter_error(err_text: str) -> bool:
    return MAXITER_NEEDLE in err_text.lower()


def is_failed_edit_result(result_text: str) -> bool:
    """Return True if an edit action failed because the old_str wasn't found (hallucinated content)."""
    return any(needle in result_text for needle in FAILED_EDIT_NEEDLES)


def is_invalid_path_edit(result_text: str) -> bool:
    """Return True if an edit action failed because the path was wrong/invalid (A3)."""
    return any(needle in result_text for needle in INVALID_PATH_EDIT_NEEDLES)


def classify_step_hallucination(
    action_raw: str,
    cmd: str,
    is_failed_edit: bool,
    is_invalid_path: bool,
    is_empty_failed_search: bool,
    is_repeated_cmd: bool,
    result_success: str,
    result_text: str,
    action_category: str,
) -> str:
    """
    Assign a hallucination type to a single step.
    Priority A1 > A3 > A2 > B2 > B3 > B1 > none.
    C1/C2 are assigned later by postprocess_stall_types().
    """
    if is_failed_edit:
        return "A1_invented_file_content"
    if is_invalid_path:
        return "A3_invalid_edit_path"
    if is_empty_failed_search:
        # differentiate: grep for nonexistent string vs ls/find on wrong path
        if GREP_CMD_RE.search(cmd):
            return "A2_invented_search_target"
        if LS_FIND_RE.search(cmd):
            return "B2_wrong_path"
        return "A2_invented_search_target"   # default for unknown search
    # B2: command ran but shell reported the path doesn't exist
    if action_raw == "run" and any(n in result_text for n in B2_NEEDLES):
        return "B2_wrong_path"
    # B3: non-search run command failed outright (wrong tool usage / bad env assumptions)
    # Exclude: pure search failures (covered by A2/B2), git inspect (often output piped
    # to grep returning empty), and test failures (those are real test outcomes)
    if (action_raw == "run"
            and result_success == "False"
            and action_category not in ("Search", "Test", "Git/Inspect")):
        return "B3_failed_command"
    if is_repeated_cmd:
        return "B1_state_amnesia"
    return "none"


def postprocess_stall_types(step_rows: List[Dict[str, Any]]) -> None:
    """
    Second pass: mark stall steps that are part of a consecutive run of >= CONSEC_STALL_THRESHOLD
    as C1_think_loop or C2_tasktrack_loop.  Only updates steps still labelled 'none'.
    Steps within a run are guaranteed adjacent because extract_from_output_jsonl processes
    instances sequentially.
    """
    i = 0
    n = len(step_rows)
    while i < n:
        row = step_rows[i]
        if not row.get("is_stall_action", False):
            i += 1
            continue
        # collect the full run of consecutive stalls for this instance
        inst = row["instance_id"]
        j = i
        run_indices: List[int] = []
        while (j < n
               and step_rows[j]["instance_id"] == inst
               and step_rows[j].get("is_stall_action", False)):
            run_indices.append(j)
            j += 1
        if len(run_indices) >= CONSEC_STALL_THRESHOLD:
            dominant_action = Counter(
                step_rows[k]["action_raw"] for k in run_indices
            ).most_common(1)[0][0]
            ctype = (
                "C2_tasktrack_loop"
                if dominant_action == "task_tracking"
                else "C1_think_loop"
            )
            for k in run_indices:
                if step_rows[k]["hallucination_type"] == "none":
                    step_rows[k]["hallucination_type"] = ctype
        i = j if j > i else i + 1


def attach_halluc_type_counts_to_instances(
    instance_rows: List[Dict[str, Any]],
    step_rows: List[Dict[str, Any]],
) -> None:
    """
    After postprocess_stall_types(), aggregate per-instance type counts and
    dominant_halluc_type, then merge them into instance_rows.
    """
    # index instance_rows by instance_id
    inst_index: Dict[str, Dict[str, Any]] = {r["instance_id"]: r for r in instance_rows}
    # accumulate
    type_counts: Dict[str, Counter] = defaultdict(Counter)
    for sr in step_rows:
        ht = sr.get("hallucination_type", "none")
        if ht != "none":
            type_counts[sr["instance_id"]][ht] += 1
    # attach to instance rows
    for r in instance_rows:
        inst = r["instance_id"]
        ctr = type_counts.get(inst, Counter())
        for ht in HALLUC_TYPES:
            if ht == "none":
                continue
            r[f"n_{ht}"] = ctr.get(ht, 0)
        dominant = ctr.most_common(1)[0][0] if ctr else "none"
        r["dominant_halluc_type"] = dominant


def hallucination_score(n_failed_edits: int, n_empty_failed_searches: int,
                        n_repeated_cmd_steps: int, max_consec_stalls: int,
                        multi_cmd_error_count: int) -> float:
    """
    Composite hallucination score (higher = more hallucination evidence).
    Weights:
      - failed_edit          x3  (strong: agent invented file content)
      - repeated_cmd         x2  (agent stuck in a loop)
      - multi_cmd_error      x2  (agent issued unparseable multi-command)
      - empty_failed_search  x1  (mild: searched for nonexistent content)
      - max_consec_stalls    x0.5 per stall step (prolonged think loops)
    """
    return (
        3.0 * n_failed_edits
        + 2.0 * n_repeated_cmd_steps
        + 2.0 * multi_cmd_error_count
        + 1.0 * n_empty_failed_searches
        + 0.5 * max_consec_stalls
    )


# -----------------------------
# Completion token aggregation (best-effort)
# -----------------------------

@dataclass
class TokenAgg:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_completions: int = 0

    def add(self, p: int, c: int, t: int) -> None:
        self.prompt_tokens += int(p or 0)
        self.completion_tokens += int(c or 0)
        self.total_tokens += int(t or 0)
        self.n_completions += 1


INSTANCE_ID_RE = re.compile(
    r"(astropy__astropy-\d+|django__django-\d+|[a-zA-Z0-9_.-]+__\w+-\d+)"
)


def infer_instance_id_from_completion(obj: Any) -> Optional[str]:
    """
    Best-effort inference of instance_id from a completion JSON.
    We search common fields and fall back to scanning serialized text.
    """
    if isinstance(obj, dict):
        # common spots
        for k in ["instance_id", "task_id", "problem_id", "id"]:
            v = obj.get(k)
            if isinstance(v, str) and "__" in v:
                return v

        # scan messages if present
        for mk in ["messages", "input", "prompt"]:
            v = obj.get(mk)
            if isinstance(v, (list, dict, str)):
                s = stringify(v)
                m = INSTANCE_ID_RE.search(s)
                if m:
                    return m.group(1)

        # fallback: scan whole dict
        s = stringify(obj)
        m = INSTANCE_ID_RE.search(s)
        if m:
            return m.group(1)

    # fallback: string scan
    s = stringify(obj)
    m = INSTANCE_ID_RE.search(s)
    if m:
        return m.group(1)
    return None


def extract_tokens_from_completion(obj: Any) -> Tuple[int, int, int]:
    """
    Returns (prompt_tokens, completion_tokens, total_tokens) if present.
    Supports a few common schema variants.
    """
    p = c = t = 0
    if not isinstance(obj, dict):
        return p, c, t

    # OpenAI style
    usage = obj.get("usage")
    if isinstance(usage, dict):
        p = usage.get("prompt_tokens", 0) or 0
        c = usage.get("completion_tokens", 0) or 0
        t = usage.get("total_tokens", 0) or 0
        return int(p), int(c), int(t)

    # Other variants
    for path in [
        ("response", "usage"),
        ("result", "usage"),
        ("metadata", "usage"),
    ]:
        cur = obj
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, dict):
            p = cur.get("prompt_tokens", 0) or 0
            c = cur.get("completion_tokens", 0) or 0
            t = cur.get("total_tokens", 0) or 0
            return int(p), int(c), int(t)

    return 0, 0, 0


# -----------------------------
# Main extraction from output.jsonl
# -----------------------------

def extract_from_output_jsonl(output_jsonl: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (instance_rows, step_rows)
    """
    instance_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for rec in iter_jsonl(output_jsonl):
        instance_id = rec.get("instance_id")
        history = rec.get("history") or []
        err = rec.get("error")
        err_str = stringify(err)
        metrics = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
        test_result = rec.get("test_result") if isinstance(rec.get("test_result"), dict) else {}
        git_patch = test_result.get("git_patch", "") if isinstance(test_result.get("git_patch"), str) else ""

        # Iterations: count tool-action starts (action not None)
        n_iterations = 0
        n_steps_total = 0

        # Tool misuse evidence
        history_text = stringify(history)
        multi_cmd_error_count = detect_tool_misuse(history_text)

        # Hallucination counters
        n_failed_edits_inst = 0
        n_empty_failed_searches_inst = 0
        n_repeated_cmd_steps_inst = 0
        max_consec_stalls_inst = 0
        _consec_stalls = 0        # running consecutive stall counter
        _prev_cmd = ""            # last non-empty command seen

        # Patch metrics
        patch_lines = len(git_patch.splitlines()) if git_patch else 0
        patch_chars = len(git_patch) if git_patch else 0

        # Basic resolved heuristic (strict resolved needs real test outcome, not available here)
        is_resolved = err is None

        # Error type
        error_type = ""
        if err_str:
            if detect_maxiter_error(err_str):
                error_type = "maxiter"
            elif STUCK_IN_LOOP_NEEDLE in err_str:
                error_type = "stuck_in_loop"
            elif any(n in err_str for n in TIMEOUT_NEEDLES):
                error_type = "api_timeout"
            elif LLM_SERVER_ERROR_NEEDLE in err_str:
                error_type = "llm_server_error"
            elif MODEL_NOT_FOUND_NEEDLE in err_str:
                error_type = "model_not_found"
            elif "Traceback" in err_str:
                error_type = "runtime"
            else:
                error_type = "other"

        # Walk history to extract step rows
        # We treat each dict item with action != None as a step "start"
        # Then we try to attach the immediate next item with action == None as its result.
        idx = 0
        step_idx = 0
        while idx < len(history):
            item = history[idx]
            n_steps_total += 1
            if isinstance(item, dict) and item.get("action") is not None:
                n_iterations += 1
                action_raw = item.get("action")
                observation = item.get("observation", "")
                # Best effort command extraction
                cmd = ""
                # Common patterns in OpenHands logs
                if isinstance(item.get("args"), dict):
                    args = item["args"]
                    for ck in ["command", "cmd", "shell_command"]:
                        if ck in args and isinstance(args[ck], str):
                            cmd = args[ck]
                            break
                if not cmd and isinstance(item.get("command"), str):
                    cmd = item["command"]
                if not cmd and isinstance(item.get("content"), str) and action_raw == "run":
                    cmd = item["content"]

                # Attach result if next item is an observation (action None)
                result_success = ""
                result_content = ""
                result_observation = ""
                if idx + 1 < len(history):
                    nxt = history[idx + 1]
                    if isinstance(nxt, dict) and nxt.get("action") is None:
                        # success often exists on result items
                        if "success" in nxt:
                            result_success = stringify(nxt.get("success"))
                        result_content = stringify(nxt.get("content"))
                        result_observation = stringify(nxt.get("observation"))
                category = "Edit" if action_raw == "edit" else categorize_command(cmd) if action_raw == "run" else "Read" if action_raw == "read" else "Other"

                # ---- Hallucination detection (step level) ----
                _is_stall = action_raw in STALL_ACTIONS

                if _is_stall:
                    _consec_stalls += 1
                else:
                    max_consec_stalls_inst = max(max_consec_stalls_inst, _consec_stalls)
                    _consec_stalls = 0

                _result_text = result_content + " " + result_observation

                # Failed edit: agent tried to replace text that doesn't exist
                _is_failed_edit = False
                _is_invalid_path = False
                if action_raw == "edit":
                    if is_failed_edit_result(_result_text):
                        _is_failed_edit = True
                        n_failed_edits_inst += 1
                    elif is_invalid_path_edit(_result_text):
                        _is_invalid_path = True

                # Empty-result search: grep/find returned nothing (result_success=False)
                _is_empty_failed_search = False
                if action_raw == "run" and category == "Search" and result_success == "False":
                    _is_empty_failed_search = True
                    n_empty_failed_searches_inst += 1

                # Repeated command: same non-empty command issued again
                _is_repeated_cmd = False
                if cmd and not _is_stall:
                    if cmd == _prev_cmd:
                        _is_repeated_cmd = True
                        n_repeated_cmd_steps_inst += 1
                    _prev_cmd = cmd

                # Classify hallucination type (C1/C2 filled in by postprocess_stall_types later)
                _halluc_type = (
                    "none"
                    if _is_stall
                    else classify_step_hallucination(
                        action_raw, cmd,
                        _is_failed_edit, _is_invalid_path,
                        _is_empty_failed_search,
                        _is_repeated_cmd, result_success,
                        _result_text, category,
                    )
                )

                step_rows.append({
                    "instance_id": instance_id,
                    "step_idx": step_idx,
                    "action_raw": action_raw,
                    "action_category": category,
                    "command": clip(cmd, 500),
                    "observation": clip(stringify(observation), 120),
                    "result_success": result_success,
                    "result_observation": clip(result_observation, 80),
                    "result_preview": clip(result_content, 300),
                    # Hallucination flags (raw signals)
                    "is_stall_action": _is_stall,
                    "is_failed_edit": _is_failed_edit,
                    "is_empty_failed_search": _is_empty_failed_search,
                    "is_repeated_cmd": _is_repeated_cmd,
                    # Taxonomy label (A1/A2/B1/B2/C1/C2/none)
                    "hallucination_type": _halluc_type,
                })
                step_idx += 1

            idx += 1

        # Flush any trailing stall run
        max_consec_stalls_inst = max(max_consec_stalls_inst, _consec_stalls)

        _hscore = hallucination_score(
            n_failed_edits_inst, n_empty_failed_searches_inst,
            n_repeated_cmd_steps_inst, max_consec_stalls_inst,
            multi_cmd_error_count,
        )

        # Save instance row
        instance_rows.append({
            "instance_id": instance_id,
            "is_resolved": is_resolved,
            "n_iterations": n_iterations,
            "history_len": len(history),
            "multi_cmd_error_count": multi_cmd_error_count,
            "patch_lines": patch_lines,
            "patch_chars": patch_chars,
            "error_type": error_type,
            "error_preview": clip(err_str, 120),
            "model_cost": metrics.get("cost", ""),
            "model_latency": metrics.get("latency", ""),
            # Hallucination metrics
            "n_failed_edits": n_failed_edits_inst,
            "n_empty_failed_searches": n_empty_failed_searches_inst,
            "n_repeated_cmd_steps": n_repeated_cmd_steps_inst,
            "max_consec_stalls": max_consec_stalls_inst,
            "hallucination_score": round(_hscore, 2),
        })

    return instance_rows, step_rows


def attach_token_agg_to_instances(
    instance_rows: List[Dict[str, Any]],
    completions_glob: Optional[str],
) -> None:
    """
    Best-effort token aggregation:
    - Parse each completion json
    - Infer instance_id (if possible)
    - Sum tokens per instance_id
    Then attach prompt_tokens_sum, completion_tokens_sum, total_tokens_sum, n_completions
    """
    if not completions_glob:
        # still add empty cols for consistency
        for r in instance_rows:
            r["prompt_tokens_sum"] = ""
            r["completion_tokens_sum"] = ""
            r["total_tokens_sum"] = ""
            r["n_completions"] = ""
        return

    paths = sorted(glob.glob(completions_glob))
    agg: Dict[str, TokenAgg] = defaultdict(TokenAgg)
    unknown = 0

    for p in paths:
        try:
            obj = load_json(p)
        except Exception:
            # sometimes these are huge or partially written, try text parse fallback
            txt = safe_read_text(p, max_bytes=2_000_000)
            try:
                obj = json.loads(txt)
            except Exception:
                unknown += 1
                continue

        inst = infer_instance_id_from_completion(obj)
        pt, ct, tt = extract_tokens_from_completion(obj)
        if inst:
            agg[inst].add(pt, ct, tt)
        else:
            unknown += 1

    # attach
    for r in instance_rows:
        inst = r.get("instance_id")
        a = agg.get(inst)
        if a and a.n_completions > 0:
            r["prompt_tokens_sum"] = a.prompt_tokens
            r["completion_tokens_sum"] = a.completion_tokens
            r["total_tokens_sum"] = a.total_tokens
            r["n_completions"] = a.n_completions
        else:
            r["prompt_tokens_sum"] = 0
            r["completion_tokens_sum"] = 0
            r["total_tokens_sum"] = 0
            r["n_completions"] = 0

    # helpful note printed by main
    return


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")

    # stable field order
    fieldnames = list(rows[0].keys())
    # include any extra keys that might appear later
    extra = set()
    for r in rows[1:]:
        extra |= set(r.keys()) - set(fieldnames)
    fieldnames += sorted(extra)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_jsonl", required=True, help="Path to output.jsonl")
    ap.add_argument(
        "--completions_glob",
        default="",
        help='Glob for llm_completions json files, for token aggregation (optional). Example: "/path/llm_completions/*.json"',
    )
    ap.add_argument("--out_dir", default="./analysis_out", help="Directory to write CSV outputs")
    args = ap.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)

    instance_rows, step_rows = extract_from_output_jsonl(args.output_jsonl)
    postprocess_stall_types(step_rows)
    attach_halluc_type_counts_to_instances(instance_rows, step_rows)
    attach_token_agg_to_instances(instance_rows, args.completions_glob or None)

    instance_csv = os.path.join(out_dir, "instance_summary.csv")
    steps_csv = os.path.join(out_dir, "step_actions.csv")

    write_csv(instance_csv, instance_rows)
    write_csv(steps_csv, step_rows)

    # quick console summary
    total = len(instance_rows)
    resolved = sum(1 for r in instance_rows if r.get("is_resolved") is True)
    maxiter = sum(1 for r in instance_rows if r.get("error_type") == "maxiter")
    stuck_in_loop = sum(1 for r in instance_rows if r.get("error_type") == "stuck_in_loop")
    api_timeout = sum(1 for r in instance_rows if r.get("error_type") == "api_timeout")
    llm_server_error = sum(1 for r in instance_rows if r.get("error_type") == "llm_server_error")
    model_not_found = sum(1 for r in instance_rows if r.get("error_type") == "model_not_found")
    other_error = sum(1 for r in instance_rows if r.get("error_type") == "other")
    nonempty_patch = sum(1 for r in instance_rows if int(r.get("patch_lines") or 0) > 0)
    multi_cmd = sum(1 for r in instance_rows if int(r.get("multi_cmd_error_count") or 0) > 0)

    print(f"Wrote: {instance_csv}")
    print(f"Wrote: {steps_csv}")
    print("Summary")
    print(f"  instances: {total}")
    print(f"  resolved (heuristic): {resolved}")
    print(f"  maxiter errors (excluded): {maxiter}")
    print(f"  stuck-in-loop errors (AgentStuckInLoopError): {stuck_in_loop}")
    print(f"  api timeout errors: {api_timeout}")
    print(f"  llm server errors: {llm_server_error}")
    print(f"  model not found errors: {model_not_found}")
    if other_error:
        print(f"  other/unknown errors: {other_error}")
    print(f"  nonempty patches: {nonempty_patch}")
    print(f"  instances with multi-cmd tool error evidence: {multi_cmd}")

    # Hallucination summary
    inst_with_failed_edit = sum(1 for r in instance_rows if int(r.get("n_failed_edits") or 0) > 0)
    inst_with_empty_search = sum(1 for r in instance_rows if int(r.get("n_empty_failed_searches") or 0) > 0)
    inst_with_repeated_cmd = sum(1 for r in instance_rows if int(r.get("n_repeated_cmd_steps") or 0) > 0)
    inst_with_high_stalls = sum(1 for r in instance_rows if int(r.get("max_consec_stalls") or 0) >= 5)
    high_halluc = sum(1 for r in instance_rows if float(r.get("hallucination_score") or 0) >= 5.0)
    scores = [float(r.get("hallucination_score") or 0) for r in instance_rows]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print("Hallucination signals")
    print(f"  instances with >=1 failed edit (hallucinated file content): {inst_with_failed_edit}")
    print(f"  instances with >=1 empty failed search (searched nonexistent text): {inst_with_empty_search}")
    print(f"  instances with repeated commands (stuck in loop): {inst_with_repeated_cmd}")
    print(f"  instances with >=5 consecutive stall actions (think/task_tracking): {inst_with_high_stalls}")
    print(f"  instances with hallucination_score >= 5.0: {high_halluc}")
    print(f"  avg hallucination_score: {avg_score:.2f}")

    print("Hallucination taxonomy (per instance, steps with any halluc type != none)")
    type_labels = {
        "A1_invented_file_content": "A1 Content: invented file content (failed edit)",
        "A2_invented_search_target": "A2 Content: invented search string (empty grep)",
        "A3_invalid_edit_path": "A3 Content: wrong/invalid path in edit",
        "B1_state_amnesia": "B1 State:   repeated same command (amnesia)",
        "B2_wrong_path": "B2 State:   accessed nonexistent file/path via shell",
        "B3_failed_command": "B3 State:   non-search run command failed (bad env assumption)",
        "C1_think_loop": "C1 Loop:    prolonged think chain (>= {} consecutive)".format(CONSEC_STALL_THRESHOLD),
        "C2_tasktrack_loop": "C2 Loop:    task_tracking plan/view loop (>= {} consecutive)".format(CONSEC_STALL_THRESHOLD),
    }
    for ht, label in type_labels.items():
        col = f"n_{ht}"
        count = sum(1 for r in instance_rows if int(r.get(col) or 0) > 0)
        total_steps = sum(int(r.get(col) or 0) for r in instance_rows)
        print(f"  {label}: {count} instances, {total_steps} steps")

    dom_counter: Counter = Counter(r.get("dominant_halluc_type", "none") for r in instance_rows)
    print("  Dominant hallucination type per instance:")
    for ht, cnt in dom_counter.most_common():
        print(f"    {ht}: {cnt} instances")

    if args.completions_glob:
        paths = glob.glob(args.completions_glob)
        print(f"  completion files matched: {len(paths)}")
        print("  token aggregation is best-effort, if instance_id cannot be inferred for some completion files, those files are ignored")


if __name__ == "__main__":
    main()
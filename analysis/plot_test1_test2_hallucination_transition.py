from __future__ import annotations

import csv
from collections import Counter
from html import escape
from pathlib import Path


BASE_DIR = Path("/home/groupuser/coding_agent_project/experiments/oh_runs/analysis/out")
TEST1_CSV = BASE_DIR / "A_optimized_test_1_maxiter_20_verified" / "instance_summary.csv"
TEST2_CSV = BASE_DIR / "A_optimized_test_2_maxiter_20_verified" / "instance_summary.csv"
OUTPUT_DIR = BASE_DIR / "A_test1_vs_test2_hallucination_transition"
SVG_OUT = OUTPUT_DIR / "hallucination_transition_test1_to_test2.svg"


LABEL_TEXT = {
    "Fixed": "Fixed",
    "C1_think_loop": "C1 Think loop",
    "A2_invented_search_target": "A2 Invented search target",
    "B3_failed_command": "B3 Failed command",
    "C2_tasktrack_loop": "C2 Tasktrack loop",
    "Unknown": "Unknown",
}

COLORS = {
    "Fixed": "#59C36A",
    "C1_think_loop": "#4E79A7",
    "A2_invented_search_target": "#F2C94C",
    "B3_failed_command": "#E15759",
    "C2_tasktrack_loop": "#7F7F7F",
    "Unknown": "#B07AA1",
}

ORDER = [
    "C1_think_loop",
    "A2_invented_search_target",
    "B3_failed_command",
    "C2_tasktrack_loop",
    "Fixed",
    "Unknown",
]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["instance_id"]: row for row in csv.DictReader(f) if row.get("instance_id")}


def status_label(row: dict[str, str]) -> str:
    if row.get("is_resolved") == "True":
        return "Fixed"
    return row.get("dominant_halluc_type") or "Unknown"


def to_px(value: float, total_px: int) -> float:
    return value * total_px


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def bezier_flow_path(x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float) -> str:
    ctrl = (x1 - x0) * 0.38
    return (
        f"M {x0:.2f},{y0a:.2f} "
        f"C {x0 + ctrl:.2f},{y0a:.2f} {x1 - ctrl:.2f},{y1a:.2f} {x1:.2f},{y1a:.2f} "
        f"L {x1:.2f},{y1b:.2f} "
        f"C {x1 - ctrl:.2f},{y1b:.2f} {x0 + ctrl:.2f},{y0b:.2f} {x0:.2f},{y0b:.2f} Z"
    )


def stack_positions(counts: Counter[str], active_labels: list[str], top: float, bottom: float, gap: float) -> dict[str, tuple[float, float]]:
    total = sum(counts.values())
    slots = len(active_labels)
    usable_height = (top - bottom) - gap * (slots - 1)
    positions: dict[str, tuple[float, float]] = {}
    cursor = top
    for label in active_labels:
        height = usable_height * counts[label] / total
        positions[label] = (cursor - height, cursor)
        cursor = cursor - height - gap
    return positions


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test1 = read_rows(TEST1_CSV)
    test2 = read_rows(TEST2_CSV)
    common_ids = sorted(set(test1) & set(test2))
    if not common_ids:
        raise SystemExit("No shared instance_id values found between the two CSV files.")

    transitions = Counter((status_label(test1[iid]), status_label(test2[iid])) for iid in common_ids)
    left_counts = Counter(src for src, _ in transitions.elements())
    right_counts = Counter(dst for _, dst in transitions.elements())

    active_labels = [label for label in ORDER if left_counts[label] or right_counts[label]]
    total = len(common_ids)

    left_pos = stack_positions(left_counts, active_labels, top=0.86, bottom=0.12, gap=0.028)
    right_pos = stack_positions(right_counts, active_labels, top=0.86, bottom=0.12, gap=0.028)

    left_offsets = {label: left_pos[label][0] for label in active_labels}
    right_offsets = {label: right_pos[label][0] for label in active_labels}

    unchanged = sum(n for (src, dst), n in transitions.items() if src == dst)
    improved = sum(n for (src, dst), n in transitions.items() if dst == "Fixed" and src != "Fixed")
    regressed = sum(n for (src, dst), n in transitions.items() if src == "Fixed" and dst != "Fixed")

    width = 1600
    height = 900
    x_left = to_px(0.20, width)
    x_right = to_px(0.80, width)
    bar_w = to_px(0.05, width)
    font_family = "Arial, Helvetica, sans-serif"

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#FBF8F1"/>',
        '<stop offset="100%" stop-color="#F1EEE7"/>',
        "</linearGradient>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#4B5563" flood-opacity="0.12"/>',
        "</filter>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>',
        f'<text x="{width / 2:.1f}" y="70" text-anchor="middle" font-family="{font_family}" font-size="34" font-weight="700" fill="#20262E">Hallucination Type Transition: Test1 -&gt; Test2</text>',
        f'<text x="{width / 2:.1f}" y="112" text-anchor="middle" font-family="{font_family}" font-size="20" fill="#4B5563">Shared instances: {total} | Unchanged: {unchanged} ({unchanged / total:.0%}) | Improved to Fixed: {improved} | Regressed from Fixed: {regressed}</text>',
        f'<text x="{x_left - bar_w / 2:.1f}" y="860" text-anchor="middle" font-family="{font_family}" font-size="24" font-weight="700" fill="#24303A">Test1</text>',
        f'<text x="{x_right + bar_w / 2:.1f}" y="860" text-anchor="middle" font-family="{font_family}" font-size="24" font-weight="700" fill="#24303A">Test2</text>',
    ]

    parts.append('<g filter="url(#shadow)">')

    for label in active_labels:
        y0, y1 = left_pos[label]
        y0_px = to_px(y0, height)
        y1_px = to_px(y1, height)
        parts.append(
            f'<rect x="{x_left - bar_w:.2f}" y="{y0_px:.2f}" width="{bar_w:.2f}" height="{(y1_px - y0_px):.2f}" '
            f'rx="8" fill="{COLORS[label]}" stroke="#FFFFFF" stroke-width="3"/>'
        )

    for label in active_labels:
        y0, y1 = right_pos[label]
        y0_px = to_px(y0, height)
        y1_px = to_px(y1, height)
        parts.append(
            f'<rect x="{x_right:.2f}" y="{y0_px:.2f}" width="{bar_w:.2f}" height="{(y1_px - y0_px):.2f}" '
            f'rx="8" fill="{COLORS[label]}" stroke="#FFFFFF" stroke-width="3"/>'
        )

    sorted_transitions = sorted(
        transitions.items(),
        key=lambda item: (
            active_labels.index(item[0][0]),
            active_labels.index(item[0][1]),
        ),
    )

    for (src, dst), n in sorted_transitions:
        left_total_height = to_px(left_pos[src][1] - left_pos[src][0], height)
        right_total_height = to_px(right_pos[dst][1] - right_pos[dst][0], height)
        left_h = left_total_height * n / left_counts[src]
        right_h = right_total_height * n / right_counts[dst]
        y0a = to_px(left_offsets[src], height)
        y0b = y0a + left_h
        y1a = to_px(right_offsets[dst], height)
        y1b = y1a + right_h
        left_offsets[src] += (left_h / height)
        right_offsets[dst] += (right_h / height)

        alpha = 0.34 if src == dst else 0.26
        parts.append(
            f'<path d="{bezier_flow_path(x_left, x_right, y0a, y0b, y1a, y1b)}" fill="{rgba(COLORS[src], alpha)}" stroke="rgba(0,0,0,0.08)" stroke-width="1"/>'
        )

        if n >= 3:
            mid_y = (y0a + y0b + y1a + y1b) / 4
            parts.append(
                f'<rect x="{width * 0.5 - 18:.2f}" y="{mid_y - 16:.2f}" width="36" height="30" rx="10" fill="rgba(255,255,255,0.82)"/>'
            )
            parts.append(
                f'<text x="{width * 0.5:.2f}" y="{mid_y + 6:.2f}" text-anchor="middle" font-family="{font_family}" font-size="18" font-weight="700" fill="#1F2933">{n}</text>'
            )

    parts.append("</g>")

    for label in active_labels:
        y0, y1 = left_pos[label]
        pct = left_counts[label] / total * 100
        parts.append(
            f'<text x="{x_left - bar_w - 22:.2f}" y="{to_px((y0 + y1) / 2, height) - 8:.2f}" text-anchor="end" '
            f'font-family="{font_family}" font-size="20" font-weight="700" fill="#24303A">{escape(LABEL_TEXT[label])}</text>'
        )
        parts.append(
            f'<text x="{x_left - bar_w - 22:.2f}" y="{to_px((y0 + y1) / 2, height) + 18:.2f}" text-anchor="end" '
            f'font-family="{font_family}" font-size="18" fill="#475569">{left_counts[label]} ({pct:.0f}%)</text>'
        )

    for label in active_labels:
        y0, y1 = right_pos[label]
        pct = right_counts[label] / total * 100
        parts.append(
            f'<text x="{x_right + bar_w + 22:.2f}" y="{to_px((y0 + y1) / 2, height) - 8:.2f}" text-anchor="start" '
            f'font-family="{font_family}" font-size="20" font-weight="700" fill="#24303A">{escape(LABEL_TEXT[label])}</text>'
        )
        parts.append(
            f'<text x="{x_right + bar_w + 22:.2f}" y="{to_px((y0 + y1) / 2, height) + 18:.2f}" text-anchor="start" '
            f'font-family="{font_family}" font-size="18" fill="#475569">{right_counts[label]} ({pct:.0f}%)</text>'
        )

    note = (
        "Interpretation:\n"
        "Most cases remained in C1 think loop.\n"
        "The largest directional shift was A2 invented search target -> C1 think loop (4 cases).\n"
        "Two instances ended in Fixed in Test2, while one previously fixed instance regressed to A2."
    )
    note_lines = note.splitlines()
    parts.append('<g filter="url(#shadow)">')
    parts.append('<rect x="410" y="655" width="780" height="150" rx="18" fill="#FFFDF8" stroke="#D7D1C5" stroke-width="2"/>')
    for idx, line in enumerate(note_lines):
        parts.append(
            f'<text x="{width / 2:.1f}" y="{690 + idx * 30}" text-anchor="middle" font-family="{font_family}" '
            f'font-size="19" fill="#334155"{" font-weight=\"700\"" if idx == 0 else ""}>{escape(line)}</text>'
        )
    parts.append("</g>")

    parts.append("</svg>")
    SVG_OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {SVG_OUT}")


if __name__ == "__main__":
    main()

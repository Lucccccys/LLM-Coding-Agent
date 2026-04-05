# LLM-Coding-Agent

A research-oriented repository for studying **LLM coding agent behavior**, with a focus on **trajectory export**, **hallucination analysis**, and **success/failure comparison** using OpenHands-style execution logs.

This repository currently contains:

- `OpenHands/`: a local copy/fork of the OpenHands codebase used for experiments
- `analysis/`: Python scripts for exporting trajectories and generating quantitative reports and visualizations

---

## Repository Structure

```text
.
├── OpenHands/   # OpenHands framework and runtime code
├── analysis/    # analysis scripts for trajectory and hallucination studies
└── README.md
```

### `analysis/` scripts

- `export_trajectories.py`  
  Converts OpenHands-style `output.jsonl` logs into:
  - `instance_summary.csv`
  - `step_actions.csv`

- `quantitative_analysis.py`  
  Computes aggregate statistics such as resolution rate and hallucination-type frequencies.

- `success_failure_comparison.py`  
  Compares resolved vs failed runs and reports which hallucination patterns are more associated with failure.

- `trace_localization.py`  
  Analyzes when hallucinations first appear in the trajectory and how they evolve over time.

- `plot_test1_test2_hallucination_transition.py`  
  Generates an SVG transition figure comparing hallucination outcomes across two experiment settings.

---

## What This Project Is For

This project is useful if you want to:

- inspect OpenHands agent trajectories
- measure different hallucination/error categories
- compare successful and failed coding-agent runs
- generate reports and figures for research analysis

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Lucccccys/LLM-Coding-Agent.git
cd LLM-Coding-Agent
```

### 2. Prepare Python

Use Python `3.10+` (recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

If you want to run the full OpenHands stack, refer to the setup instructions inside `OpenHands/README.md`.

---

## Example Analysis Workflow

### Export trajectory tables

```bash
cd analysis
python3 export_trajectories.py \
  --output_jsonl /path/to/output.jsonl \
  --completions_glob "/path/to/llm_completions/*.json" \
  --out_dir ./analysis_out/my_run
```

This produces files such as:

- `analysis_out/my_run/instance_summary.csv`
- `analysis_out/my_run/step_actions.csv`

### Run quantitative summary

```bash
python3 quantitative_analysis.py \
  --instance_summary ./analysis_out/my_run/instance_summary.csv \
  --out_report ./analysis_out/my_run/quantitative_report.txt
```

### Compare success vs failure

```bash
python3 success_failure_comparison.py \
  --instance_summary ./analysis_out/my_run/instance_summary.csv \
  --out_report ./analysis_out/my_run/success_failure_report.txt
```

### Localize hallucinations in time

```bash
python3 trace_localization.py \
  --step_actions ./analysis_out/my_run/step_actions.csv \
  --instance_summary ./analysis_out/my_run/instance_summary.csv \
  --out_report ./analysis_out/my_run/trace_localization_report.txt
```

---

## Notes

- Some scripts assume **OpenHands-style log formats**.
- `plot_test1_test2_hallucination_transition.py` currently uses **hard-coded paths**, so you may need to edit the file before running it on a new machine.
- Sensitive config files and local environments should not be committed.

---

## Upstream Reference

The `OpenHands/` directory is based on the OpenHands project:

- Upstream: https://github.com/All-Hands-AI/OpenHands

Please see `OpenHands/README.md` for the original framework documentation, runtime setup, and licensing details.

---

## License

Please follow the license terms provided by the upstream OpenHands project for the code under `OpenHands/`.

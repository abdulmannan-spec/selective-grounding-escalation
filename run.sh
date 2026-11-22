#!/usr/bin/env bash
# One command: bash run.sh
# Creates a virtualenv, installs requirements, fetches the upstream artifact at a pinned commit,
# reproduces the upstream aggregate numbers, runs the experiment (two served systems), builds tables + figure.
set -euo pipefail
cd "$(dirname "$0")"

UPSTREAM_URL="https://github.com/faerber-lab/topic-timestamp-alignment"
UPSTREAM_COMMIT="63ffb7a"
UP="${UPSTREAM_DIR:-upstream}"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -d "$UP/.git" ]; then git clone -q "$UPSTREAM_URL" "$UP"; fi
git -C "$UP" checkout -q "$UPSTREAM_COMMIT"

python -m src.run_experiment --upstream "$UP" --served gpt_cid --out results                 --repeats 10 --boot 1000
python -m src.make_report    --upstream "$UP" --served gpt_cid --out results
python -m src.run_experiment --upstream "$UP" --served mis_cid --out results/served_mistral  --repeats 10 --boot 1000
python -m src.make_report    --upstream "$UP" --served mis_cid --out results/served_mistral
python -m pytest -q tests
echo "Done. See results/results_tables.md and results/risk_coverage.png"

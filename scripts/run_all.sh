#!/usr/bin/env bash
# Reproduce every number and figure in the report.
#
#   bash scripts/run_all.sh
#
# Runtime is a few minutes on a laptop CPU; no GPU is needed.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

SEEDS=${SEEDS:-"0 1 2"}

echo "==> training world models (seeds: ${SEEDS})"
for seed in ${SEEDS}; do
  "${PY}" -m src.train --seed "${seed}" --out "runs/wm_seed${seed}"
done

CKPTS=""
for seed in ${SEEDS}; do
  CKPTS="${CKPTS} runs/wm_seed${seed}/checkpoint.pt"
done

echo "==> rollout-drift experiment"
# shellcheck disable=SC2086
"${PY}" -m src.evaluate --ckpt ${CKPTS} --out results

echo "==> figures"
"${PY}" -m src.plot --raw results/rollout_raw.csv --out results

echo "==> done. see results/"
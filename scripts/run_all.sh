#!/usr/bin/env bash
# Reproduce every number and figure in the report.
#
#   bash scripts/run_all.sh
#
# The three seeds are trained concurrently, which is how the committed results
# were produced. Runtime is roughly 10-15 minutes on a multi-core laptop CPU;
# no GPU is needed. Override the seed list with SEEDS="0", or the per-process
# thread budget with THREADS=2, if you are on a smaller machine.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
THREADS=${THREADS:-4}

echo "==> training world models (seeds: ${SEEDS}, ${THREADS} threads each)"
pids=()
for seed in ${SEEDS}; do
  OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    "${PY}" -m src.train --seed "${seed}" --out "runs/wm_seed${seed}" \
    > "runs_train_${seed}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [ "${status}" -ne 0 ]; then
  echo "training failed (exit ${status}); see runs_train_*.log" >&2
  exit "${status}"
fi
for seed in ${SEEDS}; do
  echo "    seed ${seed}: $(tail -n 1 "runs_train_${seed}.log")"
done

CKPTS=()
for seed in ${SEEDS}; do
  CKPTS+=("runs/wm_seed${seed}/checkpoint.pt")
done

echo "==> rollout-drift experiment"
"${PY}" -m src.evaluate --ckpt "${CKPTS[@]}" --out results

echo "==> figures"
"${PY}" -m src.plot --raw results/rollout_raw.csv --out results

echo "==> done. see results/ and report/report.md"
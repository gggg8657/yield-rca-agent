#!/usr/bin/env bash
# Regenerate every number in this repo from scratch, in dependency order.
#
#   bash scripts/overnight.sh [PYTHON]
#
# CPU only; parallelism is capped at 16 workers and each worker is pinned to a
# single BLAS thread, because 16 processes x 16 threads on a shared box is how
# you make a 6-minute cross-validation take an hour.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JOBS=16
mkdir -p runs

echo "== 1/5 data profile =="
"$PY" scripts/prepare_data.py

echo "== 2/5 headline SECOM evaluation (baselines vs agent loop) =="
"$PY" scripts/eval_secom.py --repeats 5 --jobs "$JOBS"

echo "== 3/5 agent-loop sensitivity sweep =="
"$PY" scripts/sweep_loop.py --repeats 2 --jobs "$JOBS"

echo "== 4/5 top-5 stability KPI =="
"$PY" scripts/stability_secom.py --boot 200 --jobs "$JOBS"

echo "== 5/5 synthetic ground-truth benchmark =="
"$PY" scripts/eval_synthetic.py --seeds 10 --jobs 10

echo "== report =="
"$PY" scripts/report.py
"$PY" scripts/report.py --check

echo "== tests =="
PYTHONPATH=. "$PY" tests/test_smoke.py
PYTHONPATH=. "$PY" tests/test_real.py
echo "all done"

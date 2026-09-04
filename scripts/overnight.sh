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

echo "== 1/9 data profile =="
"$PY" scripts/prepare_data.py

echo "== 2/9 headline SECOM evaluation (baselines vs agent loop) =="
"$PY" scripts/eval_secom.py --repeats 5 --jobs "$JOBS"

echo "== 3/9 agent-loop sensitivity sweep =="
"$PY" scripts/sweep_loop.py --repeats 2 --jobs "$JOBS"

echo "== 4/9 top-5 stability KPI =="
"$PY" scripts/stability_secom.py --boot 200 --jobs "$JOBS"

echo "== 5/9 drift diagnostics =="
"$PY" scripts/drift.py --jobs "$JOBS"

echo "== 6/9 rolling-origin robustness =="
"$PY" scripts/rolling_sweep.py --jobs "$JOBS"

echo "== 7/9 synthetic ground-truth benchmark =="
"$PY" scripts/eval_synthetic.py --seeds 10 --jobs 10

# The false-discovery rate of the reported suspects under a no-causal-sensor
# null. Depends on nothing above, but is slow (240 agent-loop fits), so it sits
# after the cheap stages rather than blocking them.
echo "== 8/9 hallucination control: permuted-label false-discovery rate =="
"$PY" scripts/null_fdr.py --null 200 --real 40 --jobs "$JOBS"

# Reads runs/secom_eval.json for the suspect sets, so it must follow stage 2.
echo "== 9/9 invariance across production periods =="
"$PY" scripts/invariance.py --blocks 5 --assoc-perm 20000 --inv-perm 20000 \
    --power-rep 200 --power-perm 5000

echo "== report =="
"$PY" scripts/report.py
"$PY" scripts/make_figures.py
"$PY" scripts/report.py --check

echo "== tests =="
PYTHONPATH=. "$PY" tests/test_smoke.py
PYTHONPATH=. "$PY" tests/test_real.py
PYTHONPATH=. "$PY" tests/test_null.py
echo "all done"

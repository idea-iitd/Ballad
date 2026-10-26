#!/usr/bin/env bash
# run_inference_pipeline.sh — val eval → test metric eval, for one problem.
#
# Usage:
#   ./run_inference_pipeline.sh \
#       --problem tsp_constructive \
#       --models gemini-2.5-flash gemma-3-12b-it gemma-3-27b-it \
#       --runs "1 2 3" "1 2 3" "1 2" \
#       [--use_parallel] [--use_gpu_batch] [--timeout 60]
#
# --models and --runs must have the same count; each --runs string corresponds
# to the model at the same position.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROBLEM=""
MODELS=()
RUNS=()
ALL_PROBLEMS=()
USE_PARALLEL=false
USE_GPU_BATCH=false
TIMEOUT=60


# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)
            PROBLEM="$2"; shift 2 ;;
        --models)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MODELS+=("$1"); shift
            done ;;
        --runs)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                RUNS+=("$1"); shift
            done ;;
        --all_problems)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                ALL_PROBLEMS+=("$1"); shift
            done ;;
        --use_parallel)
            USE_PARALLEL=true; shift ;;
        --use_gpu_batch)
            USE_GPU_BATCH=true; shift ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        *)
            echo "[!] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validation ───────────────────────────────────────────────────────────────
if [[ -z "$PROBLEM" ]]; then
    echo "[!] --problem is required" >&2; exit 1
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "[!] --models is required" >&2; exit 1
fi
if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "[!] --runs is required" >&2; exit 1
fi
if [[ ${#MODELS[@]} -ne ${#RUNS[@]} ]]; then
    echo "[!] --models count (${#MODELS[@]}) must match --runs count (${#RUNS[@]})" >&2; exit 1
fi

# Default: table covers only the current problem
if [[ ${#ALL_PROBLEMS[@]} -eq 0 ]]; then
    ALL_PROBLEMS=("$PROBLEM")
fi

PARALLEL_OPT=""
[[ "$USE_PARALLEL" == true ]] && PARALLEL_OPT="--use_parallel"
GPU_BATCH_OPT=""
[[ "$USE_GPU_BATCH" == true ]] && GPU_BATCH_OPT="--use_gpu_batch"

echo ""
echo "============================================================"
echo "  Problem      : $PROBLEM"
echo "  Models       :"
for i in "${!MODELS[@]}"; do
    echo "    ${MODELS[$i]}  →  runs: ${RUNS[$i]}"
done
echo "  Table covers : ${ALL_PROBLEMS[*]}"
echo "  Parallel     : $USE_PARALLEL
  GPU batch    : $USE_GPU_BATCH"
echo "  Timeout      : ${TIMEOUT}s"
echo "============================================================"

# ── Step 1: Validation eval ──────────────────────────────────────────────────
echo ""
echo "[1/3] eval_val.py ..."
CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/inference/eval_val.py" \
    --problem_name "$PROBLEM" \
    --models "${MODELS[@]}" \
    --runs "${RUNS[@]}" \
    --timeout "$TIMEOUT" \
    $PARALLEL_OPT $GPU_BATCH_OPT

# ── Step 2: Test metric eval ─────────────────────────────────────────────────
echo ""
echo "[2/3] eval_test_best_metric.py ..."
python3 "$SCRIPT_DIR/inference/eval_test_best_metric.py" \
    --problem_name "$PROBLEM" \
    --models "${MODELS[@]}" \
    --runs "${RUNS[@]}" \
    --timeout "$TIMEOUT" \
    $PARALLEL_OPT $GPU_BATCH_OPT

echo ""
echo "[✓] Pipeline complete for problem: $PROBLEM"

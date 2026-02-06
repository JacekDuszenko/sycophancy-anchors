#!/bin/bash
set -e

OUTPUT_DIR="${1:-multimodel_test_run}"
N_SYCO="${2:-5}"
N_NONSYCO="${3:-5}"
ROLLOUTS="${4:-5}"

echo "=============================================="
echo "Multi-Model Test Run"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo "Target: $N_SYCO sycophantic + $N_NONSYCO non-sycophantic samples (min 5+5 for experiments)"
echo "Rollouts per sentence: $ROLLOUTS"
echo ""

cd "$(dirname "$0")/../.."

VLLM_WORKER_MULTIPROC_METHOD=spawn python multimodel/scripts/run_pipeline.py \
    --all-models \
    --output-dir "$OUTPUT_DIR" \
    --n-syco "$N_SYCO" \
    --n-nonsyco "$N_NONSYCO" \
    --rollouts-per-sentence "$ROLLOUTS" \
    --base-batch-size 64 \
    --rollout-batch-size 64 \
    --activation-batch-size 32

echo ""
echo "=============================================="
echo "Test run complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="

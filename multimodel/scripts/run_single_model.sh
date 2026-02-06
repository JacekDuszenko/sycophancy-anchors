#!/bin/bash
set -e

MODEL="${1:?Usage: $0 <model-name> [output-dir] [n-syco] [n-nonsyco]}"
OUTPUT_DIR="${2:-multimodel_results}"
N_SYCO="${3:-101}"
N_NONSYCO="${4:-410}"

echo "=============================================="
echo "Running pipeline for: $MODEL"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo "Target: $N_SYCO sycophantic + $N_NONSYCO non-sycophantic samples"
echo ""

cd "$(dirname "$0")/../.."

VLLM_WORKER_MULTIPROC_METHOD=spawn python multimodel/scripts/run_pipeline.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --n-syco "$N_SYCO" \
    --n-nonsyco "$N_NONSYCO" \
    --rollouts-per-sentence 20 \
    --base-batch-size 256 \
    --rollout-batch-size 128 \
    --activation-batch-size 64

echo ""
echo "=============================================="
echo "Pipeline complete for: $MODEL"
echo "Results in: $OUTPUT_DIR/$MODEL"
echo "=============================================="

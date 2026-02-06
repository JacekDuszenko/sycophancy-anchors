#!/bin/bash
# Bash wrapper for OOM recovery - automatically restarts pipeline on crash
#
# Usage:
#   ./multimodel/scripts/run_with_restart.sh MODEL_NAME [OPTIONS]
#
# Examples:
#   ./multimodel/scripts/run_with_restart.sh deepseek-qwen-1.5b
#   ./multimodel/scripts/run_with_restart.sh deepseek-qwen-1.5b --n-syco 5 --n-nonsyco 5
#   ./multimodel/scripts/run_with_restart.sh --all-models --n-syco 101 --n-nonsyco 410

set -e

# Parse arguments - first arg is model (or --all-models), rest passed through
MODEL="${1:-}"
shift || true
EXTRA_ARGS="$@"

MAX_RESTARTS=10
RESTART_DELAY=30  # seconds to wait before restart
TIMEOUT_HOURS=4   # timeout per run attempt

if [ -z "$MODEL" ]; then
    echo "Usage: $0 MODEL_NAME [OPTIONS]"
    echo "       $0 --all-models [OPTIONS]"
    echo ""
    echo "Available models:"
    echo "  - deepseek-qwen-1.5b"
    echo "  - deepseek-qwen-7b"
    echo "  - olmo-7b-think"
    echo "  - falcon-h1r-7b"
    echo ""
    echo "Options (passed to run_pipeline.py):"
    echo "  --n-syco N              Target sycophantic samples (default: 101)"
    echo "  --n-nonsyco N           Target non-sycophantic samples (default: 410)"
    echo "  --rollouts-per-sentence N  Rollouts per sentence (default: 20)"
    echo "  --base-batch-size N     Base generation batch size (default: 50)"
    echo "  --rollout-batch-size N  Rollout generation batch size (default: 20)"
    echo "  --activation-batch-size N  Activation extraction batch size (default: 8)"
    echo "  --output-dir PATH       Output directory (default: multimodel_results)"
    echo "  --stage N               Run only stage N (0, 1, 2, 2.5, 3, 4)"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

# Set vLLM environment variables
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "========================================"
echo "Multi-Model Pipeline with Auto-Restart"
echo "========================================"
echo "Model: $MODEL"
echo "Extra args: $EXTRA_ARGS"
echo "Max restarts: $MAX_RESTARTS"
echo "Timeout per run: ${TIMEOUT_HOURS}h"
echo "Restart delay: ${RESTART_DELAY}s"
echo ""

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo ""
    echo "========================================"
    echo "Attempt $((restart_count + 1)) of $MAX_RESTARTS"
    echo "Started at: $(date)"
    echo "========================================"
    echo ""

    # Build command
    if [ "$MODEL" = "--all-models" ]; then
        CMD="python -m multimodel.scripts.run_pipeline --all-models $EXTRA_ARGS"
    else
        CMD="python -m multimodel.scripts.run_pipeline --model $MODEL $EXTRA_ARGS"
    fi

    echo "Running: $CMD"
    echo ""

    # Run the pipeline with timeout
    set +e
    timeout "${TIMEOUT_HOURS}h" $CMD
    exit_code=$?
    set -e

    # Check exit status
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Pipeline completed successfully!"
        echo "Finished at: $(date)"
        echo "========================================"
        exit 0
    elif [ $exit_code -eq 124 ]; then
        echo ""
        echo "Pipeline timed out after ${TIMEOUT_HOURS} hours"
    else
        echo ""
        echo "Pipeline exited with code: $exit_code"
    fi

    restart_count=$((restart_count + 1))

    if [ $restart_count -lt $MAX_RESTARTS ]; then
        echo ""
        echo "Waiting ${RESTART_DELAY}s before restart..."
        echo "Progress is checkpointed - will resume from last completed sample"
        sleep $RESTART_DELAY

        # Clear GPU memory (optional, depends on system)
        if command -v nvidia-smi &> /dev/null; then
            echo "Clearing GPU memory..."
            nvidia-smi --gpu-reset 2>/dev/null || true
        fi
    fi
done

echo ""
echo "========================================"
echo "Max restarts ($MAX_RESTARTS) exceeded"
echo "Pipeline failed after $restart_count attempts"
echo "========================================"
exit 1

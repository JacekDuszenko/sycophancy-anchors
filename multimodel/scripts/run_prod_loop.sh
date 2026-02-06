#!/bin/bash
set -e

OUTPUT_DIR="${1:-multimodel_prod}"
RESTART_INTERVAL_SECONDS=7200  # 2 hours

MODELS=(
    "deepseek-qwen-1.5b"
    "deepseek-qwen-7b"
    "olmo-7b-think"
    "falcon-h1r-7b"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/pipeline_loop.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_pipeline_complete() {
    local model="$1"
    local model_dir="$OUTPUT_DIR/$model"
    if [[ -f "$model_dir/summary.json" ]]; then
        return 0
    fi
    return 1
}

run_pipeline() {
    local model="$1"
    log "Starting pipeline run for $model..."
    
    source multimodel/.venv/bin/activate
    
    timeout "${RESTART_INTERVAL_SECONDS}s" \
        env VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python -m multimodel.scripts.run_pipeline \
            --model "$model" \
            --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE" || {
            local exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                log "Pipeline timed out after 2 hours, will restart..."
                pkill -f "run_pipeline.*$model" 2>/dev/null || true
                sleep 10
                return 1
            else
                log "Pipeline exited with code $exit_code"
                return $exit_code
            fi
        }
    
    return 0
}

cleanup_gpu() {
    log "Cleaning up GPU memory..."
    pkill -f "vllm" 2>/dev/null || true
    pkill -f "EngineCore" 2>/dev/null || true
    sleep 10
}

log "============================================"
log "Starting production pipeline loop"
log "Models: ${MODELS[*]}"
log "Output: $OUTPUT_DIR"
log "Restart interval: ${RESTART_INTERVAL_SECONDS}s (2 hours)"
log "============================================"

for MODEL in "${MODELS[@]}"; do
    log ""
    log "============================================"
    log "Processing model: $MODEL"
    log "============================================"
    
    if check_pipeline_complete "$MODEL"; then
        log "Model $MODEL already complete, skipping..."
        continue
    fi
    
    iteration=1
    while true; do
        log "--- $MODEL: Iteration $iteration ---"
        
        if check_pipeline_complete "$MODEL"; then
            log "Pipeline complete for $MODEL!"
            break
        fi
        
        run_pipeline "$MODEL"
        pipeline_result=$?
        
        cleanup_gpu
        
        if [[ $pipeline_result -eq 0 ]] && check_pipeline_complete "$MODEL"; then
            log "Pipeline completed successfully for $MODEL!"
            break
        fi
        
        log "Restarting pipeline for $MODEL in 30 seconds..."
        sleep 30
        
        ((iteration++))
    done
done

log ""
log "============================================"
log "All models processed!"
log "============================================"

for MODEL in "${MODELS[@]}"; do
    if check_pipeline_complete "$MODEL"; then
        log "  ✓ $MODEL: Complete"
    else
        log "  ✗ $MODEL: Incomplete"
    fi
done

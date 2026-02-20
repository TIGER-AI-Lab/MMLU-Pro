#!/usr/bin/env bash
# run_eval.sh — Launch 20 shards for MMLU-Pro evaluation, then merge.
#
# Usage:
#   chmod +x run_eval.sh
#   export GOOGLE_API_KEY="your-api-key-here"
#   ./run_eval.sh

set -euo pipefail

cd ../..

NUM_SHARDS=20
OUTPUT_DIR="eval_results"
SCRIPT="eval_mmlu_pro.py"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  MMLU-Pro Evaluation — ${NUM_SHARDS} shards"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

# Check API key
if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "❌ ERROR: GOOGLE_API_KEY is not set."
    exit 1
fi

# Launch all shards in background
PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG_FILE="${LOG_DIR}/shard_${SHARD_ID}.log"
    echo "🚀 Starting shard ${SHARD_ID}/${NUM_SHARDS}  →  ${LOG_FILE}"
    python "$SCRIPT" \
        --output_dir "$OUTPUT_DIR" \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$SHARD_ID" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    # Stagger launches slightly to avoid API burst
    sleep 1
done

echo ""
echo "All ${NUM_SHARDS} shards launched. PIDs: ${PIDS[*]}"
echo "Waiting for all shards to finish..."
echo ""

# Wait and track failures
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "✅ Shard ${i} (PID ${PID}) finished successfully."
    else
        EXIT_CODE=$?
        echo "❌ Shard ${i} (PID ${PID}) failed with exit code ${EXIT_CODE}. Check ${LOG_DIR}/shard_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "⚠️  ${FAILED} shard(s) failed. You can re-run the script — resume will skip completed questions."
    echo "   Merging available results anyway..."
fi

# Merge all shard results
echo ""
echo "📊 Merging results..."
python "$SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_SHARDS" \
    --merge

echo ""
echo "✅ Done. Results in ${OUTPUT_DIR}/merged/"
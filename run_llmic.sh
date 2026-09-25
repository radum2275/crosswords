#!/bin/bash
# Sweep a base model (default: LLMic_v2) over both datasets and every hint
# condition that has a base-model prompt: v1, v2 and v3 with k = 1..4.
#
# Unlike the other run_*.sh scripts this uses repo-relative paths, so run it
# from the repository root on any machine:
#
#   ./run_llmic.sh                       # llmic-v2, batch size 64
#   ./run_llmic.sh llmic-v2 128          # larger batches on a big GPU
#   MODEL_PATH=/scratch/LLMic_v2 ./run_llmic.sh   # pre-downloaded weights
#   NUM_SAMPLES=50 OUT=/tmp/llmic_smoke ./run_llmic.sh   # quick smoke test
#
# Each run resumes from its output file, so re-running after a crash or
# preemption only does the missing items.

set -euo pipefail

MODEL=${1:-llmic-v2}
BATCH=${2:-64}
OUT=${OUT:-data/results}
LOGS=$OUT/logs_${MODEL}
mkdir -p "$LOGS"

EXTRA=()
if [[ -n "${MODEL_PATH:-}" ]]; then
    EXTRA+=(--model_path "$MODEL_PATH")
fi
if [[ -n "${NUM_SAMPLES:-}" ]]; then
    EXTRA+=(--num_samples "$NUM_SAMPLES" --no_bertscore)
fi

run() {
    local file=$1 type=$2 version=$3 k=$4
    local log="$LOGS/${type}_${version}_${k}.log"
    echo "[$(date '+%F %T')] $type $version k=$k -> $log"
    if ! PYTHONPATH=src python src/crosswords/solve_base_ro.py \
        --model_id "$MODEL" ${EXTRA[@]+"${EXTRA[@]}"} \
        --dataset_file "$file" --dataset_type "$type" \
        --version "$version" --prefix_len "$k" \
        --batch_size "$BATCH" --output_dir "$OUT" --resume > "$log" 2>&1; then
        echo "  FAILED, see $log:"
        tail -3 "$log"
        exit 1
    fi
    grep "Evaluation results" "$log"
}

for spec in "data/polycross_sample3000.json clues" "data/themcross_data.json baseline"; do
    read -r file type <<< "$spec"
    run "$file" "$type" v1 0
    run "$file" "$type" v2 0
    for k in 1 2 3 4; do
        run "$file" "$type" v3 "$k"
    done
done
echo "All runs finished. Rescore with: PYTHONPATH=src python -m crosswords.analysis.rescore"

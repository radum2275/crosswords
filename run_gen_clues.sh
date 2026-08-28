#!/bin/bash

# Run the clue-generation script (src/crosswords/gen_clues_ro.py).
#
# Given a dataset of (answer, clue) pairs, the script asks an LLM (via RITS) to
# generate polysemantic Romanian clues for each answer, then scores them against
# the ground-truth clue with BERTScore. It sweeps the number of ground-truth
# hint words (h0, h1, h2, ...) so the effect of hinting can be compared.
#
# Arguments:
#   1st: model id        (e.g., llama, granite, mistral, gpt-oss)
#   2nd: dataset type     (clues | baseline)
#   3rd: num candidates   (clues generated per answer, e.g., 3) [optional, default 3]
#   4th: num samples      (limit number of answers, empty = all)  [optional]

m=$1
t=$2
cand=${3:-3}
samples=$4

# Select the input dataset based on the dataset type.
base=/home/radu/storage/git/crosswords
if [[ "$t" == "clues" ]]; then
    dataset=${base}/data/extracted_data.json
elif [[ "$t" == "baseline" ]]; then
    dataset=${base}/data/baseline-dataset.json
else
    echo "Unknown dataset type: $t (expected 'clues' or 'baseline')"
    exit 1
fi

# Optional cap on the number of answers processed.
sample_arg=""
if [[ -n "$samples" ]]; then
    sample_arg="--num_samples $samples"
fi

# Sweep the number of ground-truth hint words.
for h in 0 1 2; do
    l='log_gen_clues_'${m}'_'${t}'_h'${h}'.txt'
    ./timeout -m 30000000 python src/crosswords/gen_clues_ro.py \
        --model_id $m \
        --dataset_file $dataset \
        --dataset_type $t \
        --num_candidates $cand \
        --num_hints $h \
        $sample_arg \
        --output_name gen_clues \
        --output_dir ${base}/data/results \
        --batch_size 50 \
        --rate_limit 1500 >& $l
done

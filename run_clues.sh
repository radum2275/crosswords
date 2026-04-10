#!/bin/bash

# 1st argument is the model id (e.g., llama, granite, gpt)
# 2nd argument is the prompt version (e.g., v1, v2, v3, v4)

m=$1
p=$2

if [[ "$p" == "v1" ]]; then
    l='log_clues_'${m}'_'${p}'_s0.txt'
    ./timeout -m 30000000 python src/crosswords/clues_ro.py --model_id $m --dataset_file /home/radu/storage/git/crosswords/data/extracted_data.json --output_dir /home/radu/storage/git/crosswords/data/results --dataset_type clues --version $p --output_name clues --batch_size 200 >& $l
fi
if [[ "$p" == "v2" ]]; then
    l='log_clues_'${m}'_'${p}'_s0.txt'
    ./timeout -m 30000000 python src/crosswords/clues_ro.py --model_id $m --dataset_file /home/radu/storage/git/crosswords/data/extracted_data.json --output_dir /home/radu/storage/git/crosswords/data/results --dataset_type clues --version $p --output_name clues --batch_size 200 >& $l
fi
if [[ "$p" == "v3" ]]; then
    for s in 0 1 2 3 4; do
        l='log_clues_'${m}'_'${p}'_s'${s}'.txt'
        ./timeout -m 30000000 python src/crosswords/clues_ro.py --model_id $m --dataset_file /home/radu/storage/git/crosswords/data/extracted_data.json --output_dir /home/radu/storage/git/crosswords/data/results --dataset_type clues --version $p --prefix_len $s --output_name clues --batch_size 200 >& $l
    done
fi
if [[ "$p" == "v4" ]]; then
    for s in 0 1 2 3 4; do
        l='log_clues_'${m}'_'${p}'_s'${s}'.txt'
        ./timeout -m 30000000 python src/crosswords/clues_ro.py --model_id $m --dataset_file /home/radu/storage/git/crosswords/data/extracted_data.json --output_dir /home/radu/storage/git/crosswords/data/results --dataset_type clues --version $p --prefix_len $s --output_name clues --batch_size 200 >& $l
    done
fi





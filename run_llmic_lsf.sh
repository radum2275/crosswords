#!/bin/bash
# LSF wrapper for run_llmic.sh: activates the Python env and runs the full
# LLMic_v2 sweep (polycross 3000 + themcross, v1/v2/v3 k=1..4) on one GPU.
#
#   bsub -G grp_big_job_lease -q normal -gpu "num=1" -n 8 -R "rusage[mem=64GB]" \
#        -J llmic -o data/results/logs_llmic-v2/lsf_%J.out ./run_llmic_lsf.sh
#
# Weights must already be in the HF cache (the job runs with HF_HUB_OFFLINE=1).
# Re-submitting resumes: finished runs are skipped item-by-item via --resume.

set -euo pipefail
cd "$(dirname "$0")"
source "${ENV:-/u/radu/envs/crosswords}/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "host: $(hostname)  job: ${LSB_JOBID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

./run_llmic.sh "${MODEL:-llmic-v2}" "${BATCH:-128}"

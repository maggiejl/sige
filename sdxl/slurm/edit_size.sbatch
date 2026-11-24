#!/bin/bash
# Edit-size ablation (TODOS.txt): 1-60% edits at the 512 px base point, all three configs,
# efficiency + quality, averaged over three prompts (one seed each).
# Submit from ~/orcd/scratch:  sbatch -p mit_preemptable --gres=gpu:h200:1 sige/sdxl/slurm/edit_size.sbatch
#SBATCH -J sige-editsize
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/maggiejl/orcd/scratch/logs/%x-%j.out
set -e
module load cuda/12.9.1
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate sige-sdxl
export HF_HOME=$HOME/orcd/scratch/hf HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
cd $HOME/orcd/scratch/sige/sdxl

PROMPTS=(
  "a photograph of a cozy wooden cabin by a lake in the mountains, highly detailed"
  "a photograph of a city street at dusk with cars and storefronts, highly detailed"
  "a photograph of a bowl of fruit on a wooden kitchen table, natural light, highly detailed"
)
for i in "${!PROMPTS[@]}"; do
  echo "=== prompt $i: ${PROMPTS[$i]} ==="
  python benchmark.py --prompt "${PROMPTS[$i]}" --resolutions 512 \
    --edit_ratios 0.01 0.05 0.1 0.2 0.4 0.6 \
    --steps 50 --strength 0.8 --guidance 5.0 --repeats 10 --seed 0 \
    --text_encoders_on_cpu --preserve_unedited \
    --output_dir outputs/edit_size/prompt$i
done
python aggregate.py outputs/edit_size/prompt*/results.csv --out outputs/edit_size/summary.csv
echo JOB_DONE

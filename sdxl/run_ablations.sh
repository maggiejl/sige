#!/bin/bash
# Baseline, edit-size and kernel-size (SIGE block size) ablations for SIGE-SDXL at the 512 px base point.
# Every cell runs dense / sige_conv / sige_conv_attn on three prompts (one seed each) and is averaged
# with aggregate.py.
#
# Local (Git Bash on Windows, or any Linux shell):
#     bash sdxl/run_ablations.sh                      # all three ablations
#     bash sdxl/run_ablations.sh baseline block_size  # a subset
# Slurm (submit from the repo root, from sdxl/, or from the directory that contains sige/):
#     sbatch sdxl/run_ablations.sh
#     sbatch -p mit_preemptable --gres=gpu:h200:1 sdxl/run_ablations.sh edit_size
#
# Environment overrides:
#     MACHINE=a100     tag for the output directory (default: derived from the GPU name)
#     PYTHON=...       interpreter (default: the repo's .venv if present, else `python`)
#     SDXL_DIR=...     path to sige/sdxl (only needed if it cannot be found)
#     CONDA_ENV=...    conda env to activate under Slurm (default sige-sdxl)
#     QUICK=1          smoke test: 1 prompt, efficiency only, 2 repeats, separate output directory
#     FORCE=1          rerun cells whose results.csv is already complete
#
# Results: outputs/ablations/$MACHINE/{baseline,edit_size,block_size}/summary.csv
# Finished cells are skipped on rerun, so an interrupted or preempted job can simply be resubmitted.
#
#SBATCH -J sige-ablations
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%j.out
set -eo pipefail

# ---------------------------------------------------------------- locate sige/sdxl
# Under Slurm the script is copied to a spool directory, so BASH_SOURCE is useless there.
find_sdxl_dir() {
  local d
  for d in "${SDXL_DIR:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/sdxl" "${SLURM_SUBMIT_DIR:-}/sige/sdxl" \
           "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$PWD" "$PWD/sdxl"; do
    if [ -n "$d" ] && [ -f "$d/benchmark.py" ] && [ -f "$d/aggregate.py" ]; then
      (cd "$d" && pwd)
      return 0
    fi
  done
  echo "Cannot find sige/sdxl (benchmark.py + aggregate.py); set SDXL_DIR." >&2
  return 1
}
SDXL_DIR=$(find_sdxl_dir)
REPO=$(cd "$SDXL_DIR/.." && pwd)
cd "$SDXL_DIR"

# ---------------------------------------------------------------- environment
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if command -v module >/dev/null 2>&1; then module load "${CUDA_MODULE:-cuda/12.9.1}" || true; fi
  CONDA_SH=${CONDA_SH:-/orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh}
  if [ -z "${PYTHON:-}" ] && [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-sige-sdxl}"
  fi
  # Compute nodes usually have no internet: use the pre-downloaded Hugging Face cache if there is one.
  if [ -z "${HF_HOME:-}" ] && [ -d "$HOME/orcd/scratch/hf" ]; then
    export HF_HOME=$HOME/orcd/scratch/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  fi
fi

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/.venv/Scripts/python.exe" ]; then PYTHON="$REPO/.venv/Scripts/python.exe"   # Windows venv
  elif [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"
  else PYTHON=python; fi
fi

# `sige` may be built in place rather than pip-installed, so the repo root must be on sys.path.
# This is done by a bootstrap instead of PYTHONPATH: WSL bash (what `bash` starts from PowerShell) does
# not reliably pass environment variables to a Windows python.exe, and that python needs a Windows path
# (cygpath in Git Bash, wslpath in WSL).
REPO_PY=$REPO
if command -v cygpath >/dev/null 2>&1; then REPO_PY=$(cygpath -w "$REPO")
elif [[ "$PYTHON" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then REPO_PY=$(wslpath -w "$REPO"); fi
PY_BOOT="import sys, runpy; sys.path.insert(0, r'''$REPO_PY'''); a = sys.argv[1:]; sys.argv = a
if a[0] == '-c': exec(a[1])
else: runpy.run_path(a[0], run_name='__main__')"
# py script.py args...   or   py -c "code"
py() { "$PYTHON" -u -c "$PY_BOOT" "$@"; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')
if [ -z "${MACHINE:-}" ]; then
  MACHINE=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -e 's/nvidia//; s/geforce//; s/gpu//; s/[^a-z0-9]//g')
fi

# Text encoders always run on the CPU (it does not touch the U-Net measurements and keeps the three
# machines on the same flags). The fp32 U-Net plus SIGE caches need the VAE off the GPU on <16 GB cards.
MEM_FLAGS=(--text_encoders_on_cpu)
if [ "${GPU_MEM_MB:-0}" -lt 16000 ]; then MEM_FLAGS+=(--offload_vae); fi

# ---------------------------------------------------------------- experiment grid
PROMPTS=(
  "a photograph of a cozy wooden cabin by a lake in the mountains, highly detailed"
  "a photograph of a city street at dusk with cars and storefronts, highly detailed"
  "a photograph of a bowl of fruit on a wooden kitchen table, natural light, highly detailed"
)
BASE_RATIO=0.1
EDIT_RATIOS=(0.01 0.05 0.1 0.2 0.4 0.6)
# "conv attn" block sizes, one factor varied at a time around the default 6 / 4 (which is the baseline).
BLOCK_VARIANTS=("4 4" "8 4" "6 2" "6 8")
COMMON=(--resolutions 512 --steps 50 --strength 0.8 --guidance 5.0 --seed 0 --preserve_unedited)
REPEATS=10
OUT=outputs/ablations/$MACHINE

if [ -n "${QUICK:-}" ]; then
  PROMPTS=("${PROMPTS[0]}")
  EDIT_RATIOS=(0.05 0.2)
  REPEATS=2
  COMMON+=(--skip_quality)
  OUT=outputs/ablations_quick/$MACHINE
fi

echo "sdxl dir : $SDXL_DIR"
echo "python   : $PYTHON"
echo "gpu      : $GPU_NAME (${GPU_MEM_MB} MiB) -> ${MEM_FLAGS[*]}"
echo "output   : $SDXL_DIR/$OUT"
py -c "import torch, sige.nn; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# run_cell <output dir> <benchmark.py args...>: one benchmark.py call per prompt, then the average.
# Expects the edit ratios of the cell in the RATIOS array.
run_cell() {
  local dir=$1; shift
  local expected=$((1 + 3 * ${#RATIOS[@]})) i
  for i in "${!PROMPTS[@]}"; do
    local csv="$dir/prompt$i/results.csv"
    if [ -z "${FORCE:-}" ] && [ -f "$csv" ] && [ "$(grep -c . "$csv")" -ge "$expected" ]; then
      echo "--- skip $dir/prompt$i (complete)"
      continue
    fi
    echo "--- $dir/prompt$i: ${PROMPTS[$i]}"
    py benchmark.py --prompt "${PROMPTS[$i]}" --edit_ratios "${RATIOS[@]}" --repeats "$REPEATS" \
      "${COMMON[@]}" "${MEM_FLAGS[@]}" --output_dir "$dir/prompt$i" "$@"
  done
  py aggregate.py "$dir"/prompt*/results.csv --out "$dir/summary.csv"
}

run_baseline() {
  echo "=== baseline: ${BASE_RATIO} edit, default block sizes ==="
  RATIOS=("$BASE_RATIO")
  run_cell "$OUT/baseline"
}

run_edit_size() {
  echo "=== edit size: ${EDIT_RATIOS[*]} ==="
  RATIOS=("${EDIT_RATIOS[@]}")
  run_cell "$OUT/edit_size"
}

run_block_size() {
  # The default 6 / 4 point is the baseline, so make sure it exists (skipped if already complete).
  run_baseline
  echo "=== block size (kernel size): conv 4 / 6 / 8, attention 2 / 4 / 8 at ${BASE_RATIO} edit ==="
  RATIOS=("$BASE_RATIO")
  local v conv attn
  for v in "${BLOCK_VARIANTS[@]}"; do
    read -r conv attn <<< "$v"
    run_cell "$OUT/block_size/conv${conv}_attn${attn}" \
      --main_block_size "$conv" --upsample_block_size "$conv" --attn_block_size "$attn"
  done
  # One table for the whole ablation: the per-variant summaries with the block sizes as leading columns.
  local combined="$OUT/block_size/summary.csv" f
  : > "$combined"
  for v in "6 4" "${BLOCK_VARIANTS[@]}"; do
    read -r conv attn <<< "$v"
    f="$OUT/block_size/conv${conv}_attn${attn}/summary.csv"
    if [ "$v" = "6 4" ]; then f="$OUT/baseline/summary.csv"; fi
    if [ ! -s "$combined" ]; then echo "conv_block_size,attn_block_size,$(head -n 1 "$f" | tr -d '\r')" > "$combined"; fi
    tail -n +2 "$f" | tr -d '\r' | sed "s/^/$conv,$attn,/" >> "$combined"
  done
  echo "Wrote $combined"
}

if [ $# -eq 0 ]; then set -- baseline edit_size block_size; fi
for ablation in "$@"; do
  case "$ablation" in
    baseline) run_baseline ;;
    edit_size) run_edit_size ;;
    block_size | kernel_size) run_block_size ;;
    *) echo "Unknown ablation '$ablation' (choose from: baseline edit_size block_size)" >&2; exit 2 ;;
  esac
done
# Paper tables (Markdown + LaTeX) over every machine whose results are in the same output root,
# so copying other machines' outputs/ablations/<machine> folders here and rerunning this gives the
# cross-GPU comparison too:  python make_tables.py outputs/ablations
py make_tables.py "$(dirname "$OUT")"
echo "JOB_DONE: summaries in $SDXL_DIR/$OUT, tables in $SDXL_DIR/$(dirname "$OUT")/tables.{md,tex}"

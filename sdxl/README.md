# SIGE + Stable Diffusion XL

Spatially sparse SDEdit for SDXL: the pretrained SDXL base U-Net (from diffusers) is patched in place so that,
for an edited image, only the regions that changed are recomputed and everything else is reused from a pass over
the original image. This is the base implementation for the experiments in the project proposal.

## What is implemented

| Piece | File | Notes |
|---|---|---|
| Sparse conv layers | `sige_sdxl/modules.py` | `ResnetBlock2D`, `Downsample2D`, `Upsample2D` wrapped with SIGE gather / scatter / scatter-gather; GroupNorm + SiLU fused into the gathers using the original image's statistics (same as SIGE SD1.x). |
| Sparse attention | `sige_sdxl/modules.py` (`SIGETransformer2D`) | proj_in / LayerNorm / FFN / proj_out on active tokens only. **Self-attention:** Q/K/V computed for active tokens; active K/V are written into the cached K/V of the original image; active queries attend over the full grid. **Cross-attention:** active queries attend to the full text context. K/V are cached per transformer block, because SDXL stacks 2–10 blocks per layer and SIGE's SD1.x port only handled depth 1. |
| U-Net patcher and configs | `sige_sdxl/patch.py` | `patch_unet(unet)` wraps 17 resnets, 2 downsamplers, 2 upsamplers and 11 transformers (70 transformer blocks) of SDXL base without copying weights. Runtime configs: `dense` (stock SDXL), `sige_conv` (sparse convs, dense transformers), `sige_conv_attn` (both). |
| SDEdit pipeline | `sige_sdxl/pipeline.py` | Original and edited images are encoded and noised with the **same** noise. At every DDIM step the original latent runs a `full` pass (fills caches) and the edited latent a `sparse` pass. Also has `profile_unet` for MACs and latency. |
| Correctness test | `test_equivalence.py` | Checks that sparse equals dense when nothing is edited (all-active and partial masks), and reports the approximation error for real edits. |
| Experiments | `benchmark.py`, `run.py`, `edits.py`, `metrics.py` | Edit-size × resolution × config sweep → `results.csv`; single-edit runner; synthetic stroke edits of controlled size; PSNR / LPIPS. |

The stem (`conv_in`), output head and time/text embeddings stay dense, as in SIGE's SD1.x port. The VAE is dense
(the proposal measures the U-Net).

## Setup (Colab, GPU runtime: A100 recommended, L4 should work)

SIGE's CUDA kernels are **float32 only**, so the SDXL U-Net (2.6B parameters, ~10 GB) runs in fp32, and so does
the dense baseline. On GPUs with less than 24 GB, pass `--text_encoders_on_cpu`.

```bash
git clone <this repo> sige && cd sige
pip install -e .          # builds SIGE's CUDA kernels (a few minutes)
pip install diffusers transformers accelerate lpips
cd sdxl
```

## Usage

1. Check the port (CPU, tiny random SDXL-architecture U-Net, a few seconds):

   ```bash
   python test_equivalence.py
   ```

   Add `--pretrained` to run the same checks on the real SDXL U-Net on GPU. Checks 1 and 2 should be about
   1e-5 or lower.

2. Run one edit and compare configs:

   ```bash
   python run.py --init_img original.png --edited_img edited.png \
     --prompt "a photograph of a cabin by a lake" --configs dense sige_conv sige_conv_attn
   ```

3. Main sweep (edit size × resolution × config):

   ```bash
   python benchmark.py --resolutions 512 768 1024 --edit_ratios 0.01 0.05 0.1 0.2 0.4
   ```

   Writes `outputs/benchmark/results.csv` plus every original, edited and output image. Per row:
   `gmacs`, `latency_ms` (one CFG U-Net call on the edited latent), `macs_reduction`, `speedup`, and
   `psnr_vs_dense` / `lpips_vs_dense` (full SDEdit output vs dense SDXL with the same input, prompt, noise and
   seed). `--skip_quality` measures efficiency only.

## Measurement conventions

- **Latency** is for the sparse U-Net pass on the edited image, as in the SIGE paper. The cache-filling `full`
  pass on the original image is assumed to be precomputed once per image in an interactive editor.
  `EditResult.full_ms` records its cost if you want to report it.
- **MACs** are counted with `torch.utils.flop_counter.FlopCounterMode` (convs, linears, attention). Gather and
  scatter are memory operations and add latency but no MACs, which is why both numbers are reported.
- **Masks** follow SIGE SD1.x: pixel difference > 0.02, dilated by 5 px, downsampled to every U-Net resolution
  (threshold 0.3, dilation 1). Block sizes: 6 (3×3 convs), 4 (1×1 shortcuts and attention), 5 (stride-2
  downsample).
- **Sampler**: DDIM (eta = 0), 50 steps, strength 0.8 (40 denoising steps), guidance 5.0. Use DDIM or another
  deterministic sampler so that both trajectories see the same noise.

## Known limitations / next steps

- fp32 only: half-precision gather/scatter kernels would be needed to compare against the usual fp16 SDXL
  deployment.
- The `full` pass is rerun at every step, so the end-to-end wall-clock of `edit()` is higher than dense. Only the
  sparse-pass latency reflects an interactive setting.
- The sparse VAE encoder/decoder from SIGE SD1.x (`stable-diffusion/ldm/models/sige_autoencoder.py`) is not
  ported to the SDXL VAE yet.

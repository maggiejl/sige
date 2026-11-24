# SIGE-SDXL ablation tables

### Hardware comparison

Baseline latency of one CFG U-Net call on each GPU (512 px, 10% edit, fp32). MACs are hardware independent; the speedup shows how much of the MACs reduction each GPU realizes.

| GPU | Dense (ms) | SIGE conv (ms) | SIGE conv+attn (ms) | Speedup conv | Speedup conv+attn | MACs red. conv+attn |
|---|---:|---:|---:|---:|---:|---:|
| a100 | 179.2 ± 0.3 | 175.5 ± 0.0 | 106.0 ± 0.0 | 1.02× ± 0.00 | 1.69× ± 0.00 | 2.12× |
| h200 | 92.7 ± 0.3 | 91.6 ± 0.2 | 66.7 ± 0.2 | 1.01× ± 0.00 | 1.39× ± 0.00 | 2.12× |
| rtx5070tilaptop | 368.6 ± 1.9 | 341.9 ± 6.9 | 237.3 ± 27.8 | 1.08× ± 0.02 | 1.58× ± 0.20 | 2.12× |

### Baseline (a100)

SIGE vs. dense SDXL at the base point (a100): 512 px, 10% edit, 50 DDIM steps, strength 0.8, guidance 5.0, mean ± std over 3 prompts. PSNR / LPIPS compare the full SDEdit output with dense SDXL given the same input, prompt, noise and seed.

| Method | GMACs | MACs red. | Latency (ms) | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense SDXL | 1589 | 1.00× | 179.2 ± 0.3 | 1.00× | -- | -- |
| SIGE (conv) | 1352 | 1.18× | 175.5 ± 0.0 | 1.02× ± 0.00 | 27.98 ± 2.57 | 0.055 ± 0.020 |
| SIGE (conv+attn) | 748 | 2.12× | 106.0 ± 0.0 | 1.69× ± 0.00 | 28.34 ± 2.81 | 0.049 ± 0.030 |

### Edit size (a100)

Effect of edit size (a100, 512 px). Edit = measured fraction of changed pixels; MACs reduction and speedup are relative to dense SDXL; quality is vs. dense SDXL, mean ± std over 3 prompts.

| Edit | Method | GMACs | MACs red. | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---:|---|---:|---:|---:|---:|---:|
| 1.3% | SIGE (conv) | 1255 | 1.27× | 1.03× ± 0.00 | 47.54 ± 4.43 | 0.002 ± 0.001 |
|  | SIGE (conv+attn) | 378 | 4.20× | 2.26× ± 0.00 | 46.48 ± 4.47 | 0.002 ± 0.001 |
| 5.9% | SIGE (conv) | 1303 | 1.22× | 1.03× ± 0.00 | 33.92 ± 1.72 | 0.013 ± 0.005 |
|  | SIGE (conv+attn) | 502 | 3.16× | 2.07× ± 0.00 | 29.99 ± 2.67 | 0.024 ± 0.007 |
| 11.4% | SIGE (conv) | 1352 | 1.18× | 1.02× ± 0.00 | 27.98 ± 2.57 | 0.055 ± 0.020 |
|  | SIGE (conv+attn) | 748 | 2.12× | 1.70× ± 0.00 | 28.34 ± 2.81 | 0.049 ± 0.030 |
| 22.3% | SIGE (conv) | 1422 | 1.12× | 1.01× ± 0.00 | 27.27 ± 5.24 | 0.075 ± 0.049 |
|  | SIGE (conv+attn) | 1021 | 1.55× | 1.28× ± 0.00 | 27.17 ± 4.61 | 0.080 ± 0.053 |
| 43.7% | SIGE (conv) | 1566 | 1.01× | 0.98× ± 0.00 | 25.28 ± 2.52 | 0.097 ± 0.007 |
|  | SIGE (conv+attn) | 1386 | 1.15× | 1.02× ± 0.00 | 24.15 ± 2.36 | 0.122 ± 0.006 |
| 60.1% | SIGE (conv) | 1610 | 0.99× | 0.98× ± 0.00 | 22.35 ± 1.18 | 0.169 ± 0.010 |
|  | SIGE (conv+attn) | 1579 | 1.01× | 0.98× ± 0.00 | 19.62 ± 3.54 | 0.280 ± 0.126 |

### Block size (a100)

Effect of the SIGE block size (a100, 512 px, 10% edit). Conv = block size of the 3×3 convolutions (resnet and upsample), Attn = token block size of the transformers; 1×1 shortcuts (4) and stride-2 downsamplers (5) are fixed. The attention block size does not affect SIGE (conv), whose transformers are dense. Defaults are 6 / 4.

| Conv | Attn | Method | GMACs | MACs red. | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | **4** | SIGE (conv) | 1312 | 1.21× | 1.02× ± 0.00 | 27.87 ± 2.88 | 0.055 ± 0.021 |
|  |  | SIGE (conv+attn) | 708 | 2.24× | 1.69× ± 0.00 | 27.09 ± 2.80 | 0.063 ± 0.025 |
| **6** | 2 | SIGE (conv) | 1352 | 1.18× | 1.02× ± 0.00 | 27.98 ± 2.57 | 0.055 ± 0.020 |
|  |  | SIGE (conv+attn) | 556 | 2.86× | 2.03× ± 0.00 | 24.98 ± 1.61 | 0.073 ± 0.028 |
| **6** | **4** | SIGE (conv) | 1352 | 1.18× | 1.02× ± 0.00 | 27.98 ± 2.57 | 0.055 ± 0.020 |
|  |  | SIGE (conv+attn) | 748 | 2.12× | 1.69× ± 0.00 | 28.34 ± 2.81 | 0.049 ± 0.030 |
| **6** | 8 | SIGE (conv) | 1352 | 1.18× | 1.02× ± 0.00 | 27.98 ± 2.57 | 0.055 ± 0.020 |
|  |  | SIGE (conv+attn) | 1258 | 1.26× | 1.07× ± 0.00 | 27.90 ± 2.09 | 0.054 ± 0.019 |
| 8 | **4** | SIGE (conv) | 1387 | 1.15× | 1.02× ± 0.00 | 28.23 ± 3.14 | 0.054 ± 0.022 |
|  |  | SIGE (conv+attn) | 783 | 2.03× | 1.69× ± 0.00 | 27.69 ± 2.75 | 0.057 ± 0.027 |

### Baseline (h200)

SIGE vs. dense SDXL at the base point (h200): 512 px, 10% edit, 50 DDIM steps, strength 0.8, guidance 5.0, mean ± std over 3 prompts. PSNR / LPIPS compare the full SDEdit output with dense SDXL given the same input, prompt, noise and seed.

| Method | GMACs | MACs red. | Latency (ms) | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense SDXL | 1589 | 1.00× | 92.7 ± 0.3 | 1.00× | -- | -- |
| SIGE (conv) | 1352 | 1.18× | 91.6 ± 0.2 | 1.01× ± 0.00 | 28.16 ± 3.06 | 0.054 ± 0.020 |
| SIGE (conv+attn) | 748 | 2.12× | 66.7 ± 0.2 | 1.39× ± 0.00 | 28.22 ± 2.74 | 0.052 ± 0.028 |

### Edit size (h200)

Effect of edit size (h200, 512 px). Edit = measured fraction of changed pixels; MACs reduction and speedup are relative to dense SDXL; quality is vs. dense SDXL, mean ± std over 3 prompts.

| Edit | Method | GMACs | MACs red. | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---:|---|---:|---:|---:|---:|---:|
| 1.3% | SIGE (conv) | 1255 | 1.27× | 1.03× ± 0.00 | 49.28 ± 4.63 | 0.001 ± 0.001 |
|  | SIGE (conv+attn) | 378 | 4.20× | 1.88× ± 0.00 | 47.48 ± 4.58 | 0.001 ± 0.001 |
| 5.9% | SIGE (conv) | 1303 | 1.22× | 1.02× ± 0.00 | 34.10 ± 1.41 | 0.011 ± 0.003 |
|  | SIGE (conv+attn) | 502 | 3.16× | 1.70× ± 0.00 | 29.87 ± 2.90 | 0.024 ± 0.010 |
| 11.4% | SIGE (conv) | 1352 | 1.18× | 1.01× ± 0.00 | 28.16 ± 3.06 | 0.054 ± 0.020 |
|  | SIGE (conv+attn) | 748 | 2.12× | 1.39× ± 0.00 | 28.22 ± 2.74 | 0.052 ± 0.028 |
| 22.3% | SIGE (conv) | 1422 | 1.12× | 1.00× ± 0.00 | 27.04 ± 4.34 | 0.076 ± 0.046 |
|  | SIGE (conv+attn) | 1021 | 1.55× | 1.29× ± 0.00 | 26.82 ± 4.28 | 0.080 ± 0.051 |
| 43.7% | SIGE (conv) | 1566 | 1.01× | 0.99× ± 0.00 | 27.04 ± 2.78 | 0.074 ± 0.022 |
|  | SIGE (conv+attn) | 1386 | 1.15× | 1.05× ± 0.00 | 24.67 ± 2.54 | 0.103 ± 0.013 |
| 60.1% | SIGE (conv) | 1610 | 0.99× | 0.98× ± 0.00 | 22.31 ± 1.24 | 0.169 ± 0.004 |
|  | SIGE (conv+attn) | 1579 | 1.01× | 0.96× ± 0.00 | 20.56 ± 2.07 | 0.197 ± 0.028 |

### Block size (h200)

Effect of the SIGE block size (h200, 512 px, 10% edit). Conv = block size of the 3×3 convolutions (resnet and upsample), Attn = token block size of the transformers; 1×1 shortcuts (4) and stride-2 downsamplers (5) are fixed. The attention block size does not affect SIGE (conv), whose transformers are dense. Defaults are 6 / 4.

| Conv | Attn | Method | GMACs | MACs red. | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | **4** | SIGE (conv) | 1312 | 1.21× | 1.00× ± 0.00 | 28.42 ± 2.71 | 0.052 ± 0.022 |
|  |  | SIGE (conv+attn) | 708 | 2.24× | 1.37× ± 0.00 | 27.03 ± 2.55 | 0.066 ± 0.023 |
| **6** | 2 | SIGE (conv) | 1352 | 1.18× | 1.01× ± 0.00 | 28.16 ± 3.06 | 0.054 ± 0.020 |
|  |  | SIGE (conv+attn) | 556 | 2.86× | 1.67× ± 0.00 | 25.15 ± 1.70 | 0.071 ± 0.029 |
| **6** | **4** | SIGE (conv) | 1352 | 1.18× | 1.01× ± 0.00 | 28.16 ± 3.06 | 0.054 ± 0.020 |
|  |  | SIGE (conv+attn) | 748 | 2.12× | 1.39× ± 0.00 | 28.22 ± 2.74 | 0.052 ± 0.028 |
| **6** | 8 | SIGE (conv) | 1352 | 1.18× | 1.01× ± 0.00 | 28.16 ± 3.06 | 0.054 ± 0.020 |
|  |  | SIGE (conv+attn) | 1258 | 1.26× | 1.04× ± 0.00 | 28.05 ± 2.82 | 0.054 ± 0.020 |
| 8 | **4** | SIGE (conv) | 1387 | 1.15× | 1.01× ± 0.00 | 27.92 ± 2.97 | 0.055 ± 0.020 |
|  |  | SIGE (conv+attn) | 783 | 2.03× | 1.39× ± 0.00 | 27.67 ± 2.66 | 0.058 ± 0.026 |

### Baseline (rtx5070tilaptop)

SIGE vs. dense SDXL at the base point (rtx5070tilaptop): 512 px, 10% edit, 50 DDIM steps, strength 0.8, guidance 5.0, mean ± std over 3 prompts. PSNR / LPIPS compare the full SDEdit output with dense SDXL given the same input, prompt, noise and seed.

| Method | GMACs | MACs red. | Latency (ms) | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense SDXL | 1589 | 1.00× | 368.6 ± 1.9 | 1.00× | -- | -- |
| SIGE (conv) | 1352 | 1.18× | 341.9 ± 6.9 | 1.08× ± 0.02 | 28.48 ± 3.83 | 0.053 ± 0.025 |
| SIGE (conv+attn) | 748 | 2.12× | 237.3 ± 27.8 | 1.58× ± 0.20 | 28.00 ± 2.88 | 0.057 ± 0.026 |

### Edit size (rtx5070tilaptop)

Effect of edit size (rtx5070tilaptop, 512 px). Edit = measured fraction of changed pixels; MACs reduction and speedup are relative to dense SDXL; quality is vs. dense SDXL, mean ± std over 3 prompts.

| Edit | Method | GMACs | MACs red. | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---:|---|---:|---:|---:|---:|---:|
| 1.3% | SIGE (conv) | 1255 | 1.27× | 1.10× ± 0.02 | 46.58 ± 6.30 | 0.003 ± 0.002 |
|  | SIGE (conv+attn) | 378 | 4.20× | 1.81× ± 0.03 | 46.50 ± 6.38 | 0.003 ± 0.002 |
| 5.9% | SIGE (conv) | 1303 | 1.22× | 1.09× ± 0.03 | 33.99 ± 1.66 | 0.011 ± 0.003 |
|  | SIGE (conv+attn) | 502 | 3.16× | 1.62× ± 0.27 | 30.01 ± 2.78 | 0.023 ± 0.009 |
| 11.4% | SIGE (conv) | 1352 | 1.18× | 1.05× ± 0.02 | 28.48 ± 3.83 | 0.053 ± 0.025 |
|  | SIGE (conv+attn) | 748 | 2.12× | 1.69× ± 0.20 | 28.00 ± 2.88 | 0.057 ± 0.026 |
| 22.3% | SIGE (conv) | 1422 | 1.12× | 1.03× ± 0.04 | 28.10 ± 4.46 | 0.062 ± 0.036 |
|  | SIGE (conv+attn) | 1021 | 1.55× | 1.35× ± 0.06 | 27.68 ± 4.19 | 0.073 ± 0.046 |
| 43.7% | SIGE (conv) | 1566 | 1.01× | 0.99× ± 0.00 | 25.24 ± 2.58 | 0.119 ± 0.011 |
|  | SIGE (conv+attn) | 1386 | 1.15× | 1.06× ± 0.03 | 24.67 ± 2.09 | 0.116 ± 0.009 |
| 60.1% | SIGE (conv) | 1610 | 0.99× | 0.99× ± 0.03 | 22.48 ± 0.91 | 0.170 ± 0.001 |
|  | SIGE (conv+attn) | 1579 | 1.01× | 1.01× ± 0.01 | 22.36 ± 1.54 | 0.171 ± 0.026 |

### Block size (rtx5070tilaptop)

Effect of the SIGE block size (rtx5070tilaptop, 512 px, 10% edit). Conv = block size of the 3×3 convolutions (resnet and upsample), Attn = token block size of the transformers; 1×1 shortcuts (4) and stride-2 downsamplers (5) are fixed. The attention block size does not affect SIGE (conv), whose transformers are dense. Defaults are 6 / 4.

| Conv | Attn | Method | GMACs | MACs red. | Speedup | PSNR (dB) ↑ | LPIPS ↓ |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | **4** | SIGE (conv) | 1312 | 1.21× | 1.02× ± 0.03 | 27.90 ± 3.19 | 0.055 ± 0.023 |
|  |  | SIGE (conv+attn) | 708 | 2.24× | 1.50× ± 0.07 | 27.32 ± 2.97 | 0.066 ± 0.024 |
| **6** | 2 | SIGE (conv) | 1352 | 1.18× | 1.03× ± 0.01 | 28.48 ± 3.83 | 0.053 ± 0.025 |
|  |  | SIGE (conv+attn) | 556 | 2.86× | 1.47× ± 0.02 | 25.57 ± 1.69 | 0.066 ± 0.028 |
| **6** | **4** | SIGE (conv) | 1352 | 1.18× | 1.08× ± 0.02 | 28.48 ± 3.83 | 0.053 ± 0.025 |
|  |  | SIGE (conv+attn) | 748 | 2.12× | 1.58× ± 0.20 | 28.00 ± 2.88 | 0.057 ± 0.026 |
| **6** | 8 | SIGE (conv) | 1352 | 1.18× | 1.04× ± 0.02 | 28.48 ± 3.83 | 0.053 ± 0.025 |
|  |  | SIGE (conv+attn) | 1258 | 1.26× | 1.11× ± 0.06 | 28.06 ± 3.04 | 0.054 ± 0.021 |
| 8 | **4** | SIGE (conv) | 1387 | 1.15× | 1.04× ± 0.01 | 28.30 ± 3.85 | 0.054 ± 0.024 |
|  |  | SIGE (conv+attn) | 783 | 2.03× | 1.43× ± 0.03 | 27.69 ± 3.01 | 0.060 ± 0.026 |

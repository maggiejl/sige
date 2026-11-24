# SIGE-SDXL ablation tables

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

"""Run a single SDEdit edit with SDXL, dense and/or SIGE.

Example:
    python run.py --init_img original.png --edited_img edited.png \
        --prompt "a photograph of a mountain lake at sunset" --configs dense sige_conv sige_conv_attn
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from metrics import lpips, psnr
from sige_sdxl.pipeline import SIGESDXLEditor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_img", required=True)
    p.add_argument("--edited_img", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--configs", nargs="+", default=["dense", "sige_conv_attn"])
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preserve_unedited", action="store_true",
                   help="keep the image outside the edit mask (inpainting-style latent blending)")
    p.add_argument("--output_dir", default="outputs/run")
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--text_encoders_on_cpu", action="store_true", help="save ~3 GB of GPU memory")
    p.add_argument("--offload_vae", action="store_true",
                   help="keep the VAE off the GPU during denoising (needed on 12 GB cards)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    init = Image.open(args.init_img).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    edited = Image.open(args.edited_img).convert("RGB").resize((args.size, args.size), Image.LANCZOS)

    editor = SIGESDXLEditor(args.model_id, args.device, text_encoders_on_cpu=args.text_encoders_on_cpu,
                            offload_vae=args.offload_vae)
    results = {}
    for cfg in args.configs:
        r = editor.edit(
            init, edited, args.prompt, config=cfg, negative_prompt=args.negative_prompt,
            strength=args.strength, steps=args.steps, guidance=args.guidance, seed=args.seed,
            preserve_unedited=args.preserve_unedited,
        )
        r.image.save(os.path.join(args.output_dir, "%s.png" % cfg))
        results[cfg] = r
        ms = np.mean(r.unet_ms[2:]) if len(r.unet_ms) > 2 else np.mean(r.unet_ms)
        print("[%s] edit ratio %.2f%% | U-Net (edited image, CFG batch) %.1f ms/step" % (cfg, 100 * r.edit_ratio, ms))

    summary = {}
    if "dense" in results:
        ref = results["dense"]
        for cfg, r in results.items():
            summary[cfg] = {
                "unet_ms": float(np.mean(r.unet_ms[2:])),
                "psnr_vs_dense": psnr(r.image, ref.image) if cfg != "dense" else None,
                "lpips_vs_dense": lpips(r.image, ref.image, args.device) if cfg != "dense" else None,
            }
        print(json.dumps(summary, indent=2))
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

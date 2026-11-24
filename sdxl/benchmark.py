"""Main experiment: SIGE-SDXL vs dense SDXL across edit sizes and resolutions.

For each (resolution, edit ratio, config) this records
  * U-Net MACs and latency of one CFG U-Net call on the edited latent (``profile_unet``), and
  * PSNR / LPIPS of the full SDEdit result against dense SDXL with the same input, prompt, noise
    and seed (``--skip_quality`` to only measure efficiency).

Original images are generated with dense SDXL from ``--prompt`` (or loaded with ``--init_img``),
and edits are synthetic strokes of a controlled size (see ``edits.py``).

Example (Colab A100):
    python benchmark.py --resolutions 512 768 1024 --edit_ratios 0.01 0.05 0.1 0.2 0.4
"""

import argparse
import csv
import os

import torch
from PIL import Image

from edits import make_stroke_edit
from metrics import lpips, psnr
from sige_sdxl.pipeline import SIGESDXLEditor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a photograph of a cozy wooden cabin by a lake in the mountains, highly detailed")
    p.add_argument("--init_img", default=None, help="use this image instead of generating one")
    p.add_argument("--resolutions", type=int, nargs="+", default=[1024])
    p.add_argument("--edit_ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.4])
    p.add_argument("--configs", nargs="+", default=["dense", "sige_conv", "sige_conv_attn"])
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--skip_quality", action="store_true")
    p.add_argument("--output_dir", default="outputs/benchmark")
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--text_encoders_on_cpu", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    editor = SIGESDXLEditor(args.model_id, args.device, text_encoders_on_cpu=args.text_encoders_on_cpu)
    csv_path = os.path.join(args.output_dir, "results.csv")
    fields = ["resolution", "target_ratio", "edit_ratio", "config", "gmacs", "latency_ms", "latency_std_ms",
              "macs_reduction", "speedup", "psnr_vs_dense", "lpips_vs_dense"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for res in args.resolutions:
        img_dir = os.path.join(args.output_dir, str(res))
        os.makedirs(img_dir, exist_ok=True)
        orig_path = os.path.join(img_dir, "original.png")
        if args.init_img is not None:
            init = Image.open(args.init_img).convert("RGB").resize((res, res), Image.LANCZOS)
        elif os.path.exists(orig_path):
            init = Image.open(orig_path).convert("RGB")
        else:
            print("Generating original image at %d..." % res)
            init = editor.generate(args.prompt, size=res, seed=args.seed)
        init.save(orig_path)

        for ratio in args.edit_ratios:
            edited = make_stroke_edit(init, ratio, seed=args.seed)
            tag = "r%.2f" % ratio
            edited.save(os.path.join(img_dir, "%s_edited.png" % tag))

            prof = {cfg: editor.profile_unet(init, edited, args.prompt, cfg, repeats=args.repeats) for cfg in args.configs}
            dense_img = None
            rows = []
            for cfg in args.configs:
                row = dict(resolution=res, target_ratio=ratio, **prof[cfg])
                if "dense" in prof:
                    row["macs_reduction"] = prof["dense"]["gmacs"] / row["gmacs"]
                    row["speedup"] = prof["dense"]["latency_ms"] / row["latency_ms"]
                if not args.skip_quality:
                    out = editor.edit(init, edited, args.prompt, config=cfg, strength=args.strength,
                                      steps=args.steps, guidance=args.guidance, seed=args.seed)
                    out.image.save(os.path.join(img_dir, "%s_%s.png" % (tag, cfg)))
                    if cfg == "dense":
                        dense_img = out.image
                    elif dense_img is not None:
                        row["psnr_vs_dense"] = psnr(out.image, dense_img)
                        row["lpips_vs_dense"] = lpips(out.image, dense_img, args.device)
                rows.append(row)
                print(" | ".join("%s=%s" % (k, ("%.4g" % v) if isinstance(v, float) else v) for k, v in row.items()))
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in fields})
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()

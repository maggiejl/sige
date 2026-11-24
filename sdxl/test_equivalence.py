"""Correctness checks for the SIGE SDXL port.

Runs on CPU in seconds with a tiny, randomly initialized U-Net that has SDXL's architecture
(DownBlock2D / CrossAttnDownBlock2D x2, linear-projection transformers with depth > 1, text_time
conditioning). Pass ``--pretrained`` to run the same checks on the real SDXL U-Net (needs a GPU).

Checks:
 1. Unedited input + all-active mask: sparse == dense. Every block goes through the gather/scatter
    kernels, so this validates the port (block sizes, halos, GroupNorm folding, K/V scatter).
 2. Unedited input + partial mask: sparse == dense (inactive regions come from the cache).
 3. Edited input + all-active mask: only differs from dense by the GroupNorm statistics, which SIGE
    reuses from the original image.
 4. Edited input + edit mask: the SIGE approximation. Reported  against dense(edited), with the error
    of simply returning dense(original) as a reference point.
"""

import argparse

import torch

from sige.utils import downsample_mask
from sige_sdxl import patch_unet


def tiny_sdxl_unet():
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64, 128),
        layers_per_block=2,
        transformer_layers_per_block=(1, 2, 3),
        attention_head_dim=(2, 4, 8),
        cross_attention_dim=64,
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=8 * 6 + 32,
        norm_num_groups=32,
    )


def make_inputs(unet, batch, res, device, pooled_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    cdim = unet.config.cross_attention_dim
    x = torch.randn(batch, 4, res, res, generator=g).to(device)
    ctx = torch.randn(batch, 77, cdim, generator=g).to(device)
    pooled = torch.randn(batch, pooled_dim, generator=g).to(device)
    time_ids = torch.tensor([[res * 8, res * 8, 0, 0, res * 8, res * 8]] * batch, dtype=torch.float32, device=device)
    t = torch.tensor([500] * batch, device=device)
    kwargs = dict(encoder_hidden_states=ctx, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids})
    return x, t, kwargs


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


@torch.no_grad()
def run_checks(unet, sige, res, device, pooled_dim, config, tol):
    sige.set_config(config)
    x, t, kw = make_inputs(unet, 2, res, device, pooled_dim)  # batch 2 mimics CFG

    def run(mode, inp):
        sige.set_mode(mode)
        return unet(inp, t, **kw).sample

    dense_x = run("dense", x)

    # local edit in the latent: a square in the upper-left quadrant
    x_edit = x.clone()
    s, e = res // 8, res // 8 + res // 6
    x_edit[:, :, s:e, s:e] += 1.5 * torch.randn_like(x_edit[:, :, s:e, s:e])
    dense_edit = run("dense", x_edit)

    edit_mask = torch.zeros(res, res, dtype=torch.bool, device=device)
    edit_mask[s:e, s:e] = True
    full_mask = torch.ones(res, res, dtype=torch.bool, device=device)

    results = {}
    for name, inp, mask, ref in [
        ("1 unedited, all-active mask", x, full_mask, dense_x),
        ("2 unedited, partial mask   ", x, edit_mask, dense_x),
        ("3 edited,   all-active mask", x_edit, full_mask, dense_edit),
        ("4 edited,   edit mask      ", x_edit, edit_mask, dense_edit),
    ]:
        run("full", x)  # cache activations of the original input
        sige.set_masks(downsample_mask(mask, min_res=(4, 4), dilation=1))
        out = run("sparse", inp)
        results[name] = rel_err(out, ref)

    print("\n[%s] relative L2 error vs dense SDXL U-Net output" % config)
    for k, v in results.items():
        print("  %s : %.2e" % (k, v))
    print("  (reference) dense(original) vs dense(edited): %.2e" % rel_err(dense_x, dense_edit))

    ok = True
    for k in list(results)[:2]:
        if results[k] > tol:
            print("  FAIL: %s exceeds tolerance %.0e" % (k.strip(), tol))
            ok = False
    if results[list(results)[3]] >= rel_err(dense_x, dense_edit):
        print("  FAIL: SIGE output is no closer to dense(edited) than dense(original) is")
        ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="use the real SDXL U-Net")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--res", type=int, default=None, help="latent resolution (default 32 tiny / 128 SDXL)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.pretrained:
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float32)
        res = args.res or 128
        pooled_dim = unet.config.projection_class_embeddings_input_dim - 6 * unet.config.addition_time_embed_dim
        tol = 1e-3
    else:
        unet = tiny_sdxl_unet()
        res = args.res or 32
        pooled_dim = 32
        tol = 1e-4
    unet = unet.to(args.device).eval()
    sige = patch_unet(unet)

    ok = True
    for config in ("sige_conv", "sige_conv_attn"):
        ok &= run_checks(unet, sige, res, args.device, pooled_dim, config, tol)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

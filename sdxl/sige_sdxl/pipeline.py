"""SDEdit with SDXL, with and without SIGE.

Mirrors ``stable-diffusion/runners/sdedit_runner.py`` + ``DDIMSampler.sige_img2img_decode``:
the original and edited images are encoded, noised with the *same* noise, and denoised side by side.
At every step the original latent goes through a ``full`` pass (populating SIGE's caches) and the
edited latent through a ``sparse`` pass. As in the SIGE paper, the reported SIGE latency is the sparse
pass alone: in an interactive editor the original image's activations are computed once and reused
across edits.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from sige.utils import compute_difference_mask, dilate_mask, downsample_mask
from .patch import CONFIGS, patch_unet


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None] * 2.0 - 1.0


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = ((x[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((x * 255).round().astype(np.uint8))


def compute_masks(init: torch.Tensor, edited: torch.Tensor) -> Tuple[Dict, float]:
    """Pixel-space difference mask -> masks at every U-Net resolution (same recipe as SIGE SD1.x)."""
    diff = compute_difference_mask(init, edited)
    ratio = diff.float().mean().item()
    diff = dilate_mask(diff, 5)
    return downsample_mask(diff, min_res=(4, 4), dilation=1), ratio


@dataclass
class EditResult:
    image: Image.Image
    config: str
    edit_ratio: float = 0.0
    unet_ms: List[float] = field(default_factory=list)  # per-step latency of the edited-image U-Net call
    full_ms: List[float] = field(default_factory=list)  # per-step latency of the cache-filling pass (SIGE only)
    original_image: Optional[Image.Image] = None  # SIGE only: SDEdit result for the unedited image
    latents: Optional[torch.Tensor] = None


class SIGESDXLEditor:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        text_encoders_on_cpu: bool = False,
        patch_kwargs: Optional[dict] = None,
    ):
        from diffusers import DDIMScheduler, StableDiffusionXLPipeline

        # SIGE's gather/scatter kernels are float32-only, so the whole U-Net runs in float32
        # (the dense baseline too, for a fair comparison).
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.to(device).eval()
        pipe.vae.to(device).eval()
        te_device = "cpu" if text_encoders_on_cpu else device
        pipe.text_encoder.to(te_device)
        pipe.text_encoder_2.to(te_device)
        self.te_device = te_device
        self.pipe = pipe
        self.device = torch.device(device)
        self.sige = patch_unet(pipe.unet, **(patch_kwargs or {}))
        self._prompt_cache = {}

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        key = (prompt, negative_prompt)
        if key not in self._prompt_cache:
            pe, npe, pooled, npooled = self.pipe.encode_prompt(
                prompt=prompt,
                device=torch.device(self.te_device),
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            self._prompt_cache[key] = tuple(t.to(self.device, torch.float32) for t in (pe, npe, pooled, npooled))
        return self._prompt_cache[key]

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        return vae.encode(img.to(self.device)).latent_dist.mode() * vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> Image.Image:
        vae = self.pipe.vae
        return tensor_to_pil(vae.decode(z / vae.config.scaling_factor).sample)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _conditioning(self, prompt: str, negative_prompt: str, H: int, W: int):
        pe, npe, pooled, npooled = self.encode_prompt(prompt, negative_prompt)
        ctx = torch.cat([npe, pe])
        time_ids = torch.tensor([[H, W, 0, 0, H, W]], dtype=torch.float32, device=self.device).repeat(2, 1)
        return ctx, {"text_embeds": torch.cat([npooled, pooled]), "time_ids": time_ids}

    def _cfg_eps(self, sched, ctx, added, guidance):
        unet = self.pipe.unet

        def eps(z, t):
            zin = sched.scale_model_input(torch.cat([z, z]), t)
            out = unet(zin, t, encoder_hidden_states=ctx, added_cond_kwargs=added, return_dict=False)[0]
            u, c = out.chunk(2)
            return u + guidance * (c - u)

        return eps

    @torch.no_grad()
    def generate(
        self, prompt: str, size: int = 1024, steps: int = 30, guidance: float = 5.0, seed: int = 0, negative_prompt: str = ""
    ) -> Image.Image:
        """Plain (dense) SDXL text-to-image, used to create original images for the experiments."""
        self.sige.set_config("dense")
        self.sige.set_mode("dense")
        ctx, added = self._conditioning(prompt, negative_prompt, size, size)
        sched = copy.deepcopy(self.pipe.scheduler)
        sched.set_timesteps(steps, device=self.device)
        eps = self._cfg_eps(sched, ctx, added, guidance)
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(1, 4, size // 8, size // 8, generator=g).to(self.device) * sched.init_noise_sigma
        for t in sched.timesteps:
            z = sched.step(eps(z, t), t, z).prev_sample
        return self.decode_latents(z)

    # ------------------------------------------------------------------ SDEdit
    @torch.no_grad()
    def edit(
        self,
        init_image: Image.Image,
        edited_image: Image.Image,
        prompt: str,
        config: str = "sige_conv_attn",
        negative_prompt: str = "",
        strength: float = 0.8,
        steps: int = 50,
        guidance: float = 5.0,
        seed: int = 0,
        decode: bool = True,
    ) -> EditResult:
        assert config in CONFIGS, config
        assert init_image.size == edited_image.size
        W, H = edited_image.size
        assert H % 64 == 0 and W % 64 == 0, "use image sizes that are multiples of 64"

        init_px = pil_to_tensor(init_image).to(self.device)
        edit_px = pil_to_tensor(edited_image).to(self.device)
        masks, ratio = compute_masks(init_px, edit_px)

        ctx, added = self._conditioning(prompt, negative_prompt, H, W)
        sched = copy.deepcopy(self.pipe.scheduler)
        sched.set_timesteps(steps, device=self.device)
        n_denoise = min(int(steps * strength), steps)
        timesteps = sched.timesteps[steps - n_denoise :]

        g = torch.Generator(device="cpu").manual_seed(seed)
        z_edit0 = self.encode_image(edit_px)
        noise = torch.randn(z_edit0.shape, generator=g).to(self.device)
        z_edit = sched.add_noise(z_edit0, noise, timesteps[:1])
        use_sige = config != "dense"
        if use_sige:
            sched_init = copy.deepcopy(sched)
            z_init = sched.add_noise(self.encode_image(init_px), noise, timesteps[:1])

        self.sige.set_config(config)
        eps = self._cfg_eps(sched, ctx, added, guidance)

        res = EditResult(image=None, config=config, edit_ratio=ratio)
        for i, t in enumerate(timesteps):
            if use_sige:
                self.sige.set_mode("full")
                self._sync()
                t0 = time.perf_counter()
                e_init = eps(z_init, t)
                self._sync()
                res.full_ms.append((time.perf_counter() - t0) * 1e3)
                z_init = sched_init.step(e_init, t, z_init).prev_sample
                if i == 0:
                    self.sige.set_masks(masks)
                self.sige.set_mode("sparse")
            else:
                self.sige.set_mode("dense")
            self._sync()
            t0 = time.perf_counter()
            e_edit = eps(z_edit, t)
            self._sync()
            res.unet_ms.append((time.perf_counter() - t0) * 1e3)
            z_edit = sched.step(e_edit, t, z_edit).prev_sample

        self.sige.set_mode("dense")
        self.sige.clear_cache()
        res.latents = z_edit
        if decode:
            res.image = self.decode_latents(z_edit)
            if use_sige:
                res.original_image = self.decode_latents(z_init)
        return res

    # ------------------------------------------------------------------ profiling
    @torch.no_grad()
    def profile_unet(
        self, init_image: Image.Image, edited_image: Image.Image, prompt: str, config: str, t: int = 500, warmup: int = 3, repeats: int = 10
    ) -> Dict[str, float]:
        """MACs (via torch's FlopCounterMode) and latency of one CFG U-Net call on the edited latent."""
        from torch.utils.flop_counter import FlopCounterMode

        W, H = edited_image.size
        init_px = pil_to_tensor(init_image).to(self.device)
        edit_px = pil_to_tensor(edited_image).to(self.device)
        masks, ratio = compute_masks(init_px, edit_px)
        ctx, added = self._conditioning(prompt, "", H, W)
        z_init = self.encode_image(init_px).repeat(2, 1, 1, 1)
        z_edit = self.encode_image(edit_px).repeat(2, 1, 1, 1)
        tt = torch.tensor(t, device=self.device)
        unet = self.pipe.unet

        def call(z):
            return unet(z, tt, encoder_hidden_states=ctx, added_cond_kwargs=added, return_dict=False)[0]

        self.sige.set_config(config)
        if config == "dense":
            self.sige.set_mode("dense")
        else:
            self.sige.set_mode("full")
            call(z_init)
            self.sige.set_masks(masks)
            self.sige.set_mode("sparse")

        with FlopCounterMode(display=False) as fc:
            call(z_edit)
        macs = fc.get_total_flops() / 2

        for _ in range(warmup):
            call(z_edit)
        times = []
        for _ in range(repeats):
            self._sync()
            t0 = time.perf_counter()
            call(z_edit)
            self._sync()
            times.append((time.perf_counter() - t0) * 1e3)

        self.sige.set_mode("dense")
        self.sige.clear_cache()
        return {
            "config": config,
            "edit_ratio": ratio,
            "gmacs": macs / 1e9,
            "latency_ms": float(np.mean(times)),
            "latency_std_ms": float(np.std(times)),
        }

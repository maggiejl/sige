"""Patch a diffusers SDXL ``UNet2DConditionModel`` in place with SIGE wrappers."""

from typing import Dict, Tuple

import torch
from torch import nn

from sige.nn import SIGEModel, SIGEModule
from .modules import SIGEDownsample2D, SIGEResnetBlock2D, SIGETransformer2D, SIGEUpsample2D, _SIGEWrapper

CONV_WRAPPERS = (SIGEResnetBlock2D, SIGEDownsample2D, SIGEUpsample2D)

# Configurations compared in the experiments.
CONFIGS = ("dense", "sige_conv", "sige_conv_attn")


class SIGEUNet(SIGEModel):
    """Controller for a patched U-Net. ``self.unet`` is the (patched) diffusers model; call it as usual.

    Usage per denoising step (see ``pipeline.SIGESDXLEditor``)::

        sige_unet.set_config("sige_conv_attn")
        sige_unet.set_mode("full");   unet(original latents)   # fills caches
        sige_unet.set_masks(masks)                             # once, after the first full pass
        sige_unet.set_mode("sparse"); unet(edited latents)     # only active blocks are recomputed

    ``set_config("dense")`` makes every wrapper call its original module, i.e. stock SDXL.
    ``set_config("sige_conv")`` keeps the transformers dense (they recompute every token every pass)
    and sparsifies only the convolutional layers.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        self.config_name = "sige_conv_attn"

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)

    def wrappers(self):
        return [m for m in self.unet.modules() if isinstance(m, _SIGEWrapper)]

    def set_config(self, config_name: str):
        assert config_name in CONFIGS, config_name
        self.config_name = config_name
        self.set_mode(self.mode)

    def _wrapper_mode(self, wrapper: _SIGEWrapper, mode: str) -> str:
        if self.config_name == "dense" or mode == "dense":
            return "dense"
        if self.config_name == "sige_conv" and isinstance(wrapper, SIGETransformer2D):
            return "dense"
        return mode

    def set_mode(self, mode: str):
        assert mode in ("dense", "full", "sparse"), mode
        self.mode = mode
        for w in self.wrappers():
            m = self._wrapper_mode(w, mode)
            for sub in w.modules():
                if isinstance(sub, SIGEModule):
                    sub.set_mode(m)

    def set_masks(self, masks: Dict[Tuple[int, int], torch.Tensor]):
        """Reduce the masks to active block indices for every wrapper that is not dense."""
        self.timestamp += 1
        cache = {}
        for w in self.wrappers():
            if w.mode == "dense":
                continue
            for sub in w.modules():
                if isinstance(sub, SIGEModule):
                    sub.set_mask(masks, cache, self.timestamp)

    def clear_cache(self):
        for m in self.unet.modules():
            if isinstance(m, SIGEModule):
                m.clear_cache()


def patch_unet(
    unet: nn.Module,
    main_block_size: int = 6,
    shortcut_block_size: int = 4,
    downsample_block_size: int = 5,
    upsample_block_size: int = 6,
    attn_block_size: int = 4,
    verbose: bool = True,
) -> SIGEUNet:
    """Replace ResnetBlock2D / Downsample2D / Upsample2D / Transformer2DModel children with SIGE wrappers.

    The stem (``conv_in``), the output head (``conv_norm_out``/``conv_out``) and the time/text embeddings
    stay dense, as in SIGE's SD1.x port.
    """
    counts = {"ResnetBlock2D": 0, "Downsample2D": 0, "Upsample2D": 0, "Transformer2DModel": 0}
    for parent in list(unet.modules()):
        if isinstance(parent, _SIGEWrapper):
            continue
        for name, child in list(parent.named_children()):
            cls = type(child).__name__
            if cls == "ResnetBlock2D":
                new = SIGEResnetBlock2D(child, main_block_size, shortcut_block_size)
            elif cls == "Downsample2D":
                new = SIGEDownsample2D(child, downsample_block_size)
            elif cls == "Upsample2D":
                new = SIGEUpsample2D(child, upsample_block_size)
            elif cls == "Transformer2DModel":
                new = SIGETransformer2D(child, attn_block_size)
            else:
                continue
            setattr(parent, name, new)
            counts[cls] += 1
    if verbose:
        print("[SIGE] patched U-Net: " + ", ".join("%d %s" % (v, k) for k, v in counts.items()))
    return SIGEUNet(unet)

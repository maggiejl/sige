"""SIGE wrappers for the building blocks of diffusers' SDXL U-Net.

Each wrapper holds a reference to the original diffusers module (``self.orig``) and reuses its
weights, so patching does not copy any parameters. Every wrapper supports four modes:

* ``dense``  -- call the untouched diffusers module (standard SDXL; used as the baseline).
* ``full``   -- dense computation on the *original* input that also fills SIGE's caches
                (activations, GroupNorm statistics, self-attention keys/values).
* ``sparse`` -- recompute only the blocks that overlap the edit mask and scatter them into the
                cached activations from the last ``full`` pass.

The structure follows ``stable-diffusion/ldm/modules/diffusionmodules/sige_openaimodel.py`` and
``stable-diffusion/ldm/modules/sige_attention.py`` (SIGE's Stable Diffusion 1.x port).
"""

from types import SimpleNamespace
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEModule


def group_norm_with_stats(x: torch.Tensor, norm: nn.GroupNorm) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GroupNorm that also returns the per-channel affine ``(scale, shift)`` it applied.

    In sparse mode SIGE reuses these statistics from the original image instead of recomputing them,
    because the statistics of a few gathered blocks are not the statistics of the full feature map.
    """
    n, c, h, w = x.shape
    g = norm.num_groups
    gs = c // g
    xg = x.view(n, g, gs, h, w)
    var, mean = torch.var_mean(xg, unbiased=False, dim=[2, 3, 4], keepdim=True)
    std = torch.sqrt(var + norm.eps)
    xg = (xg - mean) / std
    scale = (1 / std).view(n, g, 1, 1).repeat_interleave(gs, dim=1)
    shift = (-mean / std).view(n, g, 1, 1).repeat_interleave(gs, dim=1)
    x = xg.view(n, c, h, w)
    if norm.affine:
        w_ = norm.weight.view(1, -1, 1, 1)
        b_ = norm.bias.view(1, -1, 1, 1)
        x = x * w_ + b_
        scale = scale * w_
        shift = shift * w_ + b_
    return x, scale, shift


def conv_no_pad(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Apply ``conv`` to gathered blocks. The blocks already contain the halo, so no padding."""
    return F.conv2d(x, conv.weight, conv.bias, conv.stride, 0, conv.dilation, conv.groups)


def _check_float32(*tensors):
    for t in tensors:
        if t is not None and t.dtype != torch.float32:
            raise NotImplementedError("SIGE kernels only support float32 (got %s)." % t.dtype)


class _SIGEWrapper(SIGEModule):
    """Base for wrappers: keeps the original module and adds the ``dense`` mode."""

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.orig = orig

    @property
    def is_dense(self) -> bool:
        return self.mode == "dense"


class SIGEResnetBlock2D(_SIGEWrapper):
    """Sparse version of ``diffusers.models.resnet.ResnetBlock2D`` (SDXL configuration).

    full:   h = conv2(swish(norm2(conv1(swish(norm1(x))) + temb))) ; out = shortcut(x) + h
    sparse: gather(x) -> [norm1 + swish fused into gather] -> conv1 -> scatter_gather
            -> [norm2 (with temb folded into the shift) + swish fused] -> conv2 -> scatter(+ shortcut)
    """

    def __init__(self, orig: nn.Module, main_block_size: int = 6, shortcut_block_size: int = 4):
        super().__init__(orig)
        assert not getattr(orig, "up", False) and not getattr(orig, "down", False), "up/down resnets not supported"
        assert getattr(orig, "time_embedding_norm", "default") == "default"
        assert getattr(orig, "output_scale_factor", 1.0) == 1.0
        assert isinstance(orig.norm1, nn.GroupNorm) and isinstance(orig.norm2, nn.GroupNorm)
        assert isinstance(orig.nonlinearity, nn.SiLU), "SIGE kernels fuse swish only"

        self.main_gather = Gather(orig.conv1, main_block_size, activation_name="swish")
        self.scatter_gather = ScatterGather(self.main_gather, activation_name="swish")
        if orig.conv_shortcut is not None:
            assert orig.conv_shortcut.kernel_size == (1, 1)
            self.shortcut_gather = Gather(orig.conv_shortcut, shortcut_block_size)
            self.scatter = ScatterWithBlockResidual(self.main_gather, self.shortcut_gather)
        else:
            self.shortcut_gather = None
            self.scatter = Scatter(self.main_gather)

        self.scale1 = self.shift1 = None
        self.scale2 = self.shift2 = None

    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs):
        if self.is_dense:
            return self.orig(input_tensor, temb, *args, **kwargs)
        _check_float32(input_tensor)
        if self.mode == "full":
            return self._full_forward(input_tensor, temb)
        elif self.mode == "sparse":
            return self._sparse_forward(input_tensor)
        raise NotImplementedError("Unknown mode [%s]" % self.mode)

    def _full_forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        o = self.orig
        h = x
        if self.shortcut_gather is not None:
            x = self.shortcut_gather(x)
            x = o.conv_shortcut(x)

        h = self.main_gather(h)  # records the input resolution
        h, self.scale1, self.shift1 = group_norm_with_stats(h, o.norm1)
        h = o.nonlinearity(h)
        h = o.conv1(h)
        h = self.scatter_gather(h)  # caches conv1 output

        if temb is not None and o.time_emb_proj is not None:
            t = o.time_emb_proj(o.nonlinearity(temb))[:, :, None, None]
            h = h + t
            h, scale, shift = group_norm_with_stats(h, o.norm2)
            # norm2(h + t) == h * scale + (t * scale + shift): fold temb into the shift
            shift = t * scale + shift
        else:
            h, scale, shift = group_norm_with_stats(h, o.norm2)
        self.scale2, self.shift2 = scale, shift

        h = o.nonlinearity(h)
        h = o.dropout(h)
        h = o.conv2(h)
        return self.scatter(h, x)  # x + h, cached

    def _sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.orig
        h = x
        if self.shortcut_gather is not None:
            x = self.shortcut_gather(x)
            x = conv_no_pad(o.conv_shortcut, x)
        h = self.main_gather(h, self.scale1, self.shift1)
        h = conv_no_pad(o.conv1, h)
        h = self.scatter_gather(h, self.scale2, self.shift2)
        h = conv_no_pad(o.conv2, h)
        return self.scatter(h, x)


class SIGEDownsample2D(_SIGEWrapper):
    """Sparse version of ``diffusers.models.downsampling.Downsample2D`` (3x3 conv, stride 2, pad 1)."""

    def __init__(self, orig: nn.Module, block_size: int = 5):
        super().__init__(orig)
        assert orig.use_conv and getattr(orig, "norm", None) is None
        assert orig.padding == 1, "only symmetric padding is supported"
        self.gather = Gather(orig.conv, block_size)
        self.scatter = Scatter(self.gather)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self.is_dense:
            return self.orig(hidden_states, *args, **kwargs)
        _check_float32(hidden_states)
        x = self.gather(hidden_states)
        x = self.orig.conv(x) if self.mode == "full" else conv_no_pad(self.orig.conv, x)
        return self.scatter(x)


class SIGEUpsample2D(_SIGEWrapper):
    """Sparse version of ``diffusers.models.upsampling.Upsample2D`` (nearest x2 then 3x3 conv)."""

    def __init__(self, orig: nn.Module, block_size: int = 6):
        super().__init__(orig)
        assert orig.use_conv and not orig.use_conv_transpose and getattr(orig, "norm", None) is None
        assert getattr(orig, "interpolate", True)
        self.gather = Gather(orig.conv, block_size)
        self.scatter = Scatter(self.gather)

    def forward(self, hidden_states: torch.Tensor, output_size=None, *args, **kwargs):
        if self.is_dense:
            return self.orig(hidden_states, output_size, *args, **kwargs)
        _check_float32(hidden_states)
        # The upsample itself runs on the full (cached + scattered) map; it is memory-bound and cheap.
        if output_size is None:
            x = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(hidden_states, size=output_size, mode="nearest")
        x = self.gather(x)
        x = self.orig.conv(x) if self.mode == "full" else conv_no_pad(self.orig.conv, x)
        return self.scatter(x)


def _attention(attn: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Same math as diffusers' ``AttnProcessor2_0`` for SDXL's ``Attention`` modules."""
    b = q.shape[0]
    heads = attn.heads
    d = q.shape[-1] // heads
    q = q.view(b, -1, heads, d).transpose(1, 2)
    k = k.view(b, -1, heads, d).transpose(1, 2)
    v = v.view(b, -1, heads, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, scale=attn.scale)
    out = out.transpose(1, 2).reshape(b, -1, heads * d)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return out


class _Output(tuple):
    """Stand-in for ``Transformer2DModelOutput`` that works with both ``[0]`` and ``.sample``."""

    @property
    def sample(self):
        return self[0]


class SIGETransformer2D(_SIGEWrapper):
    """Sparse version of ``diffusers`` ``Transformer2DModel`` as used in SDXL.

    Sparse mode only processes the tokens inside active blocks (block size ``block_size`` x ``block_size``
    on the transformer's input resolution):

    * GroupNorm uses the cached statistics of the original image (fused into the gather).
    * proj_in / LayerNorms / FFN / proj_out are token-wise, so they run on active tokens only.
    * Self-attention: queries, keys and values are computed for active tokens; the active keys/values are
      written into the cached keys/values of the original image, and active queries attend over the full
      spatial grid. (SDXL stacks 2-10 transformer blocks per layer, so unlike SIGE's SD1.x port we cache
      K/V per block rather than reusing the proj_in output as context.)
    * Cross-attention: active queries attend to the full text context.
    """

    def __init__(self, orig: nn.Module, block_size: int = 4):
        super().__init__(orig)
        assert getattr(orig, "is_input_continuous", True), "only continuous (image) inputs are supported"
        assert isinstance(orig.norm, nn.GroupNorm)
        self.use_linear_projection = orig.use_linear_projection
        for blk in orig.transformer_blocks:
            assert getattr(blk, "norm_type", "layer_norm") == "layer_norm", "only SDXL-style blocks"
            assert getattr(blk, "pos_embed", None) is None
            assert not getattr(blk, "only_cross_attention", False)
            assert blk.attn2 is not None
            for attn in (blk.attn1, blk.attn2):
                assert getattr(attn, "norm_q", None) is None and getattr(attn, "norm_k", None) is None
                assert not getattr(attn, "residual_connection", False)
                assert getattr(attn, "rescale_output_factor", 1.0) == 1.0

        # A 1x1 "conv" spec for the gather: proj_in is token-wise, so blocks need no halo.
        spec = SimpleNamespace(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.gather = Gather(spec, block_size)
        self.scatter = Scatter(self.gather)

        self.scale = self.shift = None
        self.cached_kv = [None] * len(orig.transformer_blocks)
        self.token_idx: Optional[torch.Tensor] = None  # flat spatial index of each valid active token
        self.token_pos: Optional[torch.Tensor] = None  # position of that token in the gathered token list

    def clear_cache(self):
        self.cached_kv = [None] * len(self.orig.transformer_blocks)

    def set_mask(self, masks, cache, timestamp):
        if self.timestamp == timestamp:
            return
        super().set_mask(masks, cache, timestamp)
        self.gather.set_mask(masks, cache, timestamp)
        H, W = self.gather.input_res
        bh, bw = self.gather.block_size
        key = ("token_index", H, W, bh, bw, self.gather.active_indices.data_ptr())
        if key not in cache:
            ai = self.gather.active_indices.long()
            device = ai.device
            hh = ai[:, 0, None, None] + torch.arange(bh, device=device)[None, :, None]
            ww = ai[:, 1, None, None] + torch.arange(bw, device=device)[None, None, :]
            hh, ww = torch.broadcast_tensors(hh, ww)
            valid = ((hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)).reshape(-1)
            flat = (hh * W + ww).reshape(-1)
            cache[key] = (flat[valid].contiguous(), torch.nonzero(valid).squeeze(1).contiguous())
        self.token_idx, self.token_pos = cache[key]

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, *args, **kwargs):
        if self.is_dense:
            return self.orig(hidden_states, encoder_hidden_states, *args, **kwargs)
        assert kwargs.get("attention_mask") is None and kwargs.get("encoder_attention_mask") is None
        _check_float32(hidden_states)
        if self.mode == "full":
            out = self._full_forward(hidden_states, encoder_hidden_states)
        elif self.mode == "sparse":
            out = self._sparse_forward(hidden_states, encoder_hidden_states)
        else:
            raise NotImplementedError("Unknown mode [%s]" % self.mode)
        return _Output((out,))

    # -- full -------------------------------------------------------------------------------------
    def _full_forward(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        o = self.orig
        b, _, h, w = x.shape
        residual = x
        x = self.gather(x)  # records the input resolution
        x, self.scale, self.shift = group_norm_with_stats(x, o.norm)
        if not self.use_linear_projection:
            x = o.proj_in(x)
        c = x.shape[1]
        t = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.use_linear_projection:
            t = o.proj_in(t)

        for i, blk in enumerate(o.transformer_blocks):
            n = blk.norm1(t)
            k = blk.attn1.to_k(n)
            v = blk.attn1.to_v(n)
            self.cached_kv[i] = (k.contiguous(), v.contiguous())
            t = _attention(blk.attn1, blk.attn1.to_q(n), k, v) + t
            t = self._cross_attn_ff(blk, t, context)

        if self.use_linear_projection:
            t = o.proj_out(t)
        c = t.shape[-1]
        x = t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if not self.use_linear_projection:
            x = o.proj_out(x)
        return self.scatter(x, residual)

    # -- sparse -----------------------------------------------------------------------------------
    def _sparse_forward(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        o = self.orig
        b = x.shape[0]
        bh, bw = self.gather.block_size
        residual = x
        xb = self.gather(x, self.scale, self.shift)  # [b * nb, c, bh, bw], normalized with cached stats
        if not self.use_linear_projection:
            xb = conv_no_pad(o.proj_in, xb)
        c = xb.shape[1]
        nb = xb.shape[0] // b
        t = xb.view(b, nb, c, bh * bw).permute(0, 1, 3, 2).reshape(b, nb * bh * bw, c)
        if self.use_linear_projection:
            t = o.proj_in(t)

        idx, pos = self.token_idx, self.token_pos
        for i, blk in enumerate(o.transformer_blocks):
            n = blk.norm1(t)
            k_full, v_full = self.cached_kv[i]
            k_full = k_full.clone()
            v_full = v_full.clone()
            k_full[:, idx] = blk.attn1.to_k(n)[:, pos]
            v_full[:, idx] = blk.attn1.to_v(n)[:, pos]
            t = _attention(blk.attn1, blk.attn1.to_q(n), k_full, v_full) + t
            t = self._cross_attn_ff(blk, t, context)

        if self.use_linear_projection:
            t = o.proj_out(t)
        c = t.shape[-1]
        xb = t.view(b, nb, bh * bw, c).permute(0, 1, 3, 2).reshape(b * nb, c, bh, bw)
        if not self.use_linear_projection:
            xb = conv_no_pad(o.proj_out, xb)
        return self.scatter(xb, residual)

    @staticmethod
    def _cross_attn_ff(blk: nn.Module, t: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        n = blk.norm2(t)
        ctx = n if context is None else context
        t = _attention(blk.attn2, blk.attn2.to_q(n), blk.attn2.to_k(ctx), blk.attn2.to_v(ctx)) + t
        t = blk.ff(blk.norm3(t)) + t
        return t

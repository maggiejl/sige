"""SD 1.x vs SDXL U-Net, and where SIGE is injected. Styled to match Figure 1 of the report.

    python make_arch_figure.py ../sdxl_sige_architecture   # writes .pdf and .png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK = "#2b2f33"
TEAL_TXT = "#2f6466"
PANEL = "#f2f4f3"
CONV = "#c5dcdc"      # SIGE sparse conv
ATTN = "#efc98f"      # SIGE sparse attention
PLAIN = "#ffffff"     # SD reference (no SIGE)
DENSE = "#e4e6e8"     # stays dense
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, ax = plt.subplots(figsize=(13, 7.4))
ax.set_xlim(0, 26)
ax.set_ylim(-0.6, 14.3)
ax.axis("off")


def box(x, y, w, h, fc, ec=INK, lw=1.0, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(x0, y0, x1, y1, color=INK, ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls, shrinkA=0, shrinkB=0)).set_zorder(1.5)


CHIP_H = 0.62


def stage(x, y, chips, sige):
    """chips: list of (label, kind) with kind in conv/attn. Returns (x, y, w, h) of the stage box."""
    widths = [0.3 + 0.158 * len(lbl.split("\n")[0]) for lbl, _ in chips]
    w = sum(widths) + 0.18 * (len(chips) + 1)
    h = CHIP_H + 0.36
    box(x, y, w, h, "#ffffff", lw=0.8)
    cx = x + 0.18
    for (lbl, kind), cw in zip(chips, widths):
        fc = (CONV if kind == "conv" else ATTN) if sige else PLAIN
        box(cx, y + 0.18, cw, CHIP_H, fc, lw=0.8, z=3)
        ax.text(cx + cw / 2, y + 0.18 + CHIP_H / 2, lbl, ha="center", va="center", fontsize=8.3, zorder=4)
        cx += cw + 0.18
    return x, y, w, h


ROWS = {64: 9.55, 32: 8.15, 16: 6.75, 8: 5.35}
MID_Y = 3.75


# ---- build each U-Net explicitly so ups can be right-aligned ---------------------------------------------------
def draw_unet(x0, title, down, mid, up, sige, note_rows):
    box(x0, 2.55, 12.3, 9.55, PANEL, lw=1.0, z=0)
    ax.text(x0 + 6.15, 11.65, title, ha="center", va="center", fontsize=12, weight="bold", color=INK)
    ax.text(x0 + 0.35, 11.1, "Encoder", ha="left", va="center", fontsize=8.5, color="#555")
    ax.text(x0 + 11.95, 11.1, "Decoder", ha="right", va="center", fontsize=8.5, color="#555")
    dboxes = {}
    for res, chips in down:
        dboxes[res] = stage(x0 + 0.35, ROWS[res], chips, sige)
    uboxes = {}
    for res, chips in up:
        widths = [0.3 + 0.158 * len(l.split("\n")[0]) for l, _ in chips]
        w = sum(widths) + 0.18 * (len(chips) + 1)
        uboxes[res] = stage(x0 + 11.95 - w, ROWS[res], chips, sige)
    widths = [0.3 + 0.158 * len(l.split("\n")[0]) for l, _ in mid]
    w = sum(widths) + 0.18 * (len(mid) + 1)
    mbox = stage(x0 + 6.15 - w / 2, MID_Y, mid, sige)
    ax.text(x0 + 6.15, MID_Y - 0.3, "Middle block", ha="center", va="center", fontsize=8.5, color="#555")
    # level labels in the centre gutter
    for res in dboxes:
        ax.text(x0 + 6.15, ROWS[res] + 0.49, f"{res}×{res}", ha="center", va="center", fontsize=9, color=INK, zorder=4,
                bbox=dict(fc=PANEL, ec="none", pad=2.5))
    # arrows down the encoder, up the decoder
    dres = sorted(dboxes, reverse=True)
    for a, b in zip(dres, dres[1:]):
        xa = dboxes[a][0] + 0.6
        arrow(xa, ROWS[a], xa, ROWS[b] + dboxes[b][3])
    last = dboxes[dres[-1]]
    arrow(last[0] + 0.6, last[1], mbox[0], MID_Y + mbox[3] / 2)
    ures = sorted(uboxes)
    first = uboxes[ures[0]]
    fx = first[0] + first[2] - 0.6
    arrow(mbox[0] + mbox[2], MID_Y + mbox[3] / 2, fx, first[1])
    for a, b in zip(ures, ures[1:]):
        xb = uboxes[b][0] + uboxes[b][2] - 0.6
        xa = uboxes[a][0] + uboxes[a][2] - 0.6
        arrow(xa, ROWS[a] + uboxes[a][3], xb, ROWS[b])
    # skip connections
    for res in dboxes:
        if res in uboxes:
            d, u = dboxes[res], uboxes[res]
            y = ROWS[res] + d[3] / 2
            ax.plot([d[0] + d[2], u[0]], [y, y], color="#8aa", lw=0.9, ls=(0, (3, 3)), zorder=1.5)
    for res, txt in note_rows:
        ax.text(x0 + 6.15, ROWS[res] + 0.05 if res in ROWS else res, txt, ha="center", va="center",
                fontsize=8.2, color=TEAL_TXT, style="italic")
    return dboxes, uboxes, mbox


# Stable Diffusion 1.x (reference, what the SIGE paper sparsified)
draw_unet(
    0.3, "Stable Diffusion 1.x U-Net (SIGE paper)",
    down=[(64, [("2× Res", "conv"), ("2× Attn", "attn"), ("Down", "conv")]),
          (32, [("2× Res", "conv"), ("2× Attn", "attn"), ("Down", "conv")]),
          (16, [("2× Res", "conv"), ("2× Attn", "attn"), ("Down", "conv")]),
          (8, [("2× Res", "conv")])],
    mid=[("Res", "conv"), ("Attn", "attn"), ("Res", "conv")],
    up=[(8, [("3× Res", "conv"), ("Up", "conv")]),
        (16, [("3× Res", "conv"), ("3× Attn", "attn"), ("Up", "conv")]),
        (32, [("3× Res", "conv"), ("3× Attn", "attn"), ("Up", "conv")]),
        (64, [("3× Res", "conv"), ("3× Attn", "attn")])],
    sige=False,
    note_rows=[],
)

# SDXL with our SIGE layers
draw_unet(
    13.4, "SDXL U-Net (ours: SIGE injected)",
    down=[(64, [("2× Res", "conv"), ("Down", "conv")]),
          (32, [("2× Res", "conv"), ("2× Attn ×2", "attn"), ("Down", "conv")]),
          (16, [("2× Res", "conv"), ("2× Attn ×10", "attn")])],
    mid=[("Res", "conv"), ("Attn ×10", "attn"), ("Res", "conv")],
    up=[(16, [("3× Res", "conv"), ("3× Attn ×10", "attn"), ("Up", "conv")]),
        (32, [("3× Res", "conv"), ("3× Attn ×2", "attn"), ("Up", "conv")]),
        (64, [("3× Res", "conv")])],
    sige=True,
    note_rows=[],
)

# Callouts on SDXL differences
callouts = [
    (19.55, 9.36, "no attention at 64×64"),
    (19.55, 6.45, "10-layer transformers: 62% of all MACs"),
    (19.55, 5.55, "no 8×8 level"),
]
for x, y, t in callouts:
    ax.text(x, y, t, ha="center", va="center", fontsize=8.5, color=TEAL_TXT, style="italic")
ax.text(6.45, 3.05, "attention: 48% of MACs, spread over 64², 32², 16²", ha="center", fontsize=8.5,
        color=TEAL_TXT, style="italic")
ax.text(19.55, 3.05, "attention: 74% of MACs, mostly on a 16×16 grid", ha="center", fontsize=8.5,
        color=TEAL_TXT, style="italic")

# Conditioning row (top)
def cond(x, w, lines, new=False, fs=8.8):
    box(x, 12.55, w, 1.45, DENSE if not new else "#ffffff", lw=1.0, ls="--" if new else "-")
    ax.text(x + w / 2, 13.27, lines, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.4)

cond(0.3, 3.9, "VAE encoder / decoder\n(dense)")
cond(4.45, 8.15, "Text encoder: CLIP ViT-L (768-d)\nrun once per prompt (dense)")
cond(13.4, 3.9, "VAE encoder / decoder\n(dense, same design)")
cond(17.55, 8.15, "New in SDXL: two text encoders (CLIP ViT-L +\nOpenCLIP ViT-bigG, 2048-d) + size/crop conditioning",
     new=True, fs=8.2)
for xc in (6.45, 19.55):
    arrow(xc, 12.55, xc, 12.1, color="#888", ls="--")

# Legend
pos = [(0.3, 1.6), (9.3, 1.6), (18.3, 1.6), (0.3, 1.0), (9.3, 1.0)]
items = [(CONV, "-", "SIGE sparse conv (both SIGE configs)"),
         (ATTN, "-", "SIGE sparse attention (conv+attn config only)"),
         (PLAIN, "-", "SD 1.x reference layer"),
         (DENSE, "-", "Unchanged, runs dense"),
         ("#ffffff", "--", "Added in SDXL")]
for (fc, ls, lbl), (lx, ly) in zip(items, pos):
    box(lx, ly, 0.55, 0.4, fc, lw=0.9, ls=ls)
    ax.text(lx + 0.75, ly + 0.2, lbl, va="center", fontsize=8.8)
ax.text(0.3, -0.05, "Res = ResNet block (3×3 convs).  Attn ×N = transformer block with N layers (self-attn, cross-attn, MLP).  "
        "Down / Up = stride-2 conv / upsample + conv.\nSizes are latent resolutions for a 512×512 image.  "
        "Dashed lines are skip connections.", fontsize=8.3, color="#555", linespacing=1.5)

out = sys.argv[1]
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")

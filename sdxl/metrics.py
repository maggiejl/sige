import numpy as np
import torch
from PIL import Image

_lpips_model = None


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB")).astype(np.float64)
    y = np.asarray(b.convert("RGB")).astype(np.float64)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def lpips(a: Image.Image, b: Image.Image, device: str = "cuda") -> float:
    """LPIPS (AlexNet), as in the SIGE paper. Returns NaN if the ``lpips`` package is not installed."""
    global _lpips_model
    try:
        import lpips as lpips_lib
    except ImportError:
        return float("nan")
    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()

    def t(img):
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)

    with torch.no_grad():
        return float(_lpips_model(t(a), t(b)).item())

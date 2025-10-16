# style_sifid_gram_mps.py
# Style-SIFID (↓) using VGG16 Gram features at relu2_2/3_3/4_3.
# MPS-ready (Apple Silicon). PyTorch >= 2.1, torchvision >= 0.15.

import os, math, argparse, json
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ------------------ device helpers ------------------
def pick_device(dev_arg: str | None):
    if dev_arg: return torch.device(dev_arg)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def to32(x: torch.Tensor):  # MPS likes float32
    return x.float() if x.dtype != torch.float32 else x

# ------------------ VGG16 features ------------------
_VGG_LAYERS = {3:"relu1_2", 8:"relu2_2", 15:"relu3_3", 22:"relu4_3", 29:"relu5_3"}

class VGG16_Style(nn.Module):
    def __init__(self, layers=('relu2_2','relu3_3','relu4_3'), device='cpu'):
        super().__init__()
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        idxs = sorted([k for k,v in _VGG_LAYERS.items() if v in layers])
        self.blocks = nn.ModuleList()
        last = 0
        for i in idxs:
            self.blocks.append(base[last:i+1])
            last = i+1
        self.eval().to(device)
        for p in self.parameters(): p.requires_grad_(False)
        self.pre = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        self.device = device

    @torch.no_grad()
    def forward(self, imgs):
        if isinstance(imgs, list):
            x = torch.stack([self.pre(im.convert('RGB')) for im in imgs], 0)
        else:
            x = torch.clamp(imgs, 0, 1)
            x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485,0.456,0.406])[None,:,None,None]
            std  = torch.tensor([0.229,0.224,0.225])[None,:,None,None]
            x = (x - mean) / std
        x = to32(x).to(self.device)
        feats, h = [], x
        for b in self.blocks:
            h = b(h)
            feats.append(h)  # [B,C,H,W] float32 on device
        return feats

# ------------------ style features ------------------
def gram_features(feat, offdiag_only=True, eps=1e-6):
    B, C, H, W = feat.shape
    f = feat.view(B, C, -1) / (H*W)**0.5
    G = torch.bmm(f, f.transpose(1,2))
    if offdiag_only:
        G = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
    g = G.reshape(B, -1)
    # standardize across the batch (the K crops)
    mu = g.mean(dim=0, keepdim=True)
    sd = g.var(dim=0, unbiased=False, keepdim=True).add(eps).sqrt()
    return (g - mu) / sd


def style_vec(feats: List[torch.Tensor]):
    return torch.cat([gram_features(F) for F in feats], dim=1)

# ------------------ paired crops ------------------
def sample_matching_crops(imgA, imgB, K=100, min_frac=0.35, max_frac=0.8, seed=0):
    """
    Return K pairs of crops with matched *relative* positions/sizes.
    If images differ in size, boxes are mapped by relative coordinates so content aligns.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    Wa, Ha = imgA.size
    Wb, Hb = imgB.size
    A, B = [], []
    for _ in range(K):
        frac = rng.uniform(min_frac, max_frac)
        # square crop side as a fraction of the *shorter* side
        La = int(round(frac * min(Wa, Ha)))
        Lb = int(round(frac * min(Wb, Hb)))

        # draw xa,ya in the VALID range [0, Wa-La], [0, Ha-La]
        xa_max = max(0, Wa - La)
        ya_max = max(0, Ha - La)
        xa = rng.randint(0, xa_max + 1) if xa_max > 0 else 0
        ya = rng.randint(0, ya_max + 1) if ya_max > 0 else 0

        # map to image B by NORMALIZING with the same valid-range denominator
        xb_max = max(0, Wb - Lb)
        yb_max = max(0, Hb - Lb)
        # use relative coords in [0,1]
        xr = 0.0 if xa_max == 0 else xa / xa_max
        yr = 0.0 if ya_max == 0 else ya / ya_max
        xb = int(round(xr * xb_max))
        yb = int(round(yr * yb_max))

        A.append(imgA.crop((xa, ya, xa + La, ya + La)))
        B.append(imgB.crop((xb, yb, xb + Lb, yb + Lb)))
    return A, B


# ------------------ Gaussians & Fréchet ------------------
def fit_gaussian_diag(X: torch.Tensor, eps: float = 1e-6):
    # X: [K, D]
    mu = X.mean(dim=0)
    var = X.var(dim=0, unbiased=False) + eps
    return mu, var  # diagonal covariance as variances

def frechet_diag(mu1, var1, mu2, var2):
    diff = mu1 - mu2
    term_mean = diff.dot(diff)
    term_var = (var1 + var2 - 2.0 * (var1 * var2).sqrt()).sum()
    return term_mean + term_var

def sqrtm_psd(A: torch.Tensor, eps=1e-6):
    # If eigh is flaky on MPS, do the tiny eigendecomp on CPU and send back.
    on_mps = (A.device.type == "mps")
    B = A.detach().cpu() if on_mps else A
    vals, vecs = torch.linalg.eigh(B)
    vals = torch.clamp(vals, min=0)
    S = (vecs * torch.sqrt(vals + eps)) @ vecs.T
    return S.to(A.device)

def frechet(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean = sqrtm_psd(cov1 @ cov2)
    return diff.dot(diff) + torch.trace(cov1 + cov2 - 2 * covmean)

# ------------------ public API ------------------
@torch.no_grad()
def style_sifid_gram(style_ref: Image.Image, generated: Image.Image,
                     K=100, layers=('relu2_2','relu3_3','relu4_3'),
                     device: str | None = None, multiscale=1, seed=0) -> float:
    device = pick_device(device)
    enc = VGG16_Style(layers=layers, device=device)

    def down(im: Image.Image):
        return im.resize((max(1, im.width//2), max(1, im.height//2)), Image.BICUBIC)

    scores = []
    ref, gen = style_ref, generated
    for s in range(multiscale):
        Cr, Cg = sample_matching_crops(ref, gen, K=K, seed=seed+s)
        Fr, Fg = enc(Cr), enc(Cg)
        Xr, Xg = style_vec(Fr), style_vec(Fg)           # [K, D], float32 on device
        mu_r, var_r = fit_gaussian_diag(Xr); mu_g, var_g = fit_gaussian_diag(Xg)
        sifid = frechet_diag(mu_r, var_r, mu_g, var_g).item()
        scores.append(sifid)
        ref, gen = down(ref), down(gen)
    return float(np.median(scores))

# ------------------ CLI ------------------
def load_img(p): return Image.open(p).convert('RGB')

def run_pair(p_ref, p_gen, **kw):
    return style_sifid_gram(load_img(p_ref), load_img(p_gen), **kw)

def run_dirs(d_ref, d_gen, suffixes=(".png",".jpg",".jpeg"), **kw):
    ref = {os.path.splitext(f)[0]: os.path.join(d_ref,f)
           for f in os.listdir(d_ref) if os.path.splitext(f)[1].lower() in suffixes}
    out = {}
    for f in os.listdir(d_gen):
        name, ext = os.path.splitext(f)
        if ext.lower() in suffixes and name in ref:
            out[name] = run_pair(ref[name], os.path.join(d_gen,f), **kw)
    return out

def main():
    ap = argparse.ArgumentParser("Style-SIFID (Gram) with MPS support")
    ap.add_argument("--ref", required=True, help="style image or folder")
    ap.add_argument("--gen", required=True, help="generated image or folder")
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (auto if omitted)")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--multiscale", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kw = dict(K=args.K, device=args.device, multiscale=args.multiscale, seed=args.seed)

    if os.path.isdir(args.ref) and os.path.isdir(args.gen):
        scores = run_dirs(args.ref, args.gen, **kw)
        vals = list(scores.values())
        print(json.dumps({
            "device": str(pick_device(args.device)),
            "median": float(np.median(vals)) if vals else None,
            "iqr": float(np.percentile(vals,75)-np.percentile(vals,25)) if vals else None,
            "per_image": scores
        }, indent=2))
    else:
        s = run_pair(args.ref, args.gen, **kw)
        print(f"[{pick_device(args.device)}] Style-SIFID (Gram) ↓ = {s:.3f}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Helpful on macOS for ops not yet implemented on MPS:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()

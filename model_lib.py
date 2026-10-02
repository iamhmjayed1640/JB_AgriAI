#!/usr/bin/env python3
"""
cdfsl_runner.py
V71: Cosine Training + Learnable Temperature (NO aggressive augmentation).
Title: Attention-Enhanced Prototypical Networks for Explainable Few-Shot Rice Leaf Disease Classification.

Modes
  dryrun      Resolve datasets, match class folders, print counts, write dataset_manifest.json.
              No image decoding, no GPU. Fails loudly on any missing class.
  train       Train (supervised pretraining, then episodic) and evaluate the main protocol.
              One run = (variant, mask, img, seed). Resumable: finished runs are skipped.
  eval        Re-evaluate saved checkpoints under other classifiers (E4), shots (E5)
              and UCI counterfactual inputs (E7). No training.
  cam         Quantitative Grad-CAM statistics and class-balanced panels (E6). No training.
  efficiency  Parameters, analytic MACs, latency (fp32 and AMP), peak memory (E8-E10 cost).

Design decisions that differ from, or make explicit, what the lost original script did
  * All episode manifests are keyed on (target, seed[, shot]) only, never on the condition,
    and masked and unmasked datasets share one file order. So every condition at a seed
    sees the same images in the same episodes, masked or not.
  * Backbone weights are created under torch.manual_seed(seed) before any attention module;
    attention modules are created under a forked RNG. Backbone initialisation is therefore
    identical across all variants at a seed (the original Block 4 shortcut differed).
  * Supervised pretraining is re-run for every (condition, seed). Logged in each run.
  * No DataLoader workers; all randomness comes from seeded generators.
  * Unreadable images raise an error instead of being replaced silently.
  * Every output JSON records the SHA-1 of this script, the class lists and file-list hashes.

Nothing here tunes anything on target data. All hyperparameters are fixed in DEFAULTS.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Configuration. Paths are the ones recorded in dataset_inventory.json / mask_coverage.json.
# --------------------------------------------------------------------------------------
DEFAULTS = {
    "src_root": "/kaggle/input/datasets/anshulm257/rice-disease-dataset/Rice_Leaf_AUG",
    "uci_root": "/kaggle/input/datasets/vbookshelf/rice-leaf-diseases/rice_leaf_diseases",
    "paddy_root": ("/kaggle/input/datasets/imbikramsaha/paddy-doctor/"
                   "paddy-disease-classification/train_images"),
    # Inferred by reproduction (reproduction_class_config.json). Override with --src_classes
    # if the executed notebook is recovered and shows otherwise.
    "src_classes": ["Leaf Blast", "Leaf scald", "Sheath Blight", "Healthy Rice Leaf"],
    "uci_classes": ["Bacterial leaf blight", "Brown spot", "Leaf smut"],
    "paddy_classes": ["bacterial_leaf_blight", "blast", "brown_spot"],
    "img": 84, "n_way": 3, "k_shot": 5, "n_query": 10,
    "episodes_per_epoch": 100, "epochs": 20, "lr": 1e-4,
    "pretrain_epochs": 12, "pretrain_lr": 1e-3, "pretrain_bs": 128,
    "eval_episodes": 300, "n_groups": 4,
    "seeds": [42, 1024, 3407],
    # Fixed a priori for E4 baselines. Not tuned on any target. If a stronger fine-tuning
    # baseline is wanted, choose finetune_lr_block on held-out SOURCE episodes only.
    "linear_head_steps": 100, "linear_head_lr": 1e-2, "linear_head_wd": 1e-3,
    "finetune_steps": 25, "finetune_lr_block": 1e-3,
    "cam_episodes": 20,
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CANONICAL = {("plain", False): "C1_Plain", ("cbam", False): "C2_CBAM",
             ("plain", True): "C3_Plain_Masked", ("cbam", True): "C4_CBAM_Masked"}


def script_sha1():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------
# Image operations. apply_hsv_mask, letterbox and leaf_proxy are exact copies of the
# versions used by the pipeline and by 02_mask_coverage_audit.py.
# --------------------------------------------------------------------------------------
def apply_hsv_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    mask_non_green = cv2.bitwise_not(mask_green)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask_non_green, thresh)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=final_mask)


def letterbox(img, size, fill=255):
    h, w = img.shape[:2]
    sc = size / max(h, w)
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    top = (size - nh) // 2
    left = (size - nw) // 2
    val = fill if img.ndim == 2 else (fill, fill, fill)
    out = cv2.copyMakeBorder(r, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=val)
    return out, (top, left, nh, nw)


def leaf_proxy(img_bgr):
    """Crude foreground proxy (green OR far from modal border colour). UCI only."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
    border = np.concatenate([img_bgr[0, :, :], img_bgr[-1, :, :],
                             img_bgr[:, 0, :], img_bgr[:, -1, :]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg[None, None, :], axis=2)
    notbg = (dist > 60).astype(np.uint8) * 255
    fg = cv2.bitwise_or(green, notbg)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return fg > 0, bg


def counterfactual(img_bgr, kind):
    """
    E7 test-time interventions, UCI only. The leaf proxy is dilated by ~1% of the short
    side so that 'bg_white' never trims leaf and 'bg_only' removes the whole leaf.
      bg_white  : pixels outside the leaf set to pure white, i.e. the letterbox fill colour.
                  Removes background appearance and the photo-rectangle geometry.
      bg_only   : leaf pixels replaced by the modal border colour. No leaf content remains.
      geom_only : whole photo replaced by the modal border colour. Only the image
                  rectangle (aspect ratio / padding geometry) remains.
    """
    if kind == "none":
        return img_bgr, None
    fg, bg = leaf_proxy(img_bgr)
    h, w = fg.shape
    d = max(3, int(round(0.01 * min(h, w))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    fg = cv2.dilate(fg.astype(np.uint8), k) > 0
    out = img_bgr.copy()
    bgc = np.clip(np.round(bg), 0, 255).astype(np.uint8)
    if kind == "bg_white":
        out[~fg] = 255
    elif kind == "bg_only":
        out[fg] = bgc
    elif kind == "geom_only":
        out[:] = bgc
    else:
        raise ValueError(f"unknown counterfactual '{kind}'")
    return out, float(fg.mean())


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
def list_files(root, classes):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset root not found: {root}")
    present = sorted(os.listdir(root))
    missing = [c for c in classes if c not in present]
    if missing:
        raise FileNotFoundError(f"class folders {missing} not found under {root}. "
                                f"Present: {present}")
    files, labels = [], []
    for i, c in enumerate(classes):
        fs = sorted(f for f in os.listdir(os.path.join(root, c)) if f.endswith(IMG_EXT))
        if not fs:
            raise RuntimeError(f"class '{c}' under {root} contains no images")
        files += [os.path.join(root, c, f) for f in fs]
        labels += [i] * len(fs)
    rel = "\n".join(os.path.relpath(f, root) for f in files)
    return files, np.array(labels), hashlib.sha1(rel.encode()).hexdigest()


class ImageSet:
    """Decodes once, caches uint8 RGB letterboxed images on the compute device."""

    def __init__(self, name, root, classes, size, mask, cf, device, cache_dir=None,
                 need_leaf=False):
        if cf != "none" and mask:
            raise ValueError("counterfactuals are defined for unmasked inputs only")
        self.name, self.classes, self.size = name, classes, size
        self.files, self.y, self.file_hash = list_files(root, classes)
        key = hashlib.sha1(f"{root}|{classes}|{size}|{mask}|{cf}|{need_leaf}|"
                           f"{self.file_hash}".encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"{name}_{key}.npz") if cache_dir else None
        if path and os.path.exists(path):
            d = np.load(path)
            arr, boxes = d["x"], d["boxes"]
            leaf = d["leaf"] if need_leaf else None
            self.cf_fg = d["cf_fg"].tolist()
        else:
            arr, boxes, leafs, bad, self.cf_fg = [], [], [], [], []
            for f in self.files:
                img = cv2.imread(f)
                if img is None:
                    bad.append(f)
                    continue
                if need_leaf:
                    lf, _ = leaf_proxy(img)
                    leafs.append(letterbox(lf.astype(np.uint8) * 255, size, fill=0)[0] > 0)
                if mask:
                    img = apply_hsv_mask(img)
                img, fgfrac = counterfactual(img, cf)
                self.cf_fg.append(-1.0 if fgfrac is None else fgfrac)
                lb, box = letterbox(img, size)
                arr.append(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))
                boxes.append(box)
            if bad:
                raise RuntimeError(f"{len(bad)} unreadable images in {name}, e.g. {bad[:3]}")
            arr, boxes = np.stack(arr), np.array(boxes)
            leaf = np.stack(leafs) if need_leaf else None
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(path, x=arr, boxes=boxes, cf_fg=np.array(self.cf_fg),
                         **({"leaf": leaf} if need_leaf else {}))
        self.boxes = boxes
        self.leaf = leaf
        self.device = device
        self.x = torch.from_numpy(arr).to(device)            # N,H,W,3 uint8
        self.n = len(self.y)

    def batch(self, idx, flip_gen=None):
        x = self.x[torch.as_tensor(idx, device=self.device)]
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if flip_gen is not None:
            b = x.shape[0]
            fh = (torch.rand(b, generator=flip_gen) < 0.5).to(self.device)
            fv = (torch.rand(b, generator=flip_gen) < 0.5).to(self.device)
            x = torch.where(fh[:, None, None, None], x.flip(3), x)
            x = torch.where(fv[:, None, None, None], x.flip(2), x)
        return (x - MEAN.to(self.device)) / STD.to(self.device)


def gen_episodes(labels, n_episodes, key, cfg, k_shot=None):
    k = cfg["k_shot"] if k_shot is None else k_shot
    rng = random.Random(key)
    by_cls = {}
    for i, l in enumerate(labels):
        by_cls.setdefault(int(l), []).append(i)
    usable = sorted(c for c, v in by_cls.items() if len(v) >= k + cfg["n_query"])
    if len(usable) < cfg["n_way"]:
        raise ValueError(f"only {len(usable)} classes have >= {k + cfg['n_query']} images "
                         f"for key '{key}'")
    eps = []
    for _ in range(n_episodes):
        cls = rng.sample(usable, cfg["n_way"])
        sup, qry = [], []
        for c in cls:
            pick = rng.sample(by_cls[c], k + cfg["n_query"])
            sup += pick[:k]
            qry += pick[k:]
        eps.append((cls, sup, qry))
    return eps


def eval_key(target, seed, k, cfg):
    # k = the main shot count reuses the main manifest, so E5 at 5-shot equals E3.
    return f"{target}|{seed}" if k == cfg["k_shot"] else f"{target}|{seed}|k{k}"


# --------------------------------------------------------------------------------------
# Model. Block structure and attention modules as in 04_efficiency_benchmark.py.
# --------------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, c, ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(c // ratio, c, 1, bias=False)

    def forward(self, x):
        a = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        m = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        return torch.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return torch.sigmoid(self.conv1(s))


class AttnModule(nn.Module):
    """kind: full (CBAM), ch (channel only), sp (spatial only), res (zero-init residual CBAM)."""

    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        if kind in ("full", "ch", "res"):
            self.ca = ChannelAttention(c)
        if kind in ("full", "sp", "res"):
            self.sa = SpatialAttention()
        if kind == "res":
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = x
        if hasattr(self, "ca"):
            y = y * self.ca(y)
        if hasattr(self, "sa"):
            y = y * self.sa(y)
        if self.kind == "res":
            return x + self.gamma * (y - x)
        return y


class ResBlock(nn.Module):
    def __init__(self, cin, cout, g):
        super().__init__()
        self.planes = cout
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, cout)
        self.conv3 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(g, cout)
        self.shortcut = (nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(g, cout))
                         if cin != cout else nn.Identity())
        self.register_module("attn", None)

    def forward(self, x):
        o = F.leaky_relu(self.gn1(self.conv1(x)), 0.1)
        o = F.leaky_relu(self.gn2(self.conv2(o)), 0.1)
        o = self.gn3(self.conv3(o))
        if self.attn is not None:
            o = self.attn(o)
        return F.max_pool2d(F.leaky_relu(o + self.shortcut(x), 0.1), 2)


class Encoder(nn.Module):
    def __init__(self, g=4):
        super().__init__()
        self.block1 = ResBlock(3, 64, g)
        self.block2 = ResBlock(64, 128, g)
        self.block3 = ResBlock(128, 256, g)
        self.block4 = ResBlock(256, 512, g)

    def features3(self, x):
        return self.block3(self.block2(self.block1(x)))

    def features(self, x):
        return self.block4(self.features3(x))

    def forward(self, x):
        return self.features(x).mean((2, 3))


VARIANTS = {"plain": None, "cbam": ("block4", "full"), "cbam_ch": ("block4", "ch"),
            "cbam_sp": ("block4", "sp"), "cbam_b3": ("block3", "full"),
            "cbam_res": ("block4", "res")}


def build_encoder(variant, seed, g=4):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {list(VARIANTS)}")
    torch.manual_seed(seed)
    enc = Encoder(g)
    spec = VARIANTS[variant]
    if spec:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 99991)
            blk = getattr(enc, spec[0])
            blk.attn = AttnModule(blk.planes, spec[1])
    return enc


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def proto_logits(zs, zq, n_way, k):
    p = zs.float().view(n_way, k, -1).mean(1)
    return -torch.cdist(zq.float(), p) ** 2


def cosine_proto_logits(zs, zq, n_way, k, temp=10.0):
    """Cosine similarity proto logits — aligns training with eval metric."""
    mu = zs.mean(0, keepdim=True)
    sn = F.normalize(zs - mu, dim=-1)
    qn = F.normalize(zq - mu, dim=-1)
    P = F.normalize(sn.view(n_way, k, -1).mean(1), dim=-1)
    return torch.mm(qn, P.t()) * temp


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=(amp and device.type == "cuda"))


def pretrain(enc, src, seed, cfg, device, amp, log):
    torch.manual_seed(seed + 1000)
    head = nn.Linear(512, len(src.classes)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                           lr=cfg["pretrain_lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=(amp and device.type == "cuda"))
    order = np.random.default_rng(seed + 2000)
    flip = torch.Generator().manual_seed(seed + 3000)
    yall = torch.as_tensor(src.y, device=device)
    enc.train()
    for ep in range(cfg["pretrain_epochs"]):
        perm, tot, nb = order.permutation(src.n), 0.0, 0
        for i in range(0, src.n, cfg["pretrain_bs"]):
            idx = perm[i:i + cfg["pretrain_bs"]]
            x = src.batch(idx, flip)
            opt.zero_grad(set_to_none=True)
            with autocast(device, amp):
                loss = F.cross_entropy(head(enc(x)).float(), yall[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, nb = tot + loss.item(), nb + 1
        log["pretrain_loss"].append(round(tot / nb, 5))
    return enc


def episodic_train(enc, src, seed, cfg, device, amp, log):
    # V70: Learnable temperature for cosine similarity (Strategy 4)
    temp = torch.nn.Parameter(torch.tensor(10.0, device=device))
    opt = torch.optim.Adam(list(enc.parameters()) + [temp], lr=cfg["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=(amp and device.type == "cuda"))
    flip = torch.Generator().manual_seed(seed + 4000)
    nw, k = cfg["n_way"], cfg["k_shot"]
    yq = torch.arange(nw, device=device).repeat_interleave(cfg["n_query"])
    enc.train()
    for ep in range(cfg["epochs"]):
        eps = gen_episodes(src.y, cfg["episodes_per_epoch"], f"train|{seed}|{ep}", cfg)
        tot = 0.0
        for _, sup, qry in eps:
            x = src.batch(sup + qry, flip)
            opt.zero_grad(set_to_none=True)
            with autocast(device, amp):
                z = enc(x)
            # V71: Train with COSINE loss (Strategy 1) + learned temp (Strategy 4)
            loss = F.cross_entropy(cosine_proto_logits(z[:len(sup)], z[len(sup):], nw, k, temp), yq)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sch.step()
        log["episodic_loss"].append(round(tot / len(eps), 5))
    log["learned_temperature"] = round(float(temp.item()), 4)
    return enc


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def embed_all(enc, ds, bs=256, level="final"):
    enc.eval()
    out = []
    for i in range(0, ds.n, bs):
        x = ds.batch(np.arange(i, min(i + bs, ds.n)))
        out.append((enc(x) if level == "final" else enc.features3(x)).float())
    return torch.cat(out)


def _confusion(cls, preds, qy, n_cls):
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(qy, preds):
        cm[cls[t], cls[p]] += 1
    return cm


def evaluate(enc, ds, seed, target, cfg, classifier="proto_euclid", k=None, device=None):
    """Returns mean, episode SD, per-episode accuracy list and confusion matrix."""
    k = cfg["k_shot"] if k is None else k
    nw, nq = cfg["n_way"], cfg["n_query"]
    eps = gen_episodes(ds.y, cfg["eval_episodes"], eval_key(target, seed, k, cfg), cfg, k)
    qy_pos = np.repeat(np.arange(nw), nq)
    cm = np.zeros((len(ds.classes),) * 2, int)
    accs, extra = [], {}
    if classifier in ("proto_euclid", "proto_cosine", "proto_cosine_transductive", "linear_head"):
        Z = embed_all(enc, ds)
        S = torch.stack([Z[torch.as_tensor(s, device=Z.device)] for _, s, _ in eps])
        Q = torch.stack([Z[torch.as_tensor(q, device=Z.device)] for _, _, q in eps])
        E = len(eps)
        # Cosine and linear heads use inductive centring: subtract the SUPPORT mean (no query
        # information), then L2-normalise. Without it, GAP embeddings are nearly collinear
        # (pairwise cosine ~0.9999 observed) and both heads collapse towards chance for reasons
        # of feature geometry, which would make them unfairly weak baselines. Euclidean
        # ProtoNet is translation invariant and is left unchanged.
        mu = S.mean(1, keepdim=True)
        if classifier == "proto_euclid":
            P = S.view(E, nw, k, -1).mean(2)
            logits = -torch.cdist(Q, P) ** 2
        elif classifier == "proto_cosine":
            Sn, Qn = F.normalize(S - mu, dim=-1), F.normalize(Q - mu, dim=-1)
            P = F.normalize(Sn.view(E, nw, k, -1).mean(2), dim=-1)
            logits = torch.bmm(Qn, P.transpose(1, 2))
        elif classifier == "proto_cosine_transductive":
            Sn, Qn = F.normalize(S - mu, dim=-1), F.normalize(Q - mu, dim=-1)
            P = F.normalize(Sn.view(E, nw, k, -1).mean(2), dim=-1)
            # Soft-assignment transductive prototype refinement
            temp = 10.0
            for _ in range(3):
                logits = torch.bmm(Qn, P.transpose(1, 2)) * temp
                probs = F.softmax(logits, dim=-1)
                P_new = []
                for e_idx in range(E):
                    p_e = []
                    for c_idx in range(nw):
                        s_c = Sn[e_idx].view(nw, k, -1)[c_idx]
                        q_w = probs[e_idx, :, c_idx].unsqueeze(1)
                        q_sum = (Qn[e_idx] * q_w).sum(0)
                        c_new = (s_c.sum(0) + q_sum) / (k + q_w.sum() + 1e-6)
                        p_e.append(c_new)
                    P_new.append(torch.stack(p_e))
                P = F.normalize(torch.stack(P_new), p=2, dim=-1)
            logits = torch.bmm(Qn, P.transpose(1, 2)) * temp
        else:  # linear head on frozen centred, L2-normalised features, one head per episode
            Sn, Qn = F.normalize(S - mu, dim=-1), F.normalize(Q - mu, dim=-1)
            ys = torch.arange(nw, device=Z.device).repeat_interleave(k).repeat(E)
            with torch.enable_grad():
                W = torch.zeros(E, nw, Z.shape[1], device=Z.device, requires_grad=True)
                b = torch.zeros(E, nw, device=Z.device, requires_grad=True)
                opt = torch.optim.Adam([W, b], lr=cfg["linear_head_lr"],
                                       weight_decay=cfg["linear_head_wd"])
                for _ in range(cfg["linear_head_steps"]):
                    opt.zero_grad()
                    lg = torch.bmm(Sn, W.transpose(1, 2)) + b[:, None, :]
                    F.cross_entropy(lg.reshape(-1, nw), ys, reduction="sum").backward()
                    opt.step()
            logits = torch.bmm(Qn, W.detach().transpose(1, 2)) + b.detach()[:, None, :]
        pred = logits.argmax(-1).cpu().numpy()
        for e, (cls, _, _) in enumerate(eps):
            accs.append(float((pred[e] == qy_pos).mean() * 100))
            cm += _confusion(cls, pred[e], qy_pos, len(ds.classes))
    elif classifier == "finetune_b4":
        # Block-4 fine-tuning on the support set with the training objective itself: at each
        # step prototypes are recomputed from the current support embeddings and the ProtoNet
        # loss is applied to the support images. No new head, no temperature; with
        # finetune_steps = 0 this is exactly proto_euclid (checked in selftest).
        # SGD, not Adam: the squared-distance loss is saturated for well-separated supports, and
        # Adam rescales those near-zero gradients into full-size steps (observed collapse).
        H3 = embed_all(enc, ds, level="block3")
        base = copy.deepcopy(enc.block4).to(H3.device)
        ys = torch.arange(nw, device=H3.device).repeat_interleave(k)
        sup_before, sup_after = [], []
        for cls, sup, qry in eps:
            blk = copy.deepcopy(base).train()
            hs = H3[torch.as_tensor(sup, device=H3.device)]

            def support_logits():
                zs = blk(hs).mean((2, 3))
                return zs, -torch.cdist(zs, zs.view(nw, k, -1).mean(1)) ** 2
            with torch.no_grad():
                sup_before.append(float((support_logits()[1].argmax(1) == ys).float().mean()))
            opt = torch.optim.SGD(blk.parameters(), lr=cfg["finetune_lr_block"], momentum=0.9)
            with torch.enable_grad():
                for _ in range(cfg["finetune_steps"]):
                    opt.zero_grad()
                    F.cross_entropy(support_logits()[1], ys).backward()
                    opt.step()
            blk.eval()
            with torch.no_grad():
                zs, lg = support_logits()
                sup_after.append(float((lg.argmax(1) == ys).float().mean()))
                P = zs.view(nw, k, -1).mean(1)
                zq = blk(H3[torch.as_tensor(qry, device=H3.device)]).mean((2, 3))
                pr = (-torch.cdist(zq, P) ** 2).argmax(1).cpu().numpy()
            accs.append(float((pr == qy_pos).mean() * 100))
            cm += _confusion(cls, pr, qy_pos, len(ds.classes))
        extra = {"support_acc_before": round(float(np.mean(sup_before)) * 100, 2),
                 "support_acc_after": round(float(np.mean(sup_after)) * 100, 2)}
    else:
        raise ValueError(f"unknown classifier {classifier}")
    return {"classifier": classifier, "k_shot": k, "mean": round(float(np.mean(accs)), 4),
            "episode_sd": round(float(np.std(accs, ddof=1)), 4), "n_episodes": len(accs),
            "episode_acc": [round(a, 3) for a in accs], "confusion": cm.tolist(),
            "classes": ds.classes, **extra}


# --------------------------------------------------------------------------------------
# Grad-CAM (E6). For a GAP embedding and s_c = -||z - p_c||^2, the Grad-CAM channel weight
# is exactly -2 (z_k - p_ck) / (h w); the map is computed in closed form and checked
# against autograd in --selftest.
# --------------------------------------------------------------------------------------
def cam_closed_form(A, protos, cls_idx):
    h, w = A.shape[2:]
    z = A.mean((2, 3))
    wts = -2.0 * (z - protos[cls_idx]) / (h * w)
    return F.relu((wts[:, :, None, None] * A).sum(1))


def cam_autograd(enc, x, protos, cls_idx):
    A = enc.features(x)
    A.retain_grad()
    z = A.mean((2, 3))
    s = -((z - protos[cls_idx]) ** 2).sum(1)
    s.sum().backward()
    wts = A.grad.mean((2, 3))
    return F.relu((wts[:, :, None, None] * A).sum(1)).detach()


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def run_name(variant, mask, img):
    n = CANONICAL.get((variant, mask), f"{variant}{'_masked' if mask else ''}")
    return n if img == 84 else f"{n}_img{img}"


def env_info(device):
    return {"torch": torch.__version__, "python": platform.python_version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "script_sha1": script_sha1(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime())}


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class Data:
    """Lazily builds and memoises datasets for one process."""

    def __init__(self, cfg, device, cache_dir):
        self.cfg, self.device, self.cache_dir, self.store = cfg, device, cache_dir, {}

    def get(self, name, mask=False, img=84, cf="none", need_leaf=False):
        key = (name, mask, img, cf, need_leaf)
        if key not in self.store:
            root = self.cfg[f"{name}_root"]
            cls = self.cfg[f"{name}_classes"]
            t0 = time.time()
            self.store[key] = ImageSet(name, root, cls, img, mask, cf, self.device,
                                       self.cache_dir, need_leaf)
            print(f"  cached {name} mask={mask} img={img} cf={cf}: "
                  f"{self.store[key].n} images [{time.time() - t0:.0f}s]")
        return self.store[key]


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------
def mode_dryrun(cfg, out):
    man = {}
    for name in ("src", "uci", "paddy"):
        files, y, h = list_files(cfg[f"{name}_root"], cfg[f"{name}_classes"])
        counts = {c: int((y == i).sum()) for i, c in enumerate(cfg[f"{name}_classes"])}
        man[name] = {"root": cfg[f"{name}_root"], "classes": cfg[f"{name}_classes"],
                     "counts": counts, "n": len(files), "file_list_sha1": h}
        print(f"{name:6s} {len(files):5d} images  {counts}")
    need = cfg["k_shot"] + cfg["n_query"]
    for name in ("uci", "paddy"):
        small = [c for c, n in man[name]["counts"].items() if n < max(need, 10 + 10)]
        if small:
            print(f"  WARNING {name}: classes {small} too small for 10-shot E5")
    man["env"] = env_info(torch.device("cpu"))
    dump(os.path.join(out, "dataset_manifest.json"), man)
    print(f"wrote {os.path.join(out, 'dataset_manifest.json')}")


def mode_train(cfg, args, device):
    data = Data(cfg, device, args.cache_dir)
    masks = {"off": [False], "on": [True], "both": [False, True]}[args.mask]
    for mask in masks:
        for variant in args.variants:
            for seed in args.seeds:
                name = run_name(variant, mask, args.img)
                rdir = os.path.join(args.out, name, f"seed{seed}")
                done = os.path.join(rdir, "eval_main.json")
                if os.path.exists(done) and not args.force:
                    print(f"skip {name} seed {seed} (done)")
                    continue
                print(f"=== {name} seed {seed}")
                t0 = time.time()
                src = data.get("src", mask, args.img)
                set_seed(seed)
                enc = build_encoder(variant, seed, cfg["n_groups"]).to(device)
                log = {"pretrain_loss": [], "episodic_loss": []}
                enc = pretrain(enc, src, seed, cfg, device, args.amp, log)
                t_pre = time.time() - t0
                torch.save(enc.state_dict(), os.path.join(_mk(rdir), "ckpt_pretrain.pt"))
                enc = episodic_train(enc, src, seed, cfg, device, args.amp, log)
                torch.save(enc.state_dict(), os.path.join(rdir, "ckpt_episodic.pt"))
                t_train = time.time() - t0
                res = {"run": name, "variant": variant, "mask": mask, "img": args.img,
                       "seed": seed, "cfg": cfg, "amp": args.amp, "env": env_info(device),
                       "params": n_params(enc), "pretrain_per_seed": True,
                       "src_file_list_sha1": src.file_hash,
                       "src_counts": {c: int((src.y == i).sum()) for i, c in
                                      enumerate(src.classes)},
                       "log": log, "minutes_pretrain": round(t_pre / 60, 2),
                       "minutes_train": round(t_train / 60, 2), "targets": {}}
                if VARIANTS[variant] and VARIANTS[variant][1] == "res":
                    res["gamma_final"] = float(getattr(enc, VARIANTS[variant][0]).attn.gamma)
                for tgt in ("uci", "paddy"):
                    ds = data.get(tgt, mask, args.img)
                    r = evaluate(enc, ds, seed, tgt, cfg)
                    r["file_list_sha1"] = ds.file_hash
                    res["targets"][tgt] = r
                    print(f"  {tgt:5s} {r['mean']:.2f}  (episode SD {r['episode_sd']:.2f})")
                res["minutes_total"] = round((time.time() - t0) / 60, 2)
                dump(done, res)


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


def _runs(args):
    for name in sorted(os.listdir(args.out)):
        if not os.path.isdir(os.path.join(args.out, name)):
            continue   # dataset_manifest.json, efficiency_img*.json live here too
        for sd in sorted(os.listdir(os.path.join(args.out, name))):
            p = os.path.join(args.out, name, sd, "eval_main.json")
            if os.path.exists(p):
                with open(p) as f:
                    meta = json.load(f)
                if args.runs and name not in args.runs:
                    continue
                if args.seeds and meta["seed"] not in args.seeds:
                    continue
                yield os.path.join(args.out, name, sd), meta


def load_encoder(meta, rdir, stage, device):
    enc = build_encoder(meta["variant"], meta["seed"], meta["cfg"]["n_groups"])
    enc.load_state_dict(torch.load(os.path.join(rdir, f"ckpt_{stage}.pt"), map_location="cpu"))
    return enc.to(device).eval()


def mode_eval(cfg, args, device):
    data = Data(cfg, device, args.cache_dir)
    for rdir, meta in _runs(args):
        for stage in args.stages:
            enc = load_encoder(meta, rdir, stage, device)
            for cf in args.counterfactuals:
                if cf != "none" and meta["mask"]:
                    continue
                for tgt in args.targets:
                    if cf != "none" and tgt != "uci":
                        continue   # counterfactuals are only defined for UCI
                    ds = data.get(tgt, meta["mask"], meta["img"], cf)
                    for clf in args.classifiers:
                        for k in args.shots:
                            tag = f"{stage}_{clf}_k{k}_{cf}_{tgt}"
                            p = os.path.join(rdir, "eval", f"{tag}.json")
                            if os.path.exists(p) and not args.force:
                                continue
                            r = evaluate(enc, ds, meta["seed"], tgt, meta["cfg"], clf, k)
                            r.update({"stage": stage, "counterfactual": cf, "target": tgt,
                                      "run": meta["run"], "variant": meta["variant"],
                                      "mask": meta["mask"], "img": meta["img"],
                                      "seed": meta["seed"], "env": env_info(device)})
                            if cf != "none":
                                fg = np.array(ds.cf_fg)
                                r["cf_leaf_fraction_mean"] = round(float(fg.mean()), 4)
                                r["cf_leaf_fraction_lt2pct"] = int((fg < 0.02).sum())
                                r["cf_leaf_fraction_gt90pct"] = int((fg > 0.90).sum())
                            dump(p, r)
                            print(f"{meta['run']:18s} s{meta['seed']} {tag:40s} {r['mean']:.2f}")


def mode_cam(cfg, args, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = Data(cfg, device, args.cache_dir)
    nw, nq = cfg["n_way"], cfg["n_query"]
    for rdir, meta in _runs(args):
        enc = load_encoder(meta, rdir, "episodic", device)
        out = {"run": meta["run"], "seed": meta["seed"], "env": env_info(device), "targets": {}}
        for tgt in args.targets:
            ds = data.get(tgt, meta["mask"], meta["img"], need_leaf=(tgt == "uci"))
            k = meta["cfg"]["k_shot"]
            eps = gen_episodes(ds.y, cfg["cam_episodes"], eval_key(tgt, meta["seed"], k, cfg),
                               cfg, k)
            S = ds.size
            rows, panel = [], []
            for e, (cls, sup, qry) in enumerate(eps):
                with torch.no_grad():
                    A_s = enc.features(ds.batch(sup)).float()
                    A_q = enc.features(ds.batch(qry)).float()
                P = A_s.mean((2, 3)).view(nw, k, -1).mean(1)
                logits = -torch.cdist(A_q.mean((2, 3)), P) ** 2
                pred = logits.argmax(1)
                true = torch.arange(nw, device=device).repeat_interleave(nq)
                for which, ci in (("pred", pred), ("true", true)):
                    cam = cam_closed_form(A_q, P, ci)
                    up = F.interpolate(cam[:, None], size=(S, S), mode="bilinear",
                                       align_corners=False)[:, 0].cpu().numpy()
                    for j, qi in enumerate(qry):
                        top, left, nh, nwid = ds.boxes[qi]
                        content = np.zeros((S, S), bool)
                        content[top:top + nh, left:left + nwid] = True
                        m = up[j]
                        tot = m.sum()
                        row = {"episode": e, "query": int(qi), "which": which,
                               "true_class": ds.classes[cls[int(true[j])]],
                               "correct": bool(pred[j] == true[j]),
                               "empty": bool(cam[j].max().item() <= 0),
                               "pad_area_frac": float(1 - content.mean())}
                        if tot > 0:
                            row["mass_pad"] = float(m[~content].sum() / tot)
                            yy, xx = np.unravel_index(int(m.argmax()), m.shape)
                            row["peak_in_pad"] = bool(not content[yy, xx])
                        if ds.leaf is not None and not meta["mask"]:
                            lf = ds.leaf[qi]
                            row["leaf_area_frac"] = float(lf.mean())
                            if tot > 0:
                                row["mass_leaf"] = float(m[lf].sum() / tot)
                        rows.append(row)
                        if e == 0 and which == "pred" and j % nq == 0:
                            panel.append((qi, up[j], ds.classes[cls[int(true[j])]],
                                          ds.classes[cls[int(pred[j])]]))
            out["targets"][tgt] = summarise_cam(rows)
            out["targets"][tgt]["rows"] = rows
            fig, ax = plt.subplots(1, len(panel), figsize=(3 * len(panel), 3.2))
            for a, (qi, m, t, p) in zip(np.atleast_1d(ax), panel):
                img = ds.x[qi].cpu().numpy()
                a.imshow(img)
                a.imshow(m, cmap="jet", alpha=0.45, vmin=0, vmax=max(m.max(), 1e-8))
                a.set_title(f"true {t}\npred {p}", fontsize=7)
                a.axis("off")
            fig.suptitle(f"{meta['run']} seed {meta['seed']} {tgt} (episode 0, one query/class)",
                         fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(rdir, f"cam_panel_{tgt}.png"), dpi=200)
            plt.close(fig)
        dump(os.path.join(rdir, "cam_stats.json"), out)
        print(f"{meta['run']} s{meta['seed']}: " + ", ".join(
            f"{t} empty={v['overall']['empty_rate']:.2f}" for t, v in out["targets"].items()))


def mode_cfpanel(cfg, args):
    """E7 input verification: first images of each UCI class under every counterfactual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files, y, _ = list_files(cfg["uci_root"], cfg["uci_classes"])
    picks = [i for c in range(len(cfg["uci_classes"])) for i in np.where(y == c)[0][:args.n_per_class]]
    kinds = ["none", "bg_white", "bg_only", "geom_only"]
    fig, ax = plt.subplots(len(picks), len(kinds), figsize=(2.4 * len(kinds), 1.6 * len(picks)))
    stats_ = {k: [] for k in kinds[1:]}
    for r, i in enumerate(picks):
        img = cv2.imread(files[i])
        for c, kd in enumerate(kinds):
            out, fg = counterfactual(img, kd)
            if fg is not None:
                stats_[kd].append(fg)
            lb, _ = letterbox(out, cfg["img"])
            a = ax[r, c]
            a.imshow(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(kd, fontsize=9)
            if c == 0:
                a.set_ylabel(cfg["uci_classes"][y[i]][:14], fontsize=7)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "E7_counterfactual_panel.png"), dpi=200)
    print(f"wrote {os.path.join(args.out, 'E7_counterfactual_panel.png')}; "
          f"mean dilated leaf-proxy fraction {np.mean(stats_['bg_white']):.3f}. Inspect it: "
          "E7 is interpretable only if the leaf is intact in bg_white and absent in bg_only.")


def summarise_cam(rows):
    """Mass and enrichment statistics are computed over non-empty maps only, per map, so that
    numerator and denominator always refer to the same images. Empty-map rate uses all maps."""
    def agg(rs):
        if not rs:
            return {}
        d = {"n": len(rs), "empty_rate": float(np.mean([r["empty"] for r in rs]))}
        ne = [r for r in rs if "mass_pad" in r]
        d["n_nonempty"] = len(ne)
        if ne:
            d["mass_pad"] = float(np.mean([r["mass_pad"] for r in ne]))
            d["pad_area_frac"] = float(np.mean([r["pad_area_frac"] for r in ne]))
            en = [r["mass_pad"] / r["pad_area_frac"] for r in ne if r["pad_area_frac"] > 0]
            if en:
                d["pad_enrichment"] = float(np.mean(en))   # >1: more CAM mass in padding than area
            d["peak_in_pad_rate"] = float(np.mean([r["peak_in_pad"] for r in ne]))
            le = [r for r in ne if "mass_leaf" in r and r["leaf_area_frac"] > 0]
            if le:
                d["mass_leaf"] = float(np.mean([r["mass_leaf"] for r in le]))
                d["leaf_area_frac"] = float(np.mean([r["leaf_area_frac"] for r in le]))
                d["leaf_enrichment"] = float(np.mean([r["mass_leaf"] / r["leaf_area_frac"]
                                                      for r in le]))
        return d
    pr = [r for r in rows if r["which"] == "pred"]
    out = {"overall": agg(pr), "true_class_maps": agg([r for r in rows if r["which"] == "true"]),
           "correct": agg([r for r in pr if r["correct"]]),
           "incorrect": agg([r for r in pr if not r["correct"]]), "per_class": {}}
    for c in sorted({r["true_class"] for r in pr}):
        out["per_class"][c] = agg([r for r in pr if r["true_class"] == c])
    return out


def analytic_macs(variant, img):
    tot, hw, attn = 0, img, 0
    spec = VARIANTS[variant]
    for bi, (ci, co) in enumerate([(3, 64), (64, 128), (128, 256), (256, 512)], 1):
        tot += ci * co * 9 * hw * hw + 2 * co * co * 9 * hw * hw + (ci * co * hw * hw if ci != co
                                                                   else 0)
        if spec and spec[0] == f"block{bi}":
            if spec[1] in ("full", "ch", "res"):
                attn += 2 * (co * (co // 16) * 2)       # shared MLP, two descriptors
            if spec[1] in ("full", "sp", "res"):
                attn += 2 * 49 * hw * hw                # 7x7 conv, 2 in, 1 out, at conv res
        hw //= 2
    return tot, attn


def mode_efficiency(cfg, args, device):
    res = {"env": env_info(device), "img": args.img, "note": (
        "MACs count convolutions and the attention MLP only; elementwise rescaling, "
        "normalisation and activations are excluded. Latency: median of timed passes after "
        "warm-up, CUDA-synchronised, repeated in independent sessions."),
        "warmup": args.warmup, "timed": args.timed, "repeats": args.repeats, "variants": {}}
    for v in args.variants:
        enc = build_encoder(v, 42).to(device).eval()
        conv, attn = analytic_macs(v, args.img)
        r = {"params": n_params(enc), "conv_macs": conv, "attn_macs": attn, "latency": {}}
        for amp in ([False, True] if device.type == "cuda" else [False]):
            for b in (1, 30, 45):
                meds = []
                for _ in range(args.repeats):
                    x = torch.randn(b, 3, args.img, args.img, device=device)
                    with torch.no_grad(), autocast(device, amp):
                        for _ in range(args.warmup):
                            enc(x)
                        ts = []
                        for _ in range(args.timed):
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            t = time.perf_counter()
                            enc(x)
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            ts.append((time.perf_counter() - t) * 1e3)
                    meds.append(round(statistics.median(ts), 4))
                r["latency"][f"{'amp' if amp else 'fp32'}_b{b}"] = meds
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                enc(torch.randn(45, 3, args.img, args.img, device=device))
            r["peak_mem_mb_b45"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 2)
        res["variants"][v] = r
        print(v, r["params"], conv + attn, r["latency"])
        del enc
    dump(os.path.join(args.out, f"efficiency_img{args.img}.json"), res)


def selftest(device):
    """Closed-form CAM equals autograd Grad-CAM; initialisation is shared across variants."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 84, 84, device=device)
    for v in VARIANTS:
        enc = build_encoder(v, 7).to(device).eval()
        with torch.no_grad():
            A = enc.features(x)
        P = torch.randn(3, 512, device=device)
        ci = torch.tensor([0, 1, 2, 1], device=device)
        a = cam_closed_form(A, P, ci)
        b = cam_autograd(enc, x, P, ci)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), f"CAM mismatch for {v}"
    p0 = build_encoder("plain", 7).state_dict()
    for v in VARIANTS:
        sd = build_encoder(v, 7).state_dict()
        for key, val in p0.items():
            assert torch.equal(val, sd[key]), f"backbone init differs for {v} at {key}"
    expect = {"plain": 7996800, "cbam": 8029666}
    for v, n in expect.items():
        assert n_params(build_encoder(v, 1)) == n, v
    c, a = analytic_macs("cbam", 84)
    assert c == 2466166784 and a == 75336, (c, a)

    class _Fake:   # minimal ImageSet stand-in: 3 classes x 20 random images
        def __init__(self):
            g = torch.Generator().manual_seed(3)
            self.xs = torch.randn(60, 3, 84, 84, generator=g).to(device)
            self.y, self.n, self.classes = np.repeat(np.arange(3), 20), 60, ["a", "b", "c"]

        def batch(self, idx, flip_gen=None):
            return self.xs[torch.as_tensor(np.asarray(idx), device=device)]
    cfg = dict(DEFAULTS, eval_episodes=4)
    enc = build_encoder("cbam", 5).to(device).eval()
    ref = evaluate(enc, _Fake(), 42, "t", cfg, "proto_euclid")
    ft0 = evaluate(enc, _Fake(), 42, "t", dict(cfg, finetune_steps=0), "finetune_b4")
    assert ref["episode_acc"] == ft0["episode_acc"], "finetune_b4 with 0 steps != proto_euclid"
    ft = evaluate(enc, _Fake(), 42, "t", dict(cfg, finetune_steps=3), "finetune_b4")
    assert ft["support_acc_after"] >= 0 and len(ft["episode_acc"]) == 4
    print("selftest passed: CAM closed form == autograd, shared backbone init, "
          "params 7,996,800 / 8,029,666, MACs 2,466,166,784 + 75,336, "
          "finetune_b4(0 steps) == proto_euclid")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["dryrun", "train", "eval", "cam", "efficiency", "selftest", "cfpanel"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--variants", nargs="+", default=["plain", "cbam"])
    ap.add_argument("--mask", choices=["off", "on", "both"], default="both")
    ap.add_argument("--img", type=int, default=84)
    ap.add_argument("--seeds", default=None, help="e.g. 42-49 or 42,43")
    ap.add_argument("--runs", nargs="*", default=None, help="restrict eval/cam to these run names")
    ap.add_argument("--stages", nargs="+", default=["episodic"], choices=["pretrain", "episodic"])
    ap.add_argument("--classifiers", nargs="+", default=["proto_euclid"],
                    choices=["proto_euclid", "proto_cosine", "proto_cosine_transductive", "linear_head", "finetune_b4"])
    ap.add_argument("--shots", nargs="+", type=int, default=[5])
    ap.add_argument("--counterfactuals", nargs="+", default=["none"],
                    choices=["none", "bg_white", "bg_only", "geom_only"])
    ap.add_argument("--targets", nargs="+", default=["uci", "paddy"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test ONLY")
    ap.add_argument("--src_classes", nargs="+", default=None)
    ap.add_argument("--data_root_override", default=None,
                    help="replace the leading /kaggle/input of all roots (for local tests)")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.src_classes:
        cfg["src_classes"] = args.src_classes
    if args.data_root_override:
        for k in ("src_root", "uci_root", "paddy_root"):
            cfg[k] = cfg[k].replace("/kaggle/input", args.data_root_override, 1)
    if args.quick:
        cfg.update(epochs=1, episodes_per_epoch=3, pretrain_epochs=1, eval_episodes=6, pretrain_bs=16,
                   linear_head_steps=5, finetune_steps=2, cam_episodes=2)
    args.seeds = parse_seeds(args.seeds) if args.seeds else (None if args.mode in ("eval", "cam")
                                                             else cfg["seeds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = (not args.no_amp) and device.type == "cuda"
    print(f"device {device} | amp {args.amp} | script sha1 {script_sha1()[:12]}")
    if args.mode == "selftest":
        selftest(device)
    elif args.mode == "dryrun":
        mode_dryrun(cfg, args.out)
    elif args.mode == "train":
        mode_train(cfg, args, device)
    elif args.mode == "eval":
        mode_eval(cfg, args, device)
    elif args.mode == "cam":
        mode_cam(cfg, args, device)
    elif args.mode == "efficiency":
        mode_efficiency(cfg, args, device)
    elif args.mode == "cfpanel":
        mode_cfpanel(cfg, args)


if __name__ == "__main__":
    main()


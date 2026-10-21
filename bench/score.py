"""Score candidate upscales against the ground truth, alone and as the detail source of the Nomos2 blend.

Full-reference metrics (DISTS, LPIPS) are averaged over the four image quarters, as in the October benchmark, so the
numbers stay comparable. Low-frequency metrics catch what the blend cannot absorb: a detail source that shifts or
recolours the image gives double edges, since the blend keeps shapes and tones from Nomos2.
"""

import csv
import io
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2lab

BLEND_SIGMA = 0.75
BLEND_KS = (0.25, 0.5, 0.75, 1.0)
LF_SIGMA = 3.0
FIELDS = ["id", "set", "method", "k", "dists", "lpips", "lf_psnr", "lf_de", "shift_px", "topiq_nr"]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255


def lowpass(a, s):
    return np.stack([gaussian_filter(a[..., c], s) for c in range(3)], -1)


def blend(nomos, cand, k):
    mix = (1 - k) * nomos + k * cand
    return np.clip(lowpass(nomos, BLEND_SIGMA) + mix - lowpass(mix, BLEND_SIGMA), 0, 1)


def quarters(a):
    h, w = a.shape[0] // 2, a.shape[1] // 2
    return [a[y : y + h, x : x + w] for y in (0, h) for x in (0, w)]


def low_freq(a, gt_lp, gt_gray):
    lp = lowpass(a, LF_SIGMA)
    mse = float(np.mean((lp - gt_lp) ** 2))
    de = float(np.mean(deltaE_ciede2000(rgb2lab(lp), rgb2lab(gt_lp))))
    (dx, dy), _ = cv2.phaseCorrelate(gt_gray, cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64))
    return {
        "lf_psnr": round(10 * np.log10(1 / max(mse, 1e-10)), 3),
        "lf_de": round(de, 3),
        "shift_px": round(float(np.hypot(dx, dy)), 3),
    }


def score(set_dir, out_dir, methods, ids, csv_path, commit):
    """Score ids, saving csv_path after each image (and calling commit) so a crash resumes where it stopped."""
    dev = torch.device("cuda")
    fr = {n: pyiqa.create_metric(n, device=dev) for n in ["dists", "lpips"]}
    nr = pyiqa.create_metric("topiq_nr", device=dev)

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to(dev)

    path = Path(csv_path)
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    done = {(r["id"], r["method"]) for r in rows}
    for iid in ids:
        if all((iid, m) in done for m in methods):
            continue
        gt = load(f"{set_dir}/gt/{iid}.png")
        gq = quarters(gt)
        gt_lp = lowpass(gt, LF_SIGMA)
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float64)
        try:
            nomos = load(f"{out_dir}/{iid}__x4_nomos2.png")
        except FileNotFoundError:
            nomos = None
        for m in methods:
            if (iid, m) in done:
                continue
            try:
                cand = load(f"{out_dir}/{iid}__x4_{m}.png")
            except FileNotFoundError:
                continue
            variants = [("solo", cand)]
            if nomos is not None and m != "nomos2":
                variants += [(k, blend(nomos, cand, k)) for k in BLEND_KS]
            for k, a in variants:
                row = {"id": iid, "set": iid.split("_")[0], "method": m, "k": k}
                with torch.no_grad():
                    for n, f in fr.items():
                        row[n] = round(sum(float(f(t(x), t(g))) for x, g in zip(quarters(a), gq, strict=True)) / 4, 4)
                    row["topiq_nr"] = round(float(nr(t(a))), 4) if k == "solo" else ""
                row.update(low_freq(a, gt_lp, gt_gray))
                rows.append(row)
            torch.cuda.empty_cache()
        write(path, rows)
        commit()
        print(iid, flush=True)
    return path.read_text()


def write(path, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue())

#!/usr/bin/env python3
"""Tiny numpy software rasteriser for .mesh files written by the CharacterPreview tool.
Usage: render.py out.png file.mesh [--front|--side|--face] [more.mesh ...]  (meshes are laid out left→right)."""
import sys
import numpy as np
from PIL import Image

def load(path):
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                vs.append([float(x) for x in line.split()[1:]])
            elif line.startswith('f '):
                fs.append([int(x) for x in line.split()[1:]])
    return np.array(vs, dtype=np.float64), np.array(fs, dtype=np.int64)

def render(meshes, out, view='front', size=(1200, 900)):
    W, H = size
    img = np.zeros((H, W, 3)); img[:] = (0.13, 0.14, 0.16)
    zb = np.full((H, W), np.inf)
    n = len(meshes)
    if view == 'face':
        cam_h, scale = 1.62, H / 0.42
    else:
        cam_h, scale = 0.95, H / 2.15
    light = np.array([-0.45, 0.6, -0.65]); light /= np.linalg.norm(light)
    fill = np.array([0.6, 0.2, -0.4]); fill /= np.linalg.norm(fill)
    for idx, (v, f) in enumerate(meshes):
        P, N, C = v[:, 0:3], v[:, 3:6], v[:, 6:9]
        x_off = (idx - (n - 1) / 2) * (W / max(n, 1)) / scale
        sx = (P[:, 0] + x_off) * scale + W / 2
        sy = H / 2 - (P[:, 1] - cam_h) * scale
        sz = P[:, 2]  # camera looks along +Z from -Z: smaller z is nearer
        lam = np.clip(N @ light, 0, 1) * 0.75 + np.clip(N @ fill, 0, 1) * 0.2 + 0.18
        rim = np.clip(1 - np.abs(N[:, 2]), 0, 1) ** 3 * 0.15
        col = np.clip(C * lam[:, None] + rim[:, None], 0, 1)
        for a, b, c in f:
            xs = np.array([sx[a], sx[b], sx[c]]); ys = np.array([sy[a], sy[b], sy[c]])
            # back-face cull (front faces have normals pointing to -Z, toward the camera)
            area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
            if abs(area) < 1e-9:
                continue
            x0, x1 = int(max(0, np.floor(xs.min()))), int(min(W - 1, np.ceil(xs.max())))
            y0, y1 = int(max(0, np.floor(ys.min()))), int(min(H - 1, np.ceil(ys.max())))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((xs[1] - gx) * (ys[2] - gy) - (xs[2] - gx) * (ys[1] - gy)) / area
            w1 = ((xs[2] - gx) * (ys[0] - gy) - (xs[0] - gx) * (ys[2] - gy)) / area
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            z = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
            region = zb[y0:y1 + 1, x0:x1 + 1]
            mask = inside & (z < region)
            if not mask.any():
                continue
            region[mask] = z[mask]
            colr = (w0[..., None] * col[a] + w1[..., None] * col[b] + w2[..., None] * col[c])
            img[y0:y1 + 1, x0:x1 + 1][mask] = colr[mask]
    Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.2) * 255).astype(np.uint8)).save(out)

if __name__ == '__main__':
    out = sys.argv[1]
    view = 'front'
    files = []
    for a in sys.argv[2:]:
        if a.startswith('--'):
            view = a[2:]
        else:
            files.append(a)
    render([load(p) for p in files], out, view)

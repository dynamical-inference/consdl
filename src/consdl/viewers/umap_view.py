"""Cluster-consistency view: decoder atoms on a joint UMAP-2D, one panel per
dictionary, matched on hover. Three kinds of view:

  seed   one arch × cell, its seeds           Hungarian or nearest neighbour
  cross  one arch, seed 0 of each of its cells nearest neighbour
  arch   one cell, seed 0 of each arch         nearest neighbour

Hover a latent: it is ringed in red, its match is ringed in every other panel,
and every atom with decoder cos ≥ τ gets a thin ring (``/sims``). The right
side shows, per panel, six of the latent's top inputs (ranks 1,2 · 50,51 ·
99,100 of its top 100): images masked by where it fires, or token contexts
shaded by its activation.

Nothing is fitted here; views come from ``consdl.cli.viewer_precompute``.
SAEs, decoders and top lists load on first use and are LRU-evicted. This
module holds one backbone's state; ``consdl.viewers.explorer`` serves it.
"""
from __future__ import annotations

import io
import json
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from consdl.viewers import layout as lay
from consdl.viewers.html_view import FOOT, PAGE_TMPL
from consdl.viewers.models import encode, load_sae, unit_decoder

PICK_RANKS = (1, 2, 50, 51, 99, 100)          # 1-based, within the top 100
N_COLOURED = 30
M_COLOUR = 48
CTX_SPAN = 12
NO_TOPS = ("no top inputs computed for this checkpoint "
           "(python -m consdl.cli.viewer_precompute --stages tops)")

#: Filled by :func:`configure` and :func:`warm`.
STATE: dict = {}
#: One thread at a time on the backbone and the SAEs (the vision backbone
#: stashes its hooked activation on the instance).
GPU = threading.RLock()
#: Guards the lazy caches.
LOAD = threading.Lock()


class LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


_PAYLOAD = LRU(24)
_TOPS = LRU(64)
_DEC = LRU(96)
_SAE = LRU(24)
_FP = LRU(64)
_RECOLOUR = LRU(64)
_Z = LRU(96)                   # vision input -> (T, d_in) on the device
_CROP = LRU(4000)              # val index -> PIL crop as the backbone saw it
_IMG_OUT = LRU(6000)           # (img, key, lat, size) -> jpeg
_ZROW = LRU(320)               # LM stream row -> (T, d_in), CPU


# ── setup ─────────────────────────────────────────────────────────────


def configure(root: Path) -> None:
    """Point the server at one backbone's precompute dir."""
    root = Path(root)
    bb = json.loads((root / "backbone.json").read_text())
    STATE.update(root=root, views_dir=root / "views", fp_dir=root / "footprint", bb=bb,
                 backbone=bb["name"], label=bb["label"],
                 modality="text" if bb["modality"] == "lm" else "image")


def refresh_index() -> None:
    """Re-read the view manifests and panel keys (cheap; picks up new views)."""
    views, keys = {}, {}
    for p in sorted(STATE["views_dir"].glob("*.json")):
        if p.name.startswith("keys__") or p.name == "lm_stream.json":
            continue
        views[p.stem] = json.loads(p.read_text())
    for p in sorted(STATE["views_dir"].glob("keys__*.json")):
        keys.update(json.loads(p.read_text()))
    STATE.update(views=views, keys=keys, default_view=_default_view(views))


def _default_view(views: dict) -> str | None:
    seeds = [v for v in views if v.startswith("seed__")]
    if not seeds:
        return next(iter(sorted(views)), None)
    ref = [v for v in seeds if v.split("__")[1] == lay.REFERENCE_ARCH] or seeds
    return max(ref, key=lambda v: lay.cell_sort_key(v.split("__")[2]))


def warm() -> None:
    """Load the dataset / processor / backbone (vision) or tokenizer and
    stream (LM). SAEs load lazily."""
    import torch  # noqa: PLC0415

    refresh_index()
    if not STATE["keys"]:
        raise SystemExit(f"[fatal] no views under {STATE['views_dir']}: run "
                         f"python -m consdl.cli.viewer_precompute first")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    STATE["dev"] = dev
    bb = STATE["bb"]
    if STATE["modality"] == "text":
        from transformers import AutoTokenizer  # noqa: PLC0415
        f = STATE["views_dir"] / "lm_stream.npy"
        STATE["tok"] = AutoTokenizer.from_pretrained(bb["hf_name"])
        STATE["ids"] = np.load(f, mmap_mode="r") if f.is_file() else None
        return
    from transformers import AutoProcessor  # noqa: PLC0415

    from consdl.data.vision_backbones import Backbone  # noqa: PLC0415
    from consdl.data.vision_datasets import load_image_split  # noqa: PLC0415

    ds = load_image_split("imagenet", "val")
    STATE.update(ds=ds, labels=np.asarray(ds["label"], dtype=np.int64),
                 names=list(ds.features["label"].names),
                 proc=AutoProcessor.from_pretrained(bb["hf_name"]),
                 model=Backbone(model_hf_name=bb["hf_name"], hook_target_name=bb["hook"],
                                device=str(dev)))


# ── lazy loads ────────────────────────────────────────────────────────


def sae_for(key: str):
    """``key``'s SAE, resident on the device."""
    meta = STATE["keys"][key]
    step = meta["step"]
    with LOAD:
        hit = _SAE.get(step)
        if hit is not None:
            _SAE.move_to_end(step)
            return hit
    with GPU:
        with LOAD:
            hit = _SAE.get(step)
        if hit is None:
            hit = load_sae(Path(step), arch=meta["arch"], level=meta["level"], device=STATE["dev"])
            with LOAD:
                _SAE.put(step, hit)
    return hit


def dec(key: str) -> np.ndarray:
    with LOAD:
        if key in _DEC:
            _DEC.move_to_end(key)
            return _DEC[key]
    W = unit_decoder(Path(STATE["keys"][key]["step"]))
    with LOAD:
        _DEC.put(key, W)
    return W


def tops(key: str) -> dict | None:
    """``{ids, vals[, pos]}`` of ``key``'s top inputs, or ``None``."""
    meta = STATE["keys"][key]
    f = STATE["views_dir"] / meta["tops"]
    with LOAD:
        z = _TOPS.get(f.name)
    if z is None:
        if not f.is_file():
            return None
        z = dict(np.load(f))
        with LOAD:
            _TOPS.put(f.name, z)
    s = meta["seed"]
    if f"ids_{s}" not in z:
        return None
    return {"ids": z[f"ids_{s}"], "vals": z[f"vals_{s}"], "pos": z.get(f"pos_{s}")}


def _rank_of(size0: np.ndarray) -> np.ndarray:
    """Colour slot per anchor cluster: the N_COLOURED largest, else -1 (grey)."""
    top = np.argsort(-size0)[:N_COLOURED]
    r = np.full(size0.shape[0], -1, dtype=np.int32)
    r[top] = np.arange(top.size)
    return r


def payload(view_id: str) -> bytes:
    with LOAD:
        hit = _PAYLOAD.get(view_id)
    if hit is not None:
        return hit
    z = np.load(STATE["views_dir"] / f"{view_id}.npz")
    meta = STATE["views"][view_id]
    keys = [str(k) for k in z["keys"]]
    rk = _rank_of(z["size_anchor"])
    f64 = lambda a: np.round(a.astype(np.float64), 3)     # noqa: E731
    out = {"id": view_id, "kind": str(z["kind"]), "title": str(z["title"]),
           "arch": meta.get("arch"), "cell": meta.get("cell"),
           "backbone": STATE["backbone"], "modality": STATE["modality"],
           "gauge": meta.get("gauge"), "colour": meta.get("colour"),
           "seeds": keys, "labels": dict(zip(keys, [str(x) for x in z["labels"]])),
           "anchors": [k for k in keys if STATE["keys"].get(k, {}).get("anchor")],
           "anchor": str(z["anchor"]), "rules": [str(r) for r in z["rules"]],
           "xy": {k: f64(z[f"xy_{i}"]).ravel().tolist() for i, k in enumerate(keys)},
           "clu": {k: z[f"clu_{i}"].tolist() for i, k in enumerate(keys)},
           "col": {k: rk[z[f"clu_{i}"]].tolist() for i, k in enumerate(keys)},
           "nfire": None, "match": {}, "mcos": {}}
    for rule in out["rules"]:
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if i != j and f"m_{rule}_{i}_{j}" in z:
                    out["match"][f"{rule}_{a}_{b}"] = z[f"m_{rule}_{i}_{j}"].tolist()
                    out["mcos"][f"{rule}_{a}_{b}"] = f64(z[f"c_{rule}_{i}_{j}"]).tolist()
    b = json.dumps(out).encode()
    with LOAD:
        _PAYLOAD.put(view_id, b)
    return b


def view_keys(view_id: str) -> list[str]:
    return json.loads(payload(view_id))["seeds"]


# ── decoder geometry ──────────────────────────────────────────────────


def sims(view_id: str, key: str, lat: int, *, tau: float, k: int) -> dict:
    """Every atom with decoder cos ≥ ``tau`` to ``key``'s latent ``lat``, per
    panel (its own panel too, self excluded), at most ``k``; and per panel
    the cos of the atom to its best non-negative mix of those atoms
    (``span_cos``) against the single nearest one (``nn_cos``)."""
    from scipy.optimize import nnls  # noqa: PLC0415

    a = dec(key)[lat]
    a_arch = STATE["keys"][key]["arch"]
    out = {}
    for b in view_keys(view_id):
        Db = dec(b)
        absolute = a_arch in lay.ABS_ARCHES or STATE["keys"][b]["arch"] in lay.ABS_ARCHES
        raw = Db @ a
        s = np.abs(raw) if absolute else raw.copy()
        if b == key:
            s[lat] = -np.inf
        top = np.argpartition(-s, k - 1)[:k] if s.size > k else np.arange(s.size)
        top = top[np.argsort(-s[top])]
        keep = top[s[top] >= tau]
        row = {"nn": int(top[0]), "nn_cos": float(s[top[0]]), "gauge": "abs" if absolute else "signed",
               "idx": keep.tolist(), "cos": np.round(s[keep].astype(np.float64), 3).tolist(),
               "span_cos": None, "w": []}
        if keep.size:
            sg = np.sign(raw[keep]) if absolute else np.ones(keep.size)
            A = (Db[keep] * sg[:, None]).T.astype(np.float64)
            w, _ = nnls(A, a.astype(np.float64))
            r = A @ w
            nr = float(np.linalg.norm(r))
            row["span_cos"] = round(float(r @ a) / nr, 3) if nr > 0 else 0.0
            row["w"] = np.round(w, 3).tolist()
        out[b] = row
    return {"key": key, "lat": lat, "tau": tau, "k": k, "panels": out}


def recolour(view_id: str, anchor: str) -> bytes:
    """Colours of every panel against ANOTHER panel's seed-0 clusters
    (nearest centroid, |cos| when either side is iSAE)."""
    ck = (view_id, anchor)
    with LOAD:
        hit = _RECOLOUR.get(ck)
    if hit is not None:
        return hit
    meta = STATE["keys"][anchor]
    if not meta.get("anchor"):
        raise ValueError(f"{anchor}: only a cell's first seed has clusters")
    z = np.load(STATE["root"] / meta["arch"] / "within" / meta["cell"] / f"centroids_m{M_COLOUR}.npz")
    C0, size0 = z["C0"].astype(np.float32), z["size_seed0"]
    rk = _rank_of(size0)
    out = {"anchor": anchor, "K": int(len(size0)), "clu": {}, "col": {}}
    for b in view_keys(view_id):
        S = dec(b) @ C0.T
        absolute = meta["arch"] in lay.ABS_ARCHES or STATE["keys"][b]["arch"] in lay.ABS_ARCHES
        lab = (np.abs(S) if absolute else S).argmax(1).astype(np.int32)
        out["clu"][b], out["col"][b] = lab.tolist(), rk[lab].tolist()
    body = json.dumps(out).encode()
    with LOAD:
        _RECOLOUR.put(ck, body)
    return body


# ── footprints (vision): per-image peak |a| over the top-input pool ────


def _fp(key: str):
    """``(indptr, indices, data, n_rows)`` for ``key``'s footprint, or ``None``."""
    arch, cell, seed = key.split("/")
    d = STATE["fp_dir"] / f"{arch}__{cell}__{seed}"
    if not (d / "data.npy").is_file():
        return None
    with LOAD:
        hit = _FP.get(key)
        if hit is None:
            n = json.loads((d / "meta.json").read_text())["n_rows"]
            hit = (*(np.load(d / f"{x}.npy", mmap_mode="r") for x in ("indptr", "indices", "data")), n)
            _FP.put(key, hit)
    return hit


def fpcos(key: str, lat: int, targets: list[tuple[str, int]]) -> list:
    """Footprint cos of ``key``'s latent with each ``(key, latent)`` target;
    ``None`` where either side has no footprint."""
    q = _fp(key)
    if q is None:
        return [None] * len(targets)
    ip, ix, dt, n = q
    dense = np.zeros(n, dtype=np.float32)
    s, e = int(ip[lat]), int(ip[lat + 1])
    dense[np.asarray(ix[s:e])] = np.asarray(dt[s:e], dtype=np.float32)
    out = []
    for k, j in targets:
        f = _fp(k)
        if f is None or f[3] != n:
            out.append(None)
            continue
        s, e = int(f[0][j]), int(f[0][j + 1])
        out.append(round(float(dense[np.asarray(f[1][s:e])] @ np.asarray(f[2][s:e], dtype=np.float32)), 3))
    return out


def fpnull() -> bytes:
    p = STATE["fp_dir"] / "null.json"
    return p.read_bytes() if p.is_file() else b"{}"


# ── images ────────────────────────────────────────────────────────────
#
# Every crop is the processor's own output, de-normalised, so a patch overlay
# sits on exactly the pixels the backbone saw (CLIP: shortest edge 224 +
# centre crop; DINOv2: shortest edge 256 + centre crop 224, 14-px patches).

UPLOADS: dict = {}
UPLOADS_MAX = 24
MAX_UPLOAD = 12 * 1024 * 1024


def is_upload(img) -> bool:
    return isinstance(img, str) and img.startswith("u_")


def upload(tok: str) -> dict:
    u = UPLOADS.get(tok)
    if u is None:
        raise KeyError(f"upload {tok!r} (it may have been evicted; upload the image again)")
    return u


def pixels(img):
    """``(1, 3, H, W)`` normalised pixels of a val index or an upload token."""
    from consdl.viewers.tops import pixel_values  # noqa: PLC0415
    if is_upload(img):
        return upload(img)["px"]
    return pixel_values(STATE["proc"], STATE["ds"][int(img)]["image"].convert("RGB"))


def denormalise(px):
    """PIL image of ``(1, 3, H, W)`` normalised pixels."""
    from PIL import Image  # noqa: PLC0415
    ip = getattr(STATE["proc"], "image_processor", STATE["proc"])
    mean = np.asarray(ip.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(ip.image_std, dtype=np.float32)[:, None, None]
    a = px[0].float().cpu().numpy()
    rgb = ((a * std + mean).clip(0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(rgb)


def crop(img):
    if is_upload(img):
        return upload(img)["crop"]
    i = int(img)
    with LOAD:
        hit = _CROP.get(i)
    if hit is None:
        hit = denormalise(pixels(i))
        with LOAD:
            _CROP.put(i, hit)
    return hit


def z_for(img):
    """``(T, d_in)`` backbone activations of one image (CLS first), cached."""
    import torch  # noqa: PLC0415
    ck = img if is_upload(img) else int(img)
    with LOAD:
        hit = _Z.get(ck)
    if hit is not None:
        return hit
    with GPU, torch.no_grad():
        with LOAD:
            hit = _Z.get(ck)
        if hit is not None:
            return hit
        z = STATE["model"].forward_to_z({"pixel_values": pixels(ck).to(STATE["dev"])})
        z = z.reshape(-1, z.shape[-1])
        with LOAD:
            _Z.put(ck, z)
    return z


def atom_acts(img, key: str, lat: int) -> np.ndarray:
    """``(n_patches,)`` |a| of one latent on one image, CLS dropped."""
    with GPU:
        a = encode(sae_for(key), z_for(img))
        return a[1:, int(lat)].abs().float().cpu().numpy()


def jpeg(img, size: int) -> bytes:
    buf = io.BytesIO()
    img.resize((int(size), int(size))).save(buf, format="JPEG", quality=80, optimize=True)
    return buf.getvalue()


def masked_image(img, key: str, lat: int, size: int) -> bytes:
    """The crop dimmed by the latent's |a| per patch, min-max scaled per image
    (floor 0.1, so dim = weaker than this image's peak, not zero)."""
    from PIL import Image  # noqa: PLC0415
    ck = (img, key, lat, size)
    with LOAD:
        hit = _IMG_OUT.get(ck)
    if hit is not None:
        return hit
    act = atom_acts(img, key, lat)
    side = int(round(np.sqrt(act.size)))
    if side * side != act.size:
        raise ValueError(f"{act.size} patch tokens is not a square grid")
    lo, hi = float(act.min()), float(act.max())
    m = np.clip((act - lo) / (hi - lo + 1e-10), 0, 1) * 0.9 + 0.1
    pic = crop(img)
    arr = np.asarray(pic, dtype=np.float32)
    up = int(np.ceil(arr.shape[0] / side))
    m2 = np.kron(m.reshape(side, side), np.ones((up, up)))[:arr.shape[0], :arr.shape[1]]
    out = jpeg(Image.fromarray((arr * m2[:, :, None]).clip(0, 255).astype(np.uint8)), size)
    with LOAD:
        _IMG_OUT.put(ck, out)
    return out


def raw_image(img, size: int) -> bytes:
    return jpeg(crop(img), size)


def picks(key: str, lat: int) -> dict:
    """Six of the latent's top 100 images."""
    t = tops(key)
    if t is None:
        return {"key": key, "lat": lat, "n_top": 0, "picks": [], "error": NO_TOPS}
    ids, vals = t["ids"][lat], t["vals"][lat]
    n_have = int((ids >= 0).sum())
    labels, names = STATE["labels"], STATE["names"]
    rows = []
    for r in PICK_RANKS:
        if n_have == 0:
            break
        rr = r if n_have >= 100 else max(1, int(round(r * n_have / 100)))
        img = int(ids[rr - 1])
        c = int(labels[img])
        rows.append({"rank": rr, "img": img, "val": float(vals[rr - 1]),
                     "cls": names[c] if c < len(names) else str(c)})
    return {"key": key, "lat": lat, "n_top": n_have, "picks": rows}


# ── token contexts (LM) ───────────────────────────────────────────────
#
# Top tokens index rows of ``views/lm_stream.npy``. A window is shaded by
# re-encoding live: the LM runs over the WHOLE row (it is causal, so the
# window's prefix matters), then the SAE over the window. The residual stream
# is cached per row and shared by every checkpoint.


def lm_backbone():
    with GPU:
        bb = STATE.get("lm")
        if bb is None:
            from consdl.cli.train_lm import LMBackbone  # noqa: PLC0415
            bb = STATE["lm"] = LMBackbone(model_hf_name=STATE["bb"]["hf_name"],
                                          hook_layer_path=STATE["bb"]["hook"],
                                          device=str(STATE["dev"]))
    return bb


def lm_forward(ids):
    """``(T, d_in)`` residual stream of one token sequence, CPU."""
    import torch  # noqa: PLC0415
    x = torch.as_tensor(np.asarray(ids, dtype=np.int64))[None]
    bb = lm_backbone()
    with GPU:
        z = bb.forward_to_z({"input_ids": x, "attention_mask": torch.ones_like(x)})
    return z.float().reshape(x.shape[1], -1).cpu()


def z_rows(rows) -> dict:
    """Residual stream ``(T, d_in)`` of each stream row, cached."""
    import torch  # noqa: PLC0415
    rows = [int(r) for r in rows]
    with LOAD:
        got = {r: _ZROW[r] for r in rows if r in _ZROW}
    need = sorted(set(rows) - set(got))
    bb = lm_backbone()
    for i in range(0, len(need), 8):
        ch = need[i:i + 8]
        x = torch.from_numpy(np.asarray(STATE["ids"][ch], dtype=np.int64))
        with GPU:
            z = bb.forward_to_z({"input_ids": x, "attention_mask": torch.ones_like(x)})
        z = z.float().reshape(len(ch), x.shape[1], -1).cpu()
        for j, r in enumerate(ch):
            got[r] = z[j]
    with LOAD:
        for r, v in got.items():
            _ZROW.put(r, v)
    return {r: got[r] for r in rows}


def encode_cpu(key: str, z):
    """Signed codes ``(N, d_sae)`` of ``z`` on CPU."""
    with GPU:
        return encode(sae_for(key), z.to(STATE["dev"])).float().cpu()


def _token_acts(key: str, picks_: list) -> None:
    """Fill each pick with ``acts`` (signed, per window token) and ``chk``
    (live |a| at the peak ÷ the stored value)."""
    import torch  # noqa: PLC0415
    lat = picks_[0]["lat"]
    zs = z_rows([p["seq"] for p in picks_])
    z = torch.cat([zs[p["seq"]][p["lo"]:p["hi"]] for p in picks_])
    a = encode_cpu(key, z)[:, lat].numpy()
    off = 0
    for p in picks_:
        n = p["hi"] - p["lo"]
        v = a[off:off + n]
        off += n
        p["acts"] = [round(float(x), 4) for x in v]
        p["chk"] = round(abs(float(v[p["pos"] - p["lo"]])) / p["val"], 4) if p["val"] > 0 else None


def contexts(key: str, lat: int, acts: bool = True) -> dict:
    """Six of the latent's top-100 tokens, each with ``CTX_SPAN`` tokens of
    context either side, and (``acts``) its activation on every token."""
    t = tops(key)
    if t is None or STATE.get("ids") is None:
        return {"key": key, "lat": lat, "n_top": 0, "picks": [], "error": NO_TOPS}
    I, P, A = t["ids"][lat], t["pos"][lat], t["vals"][lat]
    valid = [k for k in range(len(I)) if I[k] >= 0 and A[k] > 0]
    n = len(valid)
    tok, rows = STATE["tok"], []
    for r in PICK_RANKS:
        if n == 0:
            break
        rr = r if n >= 100 else max(1, int(round(r * n / 100)))
        k = valid[rr - 1]
        row = np.asarray(STATE["ids"][int(I[k])])
        p = int(P[k])
        lo, hi = max(0, p - CTX_SPAN), min(len(row), p + CTX_SPAN + 1)
        rows.append({"rank": rr, "val": float(A[k]), "seq": int(I[k]), "pos": p, "lat": lat,
                     "lo": lo, "hi": hi, "toks": [tok.decode([int(x)]) for x in row[lo:hi]],
                     "left": tok.decode(row[lo:p].tolist()), "peak": tok.decode([int(row[p])]),
                     "right": tok.decode(row[p + 1:hi].tolist())})
    out = {"key": key, "lat": lat, "n_top": n, "picks": rows,
           "ref": float(A[valid[0]]) if n else None}             # rank-1 |a|: the shading scale
    if rows and acts:
        try:
            _token_acts(key, rows)
        except Exception as e:  # noqa: BLE001 -- the contexts still show without shading
            traceback.print_exc()
            out["act_error"] = f"{type(e).__name__}: {e}"
    return out


# ── nav ───────────────────────────────────────────────────────────────


def nav() -> dict:
    v = STATE["views"]
    cells = sorted({m["cell"] for m in v.values() if m.get("cell")}, key=lay.cell_sort_key)
    return {"arches": sorted({m["arch"] for m in v.values() if m.get("arch")}, key=lay.arch_rank),
            "cells": cells, "cell_labels": {c: lay.cell_label(c) for c in cells},
            "views": sorted(v), "labels": lay.ARCH_LABEL, "default": STATE["default_view"],
            "backbone": STATE["backbone"], "label": STATE["label"]}


# ── HTTP ──────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        pass

    def _bytes(self, b: bytes, ctype: str, code: int = 200, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code: int = 200):
        self._bytes(json.dumps(obj).encode(), "application/json", code)

    def _redirect(self, to: str):
        self.send_response(302)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _view_id(self, q: dict) -> str | None:
        vid = q.get("id", STATE["default_view"])
        if vid not in STATE["views"]:
            refresh_index()
        return vid if vid in STATE["views"] else None

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            self.route_view(u.path, q) or self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def route_view(self, path: str, q: dict) -> bool:
        """The cluster-consistency view's routes; ``False`` if ``path`` is not one."""
        if path == "/view":
            vid = self._view_id(q)
            if vid is None:
                self._bytes(f"no view {q.get('id')!r}".encode(), "text/plain", 404)
            else:
                self._bytes(PAGE_TMPL.replace("%%VIEW%%", vid).replace("%%FOOT%%", FOOT).encode(),
                            "text/html; charset=utf-8")
        elif path == "/nav.json":
            refresh_index()
            self._json(nav())
        elif path == "/view/data.json":
            self._bytes(payload(q["id"]), "application/json")
        elif path == "/sims":
            self._json(sims(q["id"], q["key"], int(q["lat"]), tau=float(q.get("tau", 0.4)),
                            k=max(1, min(50, int(q.get("k", 8))))))
        elif path == "/fpcos":
            tg = [(t.rsplit(":", 1)[0], int(t.rsplit(":", 1)[1]))
                  for t in q.get("with", "").split(",") if t]
            self._json({"vals": fpcos(q["key"], int(q["lat"]), tg)})
        elif path == "/fpnull":
            self._bytes(fpnull(), "application/json")
        elif path == "/recolour":
            self._bytes(recolour(q["id"], q["anchor"]), "application/json", cache=True)
        elif path == "/top":
            self._json(picks(q["key"], int(q["lat"])))
        elif path == "/masked":
            self._bytes(masked_image(_img_key(q["img"]), q["key"], int(q["lat"]),
                                     int(q.get("size", 128))), "image/jpeg", cache=True)
        elif path == "/raw":
            self._bytes(raw_image(_img_key(q["img"]), int(q.get("size", 128))), "image/jpeg",
                        cache=True)
        elif path == "/ctx":
            self._json(contexts(q["key"], int(q["lat"]), acts=q.get("acts", "1") != "0"))
        else:
            return False
        return True


def _img_key(raw: str):
    """A val index (int) or an upload token (``u_…``)."""
    s = str(raw)
    return s if s.startswith("u_") else int(s)

"""Top inputs per latent, for the viewers' "top images / top contexts" rows.

The released stats hold per-latent scalars only, so this pass encodes an
input pool once per backbone and keeps, per latent, its top ``top_k``:

* **vision**: ImageNet val images (all 50 per class, or ``per_class``
  of them), ranked by the image's peak ``|a|`` over its patch tokens (CLS
  excluded). Also writes each latent's **footprint**, its per-image peak
  ``|a|`` over the pool, as column-normalised CSC, so the server can say how
  alike two latents' firing is (footprint cos).
* **lm**: packed FineWeb sequences (``n_seqs`` × 1024 tokens, stream seed 0,
  saved once as ``views/lm_stream.npy``), ranked per token by ``|a|``. The
  LM runs over whole rows, as in training.

One backbone forward per batch is shared by every SAE of the pass.
Outputs: ``views/tops__<arch>__<cell>.npz`` (``ids_<seed>``, ``vals_<seed>``
and, for text, ``pos_<seed>``; ``-1`` where a latent has fewer firing inputs)
and ``footprint/<arch>__<cell>__<seed>/{indptr,indices,data}.npy`` plus
``footprint/null.json`` (median / p95 footprint cos of random latent pairs).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from consdl.viewers import layout as lay
from consdl.viewers.models import encode, load_sae

FINEWEB = {"hf_dataset": "HuggingFaceFW/fineweb", "hf_config": None, "split": "train",
           "text_field": "text"}
CONTEXT = 1024
NULL_PAIRS = 20000


def _log(msg: str) -> None:
    print(f"[viewer-tops] {msg}", flush=True)


def _todo(cells_by_arch: dict, views: Path, force: bool) -> list:
    out = []
    for arch, cells in cells_by_arch.items():
        for c in cells:
            if force or not (views / f"tops__{arch}__{c.name}.npz").is_file():
                out.append(c)
    return out


def _load_all(cells, device) -> list[tuple]:
    out = []
    for c in cells:
        for s in c.seed_names():
            out.append((c, s, load_sae(c.seeds[s], arch=c.arch, level=c.level, device=device)))
    return out


class _TopK:
    """Running top-k per latent over a stream of scores ``(n_items, d_sae)``."""

    def __init__(self, d_sae: int, k: int, device):
        import torch  # noqa: PLC0415
        self.k = k
        self.val = torch.full((d_sae, k), -1.0, device=device)
        self.idx = torch.full((d_sae, k), -1, dtype=torch.long, device=device)

    def add(self, score, item_ids) -> None:
        """``score`` ``(n, d_sae)`` >= 0; ``item_ids`` ``(n,)`` long."""
        import torch  # noqa: PLC0415
        kk = min(self.k, score.shape[0])
        v, j = score.T.topk(kk, dim=1)
        cat_v = torch.cat([self.val, v], 1)
        cat_i = torch.cat([self.idx, item_ids[j]], 1)
        v2, j2 = cat_v.topk(self.k, dim=1)
        self.val, self.idx = v2, cat_i.gather(1, j2)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        val = self.val.float().cpu().numpy()
        idx = self.idx.cpu().numpy()
        idx[val <= 0] = -1
        val[val <= 0] = 0.0
        return idx, val


# ── vision ────────────────────────────────────────────────────────────


def image_rows(per_class: int | None) -> np.ndarray:
    """ImageNet-val rows of the pool: every image, or ``per_class`` per class."""
    from consdl.metrics.activations import IMAGES_PER_CLASS, stratified_rows  # noqa: PLC0415
    n = 1000 * IMAGES_PER_CLASS
    if not per_class or per_class >= IMAGES_PER_CLASS:
        return np.arange(n)
    return stratified_rows(n, per_class, seed=0)


class _Images:
    """``(pixel_values, row)`` of the pool's images, through the backbone's processor."""

    def __init__(self, ds, rows, proc):
        self.ds, self.rows, self.proc = ds, rows, proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        pil = self.ds[r]["image"].convert("RGB")
        return pixel_values(self.proc, pil)[0], r


def pixel_values(proc, pil):
    """``(1, 3, H, W)`` normalised pixels from a HF processor (CLIP's wants a text)."""
    try:
        return proc(images=[pil], return_tensors="pt")["pixel_values"]
    except (TypeError, ValueError):
        return proc(images=[pil], text="a photo", return_tensors="pt")["pixel_values"]


def _save_footprint(fp_dir: Path, rows_l: list, cols_l: list, vals_l: list, n_rows: int, d_sae: int):
    """Column-normalised CSC (latent-major) of the per-image peak |a|."""
    r = np.concatenate(rows_l) if rows_l else np.zeros(0, np.int32)
    c = np.concatenate(cols_l) if cols_l else np.zeros(0, np.int32)
    v = np.concatenate(vals_l).astype(np.float32) if vals_l else np.zeros(0, np.float32)
    order = np.lexsort((r, c))
    r, c, v = r[order], c[order], v[order]
    indptr = np.zeros(d_sae + 1, dtype=np.int64)
    np.cumsum(np.bincount(c, minlength=d_sae), out=indptr[1:])
    norm = np.sqrt(np.bincount(c, weights=v.astype(np.float64) ** 2, minlength=d_sae))
    v = (v / np.maximum(norm[c], 1e-12)).astype(np.float16)
    fp_dir.mkdir(parents=True, exist_ok=True)
    np.save(fp_dir / "indptr.npy", indptr)
    np.save(fp_dir / "indices.npy", r.astype(np.int32))
    np.save(fp_dir / "data.npy", v)
    (fp_dir / "meta.json").write_text(json.dumps({"n_rows": int(n_rows), "d_sae": int(d_sae)}))


def footprint_null(fp_dir: Path, n_pairs: int = NULL_PAIRS, seed: int = 0) -> dict:
    """Median and p95 footprint cos of random latent pairs within one dictionary."""
    from scipy.sparse import csc_matrix  # noqa: PLC0415

    meta = json.loads((fp_dir / "meta.json").read_text())
    ip, ix = np.load(fp_dir / "indptr.npy"), np.load(fp_dir / "indices.npy")
    dt = np.load(fp_dir / "data.npy").astype(np.float32)
    X = csc_matrix((dt, ix, ip), shape=(meta["n_rows"], meta["d_sae"]))
    alive = np.flatnonzero(np.diff(ip) > 0)
    if alive.size < 2:
        return {"median": None, "p95": None, "n_pairs": 0}
    rng = np.random.default_rng(seed)
    a, b = rng.choice(alive, n_pairs), rng.choice(alive, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    cos = np.asarray(X[:, a].multiply(X[:, b]).sum(0)).ravel()
    return {"median": round(float(np.median(cos)), 4), "p95": round(float(np.percentile(cos, 95)), 4),
            "n_pairs": int(cos.size)}


def vision_tops(cells_by_arch: dict, root: Path, bb: lay.BackboneSpec, *, per_class: int | None,
                batch_size: int = 64, top_k: int = 100, device: str = "cuda",
                footprints: bool = True, num_workers: int = 6, force: bool = False) -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    from consdl.data.vision_backbones import Backbone  # noqa: PLC0415
    from consdl.data.vision_datasets import load_image_split  # noqa: PLC0415

    views, fpd = root / "views", root / "footprint"
    views.mkdir(parents=True, exist_ok=True)
    cells = _todo(cells_by_arch, views, force)
    if not cells:
        _log("vision tops: nothing to do")
        return
    rows = image_rows(per_class)
    if footprints:
        fpd.mkdir(parents=True, exist_ok=True)
        f = fpd / "images.npy"
        if f.is_file() and not np.array_equal(np.load(f), rows):
            raise SystemExit(f"{f} holds a different image pool; pass the same --images-per-class "
                             f"or delete {fpd}")
        np.save(f, rows)
    t0 = time.time()
    ds = load_image_split("imagenet", "val")
    proc = AutoProcessor.from_pretrained(bb.hf_name)
    backbone = Backbone(model_hf_name=bb.hf_name, hook_target_name=bb.hook, device=device)
    saes = _load_all(cells, device)
    _log(f"vision tops: {len(saes)} SAEs, {len(rows)} images  (load {time.time() - t0:.0f}s)")
    tk = [None] * len(saes)
    fp = [([], [], []) for _ in saes]
    pos_of = np.full(int(rows.max()) + 1, -1, dtype=np.int32)       # val row -> pool position
    pos_of[rows] = np.arange(len(rows), dtype=np.int32)
    loader = torch.utils.data.DataLoader(_Images(ds, rows, proc), batch_size=batch_size,
                                         num_workers=num_workers, shuffle=False)
    done = 0
    for px, rr in loader:
        with torch.no_grad():
            z = backbone.forward_to_z({"pixel_values": px.to(device)})
            z = z.reshape(px.shape[0], -1, z.shape[-1])
            T = z.shape[1]
            flat = z.reshape(-1, z.shape[-1])
            item = rr.to(device).long()
            for n, (_c, _s, sae) in enumerate(saes):
                a = encode(sae, flat).reshape(px.shape[0], T, -1)
                peak = a[:, 1:].abs().amax(1).float()                     # (B, d_sae), CLS dropped
                if tk[n] is None:
                    tk[n] = _TopK(peak.shape[1], top_k, device)
                tk[n].add(peak, item)
                if footprints:
                    b, lat = torch.nonzero(peak > 0, as_tuple=True)
                    fp[n][0].append(pos_of[rr.numpy()[b.cpu().numpy()]])
                    fp[n][1].append(lat.cpu().numpy().astype(np.int32))
                    fp[n][2].append(peak[b, lat].cpu().numpy().astype(np.float16))
        done += px.shape[0]
        if done % (batch_size * 20) == 0 or done == len(rows):
            _log(f"  {done}/{len(rows)} images  {time.time() - t0:.0f}s")
    by_cell: dict = {}
    for n, (c, s, _sae) in enumerate(saes):
        idx, val = tk[n].result()
        by_cell.setdefault((c.arch, c.name), {})[f"ids_{s}"] = idx.astype(np.int32)
        by_cell[(c.arch, c.name)][f"vals_{s}"] = val.astype(np.float32)
        if footprints:
            d = fpd / f"{c.arch}__{c.name}__{s}"
            _save_footprint(d, *fp[n], n_rows=len(rows), d_sae=idx.shape[0])
            fp[n] = None
    for (arch, cell), arrs in by_cell.items():
        np.savez_compressed(views / f"tops__{arch}__{cell}.npz", n_inputs=np.array(len(rows)),
                            ranking=np.array("peak |a| over patch tokens per image"), **arrs)
    if footprints:
        update_null(root)
    _log(f"vision tops done  {time.time() - t0:.0f}s")


def update_null(root: Path) -> None:
    """``footprint/null.json``: ``{"<arch>/<cell>": {median, p95}}`` from each
    cell's first seed."""
    fpd = root / "footprint"
    f = fpd / "null.json"
    null = json.loads(f.read_text()) if f.is_file() else {}
    firsts: dict = {}
    for d in sorted(fpd.glob("*__*__*")):
        arch, cell, seed = d.name.split("__")
        if (d / "meta.json").is_file():
            firsts.setdefault(f"{arch}/{cell}", d)
    for k, d in firsts.items():
        if k not in null:
            null[k] = footprint_null(d)
    f.write_text(json.dumps(null, indent=1))


# ── language ──────────────────────────────────────────────────────────


def lm_stream(root: Path, bb: lay.BackboneSpec, n_seqs: int) -> np.ndarray:
    """``(n_seqs, 1024)`` packed FineWeb token ids, built once per backbone."""
    views = root / "views"
    views.mkdir(parents=True, exist_ok=True)
    f, meta_f = views / "lm_stream.npy", views / "lm_stream.json"
    if f.is_file():
        ids = np.load(f, mmap_mode="r")
        if ids.shape[0] < n_seqs:
            raise SystemExit(f"{f} has {ids.shape[0]} sequences < {n_seqs}; delete it to rebuild")
        return ids
    from transformers import AutoTokenizer  # noqa: PLC0415

    from consdl.cli.train_lm import build_lm_dataloader  # noqa: PLC0415

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(bb.hf_name)
    loader = build_lm_dataloader(FINEWEB["hf_dataset"], FINEWEB["hf_config"], FINEWEB["split"],
                                 FINEWEB["text_field"], tok, CONTEXT, 32, seed=0)
    rows = []
    for batch in loader:
        rows.append(batch["input_ids"].numpy().astype(np.int32))
        if sum(len(r) for r in rows) >= n_seqs:
            break
    ids = np.concatenate(rows)[:n_seqs]
    tmp = views / f"lm_stream.{os.getpid()}.npy"
    np.save(tmp, ids)
    try:                       # first writer wins: concurrent jobs all read one stream
        os.link(tmp, f)
        meta_f.write_text(json.dumps({**FINEWEB, "context_length": CONTEXT, "stream_seed": 0,
                                      "n_seqs": int(len(ids)), "model_hf_name": bb.hf_name,
                                      "hook_target_name": bb.hook,
                                      "tokenizer_hf_name": bb.hf_name}, indent=1))
        _log(f"lm stream {ids.shape} -> {f}  {time.time() - t0:.0f}s")
    except FileExistsError:
        _log(f"lm stream: another job wrote {f} first; using it")
    finally:
        tmp.unlink(missing_ok=True)
    return np.load(f, mmap_mode="r")


def lm_tops(cells_by_arch: dict, root: Path, bb: lay.BackboneSpec, *, n_seqs: int,
            rows_per_batch: int = 8, top_k: int = 100, device: str = "cuda",
            force: bool = False) -> None:
    import torch  # noqa: PLC0415

    from consdl.cli.train_lm import LMBackbone  # noqa: PLC0415

    views = root / "views"
    ids = lm_stream(root, bb, n_seqs)[:n_seqs]
    cells = _todo(cells_by_arch, views, force)
    if not cells:
        _log("lm tops: nothing to do")
        return
    t0 = time.time()
    backbone = LMBackbone(model_hf_name=bb.hf_name, hook_layer_path=bb.hook, device=device)
    saes = _load_all(cells, device)
    _log(f"lm tops: {len(saes)} SAEs, {len(ids)} x {ids.shape[1]} tokens  (load {time.time() - t0:.0f}s)")
    T = ids.shape[1]
    tk = [None] * len(saes)
    for lo in range(0, len(ids), rows_per_batch):
        x = torch.from_numpy(np.asarray(ids[lo:lo + rows_per_batch], dtype=np.int64))
        with torch.no_grad():
            z = backbone.forward_to_z({"input_ids": x, "attention_mask": torch.ones_like(x)}).float()
            item = torch.arange(lo * T, lo * T + z.shape[0], device=device)
            for n, (_c, _s, sae) in enumerate(saes):
                score = encode(sae, z).abs().float()
                if tk[n] is None:
                    tk[n] = _TopK(score.shape[1], top_k, device)
                tk[n].add(score, item)
        if (lo // rows_per_batch) % 16 == 0:
            _log(f"  {lo + len(x)}/{len(ids)} rows  {time.time() - t0:.0f}s")
    by_cell: dict = {}
    for n, (c, s, _sae) in enumerate(saes):
        idx, val = tk[n].result()
        seq = np.where(idx >= 0, idx // T, -1).astype(np.int32)
        pos = np.where(idx >= 0, idx % T, -1).astype(np.int32)
        d = by_cell.setdefault((c.arch, c.name), {})
        d[f"ids_{s}"], d[f"pos_{s}"], d[f"vals_{s}"] = seq, pos, val.astype(np.float32)
    for (arch, cell), arrs in by_cell.items():
        np.savez_compressed(views / f"tops__{arch}__{cell}.npz", n_inputs=np.array(len(ids)),
                            ranking=np.array("|a| per token"), **arrs)
    _log(f"lm tops done  {time.time() - t0:.0f}s")

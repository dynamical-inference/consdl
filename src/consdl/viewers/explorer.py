"""Latent explorer: one input on one side, the dictionary UMAP on the other,
linked on hover. Serves ``/explore`` and the cluster-consistency view
``/view`` (``consdl.viewers.umap_view``) for one backbone.

Pick an ImageNet val image or upload one (vision), or type text / open a
FineWeb window (LM). Every panel of a precomputed view (or a subset; one
panel = one SAE) encodes it, and the page links both directions:

  input -> dictionary   hover a patch / token: its active latents are ringed
                        in every panel's UMAP, sized by |a|.
  dictionary -> input   hover a latent: the input becomes its heatmap, and its
                        matched partner in every other panel is ringed and
                        shown on that panel's input.

The input is coloured by the anchor-cluster hue of each patch's strongest
latent, by a region carve at k (``consdl.viewers.regions``), or by the
selected latent's heatmap. Codes are encoded at batch 1; support is
relative, ``|a| >= eps * max`` per patch / token.

    python -m consdl.viewers.explorer --root var/results/viewers/vision/clip-vit-b-16_layer10 --port 8761
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import time
import traceback
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from consdl.viewers import html_explorer as hle
from consdl.viewers import regions as rg
from consdl.viewers import umap_view as uv
from consdl.viewers.models import encode

EPS_REL = rg.EPS_REL
K_SWEEP = rg.K_RANGE
MAX_TOKENS = 1024                # one stream row; longer text is refused, not truncated
MAX_TEXT_BYTES = 64 * 1024
_CODES = uv.LRU(64)              # (step, input) -> (T, d_sae) float32, signed (CLS kept for images)
_REGIONS = uv.LRU(64)
_PNG = uv.LRU(512)
_TEXT = uv.LRU(64)               # text token -> record
_TZ = uv.LRU(64)                 # text token -> (T, d_in), CPU

CAVEATS = {
    "isae": "iSAE encodes in a different space from the other archs and its atom sign is "
            "unidentified: strengths are |a|, matches use |cos|.",
    "mp": "MP: encoded with a fixed number of matching-pursuit rounds = this cell's target L0.",
}


def _img_key(raw: str):
    """A val index, an upload token (``u_…``) or a text token (``t_…``)."""
    s = str(raw)
    return s if s.startswith(("t_", "u_")) else int(s)


def _is_text(img) -> bool:
    return isinstance(img, str) and img.startswith("t_")


def _caveat(key: str) -> str | None:
    return CAVEATS.get(uv.STATE["keys"].get(key, {}).get("arch", ""))


def grid_side(n_patches: int) -> int:
    side = math.isqrt(int(n_patches))
    if side * side != n_patches:
        raise ValueError(f"{n_patches} patch tokens is not a square grid (registers?)")
    return side


def pack_csr(A: np.ndarray, *, eps: float) -> dict:
    """CSR over the relative support ``|a| >= eps * max_row |a|``; values
    stay signed (float16), base64."""
    A = np.asarray(A, dtype=np.float32)
    ab = np.abs(A)
    mx = ab.max(axis=1, keepdims=True)
    keep = (ab >= float(eps) * mx) & (mx > 0)
    rows, cols = np.nonzero(keep)
    ptr = np.zeros(A.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=A.shape[0]), out=ptr[1:])
    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()  # noqa: E731
    return {"ptr": b64(ptr), "lat": b64(cols.astype(np.int32)),
            "val": b64(A[rows, cols].astype(np.float16)),
            "rowmax": np.round(mx[:, 0].astype(np.float64), 5).tolist(),
            "nnz": int(rows.size)}


def _panels(view_id: str, keys: str | None) -> list[str]:
    all_keys = uv.view_keys(view_id)
    if not keys:
        return all_keys
    want = [k for k in keys.split(",") if k]
    bad = [k for k in want if k not in all_keys]
    if bad:
        raise KeyError(f"not panels of {view_id}: {bad}")
    return want


# ── images ────────────────────────────────────────────────────────────


def codes(key: str, img) -> np.ndarray:
    """``(T, d_sae)`` signed codes of one panel on one image, CLS as row 0."""
    sae = uv.sae_for(key)
    ck = (uv.STATE["keys"][key]["step"], img)
    with uv.LOAD:
        hit = _CODES.get(ck)
    if hit is not None:
        return hit
    with uv.GPU:
        a = encode(sae, uv.z_for(img)).float().cpu().numpy()
    with uv.LOAD:
        _CODES.put(ck, a)
    return a


def add_upload(raw: bytes, *, name: str = "") -> str:
    """Decode, preprocess once with the backbone's processor, store under a token."""
    from PIL import Image  # noqa: PLC0415

    from consdl.viewers.tops import pixel_values  # noqa: PLC0415

    pil = Image.open(io.BytesIO(raw))
    pil.load()
    pil = pil.convert("RGB")
    px = pixel_values(uv.STATE["proc"], pil)
    tok = "u_" + hashlib.sha1(raw).hexdigest()[:16]
    if len(uv.UPLOADS) > uv.UPLOADS_MAX:
        uv.UPLOADS.clear()
        with uv.LOAD:
            uv._Z.clear()
            _CODES.clear()
            _REGIONS.clear()
    uv.UPLOADS[tok] = {"px": px, "crop": uv.denormalise(px), "name": name,
                       "size": f"{pil.width}x{pil.height}"}
    return tok


def crop_jpeg(img, size: int) -> bytes:
    ck = (img, size)
    with uv.LOAD:
        hit = _PNG.get(ck)
    if hit is not None:
        return hit
    b = uv.jpeg(uv.crop(img), size)
    with uv.LOAD:
        _PNG.put(ck, b)
    return b


def classes() -> dict:
    names = uv.STATE["names"]
    return {"classes": [{"cls": int(c), "name": names[c] if c < len(names) else str(c)}
                        for c in np.unique(uv.STATE["labels"])]}


def images_of(cls: int, n: int = 48) -> dict:
    idx = np.flatnonzero(uv.STATE["labels"] == int(cls))[:n]
    return {"cls": int(cls), "images": idx.astype(int).tolist()}


# ── text inputs (LM) ──────────────────────────────────────────────────
#
# An input is a sequence of token ids, resolved once and encoded by every
# panel: free text (POST /api/text, held in memory) or a window of the
# backbone's FineWeb stream (``t_s<row>_<lo>_<hi>``, rebuilt on demand).
# A window is forwarded as its full row and then sliced: the LM is causal.


def _decode_each(ids) -> list[str]:
    tok = uv.STATE["tok"]
    return [tok.decode([int(t)]) for t in ids]


def text_input(text: str, *, name: str = "") -> str:
    ids = np.asarray(uv.STATE["tok"](text)["input_ids"], dtype=np.int64)
    if ids.size == 0:
        raise ValueError("the text tokenises to nothing")
    if ids.size > MAX_TOKENS:
        raise ValueError(f"{ids.size} tokens; the limit is {MAX_TOKENS}")
    tok = "t_" + hashlib.sha1(ids.tobytes()).hexdigest()[:16]
    rec = {"kind": "text", "ids": ids, "first": 0,
           "name": name or (text[:60] + ("…" if len(text) > 60 else "")),
           "tokens": _decode_each(ids)}
    with uv.LOAD:
        _TEXT.put(tok, rec)
    return tok


def stats_input(seq: int, lo: int, hi: int) -> str:
    """A window ``[lo, hi)`` of row ``seq`` of the stream."""
    ids = uv.STATE.get("ids")
    if ids is None:
        raise ValueError("no token stream: run viewer_precompute --stages tops for this backbone")
    seq = int(seq)
    if not 0 <= seq < len(ids):
        raise ValueError(f"row {seq} outside 0..{len(ids) - 1}")
    T = ids.shape[1]
    lo, hi = max(0, int(lo)), min(T, int(hi))
    if hi - lo < 2:
        raise ValueError(f"window [{lo}, {hi}) has fewer than 2 tokens")
    tok = f"t_s{seq}_{lo}_{hi}"
    with uv.LOAD:
        if tok in _TEXT:
            return tok
    row = np.asarray(ids[seq], dtype=np.int64)
    rec = {"kind": "stats", "ids": row[lo:hi], "seq": seq, "lo": lo, "hi": hi, "first": lo,
           "name": f"FineWeb row {seq} [{lo}, {hi})", "tokens": _decode_each(row[lo:hi])}
    with uv.LOAD:
        _TEXT.put(tok, rec)
    return tok


def _text_rec(tok: str) -> dict:
    with uv.LOAD:
        rec = _TEXT.get(tok)
    if rec is not None:
        return rec
    if tok.startswith("t_s"):                 # stream windows rebuild from their name
        seq, lo, hi = (int(x) for x in tok[3:].split("@", 1)[0].split("_"))
        return _TEXT[stats_input(seq, lo, hi)]
    raise KeyError(f"text input {tok!r} (typed text is held in memory only; enter it again)")


def _text_z(tok: str):
    with uv.LOAD:
        hit = _TZ.get(tok)
    if hit is not None:
        return hit
    rec = _text_rec(tok)
    if rec["kind"] == "stats":
        z = uv.z_rows([rec["seq"]])[rec["seq"]][rec["lo"]:rec["hi"]]
    else:
        z = uv.lm_forward(rec["ids"])
    with uv.LOAD:
        _TZ.put(tok, z)
    return z


def text_codes(key: str, tok: str) -> np.ndarray:
    ck = (uv.STATE["keys"][key]["step"], tok)
    with uv.LOAD:
        hit = _CODES.get(ck)
    if hit is not None:
        return hit
    A = uv.encode_cpu(key, _text_z(tok)).numpy().astype(np.float32)
    with uv.LOAD:
        _CODES.put(ck, A)
    return A


def _text_payload(view_id: str, keys: str | None, tok: str, *, eps: float) -> dict:
    rec = _text_rec(tok)
    out = {"img": tok, "eps": eps, "panels": {}, "tokens": rec["tokens"],
           "exclude_first": rec["first"] == 0,
           "input": {k: rec.get(k) for k in ("kind", "name", "seq", "lo", "hi", "first")},
           "meta": {"support": f"|a| >= {eps:g} * max over latents, per token",
                    "backbone": uv.STATE["backbone"], "cls": "none (text)"}}
    for k in _panels(view_id, keys):
        A = text_codes(k, tok)
        d = pack_csr(A, eps=eps)
        notop = None if uv.tops(k) is not None else "No top contexts computed for this checkpoint."
        cav = " ".join(c for c in (_caveat(k), notop) if c)
        d.update(P=int(A.shape[0]), side=None, d_sae=int(A.shape[1]), caveat=cav or None)
        out["panels"][k] = d
    return out


def _text_regions(view_id: str, keys: str | None, tok: str, *, contiguous: bool) -> dict:
    ks = _panels(view_id, keys)
    ck = (view_id, tuple(ks), tok, contiguous)
    with uv.LOAD:
        hit = _REGIONS.get(ck)
    if hit is not None:
        return hit
    rec = _text_rec(tok)
    r = rg.token_labels([text_codes(k, tok) for k in ks], ks=K_SWEEP,
                        contiguous=contiguous, exclude_first=rec["first"] == 0)
    out = {"img": tok, "ks": r["ks"], "side": None, "contiguous": r["contiguous"],
           "exclude_first": r["exclude_first"],
           "panels": {k: {"labels": r["labels"][n], "colors": r["colors"][n]} for n, k in enumerate(ks)},
           "ari": {kk: {f"{ks[int(p.split('|')[0])]}|{ks[int(p.split('|')[1])]}": v
                        for p, v in pairs.items()} for kk, pairs in r["ari"].items()}}
    with uv.LOAD:
        _REGIONS.put(ck, out)
    return out


# ── payloads ──────────────────────────────────────────────────────────


def codes_payload(view_id: str, keys: str | None, img, *, eps: float) -> dict:
    if _is_text(img):
        return _text_payload(view_id, keys, img, eps=eps)
    out = {"img": str(img), "eps": eps, "panels": {},
           "meta": {"batch": 1, "support": f"|a| >= {eps:g} * max over latents, per patch",
                    "backbone": uv.STATE["backbone"], "cls": "dropped"}}
    for k in _panels(view_id, keys):
        A = codes(k, img)
        P = A.shape[0] - 1
        d = pack_csr(A[1:], eps=eps)
        notop = None if uv.tops(k) is not None else "No top images computed for this checkpoint."
        cav = " ".join(c for c in (_caveat(k), notop) if c)
        d.update(P=P, side=grid_side(P), d_sae=int(A.shape[1]), caveat=cav or None)
        out["panels"][k] = d
    return out


def regions_payload(view_id: str, keys: str | None, img, *, contiguous: bool = False) -> dict:
    """Per-panel region carve at every k, relabelled onto panel 0 (display
    only); colours encode where a region sits."""
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    if _is_text(img):
        return _text_regions(view_id, keys, img, contiguous=contiguous)
    ks = _panels(view_id, keys)
    ck = (view_id, tuple(ks), img)
    with uv.LOAD:
        hit = _REGIONS.get(ck)
    if hit is not None:
        return hit
    raw = rg.patch_labels([codes(k, img) for k in ks], ks=K_SWEEP)
    side = grid_side(raw[0][K_SWEEP[0]].size)
    out = {"img": str(img), "ks": list(K_SWEEP), "side": side, "panels": {}, "ari": {}}
    for kk in K_SWEEP:
        ref = raw[0][kk]
        for n, key in enumerate(ks):
            lab = raw[n][kk] if n == 0 else rg.match_to_reference(raw[n][kk], ref)
            p = out["panels"].setdefault(key, {"labels": {}, "colors": {}})
            p["labels"][kk] = np.asarray(lab, dtype=np.int64).tolist()
            p["colors"][kk] = rg.patch_colors(lab, side=side)
        out["ari"][kk] = {f"{ks[i]}|{ks[j]}": round(float(adjusted_rand_score(raw[i][kk], raw[j][kk])), 4)
                          for i in range(len(ks)) for j in range(i + 1, len(ks))}
    with uv.LOAD:
        _REGIONS.put(ck, out)
    return out


# ── HTTP ──────────────────────────────────────────────────────────────


class Handler(uv.Handler):
    """``/explore`` and its API first; everything else is the UMAP view's."""

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        if u.path not in ("/api/upload", "/api/text"):
            self._json({"error": "not found"}, 404)
            return
        limit = uv.MAX_UPLOAD if u.path == "/api/upload" else MAX_TEXT_BYTES
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                self._json({"error": "empty request"}, 400)
                return
            if n > limit:             # checked before the body is read
                self._json({"error": f"body is {n / 1e3:.0f} kB; limit {limit / 1e3:.0f} kB"}, 400)
                return
            body = json.loads(self.rfile.read(n))
            text = uv.STATE["modality"] == "text"
            if u.path == "/api/text":
                if not text:
                    self._json({"error": "this server encodes images"}, 400)
                    return
                try:
                    tok = text_input(str(body.get("text", "")), name=str(body.get("name", ""))[:120])
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
                    return
                self._json({"token": tok, "n_tokens": int(_text_rec(tok)["ids"].size)})
                return
            if text:
                self._json({"error": "this server encodes text; send it to /api/text"}, 400)
                return
            uri = str(body.get("image", ""))
            if "," not in uri or not uri.startswith("data:"):
                self._json({"error": "expected {image: 'data:image/...;base64,...'}"}, 400)
                return
            try:
                tok = add_upload(base64.b64decode(uri.split(",", 1)[1]),
                                 name=str(body.get("name", ""))[:120])
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"not a readable image: {type(e).__name__}: {e}"}, 400)
                return
            up = uv.UPLOADS[tok]
            self._json({"token": tok, "name": up["name"], "size": up["size"]})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._redirect(f"/explore?id={uv.STATE['default_view']}")
            elif u.path == "/explore":
                vid = self._view_id(q)
                if vid is None:
                    self._bytes(f"no view {q.get('id')!r}".encode(), "text/plain", 404)
                    return
                self._bytes(hle.page(vid, backbone=uv.STATE["label"]).encode(),
                            "text/html; charset=utf-8")
            elif u.path == "/api/codes":
                eps = float(q.get("eps", EPS_REL))
                if not 0.0 <= eps < 1.0:
                    raise ValueError(f"eps {eps} outside [0, 1)")
                self._json(codes_payload(q["id"], q.get("keys"), _img_key(q["img"]), eps=eps))
            elif u.path == "/api/regions":
                self._json(regions_payload(q["id"], q.get("keys"), _img_key(q["img"]),
                                           contiguous=q.get("contiguous", "0") == "1"))
            elif u.path == "/api/stats_ctx":
                self._json({"token": stats_input(int(q["seq"]), int(q.get("lo", 0)),
                                                 int(q.get("hi", 96)))})
            elif u.path == "/api/random_ctx":
                ids = uv.STATE.get("ids")
                if ids is None:
                    raise ValueError("no token stream: run viewer_precompute --stages tops")
                n = max(2, min(MAX_TOKENS, int(q.get("len", 96))))
                rng = np.random.default_rng()
                seq = int(rng.integers(len(ids)))
                lo = int(rng.integers(0, max(1, ids.shape[1] - n + 1)))
                self._json({"token": stats_input(seq, lo, lo + n)})
            elif u.path == "/api/n_streams":
                ids = uv.STATE.get("ids")
                self._json({"n": 0 if ids is None else int(len(ids))})
            elif u.path == "/api/crop":
                self._bytes(crop_jpeg(_img_key(q["img"]), int(q.get("s", 224))), "image/jpeg",
                            cache=True)
            elif u.path == "/api/classes":
                self._json(classes())
            elif u.path == "/api/images":
                self._json(images_of(int(q["cls"])))
            elif u.path == "/api/n_images":
                self._json({"n": int(len(uv.STATE["labels"]))})
            elif not self.route_view(u.path, q):
                self._json({"error": "not found"}, 404)
        except KeyError as e:
            self._json({"error": f"unknown: {e}. If this was an upload or typed text, "
                                 f"send it again."}, 400)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(root: Path, port: int, host: str = "0.0.0.0") -> None:
    t0 = time.time()
    uv.configure(root)
    uv.warm()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[serve] {uv.STATE['backbone']}: {len(uv.STATE['views'])} views, "
          f"{len(uv.STATE['keys'])} panels on :{port}  warm {time.time() - t0:.0f}s", flush=True)
    srv.serve_forever()


def main() -> None:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="one backbone's precompute dir (has backbone.json and views/)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    serve(a.root, a.port, a.host)


if __name__ == "__main__":
    main()

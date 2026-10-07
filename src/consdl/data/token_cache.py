"""Pre-tokenised LM token cache (uint16 memmap).

Shuffling a streaming dataset by seed changes the shard order, so with a
token budget far smaller than the corpus, different seeds would read
different documents and confound initialisation with data. This module
fixes the token stream once:

* tokenise and EOS-pack ``n_tokens`` from the stream, as
  ``build_lm_dataloader`` does, into a flat ``uint16`` memmap (asserted
  lossless for the vocabulary at build time);
* every run memory-maps that file, and ``seed`` only permutes the order
  of the fixed context-length sequences.

A build is skipped when a complete cache for the same
(dataset, config, split, tokenizer, context_length, n_tokens) exists; a
partial file from an interrupted build is detected via the sidecar
metadata and rebuilt.
"""
from __future__ import annotations


import json
import os
from pathlib import Path


import numpy as np
import torch


DTYPE = np.uint16
_VOCAB_LIMIT = 1 << 16



def cache_paths(cache_dir: str | Path, *, hf_dataset: str, hf_config: str | None,
                split: str, tokenizer_name: str, context_length: int,
                n_tokens: int) -> tuple[Path, Path]:
    """``(bin_path, meta_path)`` for one (data, tokenizer, shape) combo."""
    safe = lambda s: str(s).replace("/", "_")  # noqa: E731
    stem = (f"{safe(hf_dataset)}_{safe(hf_config or 'default')}_{safe(split)}"
            f"_{safe(tokenizer_name)}_ctx{int(context_length)}"
            f"_n{int(n_tokens)}")
    root = Path(cache_dir)
    return root / f"{stem}.u16.bin", root / f"{stem}.json"



def is_complete(bin_path: Path, meta_path: Path, n_tokens: int) -> bool:
    """True only if the sidecar says the build finished and the size matches."""
    if not (bin_path.is_file() and meta_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return False
    if not meta.get("complete") or int(meta.get("n_tokens", -1)) != int(n_tokens):
        return False
    return bin_path.stat().st_size == int(n_tokens) * np.dtype(DTYPE).itemsize



def build(cache_dir: str | Path, *, hf_dataset: str, hf_config: str | None,
          split: str, text_field: str, tokenizer, context_length: int,
          n_tokens: int, verbose: bool = True) -> Path:
    """Tokenise ``n_tokens`` from the stream into a uint16 memmap.


    Packing matches ``build_lm_dataloader``: documents concatenated with
    EOS between them, no padding, no document-boundary handling. No
    shuffle: the stream is taken in its natural order so the cache is a
    deterministic function of (dataset, split, tokenizer, n_tokens).
    """
    import datasets  # noqa: PLC0415


    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer lacks an EOS token — sequence packing needs one.")
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab >= _VOCAB_LIMIT:
        raise ValueError(
            f"vocab_size {vocab} >= {_VOCAB_LIMIT}: uint16 would truncate token "
            f"ids. Widen DTYPE before using this cache with {tokenizer}.",
        )


    bin_path, meta_path = cache_paths(
        cache_dir, hf_dataset=hf_dataset, hf_config=hf_config, split=split,
        tokenizer_name=getattr(tokenizer, "name_or_path", "tok"),
        context_length=context_length, n_tokens=n_tokens,
    )
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(bin_path, meta_path, n_tokens):
        if verbose:
            print(f"[tokcache] reuse {bin_path} "
                  f"({bin_path.stat().st_size / 1e9:.2f} GB)")
        return bin_path


    # Concurrent jobs may request the same cache: the first builds it, the
    # rest wait on the lock and then reuse it.
    from filelock import FileLock  # noqa: PLC0415


    with FileLock(str(bin_path) + ".lock"):
        if is_complete(bin_path, meta_path, n_tokens):
            if verbose:
                print(f"[tokcache] reuse {bin_path} (built by a sibling job)")
            return bin_path
        if verbose:
            print(f"[tokcache] building {n_tokens:,} tokens → {bin_path} "
                  f"({n_tokens * 2 / 1e9:.2f} GB)")
        meta_path.write_text(json.dumps({"complete": False, "n_tokens": n_tokens}))


        ds = datasets.load_dataset(hf_dataset, name=hf_config, split=split,
                                   streaming=True)
        eos_id = int(tokenizer.eos_token_id)
        tmp = bin_path.with_suffix(".bin.tmp")
        arr = np.memmap(tmp, dtype=DTYPE, mode="w+", shape=(int(n_tokens),))
        filled = 0
        try:
            from tqdm import tqdm  # noqa: PLC0415
            pbar = tqdm(total=int(n_tokens), unit="tok", unit_scale=True,
                        desc="tokenising", disable=not verbose)
            for row in ds:
                text = row.get(text_field)
                if not text:
                    continue
                ids = tokenizer(text, add_special_tokens=False).input_ids
                ids.append(eos_id)
                take = min(len(ids), int(n_tokens) - filled)
                arr[filled:filled + take] = np.asarray(ids[:take], dtype=DTYPE)
                filled += take
                pbar.update(take)
                if filled >= int(n_tokens):
                    break
            pbar.close()
        finally:
            arr.flush()
            del arr
        if filled < int(n_tokens):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"stream exhausted after {filled:,} of {n_tokens:,} tokens — "
                f"pick a smaller budget or a larger split",
            )
        os.replace(tmp, bin_path)   # atomic: no half file is ever "complete"
        meta_path.write_text(json.dumps({
            "complete": True, "n_tokens": int(n_tokens),
            "hf_dataset": hf_dataset, "hf_config": hf_config, "split": split,
            "text_field": text_field, "context_length": int(context_length),
            "tokenizer": getattr(tokenizer, "name_or_path", "tok"),
            "dtype": "uint16",
        }, indent=2))
        if verbose:
            print(f"[tokcache] done → {bin_path}")
    return bin_path



class MemmapTokenDataset(torch.utils.data.Dataset):
    """Fixed context-length sequences sliced out of the token memmap.


    Map-style so a DataLoader can shuffle it with a seeded generator:
    ``seed`` permutes the order of a fixed token set rather than choosing
    which documents are read.
    """


    def __init__(self, bin_path: str | Path, context_length: int):
        self.bin_path = str(bin_path)
        self.ctx = int(context_length)
        n = Path(self.bin_path).stat().st_size // np.dtype(DTYPE).itemsize
        self.n_seqs = int(n) // self.ctx
        self._arr: np.memmap | None = None   # opened lazily, per worker


    def __len__(self) -> int:
        return self.n_seqs


    def __getitem__(self, i: int):
        if self._arr is None:
            # Open inside the worker: a np.memmap captured before fork is
            # not safe to share across processes.
            self._arr = np.memmap(self.bin_path, dtype=DTYPE, mode="r")
        lo = int(i) * self.ctx
        chunk = np.asarray(self._arr[lo:lo + self.ctx], dtype=np.int64)
        return torch.from_numpy(chunk)



def build_dataloader(bin_path: str | Path, *, context_length: int,
                     batch_seqs: int, seed: int | None,
                     num_workers: int = 0,
                     skip_seqs: int = 0) -> torch.utils.data.DataLoader:
    """DataLoader over the memmap. ``seed`` sets the shuffle order only.


    ``skip_seqs`` drops that many sequences from the front of the shuffled
    order, so a resumed run continues the permutation an uninterrupted
    run would have used instead of replaying its start. The permutation
    is materialised and sliced (an index list is a valid sampler) to
    reproduce that exact order.
    """
    ds = MemmapTokenDataset(bin_path, context_length)
    skip_seqs = max(0, int(skip_seqs))


    def _collate(rows):
        ids = torch.stack(rows, dim=0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


    common = dict(batch_size=int(batch_seqs), collate_fn=_collate,
                  num_workers=int(num_workers), drop_last=True)


    if seed is None:
        # Unshuffled: skipping is just an offset into the natural order.
        if skip_seqs:
            return torch.utils.data.DataLoader(
                ds, sampler=range(skip_seqs, len(ds)), **common)
        return torch.utils.data.DataLoader(ds, shuffle=False, **common)


    gen = torch.Generator()
    gen.manual_seed(int(seed))
    if not skip_seqs:
        return torch.utils.data.DataLoader(
            ds, shuffle=True, generator=gen, **common)
    perm = torch.randperm(len(ds), generator=gen).tolist()
    if skip_seqs >= len(perm):
        print(f"[tokcache] skip_seqs={skip_seqs:,} >= {len(perm):,} sequences "
              f"— nothing left to serve; starting a fresh permutation")
        return torch.utils.data.DataLoader(
            ds, shuffle=True, generator=gen, **common)
    print(f"[tokcache] resuming at sequence {skip_seqs:,}/{len(perm):,} "
          f"of the seed-{seed} order")
    return torch.utils.data.DataLoader(ds, sampler=perm[skip_seqs:], **common)



__all__ = ["MemmapTokenDataset", "build", "build_dataloader", "cache_paths",
           "is_complete"]

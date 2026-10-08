"""Start the viewers: one explorer server per backbone and the hub in front.

Each backbone's server (``consdl.viewers.explorer``) loads that backbone's
model and serves two pages:

* ``/explore`` -- the latent explorer: pick an input (ImageNet val image,
  upload, typed text or a FineWeb window) and see how every SAE panel
  decomposes it, linked both ways to the panels' decoder UMAPs;
* ``/view`` -- cluster consistency: decoder atoms of several dictionaries
  (seeds, cells or architectures) on one joint UMAP, coloured by an anchor's
  k-means clusters, with cross-dictionary matches and each latent's top inputs.

The hub (``--port``) forwards to the chosen backbone and adds a bar to switch
backbone and page. Build the data first with ``consdl.cli.viewer_precompute``.

Usage::

    python -m consdl.cli.viewer --backbones clip-vit-b-16_layer10 pythia-160m_layer8 --port 8760
"""
from __future__ import annotations

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views-root", type=Path, default=None,
                   help="precompute output root (default: <repo>/var/results/viewers)")
    p.add_argument("--backbones", nargs="+", default=None, metavar="NAME",
                   help="backbone dir names or short names (default: every precomputed one)")
    p.add_argument("--port", type=int, default=8760, help="hub port")
    p.add_argument("--backend-ports", nargs="+", type=int, default=None,
                   help="one port per backbone (default: --port + 1, + 2, ...)")
    p.add_argument("--host", default="0.0.0.0", help="interface to bind")
    p.add_argument("--no-hub", action="store_true", help="only start the backbone servers")
    return p.parse_args(argv)


def _url_host() -> str:
    h = socket.gethostname()
    return h if "." in h else f"{h}.scidom.de"


def main(argv=None) -> None:
    a = parse_args(argv)
    from consdl.viewers import hub  # noqa: PLC0415
    from consdl.viewers.layout import DEFAULT_OUT_ROOT, find_backbone_dirs  # noqa: PLC0415

    root = a.views_root or DEFAULT_OUT_ROOT
    dirs = [d for d in find_backbone_dirs(root, a.backbones) if (d / "backbone.json").is_file()]
    if not dirs:
        raise SystemExit(f"no precomputed backbone under {root} matches {a.backbones}; "
                         f"run python -m consdl.cli.viewer_precompute first")
    ports = a.backend_ports or [a.port + 1 + i for i in range(len(dirs))]
    if len(ports) < len(dirs):
        raise SystemExit(f"{len(dirs)} backbones but {len(ports)} --backend-ports")

    procs: list[subprocess.Popen] = []

    def stop(*_):
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()

    atexit.register(stop)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    for d, port in zip(dirs, ports):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "consdl.viewers.explorer", "--root", str(d),
             "--port", str(port), "--host", a.host], env=os.environ.copy()))
        hub.add_backend(d.name, port)
    host = _url_host()
    print("=" * 60)
    for d, port in zip(dirs, ports):
        print(f"  {d.name:28s} http://{host}:{port}/")
    if not a.no_hub:
        print(f"  hub                          http://{host}:{a.port}/")
    print("  (servers warm up for a minute; the hub answers 502 until then)")
    print("=" * 60, flush=True)
    if a.no_hub:
        while all(pr.poll() is None for pr in procs):
            time.sleep(5)
        raise SystemExit("a backbone server exited")
    hub.serve(a.port, a.host)


if __name__ == "__main__":
    main()

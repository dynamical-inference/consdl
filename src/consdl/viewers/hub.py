"""One URL for the latent explorer and the cluster-consistency view, any backbone.

Each backbone runs its own ``consdl.viewers.explorer`` server (it holds that
backbone's model and caches, and serves ``/explore`` and ``/view``). The hub
sits in front: the chosen backbone is kept in a cookie (or ``?bb=``), every
request is forwarded to that backbone's server unchanged, and HTML pages get
a top bar to switch modality / backbone and between the two views.

    python -m consdl.viewers.hub --port 8760 \\
        --backend clip-vit-b-16_layer10=8761 --backend pythia-160m_layer8=8762
"""
from __future__ import annotations

import argparse
import html
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from consdl.viewers.layout import backbone_spec

VIEWS = {"/explore": "Latent explorer", "/view": "Cluster consistency"}
MODALITY_LABEL = {"vision": "vision", "lm": "language"}
COOKIE = "consdl_bb"
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
       "upgrade", "proxy-authenticate", "proxy-authorization", "content-length"}

#: backbone dir name -> (port, modality, label)
BACKENDS: dict[str, tuple[int, str, str]] = {}


def add_backend(name: str, port: int) -> None:
    spec = backbone_spec(name)
    mod, label = (spec.modality, spec.label) if spec else ("vision", name)
    BACKENDS[name] = (int(port), mod, label)


def _bar(bb: str, path: str, query: str) -> bytes:
    """The top bar injected into every HTML page."""
    view = path if path in VIEWS else "/explore"
    vid = parse_qs(query).get("id", [None])[0]
    mod = BACKENDS[bb][1]
    opts = "".join(
        f'<optgroup label="{MODALITY_LABEL[m]}">' + "".join(
            f'<option value="{b}"{" selected" if b == bb else ""}>{html.escape(lab)}</option>'
            for b, (_p, mm, lab) in BACKENDS.items() if mm == m)
        + "</optgroup>" for m in MODALITY_LABEL if any(v[1] == m for v in BACKENDS.values()))
    tabs = " ".join(
        f'<a href="{p}{"?" + urlencode({"id": vid}) if vid else ""}"'
        f' style="padding:3px 10px;border-radius:4px;text-decoration:none;'
        f'{"background:#333;color:#fff" if p == view else "color:#333"}">{t}</a>'
        for p, t in VIEWS.items())
    return (
        '<div id="hubbar" style="position:sticky;top:0;z-index:99999;display:flex;'
        'gap:14px;align-items:center;padding:6px 12px;background:#f3f3f3;color:#1d1d1f;'
        'border-bottom:1px solid #ccc;font:13px system-ui,sans-serif;flex:none">'
        f'<b>ConSDL viewers</b><span>modality: <b>{MODALITY_LABEL[mod]}</b></span>'
        f'<label>backbone <select onchange="location.href=\'/_hub/set?bb=\'+this.value'
        f'+\'&next={view}\'">{opts}</select></label><span>{tabs}</span></div>'
    ).encode()


class Hub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _backbone(self) -> str:
        q = parse_qs(urlparse(self.path).query).get("bb", [""])[0]
        if q in BACKENDS:
            return q
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and v in BACKENDS:
                return v
        return next(iter(BACKENDS))

    def _send(self, code: int, headers: list, body: bytes) -> None:
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, method: str) -> None:
        u = urlparse(self.path)
        if u.path == "/_hub/set":
            q = parse_qs(u.query)
            bb = q.get("bb", [""])[0]
            nxt = q.get("next", ["/explore"])[0]
            if bb not in BACKENDS or nxt not in VIEWS:
                return self._send(400, [("Content-Type", "text/plain")], b"bad backbone or view")
            return self._send(302, [("Location", nxt),
                                    ("Set-Cookie", f"{COOKIE}={bb}; Path=/; SameSite=Lax")], b"")
        if u.path == "/_hub/backends":
            import json  # noqa: PLC0415
            body = json.dumps({b: {"port": p, "modality": m, "label": lab}
                               for b, (p, m, lab) in BACKENDS.items()}).encode()
            return self._send(200, [("Content-Type", "application/json")], body)
        bb = self._backbone()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        fwd = {k: v for k, v in self.headers.items()
               if k.lower() not in HOP and k.lower() != "host"}
        conn = http.client.HTTPConnection("127.0.0.1", BACKENDS[bb][0], timeout=600)
        try:
            conn.request(method, self.path, body=body, headers=fwd)
            r = conn.getresponse()
            data = r.read()
        except OSError as e:
            return self._send(502, [("Content-Type", "text/plain")],
                              f"{bb} viewer not reachable: {e}".encode())
        finally:
            conn.close()
        headers = [(k, v) for k, v in r.getheaders() if k.lower() not in HOP]
        if "bb" in parse_qs(u.query):          # an explicit ?bb= also sticks
            headers.append(("Set-Cookie", f"{COOKIE}={bb}; Path=/; SameSite=Lax"))
        ctype = r.getheader("Content-Type") or ""
        if r.status == 200 and ctype.startswith("text/html") and b"<body" in data:
            i = data.index(b">", data.index(b"<body")) + 1
            data = data[:i] + _bar(bb, u.path, u.query) + data[i:]
        self._send(r.status, headers, data)

    def do_GET(self):  # noqa: N802
        self._proxy("GET")

    def do_POST(self):  # noqa: N802
        self._proxy("POST")


def serve(port: int, host: str = "0.0.0.0") -> None:
    print(f"[hub] :{port} -> " + ", ".join(f"{b}=:{v[0]}" for b, v in BACKENDS.items()), flush=True)
    ThreadingHTTPServer((host, port), Hub).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--backend", action="append", required=True, metavar="NAME=PORT",
                    help="a running explorer server, e.g. clip-vit-b-16_layer10=8761")
    a = ap.parse_args()
    for spec in a.backend:
        name, _, port = spec.partition("=")
        add_backend(name, int(port))
    serve(a.port)


if __name__ == "__main__":
    main()

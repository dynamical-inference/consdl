# ConSDL project page

Source of the project page for *Beyond SAEs: Consistent Concept Discovery with Sparse Coding*.

```
index.html              page content
static/css/style.css    styles (on top of Tailwind)
static/js/main.js       nav bar, lightbox, figure tabs, BibTeX copy
static/js/charts.js     ablation and results charts
static/data/results.js  numbers behind the charts (paper Tables 1 and 13–16)
static/images/          figures
```

## Preview

Open `index.html` in a browser, or serve this folder:

```bash
python3 docs/serve.py              # http://localhost:8000
python3 docs/serve.py --public     # http://<hostname>:8000, reachable from other machines
```

To share a preview before the page is public, zip `index.html` and `static/`:

```bash
cd docs && python3 -m zipfile -c ../ConSDL-preview.zip index.html static
```

## Deploy

Enable GitHub Pages: Settings → Pages → Deploy from a branch → `main` / `/docs`.

## Figures

- `consistency_lines.png`: Figure 3
- `synthetic_dgp.png`: Figure 2A
- `synthetic_recovery.png`: Figure 2B–C, cropped from the manuscript PDF
- `method.png`: Figure 1, rendered from `figures/method.pdf`
- `dictionary_cluster_consistency.png`: Figure 4, rendered from `figures/clustering.pdf`
- `region_decomposition_{clip,pythia}.jpg`: Figures 6 and 7, rendered from `figures/{clip,pythia}_qual.pdf`

Source material (`*.pdf`, `*.tex`, `figures/`, `consdl-project-content.md` at the repository root) stays local and is
git-ignored.

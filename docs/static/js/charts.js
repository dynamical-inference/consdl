// Ablation and results charts. Data comes from static/data/results.js.
document.addEventListener('DOMContentLoaded', () => {
  if (!window.Chart || !window.RESULTS) return;

  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  Chart.defaults.color = '#475569';

  // Matches the paper figures (matplotlib tab10).
  const METHOD_COLORS = {
    ReLU: '#7f7f7f', TopK: '#1f77b4', BatchTopK: '#17becf', Matryoshka: '#9467bd',
    'MP-SAE': '#8c564b', iSAE: '#2ca02c', ConSDL: '#d62728',
  };
  const COLUMNS = [
    { key: 'l0', label: 'ℓ<sub>0</sub>', fmt: 'int' },
    { key: 'r2', label: 'R²', fmt: 'frac' },
    { key: 'dcs', label: 'Dictionary consistency', fmt: 'frac' },
    { key: 'mcc', label: 'Coefficient consistency', fmt: 'frac' },
    { key: 'usable', label: 'Usable', fmt: 'frac' },
    { key: 'dup', label: '#Dup', fmt: 'int', ascending: true },
    { key: 'dead', label: '#Dead', fmt: 'int', ascending: true },
  ];
  const AVERAGE = 'Average';

  const format = (v, fmt) => (fmt === 'int'
    ? Math.round(v).toLocaleString('en-US')
    : v.toFixed(2).replace(/^0\./, '.'));
  const alpha = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  // A bar chart above a sortable table. `getRows()` returns [{ name, l0, r2, dcs, ... }].
  // With `rank`, bars and rows are ordered by the mean of the plotted columns; otherwise rows keep their order.
  function chartAndTable(root, { getRows, nameLabel, plot, colors, colorOf, highlight, ylabel, rank }) {
    root.insertAdjacentHTML('beforeend', `
      <div class="panel p-4"><div style="height:20rem"><canvas></canvas></div></div>
      <div class="panel mt-6 overflow-x-auto"><table class="data-table"><thead></thead><tbody></tbody></table></div>`);
    const plotted = COLUMNS.filter((c) => plot.includes(c.key));
    const mean = (r) => plotted.reduce((s, c) => s + r[c.key], 0) / plotted.length;
    let sort = { key: rank ? 'mean' : null, asc: false };
    let chart = null;

    const draw = () => {
      const rows = getRows();
      const ranked = rank ? rows.slice().sort((a, b) => mean(b) - mean(a)) : rows;

      const thead = root.querySelector('thead');
      thead.innerHTML = `<tr><th class="text-left">${nameLabel}</th>${COLUMNS.map((c) => `
        <th class="sortable text-right" data-key="${c.key}">${c.label}${sort.key === c.key ? (sort.asc ? ' ↑' : ' ↓') : ''}</th>`).join('')}</tr>`;
      const tableRows = !sort.key || sort.key === 'mean' ? ranked
        : rows.slice().sort((a, b) => (sort.asc ? 1 : -1) * (a[sort.key] - b[sort.key]));
      root.querySelector('tbody').innerHTML = tableRows.map((r) => `
        <tr class="${r.name === highlight ? 'row-hl' : ''}">
          <td class="text-left">${colorOf ? `<span class="dot" style="background:${colorOf(r.name)}"></span>` : ''}${r.name}</td>
          ${COLUMNS.map((c) => `<td class="text-right font-mono">${format(r[c.key], c.fmt)}</td>`).join('')}
        </tr>`).join('');

      const labels = ranked.map((r) => r.name);
      const datasets = plotted.map((c, i) => {
        const data = ranked.map((r) => r[c.key]);
        const label = c.label.replace(/<[^>]+>/g, '');
        if (!colorOf) return { label, data, backgroundColor: colors[i], borderRadius: 3 };
        // One colour per method; solid for the first metric, lighter fills for the others.
        const cols = labels.map(colorOf);
        return {
          label, data,
          backgroundColor: i === 0 ? cols : cols.map((x) => alpha(x, Math.max(0.2, 0.5 - 0.15 * (i - 1)))),
          borderColor: cols, borderWidth: i === 0 ? 0 : 1.5, borderRadius: 4,
        };
      });

      if (chart) {
        chart.data = { labels, datasets };
        chart.update();
        return;
      }
      chart = new Chart(root.querySelector('canvas'), {
        type: 'bar',
        data: { labels, datasets },
        options: {
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 1, title: { display: !!ylabel, text: ylabel }, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false } },
          },
          plugins: {
            legend: colorOf ? {
              labels: {
                generateLabels: (ch) => ch.data.datasets.map((d, i) => ({
                  text: d.label, datasetIndex: i, lineWidth: i === 0 ? 0 : 1.5,
                  fillStyle: i === 0 ? '#334155' : alpha('#334155', 0.35), strokeStyle: '#334155',
                })),
              },
            } : {},
          },
        },
      });
    };

    root.querySelector('thead').addEventListener('click', (e) => {
      const th = e.target.closest('th[data-key]');
      if (!th) return;
      const col = COLUMNS.find((c) => c.key === th.dataset.key);
      sort = sort.key === col.key ? { key: col.key, asc: !sort.asc } : { key: col.key, asc: !!col.ascending };
      draw();
    });
    return draw;
  }

  // ── Ablation (Table 1) ──
  const ablationRoot = document.getElementById('ablation-chart');
  if (ablationRoot) {
    const backbones = Object.keys(ABLATION);
    let current = backbones[0];
    ablationRoot.innerHTML = `<div class="mb-6 flex flex-wrap justify-center gap-2">${backbones.map((b, i) =>
      `<button class="tab ${i === 0 ? 'tab-active' : ''}" data-backbone="${b}">${b}</button>`).join('')}</div>`;
    const draw = chartAndTable(ablationRoot, {
      nameLabel: 'Variant',
      plot: ['r2', 'dcs', 'mcc', 'usable'],
      colors: ['#94a3b8', '#1f77b4', '#17becf', '#d62728'],
      highlight: 'Full (ConSDL)',
      getRows: () => ABLATION[current].map(([name, l0, r2, dcs, mcc, usable, dup, dead]) => ({ name, l0, r2, dcs, mcc, usable, dup, dead })),
    });
    ablationRoot.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-backbone]');
      if (!btn) return;
      current = btn.dataset.backbone;
      ablationRoot.querySelectorAll('[data-backbone]').forEach((b) => b.classList.toggle('tab-active', b === btn));
      draw();
    });
    draw();
  }

  // ── Results (Tables 13–16) ──
  const resultsRoot = document.getElementById('results-chart');
  if (resultsRoot) {
    const rows = RESULTS.map(([backbone, expansion, target, name, l0, r2, dcs, mcc, usable, dup, dead]) =>
      ({ backbone, expansion, target, name, l0, r2, dcs, mcc, usable, dup, dead }));
    const unique = (k) => [...new Set(rows.map((r) => r[k]))];
    const dims = [
      { key: 'backbone', label: 'Backbone', values: unique('backbone'), show: (v) => v },
      { key: 'expansion', label: 'Expansion', values: unique('expansion'), show: (v) => `${v}×` },
      { key: 'target', label: 'Sparsity', values: unique('target'), show: (v) => `ℓ<sub>0</sub> ≈ ${v}` },
    ];
    const selected = { backbone: AVERAGE, expansion: AVERAGE, target: AVERAGE };

    resultsRoot.innerHTML = `<div class="dim-panel">${dims.map((d) => `
      <div class="dim-row">
        <span class="dim-label">${d.label}</span>
        <div class="seg">${[AVERAGE, ...d.values].map((v) =>
          `<button class="seg-btn" data-dim="${d.key}" data-val="${v}">${v === AVERAGE ? AVERAGE : d.show(v)}</button>`).join('')}</div>
      </div>`).join('')}</div>`;

    // "Average" averages every setting that matches the other selections, with equal weight.
    const getRows = () => {
      const hits = rows.filter((r) => dims.every((d) => selected[d.key] === AVERAGE || String(r[d.key]) === selected[d.key]));
      return Object.keys(METHOD_COLORS).map((name) => {
        const own = hits.filter((r) => r.name === name);
        const avg = (k) => own.reduce((s, r) => s + r[k], 0) / own.length;
        return { name, l0: avg('l0'), r2: avg('r2'), dcs: avg('dcs'), mcc: avg('mcc'), usable: avg('usable'), dup: avg('dup'), dead: avg('dead') };
      });
    };
    const draw = chartAndTable(resultsRoot, {
      nameLabel: 'Method',
      rank: true,
      plot: ['dcs', 'mcc', 'usable'],
      colorOf: (name) => METHOD_COLORS[name] || '#94a3b8',
      highlight: 'ConSDL',
      ylabel: 'cross-seed consistency / usable fraction ↑',
      getRows,
    });
    const syncButtons = () => resultsRoot.querySelectorAll('[data-dim]').forEach((b) =>
      b.classList.toggle('seg-active', selected[b.dataset.dim] === b.dataset.val));
    resultsRoot.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-dim]');
      if (!btn) return;
      selected[btn.dataset.dim] = btn.dataset.val;
      syncButtons();
      draw();
    });
    syncButtons();
    draw();
  }
});

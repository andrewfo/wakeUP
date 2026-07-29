"""Self-contained HTML dashboard for the benchmark (Phase 6).

One command renders the whole story into a single file that opens from
``file://`` with **zero network access and zero dependencies** — the same
offline-reproducibility contract as the rest of the pipeline. All data is
embedded as JSON; all rendering is inline SVG built by vanilla JS.

Four panels:

* **Tracks** — every vessel's clean resampled track with the injected spoof
  segments overlaid in attack-family colours; click a vessel to focus it.
* **Detectors** — held-out score distributions per detector, coloured by
  attack family, next to the per-attack PR-AUC table.
* **Severity sweeps** — the degradation curves (PR-AUC vs subtlety knob,
  log-x) from ``ablation_sweeps.csv`` / ``robustness.csv`` when present.
* **Latency** — detection rate and median points-from-onset per detector and
  family from ``latency.csv`` when present.

The module is torch-free: it consumes prepared frames and artifact CSVs, so
the dashboard builds on the core install. ``scripts/run_dashboard.py``
orchestrates data generation, held-out scoring, and rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wakeUp.eval.metrics import dominant_attack_type, per_attack_metrics
from wakeUp.eval.splits import window_labels

ATTACK_COLORS = {
    "position_jump": "#ff6b6b",
    "kinematic_impossible": "#ffb020",
    "identity_swap": "#c084fc",
    "replay": "#4dd0e1",
    "gradual_drift": "#8bd450",
    "none": "#8b9bb4",
}

DETECTOR_COLORS = {
    "KinematicRule": "#8b9bb4",
    "IForest": "#5ba3e0",
    "IsolationForest": "#5ba3e0",
    "Logistic": "#8bd450",
    "ReconTransformer": "#ffb020",
    "Transformer": "#ff6b8a",
    "Hybrid": "#4dd0e1",
    "LSTM-AE": "#c084fc",
}


def _track_payload(clean_windows: pd.DataFrame, attacked: pd.DataFrame) -> list[dict]:
    """Per-vessel clean polyline + attacked spoof segments.

    The clean track is the pre-attack windows frame deduped on timestamp
    (overlapping windows repeat points, but they agree before injection).
    Spoof segments come from the attacked frame per corrupted window, so a
    segment shows exactly where the injected positions diverge from the clean
    underlay.
    """
    tracks = []
    clean_pts = (
        clean_windows.drop_duplicates(["mmsi", "timestamp"])
        .sort_values(["mmsi", "timestamp"])
    )
    seg_src = attacked[attacked["window_label"] == 1]
    for mmsi, g in clean_pts.groupby("mmsi", sort=True):
        segments = []
        vg = seg_src[seg_src["mmsi"] == mmsi]
        for _, w in vg.groupby("window_id", sort=True):
            atk = w[w["is_attack"] == 1].sort_values("point_idx")
            if len(atk) < 2:
                continue
            segments.append(
                {
                    "type": atk["attack_type"].iloc[0],
                    "pts": atk[["lat", "lon"]].round(5).to_numpy().tolist(),
                }
            )
        tracks.append(
            {
                "mmsi": int(mmsi),
                "clean": g[["lat", "lon"]].round(5).to_numpy().tolist(),
                "segments": segments,
            }
        )
    return tracks


def _score_payload(
    scores_by_detector: dict[str, np.ndarray],
    labels: np.ndarray,
    dom_types: np.ndarray,
    recall_target: float = 0.90,
) -> tuple[list[dict], list[dict]]:
    """Min-max-normalised score strips + the per-attack PR-AUC table."""
    strips, table = [], []
    for name, scores in scores_by_detector.items():
        s = np.asarray(scores, dtype=float)
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        norm = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        strips.append(
            {
                "detector": name,
                "rows": [
                    [round(float(v), 4), int(y), t]
                    for v, y, t in zip(norm, labels, dom_types)
                ],
            }
        )
        per = per_attack_metrics(s, labels, dom_types, recall_target)
        for rec in per.to_dict(orient="records"):
            table.append(
                {
                    "detector": name,
                    "attack_type": rec["attack_type"],
                    "pr_auc": None if pd.isna(rec["pr_auc"]) else round(rec["pr_auc"], 3),
                }
            )
    return strips, table


def _sweep_payload(artifacts_dir: Path) -> list[dict]:
    """Degradation curves from the richest sweep artifact available."""
    for fname in ("ablation_sweeps.csv", "robustness.csv"):
        path = artifacts_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        out = []
        for param, g in df.groupby("param", sort=False):
            series = [
                {
                    "detector": det,
                    "points": dg.sort_values("value")[["value", "pr_auc"]]
                    .round(4)
                    .to_numpy()
                    .tolist(),
                }
                for det, dg in g.groupby("detector", sort=True)
            ]
            out.append(
                {
                    "param": param,
                    "attack": g["attack_type"].iloc[0],
                    "source": fname,
                    "series": series,
                }
            )
        return out
    return []


def _latency_payload(artifacts_dir: Path) -> list[dict]:
    path = artifacts_dir / "latency.csv"
    if not path.exists():
        return []
    from wakeUp.eval.latency import summarize_latency

    summary = summarize_latency(pd.read_csv(path))
    summary = summary.replace({np.nan: None}).round(
        {"detection_rate": 3, "median_latency_points": 1, "median_latency_s": 0}
    )
    return summary.to_dict(orient="records")


def build_payload(
    clean_windows: pd.DataFrame,
    attacked: pd.DataFrame,
    score_frame: pd.DataFrame,
    scores_by_detector: dict[str, np.ndarray],
    artifacts_dir: Path,
    meta: dict | None = None,
) -> dict:
    """Assemble the embedded-JSON payload for the dashboard.

    ``score_frame`` is the (held-out) windows frame the detectors were scored
    on; labels and dominant attack types are derived here so callers cannot
    misalign them.
    """
    labels = window_labels(score_frame)
    dom = (
        dominant_attack_type(score_frame)
        .reindex(list(score_frame.groupby("window_id", sort=True).groups))
        .to_numpy()
    )
    strips, table = _score_payload(scores_by_detector, labels, dom)
    return {
        "meta": {
            "n_vessels": int(attacked["mmsi"].nunique()),
            "n_windows": int(attacked["window_id"].nunique()),
            "n_attacked": int(
                attacked.groupby("window_id")["window_label"].first().sum()
            ),
            "n_scored": int(len(labels)),
            **(meta or {}),
        },
        "attack_colors": ATTACK_COLORS,
        "detector_colors": DETECTOR_COLORS,
        "tracks": _track_payload(clean_windows, attacked),
        "strips": strips,
        "prtable": table,
        "sweeps": _sweep_payload(artifacts_dir),
        "latency": _latency_payload(artifacts_dir),
    }


def render_dashboard(payload: dict) -> str:
    """Render the payload into a single self-contained HTML document."""
    return _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False))


def write_dashboard(payload: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dashboard(payload), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# template — inline CSS/JS only; no network access, ever
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wakeUp — AIS spoofing-detection benchmark</title>
<style>
  :root {
    --bg: #0b1220; --panel: #121b2e; --panel2: #0e1626; --line: #22304a;
    --ink: #dce6f5; --dim: #8b9bb4; --accent: #5ba3e0; --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Segoe UI", Inter, sans-serif; }
  header { padding: 28px 32px 20px; border-bottom: 1px solid var(--line);
           background: linear-gradient(180deg, #101a30, var(--bg)); }
  header h1 { font-size: 22px; letter-spacing: .3px; }
  header h1 span { color: var(--accent); }
  header p { color: var(--dim); margin-top: 4px; max-width: 72ch; }
  .chips { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
  .chip { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
          padding: 6px 12px; font-family: var(--mono); font-size: 12.5px; color: var(--dim); }
  .chip b { color: var(--ink); font-weight: 600; }
  nav { position: sticky; top: 0; z-index: 5; display: flex; gap: 4px; padding: 10px 28px;
        background: rgba(11,18,32,.92); backdrop-filter: blur(6px); border-bottom: 1px solid var(--line); }
  nav a { color: var(--dim); text-decoration: none; padding: 6px 14px; border-radius: 8px; font-size: 14px; }
  nav a:hover { color: var(--ink); background: var(--panel); }
  main { padding: 26px 32px 60px; display: grid; gap: 26px; max-width: 1240px; margin: 0 auto; }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 22px 24px; }
  section > h2 { font-size: 16px; margin-bottom: 4px; }
  section > p.sub { color: var(--dim); font-size: 13.5px; margin-bottom: 16px; max-width: 90ch; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 14px; font-size: 12.5px; color: var(--dim); }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
  .tracks-grid { display: grid; grid-template-columns: 190px 1fr; gap: 18px; }
  #vessel-list { max-height: 520px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
  #vessel-list button { text-align: left; background: var(--panel2); color: var(--dim); border: 1px solid var(--line);
        border-radius: 8px; padding: 6px 10px; font-family: var(--mono); font-size: 12px; cursor: pointer; }
  #vessel-list button:hover { color: var(--ink); border-color: var(--accent); }
  #vessel-list button.active { color: var(--ink); border-color: var(--accent); background: #16233c; }
  #vessel-list button .n { float: right; color: var(--accent); }
  svg { display: block; }
  .plot-bg { fill: var(--panel2); }
  .axis line, .axis path { stroke: var(--line); }
  .axis text { fill: var(--dim); font: 11px var(--mono); }
  .grid line { stroke: #1a2740; }
  table { border-collapse: collapse; font-family: var(--mono); font-size: 12.5px; width: 100%; }
  th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 500; }
  td:first-child, th:first-child { text-align: left; color: var(--dim); }
  td.hi { color: #8bd450; font-weight: 600; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; }
  .cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .muted { color: var(--dim); font-size: 13px; padding: 18px 0; }
  #tooltip { position: fixed; pointer-events: none; background: #0a1120; border: 1px solid var(--accent);
             border-radius: 8px; padding: 6px 10px; font: 12px var(--mono); color: var(--ink);
             opacity: 0; transition: opacity .12s; z-index: 10; }
  footer { text-align: center; color: var(--dim); font-size: 12.5px; padding: 24px; }
  @media (max-width: 900px) { .cols, .cols3, .tracks-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>wake<span>Up</span> — AIS spoofing-detection benchmark</h1>
  <p>Synthetic fleet · five labeled attack families · detectors from physics rules to
     Transformers · held-out-by-vessel protocol. Fully offline: every number and pixel
     on this page is embedded in this file.</p>
  <div class="chips" id="chips"></div>
</header>
<nav>
  <a href="#tracks">Tracks</a><a href="#detectors">Detectors</a>
  <a href="#sweeps">Severity sweeps</a><a href="#latency">Latency</a>
</nav>
<main>
  <section id="tracks">
    <h2>Vessel tracks &amp; injected attacks</h2>
    <p class="sub">Grey polylines are the clean resampled tracks; coloured strokes are the
       injected spoof segments — where the reported positions diverge from truth.
       Click a vessel to focus it; click again to show the whole fleet.</p>
    <div class="legend" id="attack-legend"></div>
    <div class="tracks-grid">
      <div id="vessel-list"></div>
      <div id="map-holder"></div>
    </div>
  </section>
  <section id="detectors">
    <h2>Detector scores on held-out vessels</h2>
    <p class="sub">Each strip is one detector's normalised anomaly scores on windows from
       vessels it never trained on — clean windows grey, attacked windows coloured by
       family. Right: PR-AUC per attack family, each scored against clean windows only.</p>
    <div class="cols">
      <div id="strips"></div>
      <div id="pr-table"></div>
    </div>
  </section>
  <section id="sweeps">
    <h2>Severity sweeps — where each detector breaks</h2>
    <p class="sub">Held-out PR-AUC as each attack's subtlety knob sweeps three decades
       (log&nbsp;x, subtle&nbsp;→&nbsp;gross). The knee of each curve is the detector's
       real operating limit; headline tables sit in the saturated far right.</p>
    <div class="legend" id="sweep-legend"></div>
    <div class="cols3" id="sweep-plots"></div>
  </section>
  <section id="latency">
    <h2>Detection latency — points from onset to first alarm</h2>
    <p class="sub">Held-out windows replayed as streams at a 5% clean-window
       false-positive budget. Bars: detection rate. Label: median points from attack
       onset to the first alarm (×&nbsp;cadence&nbsp;=&nbsp;seconds).</p>
    <div id="latency-plot"></div>
  </section>
</main>
<div id="tooltip"></div>
<footer>generated by <code>scripts/run_dashboard.py</code> · seeded &amp; reproducible · MIT</footer>
<script>
const D = __PAYLOAD__;
const NS = "http://www.w3.org/2000/svg";
const $ = (s) => document.querySelector(s);
const el = (t, a = {}) => { const e = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); return e; };
const tip = $("#tooltip");
const showTip = (ev, html) => { tip.innerHTML = html; tip.style.opacity = 1;
  tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY - 10) + "px"; };
const hideTip = () => tip.style.opacity = 0;
const fmtAtk = (t) => t.replace(/_/g, " ");

/* ---------- header chips ---------- */
{
  const m = D.meta;
  const chips = [
    ["vessels", m.n_vessels], ["windows", m.n_windows],
    ["attacked windows", m.n_attacked], ["held-out scored", m.n_scored],
  ];
  if (m.seed !== undefined) chips.push(["seed", m.seed]);
  $("#chips").innerHTML = chips.map(([k, v]) => `<div class="chip">${k} <b>${v}</b></div>`).join("");
}

/* ---------- legends ---------- */
const legend = (holder, entries) => holder.innerHTML = entries
  .map(([name, c]) => `<span><i style="background:${c}"></i>${name}</span>`).join("");
legend($("#attack-legend"), Object.entries(D.attack_colors)
  .filter(([k]) => k !== "none").map(([k, c]) => [fmtAtk(k), c]));

/* ---------- tracks map ---------- */
const MAP_W = 900, MAP_H = 520;
let focused = null;
function drawMap() {
  const holder = $("#map-holder"); holder.innerHTML = "";
  const tracks = focused === null ? D.tracks : D.tracks.filter(t => t.mmsi === focused);
  // Fit the viewport to the *clean* tracks (plus segments when focused): a
  // gross spoof excursion may leave the frame, which is itself the point.
  const pts = tracks.flatMap(t => focused === null
    ? t.clean : t.clean.concat(t.segments.flatMap(s => s.pts)));
  const lats = pts.map(p => p[0]), lons = pts.map(p => p[1]);
  const la0 = Math.min(...lats), la1 = Math.max(...lats);
  const lo0 = Math.min(...lons), lo1 = Math.max(...lons);
  const cos = Math.cos(((la0 + la1) / 2) * Math.PI / 180);
  const spanX = Math.max((lo1 - lo0) * cos, 1e-6), spanY = Math.max(la1 - la0, 1e-6);
  const pad = 24;
  const sc = Math.min((MAP_W - 2 * pad) / spanX, (MAP_H - 2 * pad) / spanY);
  const X = (lon) => pad + ((lon - lo0) * cos) * sc + (MAP_W - 2 * pad - spanX * sc) / 2;
  const Y = (lat) => MAP_H - pad - (lat - la0) * sc - (MAP_H - 2 * pad - spanY * sc) / 2;
  const path = (ps) => ps.map((p, i) => (i ? "L" : "M") + X(p[1]).toFixed(1) + " " + Y(p[0]).toFixed(1)).join("");
  const svg = el("svg", { viewBox: `0 0 ${MAP_W} ${MAP_H}`, width: "100%" });
  svg.appendChild(el("rect", { width: MAP_W, height: MAP_H, rx: 10, class: "plot-bg" }));
  const clip = el("clipPath", { id: "map-clip" });
  clip.appendChild(el("rect", { width: MAP_W, height: MAP_H, rx: 10 }));
  svg.appendChild(clip);
  const layer = el("g", { "clip-path": "url(#map-clip)" });
  svg.appendChild(layer);
  for (const t of tracks) {
    const p = el("path", { d: path(t.clean), fill: "none", stroke: "#3a4a68",
      "stroke-width": focused === null ? 0.9 : 1.8,
      "stroke-opacity": focused === null ? .55 : .9 });
    p.addEventListener("mousemove", ev => showTip(ev, `MMSI ${t.mmsi} · clean track`));
    p.addEventListener("mouseleave", hideTip);
    layer.appendChild(p);
  }
  for (const t of tracks) for (const s of t.segments) {
    const p = el("path", { d: path(s.pts), fill: "none", stroke: D.attack_colors[s.type] || "#fff",
      "stroke-width": focused === null ? 1.5 : 3, "stroke-opacity": focused === null ? .8 : 1,
      "stroke-linecap": "round" });
    p.addEventListener("mousemove", ev => showTip(ev,
      `MMSI ${t.mmsi}<br><span style="color:${D.attack_colors[s.type]}">${fmtAtk(s.type)}</span> · ${s.pts.length} pts`));
    p.addEventListener("mouseleave", hideTip);
    svg.appendChild(p);
  }
  holder.appendChild(svg);
}
function drawVesselList() {
  const list = $("#vessel-list"); list.innerHTML = "";
  for (const t of D.tracks) {
    const b = document.createElement("button");
    b.innerHTML = `${t.mmsi} <span class="n">${t.segments.length ? t.segments.length + "⚠" : ""}</span>`;
    b.className = focused === t.mmsi ? "active" : "";
    b.onclick = () => { focused = focused === t.mmsi ? null : t.mmsi; drawVesselList(); drawMap(); };
    list.appendChild(b);
  }
}
drawVesselList(); drawMap();

/* ---------- score strips ---------- */
{
  const holder = $("#strips");
  const W = 560, ROW = 64;
  for (const strip of D.strips) {
    const svg = el("svg", { viewBox: `0 0 ${W} ${ROW}`, width: "100%" });
    svg.appendChild(el("rect", { y: 14, width: W, height: ROW - 22, rx: 8, class: "plot-bg" }));
    const label = el("text", { x: 2, y: 10, fill: "#dce6f5", "font-size": "12",
      "font-family": "var(--mono)" });
    label.textContent = strip.detector;
    svg.appendChild(label);
    // clean first so attacked dots draw on top; jitter is deterministic
    const rows = [...strip.rows].sort((a, b) => a[1] - b[1]);
    rows.forEach(([v, y, t], i) => {
      const jitter = ((i * 2654435761) % 1000) / 1000;
      const c = el("circle", { cx: 8 + v * (W - 16), cy: 14 + 4 + jitter * (ROW - 30),
        r: y ? 3 : 2, fill: y ? (D.attack_colors[t] || "#fff") : "#33415e",
        "fill-opacity": y ? .9 : .55 });
      c.addEventListener("mousemove", ev => showTip(ev,
        `${strip.detector}<br>score ${v.toFixed(3)} · ${y ? fmtAtk(t) : "clean"}`));
      c.addEventListener("mouseleave", hideTip);
      svg.appendChild(c);
    });
    holder.appendChild(svg);
  }
}

/* ---------- PR-AUC table ---------- */
{
  const dets = [...new Set(D.prtable.map(r => r.detector))];
  const atks = [...new Set(D.prtable.map(r => r.attack_type))]
    .filter(a => a !== "ALL").concat(["ALL"]);
  const get = (a, d) => { const r = D.prtable.find(r => r.detector === d && r.attack_type === a);
    return r && r.pr_auc !== null ? r.pr_auc : NaN; };
  let html = "<table><tr><th>PR-AUC</th>" + dets.map(d => `<th>${d}</th>`).join("") + "</tr>";
  for (const a of atks) {
    const vals = dets.map(d => get(a, d));
    const best = Math.max(...vals.filter(v => !isNaN(v)));
    html += `<tr><td>${fmtAtk(a)}</td>` + vals.map(v =>
      `<td class="${v === best ? "hi" : ""}">${isNaN(v) ? "—" : v.toFixed(2)}</td>`).join("") + "</tr>";
  }
  $("#pr-table").innerHTML = html + "</table>";
}

/* ---------- severity sweeps ---------- */
{
  const holder = $("#sweep-plots");
  if (!D.sweeps.length) holder.innerHTML = `<p class="muted">Run
    <code>scripts/run_ablation.py --sweeps</code> or <code>--robustness</code> to
    populate the degradation curves.</p>`;
  const dets = [...new Set(D.sweeps.flatMap(s => s.series.map(x => x.detector)))];
  legend($("#sweep-legend"), dets.map(d => [d, D.detector_colors[d] || "#fff"]));
  const W = 380, H = 260, m = { l: 40, r: 12, t: 26, b: 34 };
  for (const sweep of D.sweeps) {
    const vals = sweep.series.flatMap(s => s.points.map(p => p[0]));
    const x0 = Math.log10(Math.min(...vals)), x1 = Math.log10(Math.max(...vals));
    const X = (v) => m.l + (Math.log10(v) - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - v) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" });
    svg.appendChild(el("rect", { width: W, height: H, rx: 10, class: "plot-bg" }));
    const title = el("text", { x: m.l, y: 16, fill: "#dce6f5", "font-size": "12" });
    title.textContent = `${fmtAtk(sweep.attack)} — ${sweep.param}`;
    svg.appendChild(title);
    for (const gy of [0, .25, .5, .75, 1]) {
      svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: Y(gy), y2: Y(gy),
        stroke: "#1a2740" }));
      const t = el("text", { x: m.l - 6, y: Y(gy) + 3, "text-anchor": "end",
        fill: "#8b9bb4", "font-size": "10" });
      t.textContent = gy.toFixed(2); svg.appendChild(t);
    }
    for (const s of sweep.series) {
      const col = D.detector_colors[s.detector] || "#fff";
      const d = s.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("");
      svg.appendChild(el("path", { d, fill: "none", stroke: col, "stroke-width": 1.8 }));
      for (const [v, y] of s.points) {
        const c = el("circle", { cx: X(v), cy: Y(y), r: 2.6, fill: col });
        c.addEventListener("mousemove", ev => showTip(ev,
          `${s.detector}<br>${sweep.param} = ${v}<br>PR-AUC ${y.toFixed(2)}`));
        c.addEventListener("mouseleave", hideTip);
        svg.appendChild(c);
      }
    }
    const ticks = sweep.series[0].points.map(p => p[0]);
    for (const v of ticks) {
      const t = el("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle",
        fill: "#8b9bb4", "font-size": "9.5" });
      t.textContent = v < 1 ? v : v.toFixed(v % 1 ? 2 : 0); svg.appendChild(t);
    }
    holder.appendChild(svg);
  }
}

/* ---------- latency ---------- */
{
  const holder = $("#latency-plot");
  if (!D.latency.length) {
    holder.innerHTML = `<p class="muted">Run <code>scripts/run_latency.py</code> to populate.</p>`;
  } else {
    const dets = [...new Set(D.latency.map(r => r.detector))];
    const atks = [...new Set(D.latency.map(r => r.attack_type))];
    const W = 1100, BAR = 22, GAP = 6, GROUP = atks.length * (BAR + GAP) + 26;
    const H = dets.length * GROUP + 10;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" });
    dets.forEach((det, di) => {
      const y0 = di * GROUP;
      const label = el("text", { x: 0, y: y0 + 14, fill: "#dce6f5", "font-size": "12.5",
        "font-family": "var(--mono)" });
      label.textContent = det; svg.appendChild(label);
      atks.forEach((atk, ai) => {
        const r = D.latency.find(r => r.detector === det && r.attack_type === atk);
        if (!r) return;
        const y = y0 + 22 + ai * (BAR + GAP);
        const w = 180 + r.detection_rate * (W - 420);
        svg.appendChild(el("rect", { x: 170, y, width: w - 170, height: BAR, rx: 5,
          fill: D.attack_colors[atk] || "#fff", "fill-opacity": .78 }));
        const name = el("text", { x: 164, y: y + BAR - 7, "text-anchor": "end",
          fill: "#8b9bb4", "font-size": "11", "font-family": "var(--mono)" });
        name.textContent = fmtAtk(atk); svg.appendChild(name);
        const lab = el("text", { x: w + 8, y: y + BAR - 7, fill: "#dce6f5",
          "font-size": "11", "font-family": "var(--mono)" });
        const lp = r.median_latency_points;
        lab.textContent = `${(r.detection_rate * 100).toFixed(0)}% detected · ` +
          (lp === null ? "no alarms" : `median ${lp} pt${lp === 1 ? "" : "s"}` +
           (r.median_latency_s !== null ? ` (${r.median_latency_s}s)` : ""));
        svg.appendChild(lab);
      });
    });
    holder.appendChild(svg);
  }
}
</script>
</body>
</html>
"""

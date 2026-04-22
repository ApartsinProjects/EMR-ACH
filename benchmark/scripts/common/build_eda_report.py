"""
Build a comprehensive EDA HTML report over the unified Forecast Dossier (FD)
benchmark. Uses only the stdlib for HTML/SVG generation — no external libs.

Reads:
  data/unified/forecasts_filtered.jsonl
  data/unified/forecasts_dropped.jsonl
  data/unified/articles.jsonl
  data/unified/quality_meta.json
  data/unified/relevance_meta.json
  data/unified/diagnostic_report.json

Writes:
  data/unified/eda_report.html

Usage:
  python benchmark/scripts/common/build_eda_report.py
"""
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
# Fast JSONL I/O (orjson if available, stdlib json fallback) — see _fast_jsonl.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from _fast_jsonl import loads as _j_loads, dumps as _j_dumps

ROOT = Path(__file__).parent.parent
UNI = ROOT / "data" / "unified"
OUT = UNI / "eda_report.html"


def parse_date(s):
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d")
    except Exception:
        return None


# ─────────────────── SVG primitives ────────────────────────
def svg_bar_chart(data, width=720, height=260, title="", x_label="", y_label="",
                  bar_color="#4a7ab8"):
    """data: list of (label, count) tuples."""
    if not data:
        return "<p><em>No data</em></p>"
    max_v = max(v for _, v in data)
    pad_l, pad_r, pad_t, pad_b = 48, 10, 26, 36
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b
    bw = chart_w / len(data)
    bars = []
    for i, (lab, v) in enumerate(data):
        bh = (v / max_v) * chart_h if max_v else 0
        x = pad_l + i * bw + bw * 0.1
        y = pad_t + (chart_h - bh)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.8:.1f}" height="{bh:.1f}" '
            f'fill="{bar_color}"/>'
            f'<text x="{x + bw*0.4:.1f}" y="{y - 3:.1f}" font-size="9" '
            f'text-anchor="middle" fill="#333">{v}</text>'
            f'<text x="{x + bw*0.4:.1f}" y="{pad_t + chart_h + 14:.1f}" '
            f'font-size="10" text-anchor="middle" fill="#555">{html.escape(str(lab))}</text>'
        )
    # y-axis ticks
    axes = []
    for k in range(5):
        v = max_v * (k / 4)
        y = pad_t + chart_h - (chart_h * k / 4)
        axes.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" '
                    f'stroke="#e5e5e5" stroke-width="1"/>')
        axes.append(f'<text x="{pad_l - 5}" y="{y+3:.1f}" font-size="9" '
                    f'text-anchor="end" fill="#777">{v:.0f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="background:#fff;border:1px solid #eee;max-width:{width}px">'
            f'<text x="{width/2:.0f}" y="16" font-size="12" font-weight="bold" '
            f'text-anchor="middle" fill="#222">{html.escape(title)}</text>'
            f'{"".join(axes)}'
            f'{"".join(bars)}'
            f'<text x="{width/2:.0f}" y="{height-6:.0f}" font-size="10" '
            f'text-anchor="middle" fill="#555">{html.escape(x_label)}</text>'
            f'<text x="14" y="{height/2:.0f}" font-size="10" text-anchor="middle" '
            f'transform="rotate(-90 14 {height/2:.0f})" fill="#555">{html.escape(y_label)}</text>'
            f'</svg>')


def svg_pie(data, width=320, height=260, title="", colors=None):
    """data: list of (label, count) tuples."""
    if colors is None:
        colors = ["#c0392b", "#27ae60", "#f39c12", "#2980b9", "#8e44ad",
                  "#16a085", "#d35400", "#7f8c8d"]
    total = sum(v for _, v in data)
    if not total:
        return "<p><em>No data</em></p>"
    cx, cy, r = width / 2, height / 2 + 6, min(width, height) / 2 - 36
    start = 0.0
    slices = []
    legend = []
    for i, (lab, v) in enumerate(data):
        frac = v / total
        end = start + frac * 2 * math.pi
        x1 = cx + r * math.cos(start)
        y1 = cy + r * math.sin(start)
        x2 = cx + r * math.cos(end)
        y2 = cy + r * math.sin(end)
        big = 1 if frac > 0.5 else 0
        path = (f'M {cx:.2f} {cy:.2f} L {x1:.2f} {y1:.2f} '
                f'A {r:.2f} {r:.2f} 0 {big} 1 {x2:.2f} {y2:.2f} Z')
        col = colors[i % len(colors)]
        slices.append(f'<path d="{path}" fill="{col}" stroke="#fff" stroke-width="2"/>')
        # percentage label outside
        mid = (start + end) / 2
        lx = cx + (r + 14) * math.cos(mid)
        ly = cy + (r + 14) * math.sin(mid)
        slices.append(f'<text x="{lx:.1f}" y="{ly+3:.1f}" font-size="10" '
                      f'text-anchor="middle" fill="#333">{frac*100:.0f}%</text>')
        legend.append(
            f'<div style="display:flex;align-items:center;gap:6px;font-size:11px;">'
            f'<span style="width:12px;height:12px;background:{col};display:inline-block;'
            f'border-radius:2px;"></span>{html.escape(str(lab))} ({v})</div>'
        )
        start = end
    legend_html = ('<div style="display:grid;grid-template-columns:1fr 1fr;'
                   'gap:2px 14px;margin-top:8px;">' + "".join(legend) + '</div>')
    return (f'<div style="max-width:{width}px;">'
            f'<svg viewBox="0 0 {width} {height}" width="100%">'
            f'<text x="{cx:.0f}" y="18" font-size="12" font-weight="bold" '
            f'text-anchor="middle" fill="#222">{html.escape(title)}</text>'
            f'{"".join(slices)}</svg>{legend_html}</div>')


def svg_hist(values, bins, width=720, height=240, title="", x_label="", y_label="Count",
             color="#27ae60"):
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if v == bins[-1]:
                counts[-1] += 1
    labels = [f"{bins[i]:.2g}-{bins[i+1]:.2g}" for i in range(len(bins) - 1)]
    return svg_bar_chart(list(zip(labels, counts)), width=width, height=height,
                         title=title, x_label=x_label, y_label=y_label,
                         bar_color=color)


def svg_scatter(points, width=720, height=280, title="", x_label="", y_label="",
                point_color="#2980b9"):
    """points: list of (x, y) tuples, both in [0,1] ideally."""
    if not points:
        return "<p><em>No data</em></p>"
    pad_l, pad_r, pad_t, pad_b = 48, 10, 26, 36
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    if xmax == xmin: xmax = xmin + 1
    if ymax == ymin: ymax = ymin + 1
    dots = []
    for x, y in points:
        sx = pad_l + (x - xmin) / (xmax - xmin) * chart_w
        sy = pad_t + chart_h - (y - ymin) / (ymax - ymin) * chart_h
        dots.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.2" fill="{point_color}" '
                    f'fill-opacity="0.45"/>')
    # diagonal reference line (if both axes are probabilities)
    ref = ""
    if 0 <= xmin and xmax <= 1 and 0 <= ymin and ymax <= 1:
        ref = (f'<line x1="{pad_l}" y1="{pad_t+chart_h}" '
               f'x2="{pad_l+chart_w}" y2="{pad_t}" '
               f'stroke="#bbb" stroke-dasharray="4 4"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="background:#fff;border:1px solid #eee;max-width:{width}px">'
            f'<text x="{width/2:.0f}" y="16" font-size="12" font-weight="bold" '
            f'text-anchor="middle" fill="#222">{html.escape(title)}</text>'
            f'{ref}{"".join(dots)}'
            f'<text x="{width/2:.0f}" y="{height-6:.0f}" font-size="10" '
            f'text-anchor="middle" fill="#555">{html.escape(x_label)}</text>'
            f'<text x="14" y="{height/2:.0f}" font-size="10" text-anchor="middle" '
            f'transform="rotate(-90 14 {height/2:.0f})" fill="#555">{html.escape(y_label)}</text>'
            f'</svg>')


# ─────────────────── main ────────────────────────
def main():
    random.seed(7)
    fds = [_j_loads(l) for l in open(UNI / "forecasts_filtered.jsonl", encoding="utf-8")]
    drops = []
    drop_path = UNI / "forecasts_dropped.jsonl"
    if drop_path.exists():
        drops = [_j_loads(l) for l in open(drop_path, encoding="utf-8")]
    arts = {_j_loads(l)["id"]: _j_loads(l) for l in open(UNI / "articles.jsonl", encoding="utf-8")}

    quality_meta = json.loads((UNI / "quality_meta.json").read_text(encoding="utf-8"))
    diag = json.loads((UNI / "diagnostic_report.json").read_text(encoding="utf-8"))

    # Handle empty-FD state explicitly (happens when cutoff drops everything).
    if not fds:
        OUT.write_text(
            "<!DOCTYPE html><html><body style='font-family:sans-serif;padding:40px;'>"
            "<h1>EMR-ACH Benchmark EDA Report</h1>"
            "<p><strong>No accepted Forecast Dossiers.</strong> The quality filter "
            "dropped every candidate (most commonly due to the model_cutoff + "
            "buffer_days guard). Check <code>quality_meta.json</code> and "
            "<code>forecasts_dropped.jsonl</code> for the dominant drop reasons.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        print(f"Wrote minimal empty-state EDA to {OUT}")
        return

    # ─── 1. Headline numbers ───
    n_fds = len(fds)
    n_arts = len(arts)
    with_text = sum(1 for a in arts.values() if a.get("text"))
    total_links = sum(len(fd["article_ids"]) for fd in fds)
    avg_arts = total_links / n_fds if n_fds else 0
    median_arts = statistics.median(len(fd["article_ids"]) for fd in fds)
    full_text_pct = 100 * with_text / n_arts if n_arts else 0

    # ─── 2. Per-source breakdown ───
    per_src = defaultdict(list)
    for fd in fds:
        per_src[fd["source"]].append(fd)

    # ─── 3. Articles-per-FD histogram ───
    art_counts = [len(fd["article_ids"]) for fd in fds]
    art_hist = Counter(art_counts)
    art_hist_data = sorted(art_hist.items())

    # ─── 4. GT distribution ───
    gt_count = Counter(fd["ground_truth"] for fd in fds)

    # ─── 5. Source count distribution ───
    src_count = Counter(fd["source"] for fd in fds)

    # ─── 6. Crowd probability distribution ───
    crowd = [fd["crowd_probability"] for fd in fds if fd.get("crowd_probability") is not None]
    crowd_gt = [(fd["crowd_probability"], 1 if fd["ground_truth"] == "Yes" else 0)
                for fd in fds if fd.get("crowd_probability") is not None]

    # ─── 7. Article length distribution ───
    char_counts = [a["char_count"] for a in arts.values() if a.get("char_count")]
    log_chars = [math.log10(c) for c in char_counts if c > 0]

    # ─── 8. Top article domains ───
    dom_count = Counter(a.get("source_domain", "") for a in arts.values() if a.get("source_domain"))
    top_domains = dom_count.most_common(20)

    # ─── 9. Article date timeline ───
    art_dates = [parse_date(a.get("publish_date", "")) for a in arts.values()]
    art_dates = [d for d in art_dates if d]
    by_month = Counter(d.strftime("%Y-%m") for d in art_dates)
    timeline = sorted(by_month.items())

    # ─── 10. Example FDs (pick a diverse sample) ───
    picks = []
    # one example per source, Yes and No where possible
    wanted = [("polymarket", "Yes"), ("polymarket", "No"),
              ("metaculus", "Yes"), ("metaculus", "No"),
              ("manifold", "Yes"), ("manifold", "No"),
              ("infer", "No")]
    used_ids = set()
    for src, gt in wanted:
        for fd in fds:
            if fd["id"] in used_ids: continue
            if fd["source"] == src and fd["ground_truth"] == gt and len(fd["article_ids"]) >= 3:
                picks.append(fd); used_ids.add(fd["id"]); break

    # ─── 11. Drop reasons ───
    drop_reason_count = Counter()
    for d in drops:
        key = "+".join(d.get("_drop_reasons", []))
        drop_reason_count[key] += 1

    # ─── 12. Date spread per FD ───
    day_spreads = []
    for fd in fds:
        ds = sorted(d for d in (parse_date(arts[a].get("publish_date", "")) for a in fd["article_ids"] if a in arts) if d)
        if len(ds) >= 2:
            day_spreads.append((ds[-1] - ds[0]).days)

    # ═══ Render HTML ═══
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>EMR-ACH Benchmark EDA Report</title>
<style>
:root{--bg:#fafbfc;--col:#1a1a2e;--rule:#e4e4e8;--accent:#2c3e50;--kw:#c0392b;--sub:#666;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
color:var(--col);max-width:1240px;margin:0 auto;padding:30px 40px 60px;line-height:1.55;}
h1{font-size:24pt;font-weight:700;margin-bottom:4px;}
h2{font-size:16pt;margin:34px 0 10px;border-bottom:2px solid var(--rule);padding-bottom:5px;color:var(--accent);}
h3{font-size:12pt;margin:18px 0 6px;color:var(--accent);}
p{margin-bottom:8px;}
table{width:100%;border-collapse:collapse;font-size:10pt;margin:8px 0 14px;}
th,td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:left;}
th{background:#f1f2f4;font-weight:600;color:var(--accent);}
tr:hover td{background:#fdfdff;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:14px 0 22px;}
.card{background:#fff;border:1px solid var(--rule);border-radius:4px;padding:14px 16px;}
.card .v{font-size:22pt;font-weight:700;color:var(--accent);}
.card .l{font-size:9.5pt;color:var(--sub);text-transform:uppercase;letter-spacing:.04em;}
.fd-example{background:#fff;border:1px solid var(--rule);border-radius:4px;padding:14px 18px;margin:10px 0;}
.fd-example .q{font-weight:600;color:var(--accent);margin-bottom:6px;}
.fd-example .meta{font-size:9.5pt;color:var(--sub);margin-bottom:8px;}
.fd-example .pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:9pt;margin-right:4px;}
.pill.yes{background:#d5f3de;color:#116638;}
.pill.no{background:#fadcdc;color:#8a1a1a;}
.pill.src{background:#e8edf5;color:#2c3e50;}
.fd-example .arts{margin-top:8px;}
.fd-example .art{font-size:9.5pt;padding:5px 0;border-top:1px dashed #eee;}
.fd-example .art a{color:#1a5276;text-decoration:none;}
.fd-example .art a:hover{text-decoration:underline;}
.fd-example .art .dom{color:var(--sub);font-size:9pt;}
.row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;margin:8px 0 18px;}
.row>*{flex:1 1 300px;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 18px;font-size:10pt;}
.kv .k{color:var(--sub);}
.caption{font-size:9pt;color:var(--sub);margin-top:-6px;margin-bottom:10px;}
.badge{display:inline-block;padding:1px 8px;font-size:9pt;background:#d5f3de;color:#116638;border-radius:10px;font-weight:600;}
.badge.warn{background:#fef1da;color:#7a5010;}
.badge.bad{background:#fadcdc;color:#8a1a1a;}
</style></head><body>
""")
    parts.append(f"<h1>EMR-ACH Benchmark EDA Report</h1>")
    parts.append(f'<p style="color:var(--sub);font-size:10pt;">'
                 f'Generated {datetime.now().isoformat(timespec="minutes")} from <code>data/unified/</code></p>')

    # ── Headline cards ──
    parts.append('<div class="grid">')
    parts.append(f'<div class="card"><div class="v">{n_fds}</div><div class="l">Filtered Forecast Dossiers</div></div>')
    parts.append(f'<div class="card"><div class="v">{n_arts}</div><div class="l">Unique Articles in Pool</div></div>')
    parts.append(f'<div class="card"><div class="v">{avg_arts:.1f}</div><div class="l">Avg Articles / FD</div></div>')
    parts.append(f'<div class="card"><div class="v">{full_text_pct:.0f}%</div><div class="l">Articles with Full Text</div></div>')
    parts.append('</div>')

    # ── Pipeline summary ──
    parts.append("<h2>1. Pipeline &amp; Coverage Recovery</h2>")
    parts.append(
        "<p>See <code>relevance_meta.json</code> for coverage recovery at build time.</p>"
    )

    # ── Per-source ──
    parts.append("<h2>2. Per-Source Breakdown (Accepted FDs)</h2>")
    parts.append("<table><tr><th>Source</th><th>Accepted</th><th>Dropped</th><th>Avg articles/FD</th><th>Date range</th></tr>")
    for src in sorted(per_src):
        lst = per_src[src]
        dropped_n = quality_meta["per_source_dropped"].get(src, 0)
        avgk = sum(len(f["article_ids"]) for f in lst) / len(lst)
        dates = sorted(d for d in (parse_date(f["resolution_date"]) for f in lst) if d)
        parts.append(
            f"<tr><td><strong>{src}</strong></td><td>{len(lst)}</td>"
            f"<td>{dropped_n}</td><td>{avgk:.2f}</td>"
            f"<td>{dates[0].date() if dates else '?'} &rarr; {dates[-1].date() if dates else '?'}</td></tr>"
        )
    parts.append("</table>")

    parts.append('<div class="row">')
    parts.append(svg_pie([(s, c) for s, c in src_count.items()],
                         width=320, height=260, title="Accepted FDs by source"))
    parts.append(svg_pie([(g, c) for g, c in gt_count.most_common()],
                         width=320, height=260, title="Ground-truth class balance",
                         colors=["#c0392b", "#27ae60"]))
    parts.append('</div>')

    # ── Articles per FD ──
    parts.append("<h2>3. Articles per Forecast Dossier</h2>")
    parts.append(svg_bar_chart([(str(k), v) for k, v in art_hist_data],
                               width=720, height=240,
                               title="Articles linked per FD (post-filter)",
                               x_label="Articles per FD", y_label="FDs",
                               bar_color="#4a7ab8"))
    parts.append(f'<p class="caption">Median = {statistics.median(art_counts)} articles, '
                 f'mean = {statistics.mean(art_counts):.2f}, '
                 f'mode cluster at 10 (top-k cap). {art_hist.get(10, 0)} FDs saturate at 10.</p>')

    # ── Article date spread ──
    parts.append("<h2>4. Article Date Spread (within each FD)</h2>")
    spread_bins = [0, 3, 7, 14, 30, 60, 120, 365]
    parts.append(svg_hist(day_spreads, spread_bins, width=720, height=220,
                          title="Day range between earliest and latest article (per FD)",
                          x_label="Days between oldest &amp; newest article",
                          y_label="FDs", color="#8e44ad"))
    if day_spreads:
        parts.append(f'<p class="caption">Median spread = {statistics.median(day_spreads):.0f} days, '
                     f'min = {min(day_spreads)}, max = {max(day_spreads)}.</p>')

    # ── Crowd probability ──
    parts.append("<h2>5. Crowd Probability Distribution</h2>")
    p_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    parts.append(svg_hist(crowd, p_bins, width=720, height=240,
                          title="Prediction-market consensus probability of Yes",
                          x_label="crowd_probability", y_label="FDs",
                          color="#27ae60"))
    parts.append(f'<p class="caption">N = {len(crowd)}, mean = {statistics.mean(crowd):.3f}, '
                 f'median = {statistics.median(crowd):.3f}. Base rate Yes &asymp; '
                 f'{100*gt_count.get("Yes",0)/(gt_count.get("Yes",0)+gt_count.get("No",1)):.1f}%.</p>')
    parts.append(svg_scatter(crowd_gt, width=720, height=260,
                             title="Crowd probability vs. ground truth",
                             x_label="crowd_probability (Yes)", y_label="Ground truth (1=Yes, 0=No)"))
    parts.append('<p class="caption">Points near (0, 0) and (1, 1) = confident and correct; '
                 'points in upper-left / lower-right = miscalibrated crowd predictions.</p>')

    # ── Article length ──
    parts.append("<h2>6. Article Length Distribution</h2>")
    len_bins = [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]  # log10 chars
    parts.append(svg_hist(log_chars, len_bins, width=720, height=220,
                          title="Article character count (log10 scale)",
                          x_label="log10(char_count)", y_label="Articles",
                          color="#d35400"))
    parts.append(f'<p class="caption">{sum(1 for c in char_counts if c >= 1000)} articles &ge; 1k chars '
                 f'({100*sum(1 for c in char_counts if c >= 1000)/len(char_counts):.0f}%). '
                 f'{sum(1 for c in char_counts if c < 300)} articles &lt; 300 chars (title-only or paywall stub).</p>')

    # ── Top news domains ──
    parts.append("<h2>7. Top 20 News Domains in Article Pool</h2>")
    parts.append(svg_bar_chart(top_domains[:15], width=960, height=260,
                               title="Article counts by source domain (top 15)",
                               x_label="Source domain", y_label="Articles",
                               bar_color="#16a085"))
    parts.append("<table><tr><th>Rank</th><th>Domain</th><th>Articles</th></tr>")
    for i, (d, c) in enumerate(top_domains, 1):
        parts.append(f"<tr><td>{i}</td><td>{html.escape(d)}</td><td>{c}</td></tr>")
    parts.append("</table>")

    # ── Article date timeline ──
    parts.append("<h2>8. Article Volume Over Time</h2>")
    parts.append(svg_bar_chart(timeline, width=960, height=240,
                               title="Article count by publication month",
                               x_label="Publication month (YYYY-MM)", y_label="Articles",
                               bar_color="#2980b9"))
    parts.append(f'<p class="caption">Articles span {timeline[0][0]} to {timeline[-1][0]}. '
                 f'Peak month: {max(timeline, key=lambda x: x[1])[0]} with {max(v for _, v in timeline)} articles.</p>')

    # ── Leakage audit ──
    parts.append("<h2>9. Leakage Audit</h2>")
    parts.append(
        f'<p>Every linked article satisfies <code>publish_date &lt; forecast_point</code> after '
        f'the quality filter&apos;s pre-prune step. Leakage articles pruned during filtering: '
        f'<strong>{quality_meta["leakage_articles_pruned"]}</strong>. Remaining violations in '
        f'accepted FDs: <span class="badge">{diag["leakage_violations"]}</span> '
        f'(must be zero).</p>'
    )

    # ── Example Forecast Dossiers ──
    parts.append("<h2>10. Example Forecast Dossiers (selected for diversity)</h2>")
    for fd in picks:
        gt_cls = "yes" if fd["ground_truth"] == "Yes" else "no"
        parts.append(f'<div class="fd-example">')
        parts.append(f'<div class="q">{html.escape(fd["question"])}</div>')
        parts.append(f'<div class="meta">')
        parts.append(f'<span class="pill src">{fd["source"]}</span>')
        parts.append(f'<span class="pill {gt_cls}">GT: {fd["ground_truth"]}</span>')
        parts.append(f'<span class="pill src">crowd p(Yes) = '
                     f'{fd.get("crowd_probability")}</span>')
        parts.append(f'<span class="pill src">forecast_point: {fd["forecast_point"]}</span>')
        parts.append(f'<span class="pill src">resolution: {fd["resolution_date"]}</span>')
        parts.append(f'<span class="pill src">{len(fd["article_ids"])} articles</span>')
        parts.append(f'</div>')
        if fd.get("background"):
            parts.append(f'<p style="font-size:10pt;color:#333;margin:6px 0;">'
                         f'<strong>Background:</strong> {html.escape(fd["background"][:260])}'
                         f'{"&hellip;" if len(fd["background"]) > 260 else ""}</p>')
        # show up to 3 articles
        parts.append('<div class="arts"><strong style="font-size:9.5pt;">Linked articles '
                     '(first 3 of FD):</strong>')
        for aid in fd["article_ids"][:3]:
            a = arts.get(aid, {})
            if not a: continue
            title = html.escape(a.get("title", ""))[:120]
            dom = html.escape(a.get("source_domain", ""))
            url = html.escape(a.get("url", ""))
            pub = a.get("publish_date", "")
            preview = html.escape((a.get("text", "") or "")[:260]).replace("\n", " ")
            parts.append(f'<div class="art">'
                         f'<a href="{url}" target="_blank">{title}</a> '
                         f'<span class="dom">&middot; {dom} &middot; {pub}</span>'
                         f'<div style="color:#555;margin-top:2px;">'
                         f'{preview}{"&hellip;" if preview else ""}</div></div>')
        parts.append('</div></div>')

    # ── Drop reasons ──
    parts.append("<h2>11. Why Were FDs Dropped?</h2>")
    parts.append("<p>Top 10 drop-reason combinations (FDs may hit multiple thresholds):</p>")
    parts.append("<table><tr><th>Reason combination</th><th>Count</th></tr>")
    for pat, c in drop_reason_count.most_common(10):
        parts.append(f"<tr><td><code>{html.escape(pat)}</code></td><td>{c}</td></tr>")
    parts.append("</table>")
    parts.append(f'<p class="caption">Of {len(drops)} dropped FDs, most had few or no linked articles '
                 f'after cross-matching — these are typically niche sports/celebrity prediction-market '
                 f'questions that GDELT does not index well (e.g., specific player statistics, low-visibility '
                 f'markets). Quality threshold retention: {n_fds}/{n_fds+len(drops)} = '
                 f'{100*n_fds/(n_fds+len(drops)):.0f}%.</p>')

    # ── Schema reference ──
    parts.append("<h2>12. Unified Schema Reference</h2>")
    parts.append("<h3>Forecast Dossier (FD)</h3>")
    parts.append("<pre style='background:#f6f6f6;padding:10px;border-radius:4px;font-size:9.5pt;overflow-x:auto;'>"
                 "{\n"
                 "  \"id\":               \"fb_poly_12345\",\n"
                 "  \"benchmark\":        \"forecastbench\" | \"gdelt-cameo\",\n"
                 "  \"source\":           \"polymarket\" | \"metaculus\" | \"manifold\" | \"infer\" | \"gdelt-kg\",\n"
                 "  \"hypothesis_set\":   [\"Yes\", \"No\"] | [\"VC\", \"MC\", \"VK\", \"MK\"],\n"
                 "  \"question\":         \"Will X happen by Y?\",\n"
                 "  \"background\":       \"...\",\n"
                 "  \"forecast_point\":   \"2024-11-01\",          # t* (cutoff for evidence)\n"
                 "  \"resolution_date\":  \"2026-04-01\",\n"
                 "  \"ground_truth\":     \"Yes\" | \"No\" | \"MK\" | ...,\n"
                 "  \"ground_truth_idx\": 0,\n"
                 "  \"crowd_probability\": 0.62 | null,\n"
                 "  \"lookback_days\":    30 | 90,                  # T (retrieval window)\n"
                 "  \"article_ids\":      [\"art_abc123\", ...]\n"
                 "}</pre>")
    parts.append("<h3>Article</h3>")
    parts.append("<pre style='background:#f6f6f6;padding:10px;border-radius:4px;font-size:9.5pt;overflow-x:auto;'>"
                 "{\n"
                 "  \"id\":          \"art_abc123\",                 # sha1(url)[:12]\n"
                 "  \"url\":         \"https://...\",\n"
                 "  \"title\":       \"...\",\n"
                 "  \"text\":        \"...full article text...\",\n"
                 "  \"title_text\":  \"title + \\\\n + text\",\n"
                 "  \"publish_date\": \"2024-10-17\",\n"
                 "  \"source_domain\": \"reuters.com\",\n"
                 "  \"gdelt_themes\": [],\n"
                 "  \"gdelt_tone\":   -2.4,\n"
                 "  \"actors\":       [\"USA\", \"CHN\"],\n"
                 "  \"cameo_code\":   \"\",\n"
                 "  \"char_count\":   5832,\n"
                 "  \"provenance\":   [\"forecastbench\" | \"forecastbench-stepB\" | \"gdelt-cameo\"]\n"
                 "}</pre>")

    parts.append('<p style="margin-top:26px;color:var(--sub);font-size:9pt;">'
                 'EDA report generated by <code>scripts/build_eda_report.py</code>. '
                 'Source files in <code>data/unified/</code>.</p>')
    parts.append("</body></html>")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

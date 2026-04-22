"""Build an HTML report reviewing baseline dry-run outputs: rendered prompts,
parse logic, per-method config. Used for debugging code + prompts + parsing
BEFORE spending OpenAI tokens on real runs.

Input:
  /tmp/smoke_logs/{method}.log          (one per baseline, from --dry-run --smoke 2)
  benchmark/configs/baselines.yaml      (per-method config)
  benchmark/evaluation/baselines/methods/*.py (for parse-logic inspection)

Output:
  benchmark/audit/baseline_smoke_review.html
"""
import html
import json
import re
from pathlib import Path

ROOT       = Path("E:/Projects/ACH")
LOGS_DIR   = Path("C:/Users/apart/AppData/Local/Temp") if False else Path("/tmp/smoke_logs")
CONFIG     = ROOT / "benchmark" / "configs" / "baselines.yaml"
METHODS    = ROOT / "benchmark" / "evaluation" / "baselines" / "methods"
BASE_PY    = ROOT / "benchmark" / "evaluation" / "baselines" / "base.py"
BASELINES_MD = ROOT / "benchmark" / "evaluation" / "BASELINES.md"
OUT_HTML   = ROOT / "benchmark" / "audit" / "baseline_smoke_review.html"

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)


def read_text(p):
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def extract_log_sections(log_text: str) -> dict:
    """Parse the runner's dry-run log into structured sections."""
    d = {
        "banner": "",
        "method": "",
        "mode": "",
        "n_fds": 0,
        "n_articles": 0,
        "n_requests": 0,
        "requests": [],
        "errors": "",
    }
    # Banner block between the two ==== lines
    banner_m = re.search(r"=+\n(.*?)\n=+", log_text, re.DOTALL)
    if banner_m:
        d["banner"] = banner_m.group(1).strip()
    for key, pat in [
        ("method", r"method=([\w_]+)"),
        ("mode",   r"mode=([\w-]+)"),
        ("n_fds",  r"loaded (\d+) FDs"),
        ("n_requests", r"built (\d+) requests"),
    ]:
        m = re.search(pat, log_text)
        if m:
            val = m.group(1)
            d[key] = int(val) if val.isdigit() else val
    m = re.search(r"loaded \d+ FDs, (\d+) articles", log_text)
    if m:
        d["n_articles"] = int(m.group(1))

    # Pull the first 1-2 request blocks, each marked by "--- request N ---"
    req_blocks = re.findall(
        r"--- request (\d+) ---\n(.*?)(?=\n--- request \d+ ---|\Z)",
        log_text, re.DOTALL
    )
    for n, body in req_blocks[:3]:
        d["requests"].append({"n": n, "body": body.strip()})

    if "Traceback" in log_text or "Error" in log_text:
        err_m = re.search(r"(Traceback.*|Error.*)", log_text, re.DOTALL)
        if err_m:
            d["errors"] = err_m.group(1)[:2000]
    return d


def parse_method_file(py_text: str) -> dict:
    """Extract the key bits from a method module: class docstring,
    build_requests signature, round count/temperature constants."""
    out = {"doc": "", "class_name": "", "consts": [], "build_sig": ""}
    m = re.search(r'"""([\s\S]+?)"""', py_text)
    if m:
        out["doc"] = m.group(1).strip()
    m = re.search(r'class\s+(\w+)', py_text)
    if m:
        out["class_name"] = m.group(1)
    # Top-level constants (uppercase = value)
    for cm in re.finditer(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$', py_text, re.MULTILINE):
        out["consts"].append(f"{cm.group(1)} = {cm.group(2).strip()[:80]}")
    # build_requests signature
    m = re.search(r'def (build_requests(?:_round)?)\(([^)]*)\)', py_text)
    if m:
        out["build_sig"] = f"{m.group(1)}({m.group(2)})"
    return out


# ────────────────────────── load data ──────────────────────────
methods = ["b1_direct", "b2_cot", "b3_rag", "b4_self_consistency",
           "b5_multi_agent_debate", "b6_tree_of_thoughts", "b7_reflexion",
           "b8_verbalized_confidence", "b9_llm_ensemble"]

config_text = read_text(CONFIG)
try:
    import yaml
    cfg = yaml.safe_load(config_text) or {}
except Exception:
    cfg = {}

logs_parsed = {}
for m in methods:
    log_path = LOGS_DIR / f"{m}.log"
    logs_parsed[m] = {
        "log_text": read_text(log_path),
        "parsed": extract_log_sections(read_text(log_path)),
        "source": parse_method_file(read_text(METHODS / f"{m}.py")),
    }

base_py_text = read_text(BASE_PY)
# parse_probabilities body
parse_prob_m = re.search(
    r'def parse_probabilities\(([^)]*)\)([\s\S]*?)(?=\n    def |\nclass |\Z)',
    base_py_text,
)
parse_logic = parse_prob_m.group(0)[:3000] if parse_prob_m else "[not found]"


# ────────────────────────── HTML ──────────────────────────
parts = []
parts.append("""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Baseline Smoke Review</title>
<style>
:root{--accent:#2c3e50;--rule:#e4e4e8;--sub:#666;--ok:#27ae60;--warn:#d35400;--bad:#c0392b;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,-apple-system,sans-serif;background:#fafbfc;color:#1a1a2e;
max-width:1280px;margin:0 auto;padding:30px 40px 60px;line-height:1.55;}
h1{font-size:24pt;font-weight:700;}
h2{font-size:16pt;margin:34px 0 10px;border-bottom:2px solid var(--rule);padding-bottom:5px;color:var(--accent);}
h3{font-size:12pt;margin:20px 0 6px;color:var(--accent);}
p{margin-bottom:8px;}
code, pre{font-family:Menlo,Consolas,monospace;font-size:9.5pt;}
pre{background:#f4f4f7;border:1px solid var(--rule);border-radius:4px;padding:10px 14px;
margin:6px 0 12px;white-space:pre-wrap;word-break:break-word;max-height:380px;overflow-y:auto;}
table{width:100%;border-collapse:collapse;font-size:10pt;margin:8px 0 14px;}
th,td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top;}
th{background:#f1f2f4;color:var(--accent);font-weight:600;}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:9pt;font-weight:600;}
.ok{background:#d5f3de;color:#116638;}
.warn{background:#fef1da;color:#7a5010;}
.bad{background:#fadcdc;color:#8a1a1a;}
.method-card{background:#fff;border:1px solid var(--rule);border-radius:5px;
padding:16px 20px;margin:14px 0;}
.method-card h3{margin-top:0;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.caption{font-size:9pt;color:var(--sub);margin:-4px 0 10px;}
details{margin:4px 0;}
summary{cursor:pointer;font-weight:600;color:var(--accent);}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 18px;font-size:10pt;margin:6px 0;}
.kv .k{color:var(--sub);}
</style></head><body>
""")
parts.append("<h1>Baseline Smoke Review</h1>")
parts.append(f"<p class='caption'>Dry-run review of all 9 baseline methods with --smoke 2 on "
             f"<code>benchmark/data/2024-04-01/</code> (3-FD sample). Purpose: inspect rendered prompts, "
             f"request-build correctness, parse logic — BEFORE spending API budget.</p>")

# ── Summary table
parts.append("<h2>1. Summary</h2>")
parts.append("<table><tr><th>Method</th><th>Class</th><th>FDs</th><th>Requests built</th>"
             "<th>Mode</th><th>Model</th><th>Temp</th><th>Status</th></tr>")
defaults = (cfg.get("defaults") or {})
for m in methods:
    info = logs_parsed[m]
    p = info["parsed"]
    src = info["source"]
    bm_cfg = (cfg.get("baselines") or {}).get(m, {})
    model = bm_cfg.get("model") or defaults.get("model", "?")
    temp  = bm_cfg.get("temperature", defaults.get("temperature", "?"))
    if m == "b9_llm_ensemble":
        model = "(multiple - see config)"
    status = "<span class='badge ok'>dry-run OK</span>" if not p["errors"] else "<span class='badge bad'>error</span>"
    parts.append(f"<tr><td><code>{m}</code></td><td><code>{src['class_name']}</code></td>"
                 f"<td>{p['n_fds']}</td><td>{p['n_requests']}</td>"
                 f"<td>{p['mode']}</td><td>{model}</td><td>{temp}</td><td>{status}</td></tr>")
parts.append("</table>")

# ── Expected API-call counts per FD (for cost sanity)
parts.append("<h2>2. API-call footprint per FD</h2>")
parts.append("<p class='caption'>For a production run on N FDs, multiply each row's call count by N.</p>")
parts.append("<table><tr><th>Method</th><th>Calls per FD</th><th>Knob</th><th>Rationale</th></tr>")
call_table = [
    ("b1_direct",               "1",            "—",                  "single prediction call"),
    ("b2_cot",                  "1",            "—",                  "single CoT call"),
    ("b3_rag",                  "1",            "max_articles=10",    "concat articles + 1 call"),
    ("b4_self_consistency",     "k = 8",        "n_samples=8",        "k diverse samples, mean"),
    ("b5_multi_agent_debate",   "N × (R+1)",    "n_agents=3, n_rounds=2",  "3 × 3 = 9 calls"),
    ("b6_tree_of_thoughts",     "B^D",          "breadth=3, depth=2", "3² = 9 leaves scored"),
    ("b7_reflexion",            "2 × K",        "n_iterations=3",     "critique + revise each iter"),
    ("b8_verbalized_confidence","1",            "max_articles=10",    "B3 + verbalized-probability prompt"),
    ("b9_llm_ensemble",         "C",            "configs.len=6",      "one call per ensemble config"),
]
for (m, calls, knob, why) in call_table:
    parts.append(f"<tr><td><code>{m}</code></td><td><strong>{calls}</strong></td>"
                 f"<td><code>{knob}</code></td><td>{why}</td></tr>")
parts.append("</table>")

# ── Per-method deep dive
parts.append("<h2>3. Per-method dry-run review</h2>")
for m in methods:
    info = logs_parsed[m]
    p = info["parsed"]
    src = info["source"]
    bm_cfg = (cfg.get("baselines") or {}).get(m, {})
    parts.append(f"<div class='method-card'>")
    parts.append(f"<h3><code>{m}</code> · {src['class_name']}</h3>")
    if src["doc"]:
        parts.append(f"<p class='caption'>{html.escape(src['doc'][:280])}</p>")
    parts.append("<div class='kv'>")
    for k, v in bm_cfg.items():
        parts.append(f"<span class='k'>{html.escape(str(k))}:</span><span><code>{html.escape(str(v))[:200]}</code></span>")
    parts.append(f"<span class='k'>build signature:</span><span><code>{html.escape(src['build_sig'])}</code></span>")
    parts.append(f"<span class='k'>dry-run built:</span><span>{p['n_requests']} requests for {p['n_fds']} FDs</span>")
    parts.append("</div>")
    # First rendered request
    if p["requests"]:
        req = p["requests"][0]
        parts.append(f"<details open><summary>Rendered request #{req['n']}</summary>")
        parts.append(f"<pre>{html.escape(req['body'])}</pre>")
        parts.append("</details>")
    if p["errors"]:
        parts.append(f"<p><span class='badge bad'>Error</span></p>")
        parts.append(f"<pre style='background:#fee;color:#8a1a1a;'>{html.escape(p['errors'])}</pre>")
    parts.append("</div>")

# ── Parse logic
parts.append("<h2>4. Probability-parse logic (shared)</h2>")
parts.append("<p class='caption'>Defined in <code>benchmark/evaluation/baselines/base.py::parse_probabilities()</code>. "
             "Runs on every model response. If it fails, the runner assigns a uniform prior over the hypothesis set "
             "and tags the prediction as a parse failure.</p>")
parts.append(f"<pre>{html.escape(parse_logic)}</pre>")

# ── Debug stages
parts.append("<h2>5. Three-stage debug flow</h2>")
parts.append("""<table>
<tr><th>Stage</th><th>Command</th><th>Purpose</th></tr>
<tr><td>(a) Code debug</td><td><code>--method X --dry-run --smoke 3</code></td>
    <td>Builds requests without calling any API; inspect request JSON shape.</td></tr>
<tr><td>(b) Prompt + parse</td><td><code>--method X --sync --smoke 5</code></td>
    <td>Synchronous API calls, dumps per-FD prompt + response + parse to <code>{results_dir}/debug/</code>.</td></tr>
<tr><td>(c) Results analysis</td><td><code>--method X --sync --smoke 20</code></td>
    <td>Larger sample; inspect calibration + confusion matrix before full batch run.</td></tr>
<tr><td>(d) Production</td><td><code>--method X</code> (no smoke flag)</td>
    <td>Full batch API run over all FDs; writes to <code>{results_dir}/predictions.jsonl</code>.</td></tr>
</table>""")

# ── Current sample dataset caveat
parts.append("<h2>6. Sample dataset caveat</h2>")
parts.append(f"""<p>The smoke inputs are a <strong>3-FD sample</strong> at
<code>benchmark/data/2024-04-01/{{forecasts,articles}}.jsonl</code> containing one FD from each benchmark
(ForecastBench binary, GDELT-CAMEO 4-class, earnings 3-class). This is a placeholder dataset created during
the baselines scaffolding; the full 311-FD benchmark from the earlier build was transiently stashed in
<code>data/staged/</code>. Once the in-flight from-scratch 2026-01-01 build completes, the production
dataset will live at <code>benchmark/data/2026-01-01/</code> and should be used for real runs.</p>""")

parts.append(f"<p class='caption'>Generated {__import__('datetime').datetime.now().isoformat(timespec='minutes')}</p>")
parts.append("</body></html>")

OUT_HTML.write_text("".join(parts), encoding="utf-8")
print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size/1024:.0f} KB)")

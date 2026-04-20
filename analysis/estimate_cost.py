"""
Estimate API cost for all experiments before running them.

Pricing (OpenAI Batch API, 50% off standard):
  gpt-4o:      $1.25/1M input,  $5.00/1M output
  gpt-4o-mini: $0.075/1M input, $0.30/1M output

Usage:
  python analysis/estimate_cost.py
"""

N_QUERIES   = 100   # MIRAI test set
N_ARTICLES  = 10    # articles per query
N_INDICATORS = 24   # m
N_AGENTS    = 4     # default multi-agent count

# ── Token estimates (per call) ────────────────────────────────────────────────
# Measured from pilot prompt lengths (see prompts/*.yaml)
IND_IN   = 700;   IND_OUT   = 840   # indicators (per query)
PRES_IN  = 1115;  PRES_OUT  = 360   # presence scoring (per article)
INF_IN   = 850;   INF_OUT   = 620   # influence matrix (per query)
DA_IN    = 785;   DA_OUT    = 80    # deep analysis classification (per article)
ADV_IN   = 1970;  ADV_OUT   = 350   # adversarial debate round (per agent)
DIRECT_IN = 680;  DIRECT_OUT = 180  # direct prompting baseline (per query)
COT_IN   = 680;   COT_OUT   = 420   # CoT baseline (per query)
RAG_IN   = 1250;  RAG_OUT   = 300   # RAG-only baseline (per query)

GPT4O_IN   = 1.25  / 1_000_000
GPT4O_OUT  = 5.00  / 1_000_000
MINI_IN    = 0.075 / 1_000_000
MINI_OUT   = 0.30  / 1_000_000

def cost4o(n_in, n_out):
    return n_in * GPT4O_IN + n_out * GPT4O_OUT

def cost_mini(n_in, n_out):
    return n_in * MINI_IN + n_out * MINI_OUT

rows = []  # (phase, job_name, n_calls, in_tok, out_tok, model)

# ── Phase 0: Smoke tests (gpt-4o-mini, 5 queries) ────────────────────────────
S = 5
rows += [
    ("Smoke", "smoke_indicators",   S,               IND_IN,   IND_OUT,   "mini"),
    ("Smoke", "smoke_influence",    S,               INF_IN,   INF_OUT,   "mini"),
    ("Smoke", "smoke_presence",     S * N_ARTICLES,  PRES_IN,  PRES_OUT,  "mini"),
    ("Smoke", "smoke_deep_analysis",S * N_ARTICLES,  DA_IN,    DA_OUT,    "mini"),
    ("Smoke", "smoke_multiagent",   S * N_AGENTS,    ADV_IN,   ADV_OUT,   "mini"),
]

# ── Phase 1: Baselines (gpt-4o, 100 queries) ─────────────────────────────────
rows += [
    ("Baselines", "b1_direct",  N_QUERIES,              DIRECT_IN, DIRECT_OUT, "4o"),
    ("Baselines", "b2_cot",     N_QUERIES,              COT_IN,    COT_OUT,    "4o"),
    ("Baselines", "b3_rag",     N_QUERIES,              RAG_IN,    RAG_OUT,    "4o"),
]

# ── Phase 2: Shared pipeline jobs (computed ONCE, reused across Table 1 rows) ─
# indicators + influence: 1 call per query
# presence: N_QUERIES * N_ARTICLES calls (MMR+RRF retrieval)
# deep analysis: N_QUERIES * N_ARTICLES calls
# multi-agent (N=4): N_QUERIES * 4 agent calls
rows += [
    ("Pipeline (shared)", "indicators",     N_QUERIES,              IND_IN,  IND_OUT,  "4o"),
    ("Pipeline (shared)", "influence",      N_QUERIES,              INF_IN,  INF_OUT,  "4o"),
    ("Pipeline (shared)", "presence",       N_QUERIES * N_ARTICLES, PRES_IN, PRES_OUT, "4o"),
    ("Pipeline (shared)", "deep_analysis",  N_QUERIES * N_ARTICLES, DA_IN,   DA_OUT,   "4o"),
    ("Pipeline (shared)", "multiagent_n4",  N_QUERIES * N_AGENTS,   ADV_IN,  ADV_OUT,  "4o"),
]

# ── Phase 3: Ablations (only new batches needed) ─────────────────────────────
# abl_no_contrastive: new non-contrastive indicators -> new indicators+influence+presence
# abl_no_mmr: BM25 retrieval -> new presence batch
# abl_no_decay: no time decay -> new presence batch
# All other ablations (no_diag, no_absence, no_calib, no_deepanalysis, no_multiagent)
# are CPU-only recomputations from cached A/I matrices.
rows += [
    ("Ablation", "abl_no_contrastive_indicators", N_QUERIES,              IND_IN,  IND_OUT,  "4o"),
    ("Ablation", "abl_no_contrastive_influence",  N_QUERIES,              INF_IN,  INF_OUT,  "4o"),
    ("Ablation", "abl_no_contrastive_presence",   N_QUERIES * N_ARTICLES, PRES_IN, PRES_OUT, "4o"),
    ("Ablation", "abl_no_mmr_presence",           N_QUERIES * N_ARTICLES, PRES_IN, PRES_OUT, "4o"),
    ("Ablation", "abl_no_decay_presence",         N_QUERIES * N_ARTICLES, PRES_IN, PRES_OUT, "4o"),
]

# ── Phase 4: Multi-agent scaling (N=1,2,3,5,6 — N=4 already in shared) ───────
# N=1 is CPU-only (single-agent, already computed by shared pipeline).
for n in [2, 3, 5, 6]:
    rows.append(("Multi-agent scaling", f"ma_agents{n}", N_QUERIES * n, ADV_IN, ADV_OUT, "4o"))

# ── Phase 5: ForecastBench (300 questions, binary, no RAG) ───────────────────
FB_N = 300
FB_DIRECT_IN = 480;  FB_DIRECT_OUT = 160
FB_EMR_IN    = 780;  FB_EMR_OUT    = 280
rows += [
    ("ForecastBench", "fb_direct",  FB_N, FB_DIRECT_IN, FB_DIRECT_OUT, "4o"),
    ("ForecastBench", "fb_emrach",  FB_N, FB_EMR_IN,    FB_EMR_OUT,    "4o"),
]

# ── Aggregate and print ───────────────────────────────────────────────────────
print(f"\n{'Job':40s}  {'Calls':>6}  {'In $':>7}  {'Out $':>7}  {'Total $':>8}")
print("-" * 78)

phase_totals: dict[str, list] = {}
grand_total = 0.0

for phase, job, n_calls, t_in, t_out, model in rows:
    total_in  = n_calls * t_in
    total_out = n_calls * t_out
    if model == "mini":
        c = cost_mini(total_in, total_out)
        c_in  = total_in  * MINI_IN
        c_out = total_out * MINI_OUT
    else:
        c = cost4o(total_in, total_out)
        c_in  = total_in  * GPT4O_IN
        c_out = total_out * GPT4O_OUT

    label = f"  [{model}] {job}"
    print(f"{label:40s}  {n_calls:>6,}  {c_in:>7.3f}  {c_out:>7.3f}  {c:>8.3f}")

    acc = phase_totals.setdefault(phase, [0.0, 0.0, 0.0])
    acc[0] += c_in
    acc[1] += c_out
    acc[2] += c
    grand_total += c

print("-" * 78)
print("\nPhase subtotals:")
for phase, (ci, co, ct) in phase_totals.items():
    print(f"  {phase:30s}  in=${ci:.3f}  out=${co:.3f}  total=${ct:.3f}")

print(f"\n{'Grand total (batch pricing)':40s}  ${grand_total:.2f}")
print(f"{'With 20% re-run buffer':40s}  ${grand_total * 1.2:.2f}")
print()

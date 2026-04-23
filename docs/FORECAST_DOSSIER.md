# Forecast Dossier (FD) — Canonical Schema

**Version**: 2.0 (2026-04-22)
**Authoritative JSON schema**: [`docs/forecast_dossier.schema.json`](forecast_dossier.schema.json) *(optional, see §7)*
**On-disk file**: `benchmark/data/{cutoff}/forecasts.jsonl`, one FD per line.

---

## 1. What an FD is

A **Forecast Dossier** is the atomic unit of the EMR-ACH benchmark: one
forecasting instance that a system must answer. Conceptually:

```
FD  =  ⟨ date
       , question
       , ground-truth answer
       , prior-state expectations
       , evidence timeline of dated facts from pre-forecast news ⟩
```

The FD captures **both the prediction target (what to forecast)** and **the
evaluator's expectations before reading the news (what a naive status-quo
predictor would say)**. Every FD across every benchmark — geopolitics,
finance, prediction markets — follows the same schema; what varies is the
hypothesis set and the source of the prior-state expectation.

---

## 2. Schema — concrete FD JSON

```json
{
  // ─── Identity ────────────────────────────────────────────────────────
  "id":                "gdc_1234",
  "benchmark":         "gdelt-cameo",         // "gdelt-cameo" | "forecastbench" | "earnings"
  "source":            "gdelt-kg",             // underlying data source

  // ─── Temporal anchor ─────────────────────────────────────────────────
  "forecast_point":    "2026-03-01",           // "now" — no article after this date may be used
  "resolution_date":   "2026-03-01",           // when ground truth becomes observable
  "lookback_days":     90,                     // evidence window [forecast_point − 90d, forecast_point)

  // ─── Question ────────────────────────────────────────────────────────
  "question":          "Based on news from the preceding months, what is the
                        dominant intensity of interaction between Israel and
                        Palestine on or around 2026-03-01: peace, tension,
                        or violence?",
  "background":        "Forecast the conflict-intensity level between Israel (ISR)
                        and Palestine (PAL) on 2026-03-01, using only news
                        published before that date.",

  // ─── Hypothesis set + definitions (used by every baseline + EMR-ACH) ─
  "hypothesis_set":         ["Peace", "Tension", "Violence"],
  "hypothesis_definitions": {
    "Peace":    "Cooperative or neutral interaction ...",
    "Tension":  "Non-violent friction ...",
    "Violence": "Physical use of force ..."
  },

  // ─── Ground truth ────────────────────────────────────────────────────
  "ground_truth":       "Violence",           // label from hypothesis_set
  "ground_truth_idx":   2,                     // ordinal index when the hypothesis set is ordered
  "crowd_probability":  null,                  // only ForecastBench populates this

  // ─── Prior-state expectations (the "before" world) ───────────────────
  "prior_state_30d":       "Violence",         // what a status-quo predictor would say
  "prior_state_stability": 0.73,               // strength of the status quo [0, 1]
  "prior_state_n_events":  84,                 // evidence count for the prior
  "fd_type":               "stability",        // "stability" | "change" | "unknown"

  // ─── Evidence pool (pointers; see §4) ────────────────────────────────
  "article_ids":       ["art_5a916a6d2275", "art_9b7a3a98490b", ...]
}
```

All fields listed above are **required** except `crowd_probability` (populated
only for ForecastBench) and the prior-state fields for FDs where the prior
state cannot be computed (`fd_type = "unknown"`).

---

## 3. The five slots in detail

### 3.1 Date (temporal anchor)

- `forecast_point` — UTC timestamp at which the forecaster is simulated to
  be asked the question. For v2.2,
  `forecast_point = resolution_date - horizon_days`, default **14 days**
  (overridable via `--horizon-days` on `scripts/build_benchmark.py` or
  `temporal.horizon_days` in the config YAML). Every article in
  `article_ids` satisfies `publish_date <= forecast_point`; this is a hard
  leakage constraint, re-asserted locally at every fetcher
  (`scripts/fetch_*_news.py`, `scripts/link_earnings_articles.py`) and
  verified by `tests/test_v2_2_leakage.py`. v2.1 used
  `forecast_point == resolution_date` (horizon 0, retrospective); v2.2
  shifts it so the task is genuinely prospective.
- `resolution_date` — when the ground-truth outcome becomes observable in
  public news. Preserved verbatim from the upstream source; must satisfy
  `resolution_date > model_cutoff + buffer` to prevent pretraining leakage
  of the answer.
- `lookback_days` — the length of the evidence window, so fetchers query
  `[forecast_point - lookback_days, forecast_point]`. Unified to **90
  days** across all benchmarks (v2.2 default; overridable via
  `--lookback-days` / `temporal.lookback_days`).

### 3.2 Question

A natural-language prompt that any LLM can understand without external
lookups. No CAMEO codes, ticker enumerations, or domain-specific jargon
appear in the prompt string; they go into `metadata` (pipeline-internal)
only.

### 3.3 Ground-truth answer

- `ground_truth` — the class label from `hypothesis_set`.
- `ground_truth_idx` — zero-based index. For ordinal hypothesis sets
  (e.g. Peace < Tension < Violence; Beat < Meet < Miss), the index preserves
  the ordering so ordinal-error metrics are computable.

### 3.4 Prior-state expectations

The **status-quo predictor's answer** before seeing any news. Partitions FDs
into *stability* (status-quo is correct, trivial) vs *change* (status-quo is
wrong, forecasting skill required). Headline metrics are reported on the
*change* subset.

| Benchmark | `prior_state_30d` comes from                        | Stability means                          |
|-----------|-----------------------------------------------------|------------------------------------------|
| gdelt-cameo | Modal Peace/Tension/Violence class over prior 30d for same actor pair (from `data_kg.csv`) | Ground truth matches the modal prior class |
| earnings  | Mode of prior **4 quarters'** Beat/Meet/Miss for same ticker (from historical earnings data) | Current quarter matches the chronic-beater/meeter/misser pattern |
| forecastbench | `crowd_probability` — market's majority (≥0.5 → Yes, else No) | Outcome matches the market's majority call |

The annotator script `scripts/annotate_prior_state.py` computes these
purely from data we already have; no new API calls.

### 3.5 Evidence timeline

At minimum, `article_ids` — pointers into `benchmark/data/{cutoff}/articles.jsonl`.

When the ETD pipeline has run (`scripts/articles_to_facts.py` + Stage-2 dedup
+ Stage-3 FD linkage), each FD also has a sibling record in
`data/etd/fd_links.jsonl` that maps it to a chronological timeline of atomic
**facts**:

```json
{
  "fd_id":           "gdc_1234",
  "fact_ids":        ["f_001", "f_017", "f_042", ...],
  "relevance_scores":[0.91, 0.88, 0.72, ...],
  "top_k":           30,
  "leakage_dropped": 3,
  "linked_at":       "2026-04-22T13:00:00Z"
}
```

Facts are retrieved from `data/etd/facts.jsonl` (documented in
[`ETD_SPEC.md`](ETD_SPEC.md)). Each fact is atomic, dated, provenance-linked,
and multilingual-safe. The evidence timeline is **the same schema across
every domain** — only the sources that feed it differ.

---

## 4. Evidence-channel contract (retrieval)

To avoid leakage between question and evidence channels:

| Benchmark | Evidence sources | **Forbidden** |
|---|---|---|
| forecastbench | NYT + Guardian + Google News + GDELT DOC | anything that reads `crowd_probability` or `resolution_date` |
| gdelt-cameo | NYT + Guardian + Google News (editorial only) | **GDELT DOC / GKG retrieval** (that's the label source) |
| earnings | Finnhub + yfinance + Google News + GDELT DOC | anything that reads `ground_truth` or `actual EPS` |

The GDELT-CAMEO constraint is the strictest and most important: the same
GDELT indexing that emitted the quad-class label also indexes related
articles. Using that index for retrieval is a subtle leakage path. Editorial
news sources are independent.

---

## 5. Published vs pipeline-internal fields

The shipped FD (in `benchmark/data/{cutoff}/forecasts.jsonl`) contains only
the fields enumerated in §2 plus the prior-state fields. The `metadata` block
used internally (`actors`, `event_base_code`, `ticker`, `company`, etc.)
is stripped at publish time (`scripts/quality_filter.py`), because:

1. It contains retrieval-side details that could bias evaluation if an
   external system reads them.
2. It bloats the published file with pipeline-specific noise.
3. Reviewers have a simpler schema to audit.

Systems that need the raw metadata for retrieval can recover it from the
unified article pool (`articles.jsonl[i].actors`, `articles.jsonl[i].ticker`)
or rebuild from source.

---

## 6. Published companion files

Alongside `forecasts.jsonl`, the `benchmark/data/{cutoff}/` directory ships:

- `forecasts_change.jsonl` — FDs where `fd_type == "change"` (headline slice)
- `forecasts_stability.jsonl` — FDs where `fd_type == "stability"` (trivial
  floor; retained for ablation)
- `articles.jsonl` — article pool referenced by `article_ids`
- `benchmark.yaml` — full effective build config (reproducibility)
- `build_manifest.json` — provenance (timestamp, git SHA, FD counts, etc.)

Consumers running production evaluations should iterate `forecasts_change.jsonl`
as the primary slice; the full `forecasts.jsonl` is for ablations.

---

## 7. Version policy

- **Additive changes** (new optional field, new recommended value): minor
  bump; existing consumers ignore unknown fields.
- **Breaking changes** (removed/renamed field, semantics change): major
  bump; `benchmark.yaml` and `build_manifest.json` record the schema
  version, and the cutoff directory name pins the exact build.
- A formal JSON Schema (`docs/forecast_dossier.schema.json`) is **optional**
  (lower priority than ETD's schema.json since FDs are simpler and less
  evolved); worth adding if third parties adopt the format.

---

## 8. Change log

| Version | Date | Summary |
|---|---|---|
| 2.0 | 2026-04-22 | Three-benchmark unified framing. GDELT-CAMEO target reframed to Peace/Tension/Violence (3-class intensity). `prior_state` + `fd_type` (stability/change) added across all benchmarks. Evidence-channel isolation enforced (GDELT editorial-only for retrieval). 90-day analysis window unified. Paper headline = Change-subset accuracy. |
| 1.0 | 2026-04-15 | Initial three-benchmark FD schema. GDELT-CAMEO used legacy 4-class QuadClass. Per-source ad-hoc retrieval. |

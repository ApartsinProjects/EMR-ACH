# Event Timeline Dossier (ETD) — Format Specification

**Version**: 1.0
**Status**: Draft (2026-04-22)
**Owner**: EMR-ACH project
**Canonical schema file**: [`docs/etd.schema.json`](etd.schema.json)
**Canonical extraction prompt**: [`docs/prompts/etd_extraction_v1.txt`](prompts/etd_extraction_v1.txt)
**Schema field**: `"schema_version": "1.0"`

---

## 1. Motivation

The EMR-ACH task is *"forecast event outcome from prior news"* across
heterogeneous domains (geopolitics, prediction markets, earnings, and future
additions in medical / epidemiological / sports / scientific forecasting).
Each FD aggregates 5–300+ news articles in a pre-forecast lookback window.
Feeding raw article text into downstream baselines and the EMR-ACH evidence
matrix is:

- **Token-inefficient** — article boilerplate dominates signal.
- **Redundant** — wire-service syndication duplicates facts across articles.
- **Temporally opaque** — dates buried in prose.
- **Domain-ad-hoc** — each benchmark currently has its own article schema.

The ETD is a **single flat, domain-agnostic representation** of the evidence:
one line per atomic fact, with normalized time, provenance, and optional
structured payload for quantitative facts. Produced once per article via an
LLM extraction pass; consumed by every downstream baseline and by EMR-ACH.

---

## 2. Design goals

| Goal | Implication |
|---|---|
| Domain-agnostic | No geopolitics-specific required fields. Open-vocabulary `entities[].type` and `kind`. |
| Flat + grep-able | Plain JSONL; no nested trees; query with `jq`, `awk`, `grep`. |
| Deduplicated | Same fact from N wire sources collapses to one line with N citations. |
| Temporally explicit | Every fact carries `time` (when it happened) distinct from `article_date` (when reported). |
| Provenance-complete | Every fact traces back to article(s), extractor (model+prompt version), and extraction timestamp. |
| Versionable | `schema_version` enables incremental schema evolution without silent breakage. |
| Forgiving | All fields except a minimal core are optional; absent fields degrade to `null` / `[]`. |
| Standards-adjacent | Semantically compatible with ISO-TimeML events, Schema.org `Claim`+`Event`, PROV-O provenance. |

---

## 3. Storage layout

```
data/etd/
├── facts.current             # symlink → active facts file (e.g. facts.v1.jsonl)
├── facts.v1.jsonl            # cross-domain flat fact stream, append-only per v1
├── facts.errors.jsonl        # parse-failure / validation-failure records (§3.5)
├── fd_links.jsonl            # FD-to-facts linkage, one FD per line (§3.2)
└── extract_runs.jsonl        # provenance log, one extraction batch per line (§3.3)

docs/
├── ETD_SPEC.md               # this file
├── etd.schema.json           # JSON Schema for fact records (§3.6)
├── kind_vocabulary.md        # recommended (non-enforced) kind values per domain
└── prompts/
    └── etd_extraction_v1.txt # canonical Stage-1 prompt (§6.1)
```

### 3.1 `facts.<version>.jsonl`

Append-only newline-delimited JSON. Each line is one canonical fact record
(see §4). Line order is insertion order; no semantic ordering is implied —
consumers sort by `time` as needed.

**Mutation rules:**
- In-place edit is FORBIDDEN. Existing lines are never modified.
- Additions are appended.
- To correct a fact: append a new fact whose `canonical_id` points at the old
  one and whose `derived_from = [<old_fact_id>]`. Consumers preferring the
  latest canonical resolve via §4.2 `canonical_id` chain.
- A **breaking rebuild** (e.g. different extractor prompt, schema bump) creates
  a new file (`facts.v2.jsonl`). `facts.current` is re-pointed by the
  maintainer. Old files stay for reproducibility; consumers that need a
  specific version read `facts.vN.jsonl` directly.

### 3.2 `fd_links.jsonl`

One line per FD. Contains the FD identifier, the list of fact IDs that
comprise its evidence pool, per-fact relevance scores, and provenance metadata:

```json
{
  "fd_id": "gdc_123",
  "fact_ids": ["f_001","f_002","f_017"],
  "relevance_scores": [0.91, 0.88, 0.71],
  "top_k": 30,
  "leakage_dropped": 3,
  "linked_at": "2026-04-22T13:00:00Z",
  "facts_file": "facts.v1.jsonl"
}
```

- `leakage_dropped`: count of candidate facts excluded because
  `fact.article_date >= fd.forecast_point` (see §6.3 / §6.7).
- `facts_file`: which fact-store version was used (enables reproducibility
  when `facts.current` is re-pointed).

Many FDs may link the same fact (many-to-many). A fact about "2026-02-10
US-China trade talks" can be linked by a trade-market FD, a China-policy FD,
and a commodities FD.

### 3.3 `extract_runs.jsonl`

Provenance log. One line per Stage-1 batch:

```json
{
  "run_id": "2026-04-22_v1",
  "extractor": "gpt-4o-mini-2024-07-18",
  "prompt_path": "docs/prompts/etd_extraction_v1.txt",
  "prompt_sha1": "7a1e2f...",
  "started_at": "2026-04-22T12:00:00Z",
  "completed_at": "2026-04-22T12:45:00Z",
  "n_articles_input": 22349,
  "n_articles_processed": 22337,
  "n_articles_errored": 12,
  "n_facts_emitted": 84213,
  "n_facts_canonical_after_stage2": 62015,
  "quality": {
    "pct_facts_with_valid_time": 97.3,
    "pct_facts_with_entities": 94.1,
    "mean_facts_per_article": 3.77,
    "parse_failure_rate": 0.0005
  },
  "notes": "Initial full-corpus extraction."
}
```

Each fact's `extract_run` field references a row here.

### 3.4 `docs/kind_vocabulary.md`

Human-readable, non-enforced list of recommended `kind` values per domain.
Drift is tolerated; downstream tools group / display / log unknown kinds
without crashing.

### 3.5 `facts.errors.jsonl`

Every Stage-1 failure produces one line here (append-only, alongside
`facts.v1.jsonl`). Schema:

```json
{
  "article_id": "art_5a916a6d2275",
  "extract_run": "2026-04-22_v1",
  "extractor": "gpt-4o-mini-2024-07-18",
  "error_type": "json_parse | schema_violation | empty_response | api_error | validation_failed",
  "error_detail": "Expecting ',' delimiter: line 1 column 1325 (char 1324)",
  "raw_response": "<up to 4000 chars of the model's raw output, truncated>",
  "failed_at": "2026-04-22T12:34:57Z"
}
```

An article in `facts.errors.jsonl` will be re-attempted on the next Stage-1
run unless its `(article_id, extract_run)` already succeeded in
`facts.current`. Max 3 retry attempts across runs per article; after that,
operators must manually clear the error record to force another attempt.

### 3.6 `docs/etd.schema.json`

Canonical JSON Schema (draft 2020-12) describing every field in §4. Every
Stage-1 output fact SHOULD be validated against this schema before being
appended to `facts.v1.jsonl`. Validation failures go to
`facts.errors.jsonl` with `error_type = "schema_violation"`.

---

## 4. Fact schema

### 4.1 Required core

Every fact MUST have these fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier. Format: `f_<sha1(normalized_fact_text + primary_article_id + extract_run)[:12]>`. See §4.4 on collisions. |
| `schema_version` | string | Schema version (e.g. `"1.0"`). |
| `time` | string | When the fact occurred. ISO 8601 date: `YYYY-MM-DD`, less precise `YYYY-MM` / `YYYY`, or `"unknown"`. |
| `fact` | string | One-sentence, self-contained, atomic claim. Soft cap: 400 chars. No pronouns that reference other facts. |
| `article_ids` | list[string] | IDs of source article(s) attesting the fact. At least one element. Soft cap: 20 items. |
| `article_date` | string | Publish date (`YYYY-MM-DD`) of the `primary_article_id`. Used for leakage guards and time-anchoring. |
| `extractor` | string | Model identifier that produced the fact (e.g. `gpt-4o-mini-2024-07-18`, `rebel-large-2024`). |
| `extract_run` | string | FK into `extract_runs.jsonl`. Batch the fact was produced in. |
| `extracted_at` | string | Wall-clock time (ISO 8601 UTC) when extraction was performed. |

### 4.2 Optional (populate when available)

| Field | Type | Description |
|---|---|---|
| `time_end` | string (YYYY-MM-DD) or `null` | For interval facts: upper bound. |
| `time_precision` | `"day"` \| `"week"` \| `"month"` \| `"quarter"` \| `"year"` \| `"unknown"` | Explicit uncertainty marker. |
| `time_type` | `"point"` (default) \| `"interval"` \| `"ongoing"` \| `"periodic"` | See §4.5 for `periodic` semantics. |
| `primary_article_id` | string | Preferred citation from `article_ids`. Tie-break: (§4.6). |
| `source` | string | Publisher display name. Derived from primary article's domain or editorial name. |
| `language` | string (ISO 639-1, 2 chars) | Language of `fact` text. Absence means unspecified (NOT assumed English). |
| `translated_from` | string (ISO 639-1) \| `null` | Source language if `fact` was translated. |
| `entities` | list[{`name`, `type`, ...}] | Open-vocabulary tagged entities. Soft cap: 50 items. `type` recommended lowercase: `person`, `org`, `country`, `region`, `city`, `company`, `asset`, `disease`, `drug`, `treaty`, `product`, `event`. Additional domain-specific fields allowed (`ticker`, `wikidata_id`, `role`). |
| `location` | string \| object | Free text or `{country, region, lat, lon}`. |
| `metrics` | list[{`name`, `value`, `unit`, `as_of`?}] | Quantitative payload. Soft cap: 20 items. Recommended units: `percent`, `USD`/ISO 4217, `count`, `degrees_celsius`, `meters`, `bps`. |
| `kind` | string | Free-text tag. See `docs/kind_vocabulary.md` for recommended per-domain values. |
| `tags` | list[string] | Arbitrary labels. Soft cap: 20 items. |
| `polarity` | `"asserted"` (default) \| `"negated"` \| `"hypothetical"` \| `"reported"` | Epistemic status. See §4.7. |
| `attribution` | string \| `null` | Who stated the claim. Required when `polarity = "reported"`. |
| `extraction_confidence` | `"high"` \| `"medium"` \| `"low"` | LLM self-report on extraction certainty. Distinct from source reliability. |
| `source_tier` | `"primary"` \| `"aggregator"` \| `"blog"` \| `"unknown"` | Publisher tier. Primary = Reuters/AP/AFP/NYT/Guardian/etc. |
| `canonical_id` | string \| `null` | If this record is the result of a Stage-2 merge, its own ID (or the ID of the canonical predecessor if this record itself is deprecated). `null` on raw Stage-1 output. |
| `variant_ids` | list[string] | Facts collapsed into this canonical. Soft cap: 20 items; beyond that, truncated with warning in `facts.errors.jsonl`. |
| `derived_from` | list[string] | Fact IDs this fact was computed from (aggregation / translation / inference). |
| `derivation` | string \| `null` | How `derived_from` was combined (e.g. `"translation"`, `"numeric_aggregation"`, `"schema_migration"`, `"stage2_merge"`). |

### 4.3 Reserved / forbidden field names

- Custom fields MUST be namespaced with `x_emrach_<name>`.
  Example: `x_emrach_hypothesis_link`.
- Field names without this prefix beyond those listed in §4.1 / §4.2 MAY be
  rejected by future schema versions.

### 4.4 ID collision policy

`fact.id` is `f_<sha1(normalized_fact + primary_article_id + extract_run)[:12]>`.
The extract_run inclusion means two identical facts extracted in **different
runs** receive **different IDs** — runs are logically independent. Two
syntactically-identical facts extracted in the same run from the same article
receive the same ID; this is treated as a de-facto duplicate at Stage-2 (§6.2).

### 4.5 `time_type` semantics

- **`point`** (default): instant event. `time` is the date; `time_end` is `null`.
- **`interval`**: closed interval. `time` = start, `time_end` = end.
- **`ongoing`**: open-ended from `time` until unspecified future. `time_end` is `null`.
- **`periodic`**: recurring event. `time` = most recent instance. If needed,
  a free-text `periodicity` field can be added via `x_emrach_periodicity`
  (e.g. `"monthly"`, `"quarterly"`, `"annually"`).

### 4.6 `primary_article_id` tie-breaking

When multiple articles attest the same fact, select the primary as follows
(in order):
1. Highest `source_tier` (`primary > aggregator > blog > unknown`).
2. If tied, earliest `article.publish_date`.
3. If still tied, lexicographic smallest `article_id`.

### 4.7 `polarity` semantics

- **`asserted`** (default): the fact occurred / is true. Use for direct event reports.
- **`negated`**: explicit denial. E.g., *"Israel did not deploy additional troops."*
- **`hypothetical`**: conditional or counterfactual. E.g., *"If oil prices exceed $100, OPEC will convene."*
- **`reported`**: attributed to a source, not directly observed. Requires `attribution`.
  - Named individual: `attribution = "Vladimir Putin"`.
  - Collective or anonymous: `attribution = "intelligence officials"` or
    `attribution = "anonymous White House source"`. Raw phrase OK.

### 4.8 Normalization rules

- **Dates**: always ISO 8601. Relative expressions in the source ("last week")
  MUST be anchored to `article_date` during extraction.
- **Fact text**: trim whitespace, collapse internal spaces, strip leading/trailing
  quotation marks. Should end with a period.
  - Dedup preference: when merging variants, prefer the text with fewer
    pronouns. Pronoun count = tokens in `{he, she, it, they, this, that, these, those}` (case-insensitive).
- **Entity names**: preserve original capitalization (e.g. `"Apple Inc."`).
  Entity `type` values SHOULD be lowercased.
- **Unit strings**: prefer full words (`percent`, `degrees_celsius`) over
  symbols (`%`, `°C`). Currencies: ISO 4217 codes (`USD`, `EUR`, `JPY`).

---

## 5. Domain examples

All examples fit the schema without modification. See §4 for field definitions.

### 5.1 Geopolitics (GDELT-CAMEO)

```json
{
  "id":"f_5a916a6d2275", "schema_version":"1.0",
  "time":"2026-02-13", "time_precision":"day", "time_type":"point",
  "fact":"Israel deployed 3 infantry brigades to the Gaza perimeter.",
  "language":"en",
  "entities":[{"name":"Israel","type":"country"},{"name":"Gaza","type":"region"}],
  "metrics":[{"name":"brigade_count","value":3,"unit":"count"}],
  "kind":"military-deployment", "tags":["escalation"],
  "polarity":"asserted",
  "extraction_confidence":"high", "source_tier":"primary",
  "article_ids":["art_xxx","art_yyy"], "primary_article_id":"art_xxx",
  "article_date":"2026-02-18",
  "extractor":"gpt-4o-mini-2024-07-18",
  "extract_run":"2026-04-22_v1",
  "extracted_at":"2026-04-22T12:34:56Z"
}
```

### 5.2 Earnings

```json
{
  "id":"f_a1b2c3d4e5f6", "schema_version":"1.0",
  "time":"2026-01-29", "time_precision":"day", "time_type":"point",
  "fact":"Apple Inc. reported Q1 2026 EPS of $2.84, beating consensus of $2.67.",
  "language":"en",
  "entities":[{"name":"Apple Inc.","type":"company","ticker":"AAPL"}],
  "metrics":[
    {"name":"eps_actual","value":2.84,"unit":"USD"},
    {"name":"eps_estimate","value":2.67,"unit":"USD"},
    {"name":"surprise_pct","value":6.25,"unit":"percent"}
  ],
  "kind":"earnings-release", "polarity":"asserted",
  "extraction_confidence":"high", "source_tier":"primary",
  "article_ids":["art_zzz"], "primary_article_id":"art_zzz",
  "article_date":"2026-01-30",
  "extractor":"gpt-4o-mini-2024-07-18",
  "extract_run":"2026-04-22_v1",
  "extracted_at":"2026-04-22T12:34:56Z"
}
```

### 5.3 Reported claim (prediction market)

```json
{
  "id":"f_deadbeef0001", "schema_version":"1.0",
  "time":"2026-01-20", "time_precision":"day", "time_type":"point",
  "fact":"Vladimir Putin stated Russia will not escalate military operations in 2026.",
  "language":"en", "translated_from":"ru",
  "entities":[{"name":"Vladimir Putin","type":"person"},{"name":"Russia","type":"country"}],
  "kind":"policy-statement",
  "polarity":"reported", "attribution":"Vladimir Putin",
  "extraction_confidence":"high", "source_tier":"primary",
  "article_ids":["art_aaa","art_bbb","art_ccc"], "primary_article_id":"art_aaa",
  "article_date":"2026-01-20",
  "extractor":"gpt-4o-mini-2024-07-18",
  "extract_run":"2026-04-22_v1",
  "extracted_at":"2026-04-22T12:34:56Z"
}
```

### 5.4 Interval fact

```json
{
  "id":"f_interval0001", "schema_version":"1.0",
  "time":"2026-02-15", "time_end":"2026-02-17",
  "time_precision":"day", "time_type":"interval",
  "fact":"G20 finance ministers held closed-door negotiations in Riyadh.",
  "language":"en",
  "entities":[{"name":"G20","type":"org"},{"name":"Riyadh","type":"city"}],
  "location":{"country":"Saudi Arabia","region":"Riyadh"},
  "kind":"diplomatic-meeting", "polarity":"asserted",
  "extraction_confidence":"high", "source_tier":"primary",
  "article_ids":["art_fff"], "primary_article_id":"art_fff",
  "article_date":"2026-02-18",
  "extractor":"gpt-4o-mini-2024-07-18",
  "extract_run":"2026-04-22_v1",
  "extracted_at":"2026-04-22T12:34:56Z"
}
```

### 5.5 Minimal valid fact (sparse fields)

```json
{
  "id":"f_minimal0001", "schema_version":"1.0",
  "time":"2026-02-13",
  "fact":"An earthquake of magnitude 5.8 struck central Chile.",
  "article_ids":["art_ccc"],
  "article_date":"2026-02-14",
  "extractor":"gpt-4o-mini-2024-07-18",
  "extract_run":"2026-04-22_v1",
  "extracted_at":"2026-04-22T12:34:56Z"
}
```

Valid. All optional fields default to `null` / `[]`.

---

## 6. Extraction pipeline

Three stages. Stage 1 is the only expensive step (LLM calls); stages 2–3 are
CPU-only and cheap to re-run.

### 6.1 Stage 1 — Per-article fact extraction

- **Input**: one article record (`{id, url, title, text, publish_date, source_domain, language}`).
- **Output**: 0–6 `facts.current` lines plus, on failure, one `facts.errors.jsonl` line.
- **Method**: one OpenAI batch API call per article.
- **Prompt**: canonical text committed to
  [`docs/prompts/etd_extraction_v1.txt`](prompts/etd_extraction_v1.txt). Any
  change to this file requires bumping `extract_run` / `prompt_sha1` in the
  run log.
- **Cost estimate**: gpt-4o-mini batch ≈ $0.00015 / call.
  22k articles ≈ $3. 84k articles ≈ $12.

### 6.2 Stage 2 — Cross-article dedup / canonical merge

- **Input**: raw Stage-1 output in `facts.current`.
- **Output**: `facts.current` with canonical records appended and variant
  records flagged via `canonical_id`.
- **Method**: CPU only.
  1. Embed each `fact.text` with SBERT (`all-mpnet-base-v2`, already in the pipeline).
  2. Cluster facts where: cosine ≥ 0.90 **AND** `|time_A − time_B| ≤ 2 days` when both have `day` precision (otherwise allow any time agreement within the less-precise bucket) **AND** compatible `polarity` (asserted↔asserted, negated↔negated, etc.; across polarities, NEVER merge).
  3. Per cluster, pick the canonical variant:
     a. Prefer the richer record (most of `entities` / `metrics` / `location` / `time_end` populated).
     b. Tie-break: fewer pronouns in `fact.text` (§4.8).
     c. Tie-break: longest `fact.text`.
     d. Tie-break: earliest `article_date`.
     e. Tie-break: lexicographic smallest `id`.
  4. Union `article_ids`, `entities`, `metrics`, `tags` onto the canonical; set `max(extraction_confidence)`.
  5. Append a new record for the canonical with `derivation = "stage2_merge"` and `variant_ids` = the cluster minus canonical. Older Stage-1 records remain in the file; consumers resolve via `canonical_id`.
- **Cross-extract-run merging**: when a fresh extract_run produces a candidate fact that SBERT-matches a canonical from an older run, the newer record is preferred as canonical IF it is richer (step 3a); else the older stays canonical and the newer becomes a variant.
- **Cost**: zero API. ~1–2 min for 84k facts via batched SBERT on GPU.

### 6.3 Stage 3 — FD linkage

- **Input**: canonical facts + FD list.
- **Output**: `fd_links.jsonl`.
- **Method**: CPU only.
  1. Embed each FD's `question + background` with SBERT.
  2. For each FD, score all facts by cosine(question, fact) AND restrict to facts with `article_date ∈ [fd.forecast_point − lookback, fd.forecast_point)`.
  3. Keep top-K (default K=30; configurable via `configs/etd.yaml` key `fd_links_top_k`).
  4. Record `leakage_dropped` = count of candidates that would have ranked in top-K but were excluded by the `article_date < forecast_point` rule.
  5. Write one `fd_links.jsonl` line per FD.
- **Cost**: zero API. ~1 min per 1k FDs on GPU.

### 6.4 Stage 1 error handling

Errors written to `facts.errors.jsonl` (§3.5). Retry policy:
- Per `(article_id, extract_run)`: try once.
- Per article across runs: up to 3 attempts. After that, manually clear the
  error to re-attempt.
- Stage 1 runner SHOULD skip articles whose `(article_id, extract_run)` is
  already present in `facts.current` (§6.6 idempotency).

### 6.5 Quality metrics (per-run)

Every `extract_runs.jsonl` row carries a `quality` block. Recommended keys
and minimum targets:

| Metric | Formula | Target |
|---|---|---|
| `pct_facts_with_valid_time` | facts where `time` parses as ISO date ÷ total facts | ≥ 95% |
| `pct_facts_with_entities` | facts with ≥1 entity ÷ total facts | ≥ 90% |
| `mean_facts_per_article` | total facts ÷ n_articles_processed | 2.5 – 5.0 |
| `parse_failure_rate` | `facts.errors.jsonl` additions ÷ n_articles_input | ≤ 0.01 |
| `mean_intra_article_cosine_disp` | mean pairwise cosine distance of facts within one article | ≥ 0.25 (atomicity) |

Runs failing any minimum target MUST be flagged in `notes` and MUST NOT be
promoted to `facts.current` without operator sign-off.

### 6.6 Idempotency / resume

- Stage 1 SHOULD hash `(article_id, extract_run)` and skip articles already
  present in `facts.current`. Skip mode is the default; `--force` overrides.
- Partial runs (Stage 1 crashed halfway) resume from the first unprocessed
  article on next invocation.
- Stage 2 is idempotent: re-running it produces the same canonical IDs given
  the same input facts + same SBERT model.
- Stage 3 rewrites `fd_links.jsonl` from scratch each time; no resume needed.

### 6.7 Rebuilds / version bumps

A breaking change (new required field, changed semantics) bumps
`schema_version` and creates `facts.v<N+1>.jsonl`:

1. Freeze the old file.
2. Re-extract all articles with the new prompt / schema → new
   `facts.v<N+1>.jsonl`.
3. Re-run Stage 2 / Stage 3.
4. Re-point `facts.current` → new file.
5. Old file remains for reproducibility; consumers needing a specific version
   read it directly.

Non-breaking changes (new optional field, new recommended `kind`) just bump
the minor version in `schema_version` and update this doc; no rebuild.

---

## 7. Consumption patterns

### 7.1 Baselines (B1–B9)

Baselines that currently concatenate article text now read the FD's linked
facts, sort by `time` ASC, render as a markdown-style table, and embed in
the prompt:

```
Forecasting question: {question}
Candidate hypotheses: {h1}, {h2}, ...

Evidence timeline (facts in chronological order):
  2026-02-08  Reuters     Israel approves reserve call-up of 8,000 troops
  2026-02-10  NYT         Egyptian mediators propose ceasefire framework
  2026-02-18  Reuters     Israel deploys 3 brigades to Gaza perimeter [Israel]
  ...

Forecast point: {forecast_point}
Based on the evidence timeline, select the single most likely hypothesis.
```

### 7.2 EMR-ACH evidence matrix

Rows of the analysis matrix A become **facts** instead of articles:

```
A[fact_i, indicator_j] = LLM.presence(fact_i, indicator_j)
```

Benefits:
- ~10× fewer rows (dedup + atomicity) ⇒ ~10× fewer presence calls.
- Higher precision per row (atomic claim vs whole article).
- Natural integration of `time` for temporal-trend indicators.

### 7.3 External tools — `jq` examples

```bash
# All facts about Apple in Q1 2026
jq -c 'select((.entities // [])[]?.ticker == "AAPL" and (.time | startswith("2026-0")))' \
   data/etd/facts.current

# Timeline for a specific FD
FID=gdc_123
IDS=$(jq -r --arg fid "$FID" 'select(.fd_id == $fid) | .fact_ids[]' data/etd/fd_links.jsonl)
echo "$IDS" \
  | python -c "
import sys, json
want = set(l.strip() for l in sys.stdin if l.strip())
rows = [json.loads(l) for l in open('data/etd/facts.current', encoding='utf-8') if json.loads(l)['id'] in want]
rows.sort(key=lambda r: r.get('time','~'))
for r in rows:
    print(f\"{r.get('time','?'):<10}  {r.get('source','?'):<25}  {r['fact']}\")
"
```

---

## 8. Versioning policy

- **Additive** (new optional field, new recommended `kind`, new recommended
  entity `type`) → minor bump (`1.0` → `1.1`). Backward-compatible; consumers
  ignore unknown fields.
- **Breaking** (removed/renamed field, changed semantics, new required field) →
  major bump (`1.x` → `2.0`). Consumers MUST check `schema_version` and fail
  loudly on mismatch. New file `facts.v2.jsonl` per §6.7.
- **Schema validator**: `docs/etd.schema.json` is the authoritative JSON
  Schema for v1.0. Validate every Stage-1 output before append.

---

## 9. Interoperability with external standards

Pragmatic flat format, semantically compatible with:

- **ISO-TimeML** (ISO 24617-1:2012) — our `fact` ≈ `EVENT`, `time` ≈ `TIMEX3`,
  fact ordering ≈ implicit `TLINK`s.
- **Schema.org** (JSON-LD) — our fact ≈ `schema:Claim` + `schema:Event`,
  `entities` ≈ `schema:Person`/`Organization`/`Place`.
- **PROV-O** (W3C) — `extract_run` / `extractor` / `extracted_at` map to
  `prov:Activity` / `prov:SoftwareAgent` / `prov:startedAtTime`;
  `article_ids` ↔ `prov:wasDerivedFrom`.

A thin wrapper converts our JSONL to JSON-LD if Linked-Data publishing is
required. For normal use (baselines, EMR-ACH), the plain JSONL is enough.

---

## 10. Open questions / future work

- **Fact-to-fact relations** (causes, contradicts, refines) — deferred. If
  needed, add `relations.jsonl` keyed by `(from_fact_id, relation, to_fact_id)`.
  Keep `facts.current` flat.
- **Cross-lingual canonicalization** — multilingual SBERT embedding (e.g.
  `paraphrase-multilingual-mpnet-base-v2`) would merge same-fact-different-
  language; revisit if multilingual benchmarks become primary.
- **Temporal uncertainty intervals** — `time_precision = year` sorts crudely
  next to `day`-precise facts. A midpoint-plus-range API is possible but not
  specified here.
- **Fact truth-score** — out of scope. Task is forecast-from-facts, not
  fact-validation. If added, separate file (`fact_truth_scores.jsonl`).
- **Streaming extraction** — current Stage 1 is batch-mode. Live benchmarks
  may need a streaming variant.

---

## 11. Privacy & research-use note

- The benchmark is **research-only**. ETDs are derived from publicly-published
  news articles.
- Facts MAY name individuals when the underlying article does so. Consumers
  MUST NOT use ETD records to make decisions about named individuals
  (credit, employment, insurance, law enforcement, immigration, etc.) in any
  production or operational context.
- Redistribution of the ETD dataset requires preserving `article_ids` +
  `article_date` + `extractor` provenance so downstream users can verify
  against original sources.
- For any jurisdiction where GDPR / CCPA / similar applies to any fact
  about a living individual: the original article is the authoritative
  source; requests for erasure MUST be directed to the article publisher,
  not the ETD maintainers. ETD records may be expunged on request if the
  underlying article is retracted.

---

## 12. Citation

When using ETD records in a paper or derivative work:

```bibtex
@misc{emrach_etd_2026,
  author       = {{EMR-ACH Authors}},
  title        = {Event Timeline Dossier (ETD) Format Specification, v1.0},
  year         = {2026},
  url          = {https://github.com/<org>/emr-ach/blob/main/docs/ETD_SPEC.md},
  note         = {Research artifact.}
}
```

License: TBD (recommended: MIT for code, CC-BY-4.0 for extracted fact data).
See repo `LICENSE` file for the authoritative statement.

---

## 13. Change log

| Version | Date | Summary |
|---|---|---|
| 1.0 | 2026-04-22 | Initial release. Flat JSONL, domain-agnostic, with §11 privacy note, JSON Schema, and canonical extraction prompt. |

Future entries must describe what changed, why, and migration notes for any
breaking change.

---

## 14. Glossary

- **FD** (Forecast Dossier) — one forecasting instance with a question,
  hypothesis set, article pool, and ground truth. Defined in the main
  benchmark README.
- **ETD** (Event Timeline Dossier) — the per-FD structured evidence produced
  by joining `facts.current` and `fd_links.jsonl` against a specific FD ID.
  Not a persistent file; materialized on demand.
- **Canonical fact** — a fact record retained after Stage-2 dedup. Its
  `variant_ids` lists the Stage-1 records that were merged into it.
- **Atomic fact** — a fact representing one event, one actor tuple, one
  time point. Compound claims are split during Stage-1 extraction.
- **Primary article** — the highest-quality source among an article cluster
  (§4.6).
- **Extract run** — one Stage-1 batch, identified by `run_id` in
  `extract_runs.jsonl`. Determines the `extractor` and prompt version used
  for every fact produced in that run.

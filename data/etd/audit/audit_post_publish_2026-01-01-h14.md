# ETD Stage-1 Audit Report

- Input: `E:\Projects\ACH\data\etd\facts.v1_production_2026-01-01-h14.jsonl`
- Generated: 2026-04-23T19:57:20.843537Z
- Total facts: **403**
- Unique articles producing facts: **99**
- Avg facts per article: **4.07**

## Schema validity
- Schema fails: **0** (0.0%)
- Bad `polarity` values: **0**
- Bad `extraction_confidence` values: **0**

## Field distributions
- `extraction_confidence`: {'high': 403}
- `polarity`: {'asserted': 403}
- `language` top: {'en': 403}
- `kind` top: {'policy-statement': 29, 'military-activity': 22, 'military-action': 21, 'military-operation': 18, 'event': 15, 'sports-event': 11, 'box-office': 11, 'statement': 10, 'political-statement': 9, 'protest': 8, 'economic-forecast': 7, 'survey': 6, 'market-update': 6, 'performance': 5, 'political-campaign': 5}

## Date sanity
- Facts with no parseable date: **4**
- Facts dated AFTER article publish (impossible): **0**
- Facts dated >365d BEFORE article publish (likely hallucinated): **92**

## Per-article fact density
- `<10` facts/article: **37** articles
- `<2` facts/article: **4** articles
- `<20` facts/article: **1** articles
- `<5` facts/article: **57** articles

## Duplicates (within Stage-1)
- Exact duplicates (same article + same fact text): **6**
- Near duplicates (same article + normalized-equal text): **0**
  - These should be 0 after Stage-2 dedup (`scripts/etd_dedup.py`).

## Entity coverage
- Facts with >=1 entity: **364** (90.3%)
- Unique entity names: **336**
- Entity type distribution: {'person': 202, 'organization': 149, 'country': 143, 'city': 79, 'product': 33, 'team': 33, 'region': 19, 'company': 15, 'location': 7, 'asset': 6}
- Top-20 entities by mention count:
  - `Donald Trump`: 40
  - `Iran`: 32
  - `Elon Musk`: 30
  - `Russia`: 20
  - `Israel`: 14
  - `Larry Fink`: 13
  - `Oklahoma City Thunder`: 12
  - `İsrail`: 12
  - `Zootopia 2`: 11
  - `Ukraine`: 10
  - `Indiana Pacers`: 10
  - `İran`: 10
  - `Rob Jetten`: 9
  - `Mossad`: 8
  - `Tesla`: 6
  - `The Charles Schwab Corporation`: 6
  - `Северск`: 6
  - `Tehran`: 5
  - `Boston Celtics`: 5
  - `Tel Aviv`: 5

## Per-source extraction rate (top 20 by article volume)
| Source | Articles | Facts | Facts/article |
|---|---:|---:|---:|
| `(unknown)` | 99 | 403 | 4.07 |

## Recommended next steps
- If `Date sanity` shows >1% future facts -> Stage-1 prompt needs explicit "fact.time must be on or before publish_date" reminder.
- If `Duplicates` shows non-trivial near-dupes -> run `python scripts/etd_dedup.py` (Stage 2).
- If `Per-article density` shows many 0-fact articles -> run `python scripts/etd_debug_empty.py`.
- If `Per-source extraction rate` shows >2x variance -> a specific outlet may have parser issues; sample those articles for spot-check.

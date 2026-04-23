# ETD Stage-1 Audit Report

- Input: `E:\Projects\ACH\data\etd\facts.v1_production_2026-01-01.jsonl`
- Generated: 2026-04-23T14:41:08.899610Z
- Total facts: **50821**
- Unique articles producing facts: **13008**
- Avg facts per article: **3.91**

## Schema validity
- Schema fails: **0** (0.0%)
- Bad `polarity` values: **0**
- Bad `extraction_confidence` values: **0**

## Field distributions
- `extraction_confidence`: {'high': 50821}
- `polarity`: {'asserted': 50821}
- `language` top: {'en': 50821}
- `kind` top: {'military-action': 6627, 'policy-statement': 5058, 'military-deployment': 2104, 'political-statement': 1960, 'military-attack': 1777, 'statement': 1577, 'military-strike': 1312, 'military-operation': 1161, 'military-casualty': 901, 'military-incident': 778, 'diplomatic-statement': 688, 'casualty-report': 661, 'military-threat': 439, 'diplomatic-meeting': 438, 'diplomatic-communication': 419}

## Date sanity
- Facts with no parseable date: **1908**
- Facts dated AFTER article publish (impossible): **0**
- Facts dated >365d BEFORE article publish (likely hallucinated): **3436**

## Per-article fact density
- `<10` facts/article: **4965** articles
- `<2` facts/article: **788** articles
- `<20` facts/article: **15** articles
- `<5` facts/article: **7240** articles

## Duplicates (within Stage-1)
- Exact duplicates (same article + same fact text): **61**
- Near duplicates (same article + normalized-equal text): **0**
  - These should be 0 after Stage-2 dedup (`scripts/etd_dedup.py`).

## Entity coverage
- Facts with >=1 entity: **48319** (95.1%)
- Unique entity names: **12727**
- Entity type distribution: {'country': 46346, 'person': 25243, 'organization': 13881, 'city': 7602, 'location': 3600, 'region': 3136, 'product': 1873, 'asset': 1312, 'company': 573, 'event': 572}
- Top-20 entities by mention count:
  - `Iran`: 14013
  - `Israel`: 6672
  - `Donald Trump`: 3669
  - `United States`: 2975
  - `US`: 1749
  - `Lebanon`: 1398
  - `Strait of Hormuz`: 1302
  - `Pakistan`: 1264
  - `Saudi Arabia`: 1209
  - `Hezbollah`: 1104
  - `Tehran`: 1054
  - `Russia`: 944
  - `Kuwait`: 914
  - `Ayatollah Ali Khamenei`: 902
  - `Qatar`: 858
  - `U.S.`: 835
  - `Ukraine`: 803
  - `China`: 768
  - `United Arab Emirates`: 652
  - `Iraq`: 650

## Per-source extraction rate (top 20 by article volume)
| Source | Articles | Facts | Facts/article |
|---|---:|---:|---:|
| `(unknown)` | 13008 | 50821 | 3.91 |

## Recommended next steps
- If `Date sanity` shows >1% future facts -> Stage-1 prompt needs explicit "fact.time must be on or before publish_date" reminder.
- If `Duplicates` shows non-trivial near-dupes -> run `python scripts/etd_dedup.py` (Stage 2).
- If `Per-article density` shows many 0-fact articles -> run `python scripts/etd_debug_empty.py`.
- If `Per-source extraction rate` shows >2x variance -> a specific outlet may have parser issues; sample those articles for spot-check.

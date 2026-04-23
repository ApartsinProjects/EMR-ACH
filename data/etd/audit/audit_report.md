# ETD Stage-1 Audit Report

- Input: `E:\Projects\ACH\data\etd\facts.v1.jsonl`
- Generated: 2026-04-23T06:47:27.680549Z
- Total facts: **78462**
- Unique articles producing facts: **17546**
- Avg facts per article: **4.47**

## Schema validity
- Schema fails: **0** (0.0%)
- Bad `polarity` values: **0**
- Bad `extraction_confidence` values: **0**

## Field distributions
- `extraction_confidence`: {'high': 77445, 'medium': 1017}
- `polarity`: {'asserted': 67522, 'reported': 10598, 'hypothetical': 84, 'negated': 258}
- `language` top: {'en': 78462}
- `kind` top: {'military-action': 10022, 'policy-statement': 7473, 'military-deployment': 3135, 'statement': 3093, 'military-attack': 3049, 'political-statement': 3029, 'military-strike': 2017, 'casualty-report': 1839, 'military-operation': 1704, 'military-casualty': 1322, 'military-incident': 1167, 'diplomatic-statement': 1079, 'diplomatic-communication': 677, 'military-threat': 605, 'diplomatic-meeting': 605}

## Date sanity
- Facts with no parseable date: **3122**
- Facts dated AFTER article publish (impossible): **0**
- Facts dated >365d BEFORE article publish (likely hallucinated): **5118**

## Per-article fact density
- `<10` facts/article: **10420** articles
- `<2` facts/article: **622** articles
- `<20` facts/article: **2** articles
- `<5` facts/article: **6502** articles

## Duplicates (within Stage-1)
- Exact duplicates (same article + same fact text): **20**
- Near duplicates (same article + normalized-equal text): **0**
  - These should be 0 after Stage-2 dedup (`scripts/etd_dedup.py`).

## Entity coverage
- Facts with >=1 entity: **75017** (95.6%)
- Unique entity names: **18423**
- Entity type distribution: {'country': 70635, 'person': 38611, 'organization': 22971, 'city': 11634, 'location': 5391, 'region': 5152, 'product': 2758, 'asset': 1974, 'event': 856, 'company': 842}
- Top-20 entities by mention count:
  - `Iran`: 21025
  - `Israel`: 9872
  - `Donald Trump`: 5149
  - `United States`: 4475
  - `US`: 2694
  - `Lebanon`: 2599
  - `Pakistan`: 2080
  - `Russia`: 2027
  - `Strait of Hormuz`: 1842
  - `Ukraine`: 1746
  - `Saudi Arabia`: 1670
  - `Hezbollah`: 1655
  - `Tehran`: 1513
  - `U.S.`: 1295
  - `Ayatollah Ali Khamenei`: 1268
  - `Qatar`: 1253
  - `Kuwait`: 1224
  - `China`: 1144
  - `Iraq`: 970
  - `Cuba`: 917

## Per-source extraction rate (top 20 by article volume)
| Source | Articles | Facts | Facts/article |
|---|---:|---:|---:|
| `(unknown)` | 17546 | 78462 | 4.47 |

## Recommended next steps
- If `Date sanity` shows >1% future facts -> Stage-1 prompt needs explicit "fact.time must be on or before publish_date" reminder.
- If `Duplicates` shows non-trivial near-dupes -> run `python scripts/etd_dedup.py` (Stage 2).
- If `Per-article density` shows many 0-fact articles -> run `python scripts/etd_debug_empty.py`.
- If `Per-source extraction rate` shows >2x variance -> a specific outlet may have parser issues; sample those articles for spot-check.

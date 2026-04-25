# ETD Stage-1 Audit Report

- Input: `E:\Projects\ACH\data\etd\facts.v1_production_2026-01-01-h14-ccnews.jsonl`
- Generated: 2026-04-25T05:38:18.861620Z
- Total facts: **5956**
- Unique articles producing facts: **927**
- Avg facts per article: **6.43**

## Schema validity
- Schema fails: **0** (0.0%)
- Bad `polarity` values: **0**
- Bad `extraction_confidence` values: **0**

## Field distributions
- `extraction_confidence`: {'high': 5956}
- `polarity`: {'asserted': 5956}
- `language` top: {'en': 5956}
- `kind` top: {'policy-statement': 270, 'sports-event': 167, 'credit-card-offer': 154, 'statement': 93, 'political-statement': 85, 'game-result': 81, 'event': 71, 'military-action': 68, 'military-attack': 67, 'game-preview': 62, 'box-office-performance': 62, 'injury-update': 56, 'diplomatic-meeting': 56, 'player-performance': 54, 'weather-forecast': 52}

## Date sanity
- Facts with no parseable date: **104**
- Facts dated AFTER article publish (impossible): **0**
- Facts dated >365d BEFORE article publish (likely hallucinated): **104**

## Per-article fact density
- `<10` facts/article: **478** articles
- `<2` facts/article: **3** articles
- `<20` facts/article: **174** articles
- `<5` facts/article: **272** articles

## Duplicates (within Stage-1)
- Exact duplicates (same article + same fact text): **2831**
- Near duplicates (same article + normalized-equal text): **0**
  - These should be 0 after Stage-2 dedup (`scripts/etd_dedup.py`).

## Entity coverage
- Facts with >=1 entity: **4509** (75.7%)
- Unique entity names: **1973**
- Entity type distribution: {'person': 2602, 'organization': 1982, 'country': 1250, 'team': 480, 'product': 402, 'city': 372, 'region': 289, 'player': 215, 'location': 208, 'company': 161}
- Top-20 entities by mention count:
  - `Ukraine`: 209
  - `Russia`: 190
  - `China`: 148
  - `Donald Trump`: 139
  - `Meta`: 98
  - `Taiwan`: 96
  - `Venezuela`: 82
  - `Boston Celtics`: 79
  - `Israel`: 68
  - `Bank of America`: 61
  - `Trump`: 52
  - `United States`: 49
  - `Pokrovsk`: 48
  - `Japan`: 46
  - `Iran`: 44
  - `Vladimir Putin`: 44
  - `US`: 44
  - `Indiana Pacers`: 42
  - `Zootopia 2`: 36
  - `Elon Musk`: 34

## Per-source extraction rate (top 20 by article volume)
| Source | Articles | Facts | Facts/article |
|---|---:|---:|---:|
| `(unknown)` | 927 | 5956 | 6.43 |

## Recommended next steps
- If `Date sanity` shows >1% future facts -> Stage-1 prompt needs explicit "fact.time must be on or before publish_date" reminder.
- If `Duplicates` shows non-trivial near-dupes -> run `python scripts/etd_dedup.py` (Stage 2).
- If `Per-article density` shows many 0-fact articles -> run `python scripts/etd_debug_empty.py`.
- If `Per-source extraction rate` shows >2x variance -> a specific outlet may have parser issues; sample those articles for spot-check.

# Unified Forecast Dossier — Diagnostic Report
Generated: 2026-04-21T08:29

## Headline numbers
- **Filtered FDs**: 311
- **Articles in unified pool**: 1163
- **Avg articles per FD**: 8.03
- **Median articles per FD**: 9
- **Median article-date spread per FD**: 12 days
- **Median total chars of evidence per FD**: 110116
- **Leakage audit (must be 0)**: 0 violations

## Per-source breakdown
| Source | N FDs | Date range | Avg articles | Median day-spread | Median chars |
|---|---|---|---|---|---|
| infer | 14 | 2024-08-02 → 2026-04-01 | 7.6 | 8 | 169122 |
| manifold | 30 | 2024-07-27 → 2026-04-12 | 7.5 | 17 | 94482 |
| metaculus | 47 | 2024-07-31 → 2026-04-07 | 7.2 | 11 | 113022 |
| polymarket | 220 | 2024-08-10 → 2026-04-08 | 8.3 | 12 | 110706 |

## Articles-per-FD histogram
- 3 articles: 22 FDs
- 4 articles: 22 FDs
- 5 articles: 18 FDs
- 6 articles: 17 FDs
- 7 articles: 28 FDs
- 8 articles: 25 FDs
- 9 articles: 36 FDs
- 10 articles: 143 FDs

## Ground-truth class balance
- **forecastbench**: {'No': 271, 'Yes': 40}  (Yes-rate = 12.9%)

## Crowd-probability distribution (ForecastBench)
- N = 311, mean = 0.163, median = 0.057
  - [0.0, 0.10): 191
  - [0.1, 0.20): 50
  - [0.2, 0.30): 14
  - [0.3, 0.40): 9
  - [0.4, 0.50): 10
  - [0.5, 0.60): 12
  - [0.6, 0.70): 4
  - [0.7, 0.80): 9
  - [0.8, 0.90): 6
  - [0.9, 1.01): 6

## Random 20-FD spot-check
| ID | Source | n_arts | Crowd | GT | t* | Question |
|---|---|---|---|---|---|---|
| fb_36934 | metaculus | 4 | 0.5 | Yes | 2025-06-13 | Will the word "tariff" disappear from the front pages of The |
| fb_0xc5db10faff | polymarket | 10 | 0.071 | No | 2024-12-31 | Will North Korea invade South Korea in 2024? |
| fb_30960 | metaculus | 4 | 0.04 | No | 2025-12-31 | Will CDC report 10,000 or more H5 avian influenza cases in t |
| fb_0xf78b84e417 | polymarket | 10 | 0.145 | No | 2025-12-31 | Trump takes Panama Canal in 2025? |
| fb_1674 | infer | 8 | 0.0948 | Yes | 2025-12-11 | Will the Cambodia-Thailand conflict result in at least 20 de |
| fb_0x5a378fe7c6 | polymarket | 6 | 0.023 | No | 2025-06-30 | Trump overturns Biden pardon before July? |
| fb_0x3ad0371eb2 | polymarket | 9 | 0.0375 | No | 2025-06-03 | Will Han Dong-hoon be elected the next president of South Ko |
| fb_40875 | metaculus | 4 | 0.028 | No | 2026-03-04 | Will the number of manufacturing jobs in the US for February |
| fb_0x8efb4c843e | polymarket | 8 | 0.215 | No | 2025-05-07 | Will Barcelona win the treble? |
| fb_41489 | metaculus | 7 | 0.125 | No | 2026-04-01 | Will Venezuela announce a presidential election before April |
| fb_34514 | metaculus | 10 | 0.5 | No | 2026-01-01 | Will the Department of Justice announce an investigation or  |
| fb_1285 | infer | 10 | 0.0186 | No | 2025-01-01 | Will a JCPOA participant country begin the process of imposi |
| fb_1290 | infer | 9 | 0.0248 | No | 2025-01-01 | Before 1 January 2025, will Iran announce that it will leave |
| fb_0xcf8a3c45dd | polymarket | 9 | 0.0005 | No | 2025-05-18 | Will Krzysztof Stanowski be the next President of Poland? |
| fb_0x8c866938d6 | polymarket | 7 | 0.004 | No | 2025-12-07 | Will Robert Negoiță be the next Mayor of Bucharest? |
| fb_12539 | metaculus | 5 | 0.001 | No | 2025-12-19 | Will Pierre Poilievre become Prime Minister of Canada before |
| fb_0xc512e0c0fa | polymarket | 10 | 0.5865 | No | 2026-02-01 | Will Golden (Ejae and Mark Sonnenblick) win Song of the Year |
| fb_0x4ccff8c606 | polymarket | 10 | 0.705 | Yes | 2026-04-08 | US x Iran ceasefire by December 31? |
| fb_0x9fce1292be | polymarket | 8 | 0.095 | No | 2024-12-31 | Will China invade Taiwan in 2024? |
| fb_0x6987d084de | polymarket | 10 | 0.83 | No | 2026-03-31 | Will Russia capture Kostyantynivka by March 31? |

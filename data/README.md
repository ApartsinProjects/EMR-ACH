# Data Directory

## MIRAI Benchmark

The MIRAI benchmark is available from the original paper repo:
https://github.com/yecchen/MIRAI

Files needed (place in this directory):
- `mirai_test_queries.jsonl` — test set queries
- `mirai_articles.jsonl` — news article corpus

### Expected query format (one JSON per line):
```json
{
  "id": "query_001",
  "timestamp": "2023-11-04",
  "subject": "Israel",
  "relation": "Accuse",
  "object": "Palestine",
  "label": "VK",
  "label_full": "Verbal Conflict",
  "doc_ids": ["article_123", "article_456", "article_789"]
}
```

Label mapping:
- VC = Verbal Cooperation (CAMEO codes 01-04)
- MC = Material Cooperation (CAMEO codes 05-08)
- VK = Verbal Conflict (codes 09-16)
- MK = Material Conflict (codes 17-20)

### Expected article format:
```json
{
  "id": "article_123",
  "title": "Israel accuses Hamas of violating ceasefire",
  "abstract": "...",
  "text": "...",
  "date": "2023-11-03",
  "source": "Reuters",
  "country_mentions": ["Israel", "Palestine"]
}
```

## ForecastBench

Available at: https://forecastbench.org

Subset used: geopolitics + conflict categories, Oct-Dec 2024, resolved questions only.
N = 300 questions.

Place as: `forecastbench_geopolitics.jsonl`

### Expected format:
```json
{
  "id": "fb_001",
  "question": "Will Israel and Hamas agree to a ceasefire by Dec 31, 2024?",
  "resolution_date": "2024-12-31",
  "ground_truth": 0,
  "crowd_probability": 0.32,
  "category": "geopolitics"
}
```

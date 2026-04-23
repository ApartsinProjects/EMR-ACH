# scripts/archive/

Quarantine for pre-v2.0 utilities that the build orchestrator no longer
invokes (v2.2 [B14]). Per the v2.2 backlog, the following scripts are
candidates to be moved here once a follow-up PR confirms zero call
sites:

- `download_gdelt_news.py` (pre-v2.0 GDELT raw downloader; superseded
  by the GDELT DOC archive path A1)
- `download_metaculus.py` (pre-v2.0 forecastbench fetcher; superseded
  by `fetch_forecastbench_news.py`)
- `prepare_data.py` (pre-v2.0 data wrangler)
- `scrape_market_pages.py` (pre-v2.0 market-page scraper)

Files in this directory are not on the orchestrator's import path; do
not import from them in production code. To resurrect a script,
copy it back to `scripts/` and re-validate against the current
config + data schemas.

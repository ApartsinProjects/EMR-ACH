"""Stage 1 of the ETD pipeline: articles -> facts via LLM batch extraction.

Reads articles from the unified article pool, extracts atomic forecast-relevant
facts per article using the canonical ETD extraction prompt, validates against
docs/etd.schema.json, and appends to data/etd/facts.current.

Design per docs/ETD_SPEC.md §6.1:
  - OpenAI Batch API (50% discount, 24h SLA)
  - Resumable: skips articles whose (article_id, extract_run) is already
    present in facts.current, unless --force
  - Debug / smoke mode: --smoke N runs N articles synchronously (direct mode)
    and dumps raw responses + parsed facts for prompt iteration
  - Schema-validated: every parsed fact checked against docs/etd.schema.json;
    failures go to facts.errors.jsonl

Usage:
  python scripts/articles_to_facts.py --smoke 5           # debug: 5 articles, sync, dump raw
  python scripts/articles_to_facts.py --smoke 20 --dry-run  # build requests, print first few, no API
  python scripts/articles_to_facts.py --limit 100         # batch-mode on 100 articles
  python scripts/articles_to_facts.py                      # full-corpus batch

Outputs:
  data/etd/facts.current            (symlink to data/etd/facts.v1.jsonl)
  data/etd/facts.v1.jsonl           (canonical per-run)
  data/etd/facts.errors.jsonl       (parse / validation failures)
  data/etd/extract_runs.jsonl       (provenance log)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.batch_client import BatchClient, BatchRequest, BatchResult  # type: ignore
from src.config import get_config  # type: ignore

try:
    from jsonschema import Draft202012Validator  # type: ignore
    _HAVE_JSONSCHEMA = True
except ImportError:
    _HAVE_JSONSCHEMA = False

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
ETD_DIR = DATA / "etd"
ETD_DIR.mkdir(parents=True, exist_ok=True)

FACTS_FILE  = ETD_DIR / "facts.v1.jsonl"
FACTS_CURR  = ETD_DIR / "facts.current"  # plain file that mirrors FACTS_FILE (Windows-friendly)
ERRORS_FILE = ETD_DIR / "facts.errors.jsonl"
RUNS_FILE   = ETD_DIR / "extract_runs.jsonl"

ARTICLES_FILE = DATA / "unified" / "articles.jsonl"
PROMPT_PATH   = ROOT / "docs" / "prompts" / "etd_extraction_v1.txt"
SCHEMA_PATH   = ROOT / "docs" / "etd.schema.json"

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_prompt_template() -> tuple[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return text, sha


def load_schema():
    if not _HAVE_JSONSCHEMA:
        return None
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def read_articles(limit: int | None = None) -> list[dict]:
    if not ARTICLES_FILE.exists():
        raise FileNotFoundError(f"Unified articles not found at {ARTICLES_FILE}. "
                                f"Run scripts/unify_articles.py first.")
    out = []
    for line in ARTICLES_FILE.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out.append(r)
        if limit and len(out) >= limit:
            break
    return out


def already_processed_article_ids(extract_run: str) -> set[str]:
    """Load article IDs that already have at least one fact from this extract_run.
    Used for resume: skip these on the next invocation."""
    done: set[str] = set()
    target = FACTS_CURR if FACTS_CURR.exists() else FACTS_FILE
    if not target.exists():
        return done
    for line in target.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("extract_run") == extract_run:
            # Each fact's article_ids is a list; mark all attested articles as "seen"
            for aid in r.get("article_ids") or []:
                done.add(aid)
    return done


def articles_in_error_file(extract_run: str) -> dict[str, int]:
    """article_id -> retry count for this run."""
    retries: dict[str, int] = {}
    if not ERRORS_FILE.exists():
        return retries
    for line in ERRORS_FILE.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        aid = r.get("article_id", "")
        retries[aid] = retries.get(aid, 0) + 1
    return retries


def append_jsonl(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt + request building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a news-analysis assistant. Extract forecast-relevant facts only. "
    "Respond with valid JSON only. No code fences, no prose."
)


def build_user_prompt(template: str, article: dict, max_body_chars: int = 4000) -> str:
    body = (article.get("text") or "").strip().replace("\n", " ")
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + " [truncated]"
    # Prompt file contains literal JSON braces; use str.replace (not .format) so
    # those don't get parsed as placeholders.
    slots = {
        "{article_id}":   article.get("id", ""),
        "{publish_date}": article.get("publish_date", "") or "unknown",
        "{source}":       article.get("source_domain", "") or "unknown",
        "{language}":     article.get("language") or "en",
        "{title}":        (article.get("title") or "").strip() or "(no title)",
        "{body}":         body or "(no body)",
    }
    out = template
    for k, v in slots.items():
        out = out.replace(k, str(v))
    return out


def build_request(article: dict, template: str, model: str, max_tokens: int = 900) -> BatchRequest:
    user = build_user_prompt(template, article)
    return BatchRequest(
        custom_id=f"etd::{article['id']}",
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user}],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type":"json_object"},
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _normalize_fact_text(s: str) -> str:
    s = (s or "").strip().strip('"').strip()
    s = re.sub(r"\s+", " ", s)
    if s and not s.endswith((".", "?", "!")):
        s += "."
    return s


def _fact_id(normalized_fact: str, primary_article_id: str, extract_run: str) -> str:
    key = f"{normalized_fact}|{primary_article_id}|{extract_run}".encode("utf-8")
    return "f_" + hashlib.sha1(key).hexdigest()[:12]


def parse_response(content: str, article: dict, extract_run: str, extractor: str,
                   validator=None) -> tuple[list[dict], list[dict]]:
    """Returns (valid_facts, validation_errors)."""
    if not content:
        return [], [{"error_type":"empty_response","error_detail":"LLM returned empty content"}]
    try:
        data = json.loads(content)
    except Exception as e:
        return [], [{"error_type":"json_parse","error_detail":f"{type(e).__name__}: {e}"}]
    raw_facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(raw_facts, list):
        return [], [{"error_type":"schema_violation","error_detail":"response.facts not a list"}]

    publish_date = article.get("publish_date", "") or ""
    valid: list[dict] = []
    errors: list[dict] = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

    for idx, rf in enumerate(raw_facts):
        if not isinstance(rf, dict):
            errors.append({"error_type":"schema_violation",
                           "error_detail":f"fact[{idx}] not an object"})
            continue
        fact_text = _normalize_fact_text(rf.get("fact",""))
        if not fact_text:
            errors.append({"error_type":"schema_violation",
                           "error_detail":f"fact[{idx}] empty text"})
            continue
        time_val = (rf.get("time") or "").strip() or "unknown"
        # Leakage guard: reject only facts dated STRICTLY AFTER article publish_date.
        # Same-day facts (time == publish_date) are the legitimate news of the day
        # and must be kept; experiment-time leakage protection is handled separately
        # by `apply_experiment_horizon()` in the baselines runner. The previous `>=`
        # check rejected ~9.9k same-day facts and accounted for >90% of the audit
        # error pool; see data/etd/audit/error_triage.md.
        if publish_date and re.match(r"^\d{4}-\d{2}-\d{2}$", time_val) and time_val > publish_date:
            errors.append({"error_type":"validation_failed",
                           "error_detail":f"fact[{idx}] time {time_val} > publish_date {publish_date}"})
            continue

        aid = article["id"]
        rec = {
            "id": _fact_id(fact_text, aid, extract_run),
            "schema_version": SCHEMA_VERSION,
            "time": time_val,
            "time_end": rf.get("time_end") if isinstance(rf.get("time_end"), str) else None,
            "time_precision": rf.get("time_precision") or ("day" if re.match(r"^\d{4}-\d{2}-\d{2}$", time_val) else
                                                           "month" if re.match(r"^\d{4}-\d{2}$", time_val) else
                                                           "year" if re.match(r"^\d{4}$", time_val) else "unknown"),
            "time_type": rf.get("time_type") or "point",
            "fact": fact_text,
            "language": rf.get("language") or article.get("language") or "en",
            "translated_from": rf.get("translated_from"),
            "article_ids": [aid],
            "primary_article_id": aid,
            "article_date": publish_date or "unknown",
            "source": article.get("source_domain") or None,
            "entities": rf.get("entities") or [],
            "location": rf.get("location"),
            "metrics": rf.get("metrics") or [],
            "kind": rf.get("kind") or None,
            "tags": rf.get("tags") or [],
            "polarity": rf.get("polarity") or "asserted",
            "attribution": rf.get("attribution"),
            "extraction_confidence": rf.get("extraction_confidence") or "medium",
            "source_tier": rf.get("source_tier") or None,
            "canonical_id": None,
            "variant_ids": [],
            "derived_from": [],
            "derivation": None,
            "extractor": extractor,
            "extract_run": extract_run,
            "extracted_at": now,
        }
        # Schema validation
        if validator is not None:
            errs = sorted(validator.iter_errors(rec), key=lambda e: e.path)
            if errs:
                errors.append({"error_type":"schema_violation",
                               "error_detail": "; ".join(f"{list(e.path)}: {e.message}" for e in errs[:3])})
                continue
        valid.append(rec)
    return valid, errors


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------

def build_run_id(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_etd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None,
                    help="Explicit extract_run identifier. Default: timestamp.")
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N articles (after resume filter).")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Debug mode: run N articles synchronously (direct API), dump raw responses and parsed facts.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build requests, print first 3, do not call API.")
    ap.add_argument("--force", action="store_true",
                    help="Re-process articles even if already present in facts.current for this run_id.")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Max attempts per article across runs (counts from facts.errors.jsonl).")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="If >0, split requests into concurrent sub-batches of this size via "
                         "src.common.multibatch.run_multibatch (5000 is a good default for >10k runs; "
                         "completes in ~1-3h instead of 4-12h for one giant batch).")
    args = ap.parse_args()

    extract_run = build_run_id(args.run_id)
    template, prompt_sha = load_prompt_template()
    validator = load_schema()
    if validator is None:
        print("[warn] jsonschema not installed; skipping schema validation. `pip install jsonschema` to enable.")

    # Load articles
    articles = read_articles(limit=None)
    print(f"[articles] {len(articles)} total articles in {ARTICLES_FILE.relative_to(ROOT)}")

    # Resume filter
    if args.force:
        done = set()
    else:
        done = already_processed_article_ids(extract_run)
    retries = articles_in_error_file(extract_run)
    before = len(articles)
    articles = [a for a in articles if a["id"] not in done and retries.get(a["id"], 0) < args.max_retries]
    print(f"[resume] {before - len(articles)} skipped (already processed or exceeded max-retries)")
    if args.limit:
        articles = articles[: args.limit]
    if args.smoke:
        articles = articles[: args.smoke]
    if not articles:
        print("[done] nothing to process")
        return

    print(f"[plan] extract_run={extract_run}  n_articles={len(articles)}  model={args.model}  prompt_sha={prompt_sha}")

    # Build requests
    requests = [build_request(a, template, args.model, args.max_tokens) for a in articles]
    est_input = sum(len(r.messages[1]["content"]) // 4 for r in requests)
    est_output = sum(r.max_tokens // 2 for r in requests)
    in_cost = est_input / 1e6 * 0.150 * 0.5   # gpt-4o-mini batch input
    out_cost = est_output / 1e6 * 0.600 * 0.5
    print(f"[cost] ~{est_input} input tokens + ~{est_output} output -> ~${in_cost + out_cost:.3f} batch cost")

    if args.dry_run:
        print("\n[dry-run] Sample request 1:")
        print(requests[0].messages[1]["content"][:1500])
        print("...")
        if len(requests) >= 2:
            print("\n[dry-run] Sample request 2 custom_id:", requests[1].custom_id)
        return

    # Run via BatchClient (direct for smoke, batch otherwise).
    # For large corpora, --chunk-size enables concurrent sub-batches (see
    # src/common/multibatch.py for details). Typical sweet spot: 5000 per chunk.
    cfg = get_config()
    mode = "direct" if args.smoke else "batch"
    client = BatchClient(mode=mode, config=cfg)
    job_name = f"etd_{extract_run}" + ("_smoke" if args.smoke else "")
    if args.chunk_size > 0 and mode == "batch" and len(requests) > args.chunk_size:
        from src.common.multibatch import run_multibatch
        print(f"[run] mode=multibatch  chunk_size={args.chunk_size}  job_name={job_name}")
        results = run_multibatch(client, requests, job_name=job_name,
                                  chunk_size=args.chunk_size)
    else:
        print(f"[run] mode={mode} job_name={job_name}")
        results = client.run(requests, job_name=job_name)

    # Parse + validate + persist
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    n_emitted = 0
    n_articles_errored = 0
    n_articles_ok = 0
    smoke_dump: list[dict] = []

    for art in articles:
        cid = f"etd::{art['id']}"
        res = results.get(cid)
        content = res.content if isinstance(res, BatchResult) else ""
        facts, errors = parse_response(content, art, extract_run, args.model, validator)

        # Persist
        for f in facts:
            append_jsonl(FACTS_FILE, f)
            n_emitted += 1
        for e in errors:
            rec = {
                "article_id": art["id"],
                "extract_run": extract_run,
                "extractor": args.model,
                "error_type": e.get("error_type","unknown"),
                "error_detail": e.get("error_detail",""),
                "raw_response": (content or "")[:4000],
                "failed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z"),
            }
            append_jsonl(ERRORS_FILE, rec)

        if facts:
            n_articles_ok += 1
        elif errors:
            n_articles_errored += 1

        if args.smoke:
            smoke_dump.append({
                "article_id": art["id"],
                "title": art.get("title",""),
                "publish_date": art.get("publish_date",""),
                "response_raw": content,
                "facts_parsed": facts,
                "errors": errors,
            })

    # Keep facts.current in sync (plain copy on Windows, symlink-like on POSIX)
    try:
        if FACTS_CURR.exists():
            FACTS_CURR.unlink()
        FACTS_CURR.write_bytes(FACTS_FILE.read_bytes())
    except Exception as e:
        print(f"[warn] could not refresh facts.current: {e}")

    completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

    # Quality metrics
    valid_time = 0; with_entities = 0
    for f in (json.loads(l) for l in FACTS_FILE.open(encoding="utf-8") if l.strip()):
        if f.get("extract_run") != extract_run: continue
        if re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", f.get("time","")): valid_time += 1
        if f.get("entities"): with_entities += 1
    parse_fail_rate = n_articles_errored / max(1, len(articles))
    quality = {
        "pct_facts_with_valid_time": round(100 * valid_time / max(1, n_emitted), 2),
        "pct_facts_with_entities":   round(100 * with_entities / max(1, n_emitted), 2),
        "mean_facts_per_article":    round(n_emitted / max(1, len(articles)), 2),
        "parse_failure_rate":        round(parse_fail_rate, 4),
    }

    # Append run log
    run_rec = {
        "run_id":                  extract_run,
        "extractor":               args.model,
        "prompt_path":             str(PROMPT_PATH.relative_to(ROOT)),
        "prompt_sha1":             prompt_sha,
        "mode":                    mode,
        "started_at":              started_at,
        "completed_at":            completed_at,
        "n_articles_input":        len(articles),
        "n_articles_processed":    n_articles_ok + n_articles_errored,
        "n_articles_errored":      n_articles_errored,
        "n_facts_emitted":         n_emitted,
        "quality":                 quality,
        "notes":                   "smoke" if args.smoke else "full",
    }
    append_jsonl(RUNS_FILE, run_rec)

    # Summary
    print(f"\n[done] run={extract_run}")
    print(f"  articles processed : {n_articles_ok + n_articles_errored}")
    print(f"  articles w/ facts  : {n_articles_ok}")
    print(f"  articles w/ errors : {n_articles_errored}")
    print(f"  facts emitted      : {n_emitted}")
    print(f"  facts file         : {FACTS_FILE.relative_to(ROOT)}")
    print(f"  errors file        : {ERRORS_FILE.relative_to(ROOT)}")
    print(f"  quality            : {quality}")

    # Dump smoke samples
    if args.smoke and smoke_dump:
        dump_path = ETD_DIR / f"smoke_dump_{extract_run}.json"
        dump_path.write_text(json.dumps(smoke_dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  smoke dump         : {dump_path.relative_to(ROOT)}")
        print("\n--- FIRST SMOKE SAMPLE ---")
        s = smoke_dump[0]
        print(f"article_id: {s['article_id']}")
        print(f"title: {s['title'][:120]}")
        print(f"publish_date: {s['publish_date']}")
        print(f"\n[raw response, first 800 chars]:\n{s['response_raw'][:800]}")
        print(f"\n[parsed facts: {len(s['facts_parsed'])}]:")
        for i, f in enumerate(s["facts_parsed"]):
            print(f"  [{i+1}] {f.get('time','?')}  {f.get('fact','')[:120]}")
        if s["errors"]:
            print(f"\n[errors: {len(s['errors'])}]:")
            for e in s["errors"]:
                print(f"  {e['error_type']}: {e['error_detail'][:200]}")


if __name__ == "__main__":
    main()

"""
OpenAI Batch API wrapper.

Two modes:
  - "batch": submit a batch job (50% cost, ~24h turnaround). Used for full experiments.
  - "direct": call the API synchronously, one by one. Used for smoke tests.

Usage:
    client = BatchClient(mode="direct")            # for smoke tests
    client = BatchClient(mode="batch")             # for experiments

    requests = [
        BatchRequest(custom_id="q1", messages=[...], model="gpt-4o", max_tokens=1000),
        ...
    ]
    results = client.run(requests, job_name="indicators_full")
    # returns dict: custom_id -> response_text
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.config import get_config


@dataclass
class BatchRequest:
    custom_id: str
    messages: list[dict]
    model: str | None = None        # if None, uses config.experiment_model
    max_tokens: int = 2048
    temperature: float | None = None
    response_format: dict | None = None   # e.g. {"type": "json_object"}
    extra: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    custom_id: str
    content: str            # raw text response
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class BatchClient:
    def __init__(self, mode: str = "batch", config=None):
        assert mode in ("batch", "direct"), f"mode must be 'batch' or 'direct', got {mode!r}"
        self.mode = mode
        self.cfg = config or get_config()
        self._client = OpenAI(api_key=self.cfg.openai_api_key)
        self._jobs_file = Path(self.cfg.results_dir) / "batch_jobs.json"
        self._jobs_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        requests: list[BatchRequest],
        job_name: str,
    ) -> dict[str, BatchResult]:
        """Submit requests and block until all results are available.

        Returns a mapping custom_id -> BatchResult.
        Results are cached to results/raw/{job_name}.jsonl.
        Re-running with the same job_name returns cached results.
        """
        cache_path = Path(self.cfg.results_dir) / "raw" / f"{job_name}.jsonl"
        if cache_path.exists():
            print(f"[batch] Using cached results for '{job_name}': {cache_path}")
            return self._load_results(cache_path)

        if self.mode == "direct":
            results = self._run_direct(requests)
        else:
            results = self._run_batch(requests, job_name)

        self._save_results(results, cache_path)
        return results

    def estimate_cost(self, requests: list[BatchRequest]) -> dict:
        """Rough cost estimate without making any API calls."""
        total_input = sum(
            sum(len(m["content"]) // 4 for m in r.messages) for r in requests
        )
        total_output = sum(r.max_tokens // 2 for r in requests)  # assume 50% of max used
        input_cost = total_input / 1_000_000 * 2.50   # gpt-4o batch input
        output_cost = total_output / 1_000_000 * 10.0  # gpt-4o batch output
        return {
            "n_requests": len(requests),
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round((input_cost + output_cost) * 0.5, 4),  # 50% batch discount
        }

    # ------------------------------------------------------------------
    # Direct mode (smoke tests)
    # ------------------------------------------------------------------

    def _run_direct(self, requests: list[BatchRequest]) -> dict[str, BatchResult]:
        results = {}
        model = requests[0].model or self.cfg.smoke_model
        temp = self.cfg.get("model", "temperature", default=0.0)

        print(f"[direct] Running {len(requests)} requests with {model}...")
        for req in tqdm(requests, desc="direct API"):
            try:
                kwargs: dict[str, Any] = dict(
                    model=req.model or model,
                    messages=req.messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature if req.temperature is not None else temp,
                )
                if req.response_format:
                    kwargs["response_format"] = req.response_format
                kwargs.update(req.extra)

                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                results[req.custom_id] = BatchResult(
                    custom_id=req.custom_id,
                    content=content,
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                )
            except Exception as exc:
                results[req.custom_id] = BatchResult(
                    custom_id=req.custom_id,
                    content="",
                    error=str(exc),
                )
                print(f"  [ERROR] {req.custom_id}: {exc}")
        return results

    # ------------------------------------------------------------------
    # Batch mode (full experiments)
    # ------------------------------------------------------------------

    def _run_batch(self, requests: list[BatchRequest], job_name: str) -> dict[str, BatchResult]:
        # Check if a batch job already exists for this job_name
        jobs = self._load_jobs()
        existing = jobs.get(job_name)

        if existing:
            print(f"[batch] Resuming existing batch job '{job_name}': {existing['batch_id']}")
            batch = self._client.batches.retrieve(existing["batch_id"])
        else:
            model = requests[0].model or self.cfg.experiment_model
            temp = self.cfg.get("model", "temperature", default=0.0)
            print(f"[batch] Submitting {len(requests)} requests as '{job_name}' (model={model})")
            batch = self._submit_batch(requests, job_name, model, temp)
            jobs[job_name] = {"batch_id": batch.id, "job_name": job_name, "status": batch.status}
            self._save_jobs(jobs)

        # Poll until done
        batch = self._poll(batch, job_name, jobs)

        if batch.status != "completed":
            raise RuntimeError(
                f"Batch '{job_name}' finished with status '{batch.status}'. "
                f"Error: {batch.errors}"
            )

        return self._download_results(batch)

    def _submit_batch(self, requests, job_name, model, temperature) -> Any:
        # Write JSONL input file
        input_path = Path(self.cfg.results_dir) / "raw" / f"{job_name}_input.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "w") as f:
            for req in requests:
                body: dict[str, Any] = dict(
                    model=req.model or model,
                    messages=req.messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature if req.temperature is not None else temperature,
                )
                if req.response_format:
                    body["response_format"] = req.response_format
                body.update(req.extra)
                entry = {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(entry) + "\n")

        # Upload
        with open(input_path, "rb") as f:
            batch_file = self._client.files.create(file=f, purpose="batch")
        print(f"  Uploaded input file: {batch_file.id}")

        # Create batch
        batch = self._client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=self.cfg.get("batch", "completion_window", default="24h"),
            metadata={"job_name": job_name},
        )
        print(f"  Batch created: {batch.id} (status={batch.status})")
        return batch

    def _poll(self, batch, job_name, jobs) -> Any:
        interval = self.cfg.get("batch", "poll_interval_seconds", default=60)
        terminal = {"completed", "failed", "cancelled", "expired"}
        print(f"[batch] Polling '{job_name}' every {interval}s...")

        while batch.status not in terminal:
            time.sleep(interval)
            batch = self._client.batches.retrieve(batch.id)
            req_counts = batch.request_counts
            print(
                f"  status={batch.status}  "
                f"completed={req_counts.completed}  "
                f"failed={req_counts.failed}  "
                f"total={req_counts.total}"
            )
            jobs[job_name]["status"] = batch.status
            self._save_jobs(jobs)

        return batch

    def _download_results(self, batch) -> dict[str, BatchResult]:
        print(f"[batch] Downloading results (file_id={batch.output_file_id})...")
        raw = self._client.files.content(batch.output_file_id)
        results = {}
        for line in raw.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            if obj.get("error"):
                results[custom_id] = BatchResult(
                    custom_id=custom_id,
                    content="",
                    error=str(obj["error"]),
                )
            else:
                body = obj["response"]["body"]
                choice = body["choices"][0]
                usage = body.get("usage", {})
                results[custom_id] = BatchResult(
                    custom_id=custom_id,
                    content=choice["message"]["content"] or "",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_results(self, results: dict[str, BatchResult], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results.values():
                f.write(json.dumps({
                    "custom_id": r.custom_id,
                    "content": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                }) + "\n")
        print(f"[batch] Saved {len(results)} results to {path}")

    def _load_results(self, path: Path) -> dict[str, BatchResult]:
        results = {}
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                results[obj["custom_id"]] = BatchResult(**obj)
        return results

    def _load_jobs(self) -> dict:
        if self._jobs_file.exists():
            with open(self._jobs_file) as f:
                return json.load(f)
        return {}

    def _save_jobs(self, jobs: dict) -> None:
        with open(self._jobs_file, "w") as f:
            json.dump(jobs, f, indent=2)


def parse_json_response(content: str) -> dict | list | None:
    """Parse LLM response as JSON, stripping markdown code fences if present."""
    text = content.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

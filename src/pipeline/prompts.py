"""Load and render YAML prompt templates."""

from pathlib import Path
from typing import Any

import yaml

from src.config import get_config

_CACHE: dict[str, dict] = {}


def load_prompt(name: str, config=None) -> dict:
    """Load prompts/{name}.yaml, caching the result."""
    if name not in _CACHE:
        cfg = config or get_config()
        path = cfg.prompts_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path) as f:
            _CACHE[name] = yaml.safe_load(f)
    return _CACHE[name]


def render(template_str: str, **kwargs: Any) -> str:
    """Fill {variable} placeholders in a template string."""
    try:
        return template_str.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}") from e


def build_messages(
    prompt_name: str,
    variables: dict[str, Any],
    config=None,
) -> list[dict]:
    """Build OpenAI messages list from a named prompt template."""
    tmpl = load_prompt(prompt_name, config)
    system_text = render(tmpl.get("system", ""), **variables)
    user_text = render(tmpl["user"], **variables)
    msgs = []
    if system_text.strip():
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def format_indicators_list(indicators: list[dict]) -> str:
    """Format indicators list for prompt injection."""
    lines = []
    for ind in indicators:
        lines.append(f"  {ind['id']}. {ind['text']}")
    return "\n".join(lines)


def format_hypotheses_block(config=None) -> str:
    cfg = config or get_config()
    tmpl = load_prompt("indicators", cfg)
    return tmpl["hypotheses_block_template"].strip()


def format_articles_block(articles: list[dict]) -> str:
    """Format articles for multi-agent prompt injection."""
    lines = []
    for i, art in enumerate(articles, 1):
        lines.append(f"[{art['id']}] {art['title']}")
        content = art.get("abstract") or art.get("text", "")[:400]
        lines.append(f"  {content}")
        lines.append("")
    return "\n".join(lines)


def format_advocates_block(advocates: list[dict]) -> str:
    """Format advocate outputs for judge prompt injection."""
    lines = []
    for adv in advocates:
        h = adv["hypothesis"]
        lines.append(f"=== ADVOCATE FOR {h} ===")
        lines.append(f"Argument: {adv.get('argument_summary', '')}")
        lines.append(f"Key evidence: {'; '.join(adv.get('key_evidence', []))}")
        lines.append(f"Confidence: {adv.get('confidence', 0):.2f}")
        lines.append("")
    return "\n".join(lines)

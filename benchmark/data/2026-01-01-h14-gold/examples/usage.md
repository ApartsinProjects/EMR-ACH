# Five example queries (zero-dep, stdlib only)

## 1. Count FDs per benchmark and per fd_type

```python
import json, collections
fds = [json.loads(l) for l in open("forecasts.jsonl", encoding="utf-8")]
ctr = collections.Counter((d["benchmark"], d["fd_type"]) for d in fds)
for k, n in sorted(ctr.items()): print(f"{k}: {n}")
```

## 2. Find the change subset for one benchmark

```python
change = [d for d in fds if d["benchmark"] == "earnings" and d["fd_type"] == "change"]
print(f"Earnings change subset: {len(change)} FDs")
```

## 3. Look up the evidence pool for a specific FD

```python
arts = {a["id"]: a for a in (json.loads(l) for l in open("articles.jsonl", encoding="utf-8"))}
fd = next(d for d in fds if d["benchmark"] == "earnings")
for aid in fd["article_ids"]:
    a = arts[aid]
    print(a["publish_date"], a["url"])
```

## 4. Compute majority-class baseline accuracy on the change subset

```python
correct = sum(1 for d in change if d["ground_truth"] == "Surprise")
print(f"All-Surprise baseline accuracy: {correct/len(change):.2f}")
```

## 5. Render a baseline prompt without the EMR-ACH codebase

See `eval_template.py` for the full pattern. Sketch:

```python
def render_prompt(fd, articles, max_chars_per_article=600):
    arts = [articles[a] for a in fd["article_ids"][:10] if a in articles]
    hyp_block = "\n".join(f"  - {h}: {fd['hypothesis_definitions'].get(h, '')}"
                           for h in fd["hypothesis_set"])
    art_block = "\n\n".join(f"[A{i+1}] {a['publish_date']} {a.get('source_domain', '')}: "
                              f"{a['title']}\n{a.get('text', '')[:max_chars_per_article]}"
                              for i, a in enumerate(arts))
    return f"Question: {fd['question']}\n\nHypotheses:\n{hyp_block}\n\nEvidence:\n{art_block}"
```

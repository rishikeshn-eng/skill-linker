"""Skill demand index: share of postings mentioning a skill, with Wilson intervals and an early-vs-late comparison."""
from __future__ import annotations

import math
from collections import Counter


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def demand_index(postings: list[dict], min_count: int = 5) -> dict:
    """postings: [{'t': sortable time, 'skills': iterable of uris}] -> per-skill shares overall and early/late halves.

    A skill counts once per posting. 'Rising' needs the late-half share's Wilson interval to sit above the early
    half's, which is a weak test on small samples; use it as a lead, not a finding.
    """
    ps = sorted(postings, key=lambda p: p["t"])
    n = len(ps)
    half = n // 2
    early, late = ps[:half], ps[half:]
    cnt = Counter(u for p in ps for u in set(p["skills"]))
    ce = Counter(u for p in early for u in set(p["skills"]))
    cl = Counter(u for p in late for u in set(p["skills"]))
    rows = []
    for u, k in cnt.items():
        if k < min_count:
            continue
        lo, hi = wilson(k, n)
        el, eh = wilson(ce[u], len(early))
        ll, lh = wilson(cl[u], len(late))
        rows.append({"uri": u, "postings": k, "share": k / n, "ci": [lo, hi],
                     "early_share": ce[u] / max(1, len(early)), "late_share": cl[u] / max(1, len(late)),
                     "trend": "rising" if ll > eh else "falling" if lh < el else "flat"})
    rows.sort(key=lambda r: -r["postings"])
    return {"n_postings": n, "n_early": len(early), "n_late": len(late), "skills": rows}

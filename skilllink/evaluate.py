"""Score the linker against a labelled CSV: columns text, uri ('NONE' when ESCO has no equivalent), optional note."""
from __future__ import annotations

import csv
from collections import Counter

from .linker import Linker


def load_gold(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["uri"].strip()]


def evaluate(linker: Linker, gold: list[dict], thresholds=(0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)) -> dict:
    gold = [dict(g, ok=set(g["uri"].split("|"))) for g in gold]   # several URIs may be acceptable
    rows = []
    for g in gold:
        l = linker.link(g["text"])
        rows.append((g, l))
    linkable = [(g, l) for g, l in rows if g["uri"] != "NONE"]
    none = [(g, l) for g, l in rows if g["uri"] == "NONE"]

    def at(th):
        """Precision/coverage if the linker answered only when its best score >= th (exact hits always score 1.0)."""
        ans = [(g, l) for g, l in rows if l.candidates and l.candidates[0][1] >= th]
        correct = sum(1 for g, l in ans if g["uri"] != "NONE" and l.candidates[0][0] in g["ok"])
        return {"threshold": th, "answered": len(ans), "coverage": len(ans) / len(rows),
                "precision": correct / max(1, len(ans)),
                "correct_abstain_on_none": sum(1 for g, l in none if not (l.candidates and l.candidates[0][1] >= th)) / max(1, len(none))}

    top1 = sum(1 for g, l in linkable if l.uri in g["ok"])
    top5 = sum(1 for g, l in linkable if g["ok"] & {u for u, _ in l.candidates[:5]})
    by_method = Counter(l.method for _, l in rows)
    exact_ok = sum(1 for g, l in linkable if l.method == "exact" and l.uri in g["ok"])
    errors = [{"text": g["text"], "gold": g["uri"], "pred": l.uri, "method": l.method, "score": round(l.score, 2),
               "pred_label": l.label} for g, l in linkable if l.uri not in g["ok"]][:60]
    return {
        "n": len(rows), "n_linkable": len(linkable), "n_none": len(none),
        "top1_accuracy_on_linkable": top1 / max(1, len(linkable)),
        "top5_recall_on_linkable": top5 / max(1, len(linkable)),
        "exact_precision": exact_ok / max(1, sum(1 for _, l in rows if l.method == "exact")),
        "methods": dict(by_method),
        "correct_abstain_on_none": sum(1 for g, l in none if l.uri is None) / max(1, len(none)),
        "end_to_end_accuracy": (sum(1 for g, l in linkable if l.uri in g["ok"]) + sum(1 for g, l in none if l.uri is None)) / len(rows),
        "sweep": [at(t) for t in thresholds], "errors": errors,
    }

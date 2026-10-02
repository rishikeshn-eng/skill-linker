"""Pipeline: evaluate on the labelled set, extract skills from real postings, build the demand index -> docs/results.json."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import demand, esco, evaluate, extract
from .linker import Linker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold/labels.csv")
    ap.add_argument("--postings", default="data/postings_arbeitnow.json.gz")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--out", default="docs/results.json")
    a = ap.parse_args()
    skills = esco.load()
    ix = esco.Index(skills)
    lk = Linker(ix, threshold=a.threshold)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "esco": {"skills": len(skills), "labels": len(ix.labels)}, "threshold": a.threshold}

    if Path(a.gold).exists():
        import hashlib
        gold = evaluate.load_gold(a.gold)
        dev = [g for g in gold if int(hashlib.md5(g["text"].encode()).hexdigest(), 16) % 2 == 0]
        test = [g for g in gold if g not in dev]
        grid = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
        e2e = {}
        for th in grid:
            lk.threshold = th
            e2e[th] = evaluate.evaluate(lk, dev)["end_to_end_accuracy"]
        lk.threshold = max(e2e, key=e2e.get)   # chosen on dev only
        out["threshold"] = lk.threshold
        out["tuning"] = {"dev_n": len(dev), "test_n": len(test), "dev_end_to_end_by_threshold": e2e,
                         "test": {k: v for k, v in evaluate.evaluate(lk, test).items() if k not in ("errors", "sweep")}}
        print("chosen threshold (dev):", lk.threshold, "| held-out test:", {k: round(v, 3) for k, v in out["tuning"]["test"].items() if isinstance(v, float)})
        out["eval"] = evaluate.evaluate(lk, gold)
        e = out["eval"]
        print(f"gold n={e['n']} linkable={e['n_linkable']}: top1={e['top1_accuracy_on_linkable']:.3f} top5={e['top5_recall_on_linkable']:.3f} "
              f"exact_precision={e['exact_precision']:.3f} abstain_on_none={e['correct_abstain_on_none']:.3f}")

    if Path(a.postings).exists():
        jobs = json.load(gzip.open(a.postings, "rt"))
        ex = extract.Extractor(ix)
        posts, en = [], 0
        for j in jobs:
            text = extract.clean_html(j["description"]) + " " + j["title"]
            if not extract.is_english(text):
                continue
            en += 1
            found = ex.extract(text)
            posts.append({"t": j["created_at"], "skills": [u for _, u in found], "spans": found, "title": j["title"]})
        d = demand.demand_index(posts, min_count=6)
        lab = {u: ix.pref(u) for u in ix.skills}
        for r in d["skills"]:
            r["label"] = lab[r["uri"]]
        n_skill = sum(1 for p in posts if p["skills"])
        out["postings"] = {"source": "Arbeitnow public job-board API", "fetched": len(jobs), "english": en,
                           "with_any_skill": n_skill, "mean_skills_per_posting": sum(len(set(p["skills"])) for p in posts) / max(1, en),
                           "date_from": datetime.fromtimestamp(min(p["t"] for p in posts), timezone.utc).date().isoformat(),
                           "date_to": datetime.fromtimestamp(max(p["t"] for p in posts), timezone.utc).date().isoformat()}
        out["demand"] = {**d, "skills": d["skills"][:60]}
        print(f"postings: {len(jobs)} fetched, {en} English, {n_skill} with a skill; {len(d['skills'])} skills with >=6 postings")
        review = Path("gold/extraction_review.csv")
        if review.exists():
            rows = list(csv.DictReader(open(review, encoding="utf-8")))
            ok = sum(1 for r in rows if r["correct"].strip() == "1")
            out["extraction_precision"] = {"n": len(rows), "correct": ok, "precision": ok / max(1, len(rows))}
            print("extraction precision (reviewed):", out["extraction_precision"])
        # review sample for hand checking (deterministic)
        import random
        rng = random.Random(1)
        allspans = [(p["title"], s, u) for p in posts for s, u in p["spans"]]
        sample = rng.sample(allspans, min(150, len(allspans)))
        Path("data").mkdir(exist_ok=True)
        with open("data/extraction_sample.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title", "span", "uri", "label", "correct"])
            for t, s, u in sample:
                w.writerow([t, s, u, ix.pref(u), ""])
    Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()

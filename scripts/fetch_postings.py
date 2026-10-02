"""Fetch real postings from Arbeitnow's free public job-board API (https://www.arbeitnow.com/api/job-board-api).

    python scripts/fetch_postings.py --pages 6

Arbeitnow asks API users not to abuse it and to link back to the site. Descriptions are kept in data/ (git-ignored) for
local skill extraction; only extracted skills and coarse metadata are published by the pipeline.
"""
import argparse, gzip, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--pages", type=int, default=6)
a = ap.parse_args()
jobs = []
for p in range(1, a.pages + 1):
    with urllib.request.urlopen(urllib.request.Request(f"https://www.arbeitnow.com/api/job-board-api?page={p}", headers={"User-Agent": "skill-linker/0.1"}), timeout=60) as r:
        d = json.load(r)
    jobs += d["data"]
    print("page", p, len(d["data"]), flush=True)
    if not d["data"]:
        break
    time.sleep(1.5)
with gzip.open("data/postings_arbeitnow.json.gz", "wt") as f:
    json.dump(jobs, f)
print(len(jobs), "postings")

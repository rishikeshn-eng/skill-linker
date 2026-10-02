"""Harvest ESCO skills (English labels + alt labels) through ESCO's public web service.

    python scripts/harvest_esco.py --max-requests 2500

ESCO's bulk CSV download sits behind a form, so this walks the public search API instead: it queries seed words,
pages through every result list (the API's `offset` is a PAGE index), and snowballs using words found in labels
until few new skills appear. It is polite (0.3 s between calls), resumable, and writes data/esco_skills_en.json.gz.
The result may miss skills that no query surfaces; the script prints coverage against the API's own totals.

ESCO is (c) European Union; see https://esco.ec.europa.eu for licence terms and cite it when you reuse the data.
"""
import argparse, collections, gzip, json, os, re, time, urllib.parse, urllib.request

API = "https://ec.europa.eu/esco/api/search"
OUT = "data/esco_skills_en.json.gz"
SEEDS = ["python", "java", "sql", "data", "software", "communication", "management", "customer", "sales", "teaching",
         "nursing", "cooking", "driving", "welding", "accounting", "marketing", "design", "cloud", "security", "network",
         "language", "legal", "finance", "health", "safety", "machine", "maintenance", "research", "writing", "project"]
STOP = set("a an the of and or to in for on with by at from as use using apply perform work ensure be is are".split())


def get(params, tries=4):
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "skill-linker/0.1"}), timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** i)
    return None


def load():
    if os.path.exists(OUT):
        return {r["uri"]: r for r in json.load(gzip.open(OUT, "rt"))}
    return {}


def save(skills):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt") as f:
        json.dump(sorted(skills.values(), key=lambda r: r["uri"]), f)


def crawl_query(q, budget, lock, skills, words, counter):
    """Page through one query; returns (new skills, last total)."""
    page, new, total = 0, 0, None
    while counter["reqs"] < budget:
        d = get({"text": q, "type": "skill", "language": "en", "limit": 100, "offset": page, "full": "true"})
        with lock:
            counter["reqs"] += 1
        time.sleep(0.3)
        res = (d or {}).get("_embedded", {}).get("results", [])
        if not res:
            break
        total = d["total"]
        with lock:
            for x in res:
                u = x["uri"]
                en = x.get("preferredLabel", {}).get("en") or x.get("title")
                if u in skills or not en:
                    continue
                desc = x.get("description", {})
                desc = (desc.get("en", {}) or {}).get("literal", "") if isinstance(desc, dict) else ""
                skills[u] = {"uri": u, "pref": en, "alts": x.get("alternativeLabel", {}).get("en", []), "desc": desc[:200]}
                new += 1
                for w in re.findall(r"[a-z]{4,}", en.lower()):
                    words[w] += 1
        page += 1
        if page * 100 >= total:
            break
    return new, total


def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-requests", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    skills, done_q, lock, counter = load(), {"python", "java", "sql", "data"}, threading.Lock(), {"reqs": 0}
    words = collections.Counter(w for r in skills.values() for w in re.findall(r"[a-z]{4,}", r["pref"].lower()))
    queue = [q for q in SEEDS if q not in done_q]
    while queue and counter["reqs"] < a.max_requests:
        batch, queue = queue[:a.workers * 2], queue[a.workers * 2:]
        done_q.update(batch)
        with ThreadPoolExecutor(a.workers) as ex:
            for q, (new, total) in zip(batch, ex.map(lambda q: crawl_query(q, a.max_requests, lock, skills, words, counter), batch)):
                print(f"{q!r}: total={total} new={new} skills={len(skills)} requests={counter['reqs']}", flush=True)
        save(skills)
        if not queue:
            queue = [w for w, _ in words.most_common(500) if w not in done_q and w not in STOP][:24]
    print("done", len(skills), "skills,", counter["reqs"], "requests")


if __name__ == "__main__":
    main()

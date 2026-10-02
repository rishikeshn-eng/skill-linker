"""For each gold string, collect candidate ESCO skills from TWO independent sources, to label against:
  1. this linker's top 8
  2. ESCO's own public search API top 5 (so labels are not limited to what the linker can already see)
Writes gold/candidates.json. Labels (gold/labels.csv) are then assigned by reading these candidates.
"""
import json, os, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skilllink import esco
from skilllink.linker import Linker

ix = esco.Index(esco.load())
lk = Linker(ix)
strings = [s.strip() for s in open("gold/strings.txt") if s.strip()]


def api(text):
    url = "https://ec.europa.eu/esco/api/search?" + urllib.parse.urlencode(
        {"text": text, "type": "skill", "language": "en", "limit": 5, "full": "false"})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "skill-linker/0.1"}), timeout=60) as r:
            return [(x["uri"], x["title"]) for x in json.load(r)["_embedded"]["results"]]
    except Exception:
        return []

with ThreadPoolExecutor(4) as ex:
    api_hits = list(ex.map(api, strings))
out = []
for s, ah in zip(strings, api_hits):
    l = lk.link(s)
    cands = {}
    for u, sc in l.candidates:
        cands[u] = {"uri": u, "label": ix.pref(u), "src": ["linker"], "score": round(sc, 2)}
    for u, t in ah:
        cands.setdefault(u, {"uri": u, "label": t, "src": [], "score": None})["src"].append("api")
    out.append({"text": s, "linker": l.label, "candidates": list(cands.values())})
json.dump(out, open("gold/candidates.json", "w"), indent=1)
print(len(out), "strings;", sum(1 for o in out for c in o["candidates"] if "api" in c["src"] and c["uri"] not in ix.skills), "API candidates absent from the harvested subset")

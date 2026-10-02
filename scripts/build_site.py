"""Write docs/esco_skills.json (slim ESCO subset for the in-browser linker) and docs/label_pool.json (labelling tool)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skilllink import esco
from skilllink.linker import Linker

skills = esco.load()
json.dump([{"u": s["uri"].rsplit("/", 1)[-1], "p": s["pref"], "a": s.get("alts", [])} for s in skills],
          open("docs/esco_skills.json", "w"), separators=(",", ":"))
ix = esco.Index(skills)
lk = Linker(ix)
strings = [s.strip() for s in open("gold/strings.txt") if s.strip()]
pool = []
for s in strings:
    l = lk.link(s)
    pool.append({"text": s, "pick": l.uri if l.method in ("exact", "fuzzy") else None,
                 "cands": [{"uri": u, "label": ix.pref(u), "desc": ix.skills[u].get("desc", "")[:140]} for u, _ in l.candidates[:8]]})
json.dump(pool, open("docs/label_pool.json", "w"), separators=(",", ":"))
print(len(skills), "skills,", len(pool), "pool strings;", os.path.getsize("docs/esco_skills.json") // 1024, "KiB esco_skills.json")

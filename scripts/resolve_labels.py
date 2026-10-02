"""gold/label_spec.txt (idx|label[|alt label]) + gold/candidates.json -> gold/labels.csv (uri may hold several acceptable URIs joined by '|').

Labels were assigned by one LLM annotator (Claude) reading candidates from the linker and ESCO's own search API; they are
NOT human-verified. note column says so.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skilllink import esco

cands = json.load(open("gold/candidates.json"))
by_label = {}
for s in esco.load():
    by_label.setdefault(s["pref"].lower(), s["uri"])
for o in cands:
    for c in o["candidates"]:
        by_label.setdefault(c["label"].lower(), c["uri"])

def resolve(lab):
    l = lab.lower()
    if l in by_label: return by_label[l]
    pre = [u for k, u in by_label.items() if k.startswith(l) or l.startswith(k)]
    return pre[0] if len(set(pre)) >= 1 and len(set(pre)) == 1 else None

rows, bad = [], []
for line in open("gold/label_spec.txt"):
    parts = line.rstrip("\n").split("|")
    i, labs = int(parts[0]), parts[1:]
    text = cands[i]["text"]
    if labs == ["NONE"]:
        rows.append((text, "NONE")); continue
    uris = [resolve(l) for l in labs]
    if None in uris:
        bad.append((i, text, labs)); continue
    rows.append((text, "|".join(dict.fromkeys(uris))))
with open("gold/labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text", "uri", "note"])
    for t, u in rows: w.writerow([t, u, "silver: single LLM annotator, not human-verified"])
print(len(rows), "labels;", sum(1 for _, u in rows if u == "NONE"), "NONE; unresolved:", bad)

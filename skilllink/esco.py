from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from .normalize import label_variants

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "esco_skills_en.json.gz"


def load(path: str | Path = DEFAULT) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


class Index:
    """Normalised-label lookup. `exact[norm]` lists (uri, is_preferred) so preferred labels can win ties."""

    def __init__(self, skills: list[dict]):
        self.skills = {s["uri"]: s for s in skills}
        self.exact: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        self.labels: list[str] = []      # parallel arrays for the fuzzy index
        self.label_uri: list[str] = []
        self.label_pref: list[bool] = []
        for s in skills:
            seen = set()
            for lab, pref in [(s["pref"], True)] + [(a, False) for a in s.get("alts", [])]:
                for v in label_variants(lab):
                    if v in seen:
                        continue
                    seen.add(v)
                    self.exact[v].append((s["uri"], pref))
                    self.labels.append(v)
                    self.label_uri.append(s["uri"])
                    self.label_pref.append(pref)

    def pref(self, uri: str) -> str:
        return self.skills[uri]["pref"]

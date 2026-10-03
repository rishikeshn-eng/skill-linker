"""Exact -> fuzzy -> (optional) Gemini rerank, with abstention.

Order matters: exact normalised-label hits are near-certain; fuzzy char n-gram matches are scored and the linker
abstains (returns None) below a threshold instead of forcing a wrong ESCO skill; Gemini only sees the ambiguous middle.
"""
from __future__ import annotations

import difflib
import json
import os
import urllib.request
from dataclasses import dataclass, field

import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .esco import Index
from .normalize import normalize



def _post_json(url: str, key: str, body: dict, tries: int = 5) -> dict:
    """POST with exponential backoff on 429/5xx/network errors (Gemini rate-limits free keys)."""
    import time
    import urllib.error
    data = json.dumps(body).encode()
    for i in range(tries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        time.sleep(min(60, 2 ** (i + 1)))


@dataclass
class Link:
    text: str
    uri: str | None
    label: str | None
    score: float
    method: str                      # exact | fuzzy | llm | none
    candidates: list[tuple[str, float]] = field(default_factory=list)   # (uri, score) best first, distinct skills


class Linker:
    def __init__(self, index: Index, threshold: float = 0.55, llm=None, llm_band=(0.35, 0.8)):
        self.ix = index
        self.threshold = threshold
        self.llm, self.llm_band = llm, llm_band
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.M = self.vec.fit_transform(index.labels)
        self.pref = np.array(index.label_pref)
        df = Counter(t for l in set(index.labels) for t in set(l.split()))
        n = len(set(index.labels))
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.single = sorted({l for l in index.labels if " " not in l and len(l) >= 4})  # for typo matching

    def _coverage(self, qtoks: list[str], label: str) -> float:
        """IDF-weighted share of the query's words found in the label (prefix match >= 5 chars counts, so
        'budgeting' covers 'budget'). Stops 'Microsoft Excel' linking to 'Microsoft Access' on the generic word."""
        ltoks = label.split()
        tot = sum(self.idf.get(t, 8.0) for t in qtoks) or 1.0
        hit = 0.0
        for t in qtoks:
            if any(t == l or (min(len(t), len(l)) >= 5 and (t.startswith(l[:5]) and l.startswith(t[:5]))) for l in ltoks):
                hit += self.idf.get(t, 8.0)
        return hit / tot

    def candidates(self, norm: str, k: int = 8) -> list[tuple[str, float]]:
        q = self.vec.transform([norm])
        sims = (self.M @ q.T).toarray().ravel() + 0.02 * self.pref   # tiny bonus: prefer official labels
        order = np.argsort(-sims)[: k * 6]
        qtoks = norm.split()
        best: dict[str, float] = {}
        for i in order:
            u = self.ix.label_uri[i]
            sc = float(min(sims[i], 1.0))
            if self._coverage(qtoks, self.ix.labels[i]) < 0.5:
                sc *= 0.5                       # shares no informative word with the query: demote
            if sc > best.get(u, -1):
                best[u] = sc
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    def link(self, text: str, k: int = 8) -> Link:
        norm = normalize(text)
        if not norm:
            return Link(text, None, None, 0.0, "none")
        hits = self.ix.exact.get(norm)
        cands = self.candidates(norm, k)
        if hits:
            uri = sorted(hits, key=lambda h: (not h[1], h[0]))[0][0]
            return Link(text, uri, self.ix.pref(uri), 1.0, "exact", [(uri, 1.0)] + [c for c in cands if c[0] != uri][: k - 1])
        top_uri, top = cands[0] if cands else (None, 0.0)
        if self.llm and cands and self.llm_band[0] <= top < self.llm_band[1]:
            pick = self.llm.choose(text, [(u, self.ix.pref(u)) for u, _ in cands[:6]])
            if pick == NONE_PICK:
                return Link(text, None, None, top, "llm", cands)       # LLM judged: not in ESCO
            if pick is not None:
                return Link(text, pick, self.ix.pref(pick), top, "llm", cands)
        if " " not in norm and len(norm) >= 4 and top < self.threshold:
            # short single-word typos ("pyton", "excell") have too few n-grams for the cosine score; use edit similarity
            close = difflib.get_close_matches(norm, self.single, n=3, cutoff=0.85)
            if close:
                u = sorted(self.ix.exact[close[0]], key=lambda h: (not h[1], h[0]))[0][0]
                r = difflib.SequenceMatcher(None, norm, close[0]).ratio()
                cands = [(u, r)] + [c for c in cands if c[0] != u]
                return Link(text, u, self.ix.pref(u), r, "fuzzy", cands[:k])
        if top >= self.threshold:
            return Link(text, top_uri, self.ix.pref(top_uri), top, "fuzzy", cands)
        return Link(text, None, None, top, "none", cands)


NONE_PICK = "__none__"   # the model said no candidate is equivalent (distinct from an invalid/failed reply, which is None)


class GeminiRerank:
    """Picks among candidate ESCO skills or says none. Validates the answer; never trusts an out-of-range index."""

    PROMPT = ("Map this job-ad skill to the single best matching ESCO skill, or null if none is equivalent "
              "(a broader or narrower skill is NOT a match).\nSkill: {text}\nCandidates:\n{cands}\n"
              'Reply with JSON only: {{"choice": <number or null>}}')

    def __init__(self, model=None, api_key=None, transport=None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.key and transport is None:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.transport = transport or self._http
        self.calls = self.rejected = 0

    def _http(self, body):
        return _post_json(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", self.key, body)

    def choose(self, text: str, cands: list[tuple[str, str]]) -> str | None:
        prompt = self.PROMPT.format(text=text, cands="\n".join(f"{i + 1}. {lab}" for i, (_, lab) in enumerate(cands)))
        resp = self.transport({"contents": [{"parts": [{"text": prompt}]}],
                               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        self.calls += 1
        try:
            c = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])["choice"]
            if c is None:
                return NONE_PICK
            c = int(c)
            assert 1 <= c <= len(cands)
            return cands[c - 1][0]
        except Exception:
            self.rejected += 1
            return None

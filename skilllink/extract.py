"""Find ESCO skills inside free-text posting descriptions: longest-match phrase dictionary over normalised tokens."""
from __future__ import annotations

import html
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .esco import Index
from .normalize import normalize

TAG = re.compile(r"<[^>]+>")
EN_MARKERS = {"the", "and", "of", "to", "for", "with", "you", "we", "our", "is", "are", "will", "your", "experience", "have", "that"}
# one-word labels that are real words with many meanings: matching them in running text mostly yields noise
GENERIC_ONE_WORD = set(ENGLISH_STOP_WORDS) | {
    "work", "plan", "manage", "support", "data", "test", "use", "sale", "service", "system", "process", "report", "team",
    "lead", "design", "develop", "write", "read", "speak", "learn", "teach", "train", "build", "make", "help", "care",
    "business", "customer", "product", "project", "quality", "safety", "security", "analysis", "management", "control",
    "communication", "english", "german", "french", "spanish", "company", "client", "market", "marketing", "finance",
    "engineering", "software", "technology", "research", "education", "training", "operation", "maintenance", "record",
    "organise", "organize", "coordinate", "assist", "perform", "ensure", "provide", "monitor", "review", "check",
}


# Phrases that matched in a first manual review (gold/extraction_review_v1.csv) but were not skill mentions in
# job ads: another word sense ("ecosystem", "source", "scratch"), benefits boilerplate, or soft filler. Hand-built from
# that sample, so measure precision on a fresh sample (see README), not on that one.
NOISE_PHRASES = {normalize(p) for p in [
    "ecosystem", "energy", "economic", "pregnancy", "source", "personal development", "electronic", "solve problem",
    "help client", "help customer", "strategic thinking", "adjust component", "be responsible", "hybrid model", "salt",
    "check payment", "scratch", "provide evidence", "unity", "assembly", "journalism", "deliver better customer experience"]}


def clean_html(s: str) -> str:
    return html.unescape(TAG.sub(" ", s or ""))


def is_english(text: str, min_share: float = 0.12) -> bool:
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    return bool(toks) and sum(t in EN_MARKERS for t in toks) / len(toks) >= min_share


class Extractor:
    def __init__(self, index: Index, max_len: int = 6):
        self.max_len = max_len
        self.phrases: dict[tuple[str, ...], str] = {}
        for norm, hits in index.exact.items():
            toks = tuple(norm.split())
            if not toks or len(toks) > max_len:
                continue
            if norm in NOISE_PHRASES:
                continue
            if len(toks) == 1 and (norm in GENERIC_ONE_WORD or len(norm) < 3):
                continue
            pref = [u for u, p in hits if p]
            if len(toks) == 1 and not pref:        # single-word ALT labels are too ambiguous ("data", "stats")
                continue
            self.phrases[toks] = sorted(pref or [h[0] for h in hits])[0]

    def extract(self, text: str) -> list[tuple[str, str]]:
        """Returns [(matched text, uri)] using greedy longest match."""
        toks = normalize(clean_html(text)).split()
        out, i = [], 0
        while i < len(toks):
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                key = tuple(toks[i:i + n])
                if key in self.phrases:
                    out.append((" ".join(key), self.phrases[key]))
                    i += n
                    break
            else:
                i += 1
        return out

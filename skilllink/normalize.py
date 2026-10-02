"""Label/text normalisation shared by the linker and the extractor."""
from __future__ import annotations

import re
import unicodedata

_PAREN = re.compile(r"\s*\([^)]*\)")
_KEEP = re.compile(r"[^a-z0-9+#.\s]")


def _stem(w: str) -> str:
    if not w.isalpha():          # node.js, c++, c#, .net, python3 are left alone
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def normalize(text: str, drop_parens: bool = False) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    if drop_parens:
        t = _PAREN.sub("", t)
    t = t.replace("&", " and ").replace("/", " ").replace("-", " ")
    t = _KEEP.sub(" ", t)
    toks = [w.rstrip(".") for w in t.split()]
    toks = [w for w in toks if w and w != "."]
    return " ".join(_stem(w) for w in toks if w)


def label_variants(label: str) -> set[str]:
    """ESCO labels like 'Python (computer programming)' also match as plain 'python'."""
    return {v for v in {normalize(label), normalize(label, drop_parens=True)} if v}

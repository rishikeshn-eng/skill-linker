import json

import pytest

from skilllink import demand, esco, evaluate, extract, linker
from skilllink.normalize import label_variants, normalize

FIXTURE = [
    {"uri": "u/python", "pref": "Python (computer programming)", "alts": ["Python3", "Py3K"], "desc": ""},
    {"uri": "u/sql", "pref": "SQL", "alts": ["Structured Query Language"], "desc": ""},
    {"uri": "u/teamwork", "pref": "work in teams", "alts": ["teamwork", "team work", "collaborate with colleagues"], "desc": ""},
    {"uri": "u/pm", "pref": "manage projects", "alts": ["project management", "run projects"], "desc": ""},
    {"uri": "u/cust", "pref": "provide customer service", "alts": ["customer service", "customer support"], "desc": ""},
    {"uri": "u/data", "pref": "analyse data", "alts": ["data analysis", "data"], "desc": ""},
    {"uri": "u/weld", "pref": "welding techniques", "alts": ["welding"], "desc": ""},
    {"uri": "u/excel", "pref": "use spreadsheets software", "alts": ["Microsoft Excel", "MS Excel", "spreadsheets"], "desc": ""},
]


@pytest.fixture(scope="module")
def lk():
    return linker.Linker(esco.Index(FIXTURE))


def test_normalize_keeps_tech_tokens_and_stems_plurals():
    assert normalize("C++ & C#") == "c++ and c#"
    assert normalize("Node.js") == "node.js" and normalize(".NET") == ".net"
    assert normalize("Customer Services") == normalize("customer service")
    assert "python" in label_variants("Python (computer programming)")


def test_exact_match_on_alt_label_and_parenthetical_variant(lk):
    assert lk.link("python").uri == "u/python" and lk.link("python").method == "exact"
    assert lk.link("MS Excel").uri == "u/excel"
    assert lk.link("Team Work").uri == "u/teamwork"


def test_fuzzy_handles_typos_and_abstains_on_unknown(lk):
    l = lk.link("pyton")
    assert l.uri == "u/python" and l.method == "fuzzy"
    assert lk.link("Svelte").uri is None and lk.link("Svelte").method == "none"
    assert lk.link("").method == "none"


def test_candidates_are_distinct_skills_best_first(lk):
    c = lk.link("project mgmt").candidates
    assert len({u for u, _ in c}) == len(c) and c[0][0] == "u/pm"


def test_gemini_rerank_validates_choice(lk):
    def fake(choice):
        return lambda body: {"candidates": [{"content": {"parts": [{"text": json.dumps({"choice": choice})}]}}]}
    g = linker.GeminiRerank(transport=fake(1))
    assert g.choose("x", [("u/a", "A"), ("u/b", "B")]) == "u/a"
    assert linker.GeminiRerank(transport=fake(None)).choose("x", [("u/a", "A")]) is None
    bad = linker.GeminiRerank(transport=fake(9))
    assert bad.choose("x", [("u/a", "A")]) is None and bad.rejected == 1
    lk2 = linker.Linker(esco.Index(FIXTURE), threshold=0.99, llm=linker.GeminiRerank(transport=fake(1)), llm_band=(0.1, 0.99))
    assert lk2.link("pyton").method == "llm"


def test_extractor_longest_match_and_generic_words_ignored():
    ex = extract.Extractor(esco.Index(FIXTURE))
    got = ex.extract("<p>We need project management and customer service skills; you know SQL and Python. Strong data focus.</p>")
    assert [u for _, u in got] == ["u/pm", "u/cust", "u/sql", "u/python"]  # single-word alt 'data' is skipped


def test_language_filter():
    assert extract.is_english("We are looking for a developer to join our team and work with you")
    assert not extract.is_english("Wir suchen eine Buchhalterin für unser Team in Berlin und Umgebung")


def test_evaluate_counts_none_and_abstentions(lk):
    gold = [{"text": "python", "uri": "u/python"}, {"text": "pyton", "uri": "u/python"}, {"text": "Svelte", "uri": "NONE"},
            {"text": "teamwork", "uri": "u/pm"}]
    r = evaluate.evaluate(lk, gold)
    assert r["n_linkable"] == 3 and r["n_none"] == 1
    assert r["top1_accuracy_on_linkable"] == pytest.approx(2 / 3)
    assert r["correct_abstain_on_none"] == 1.0 and len(r["errors"]) == 1


def test_wilson_and_demand_trend():
    lo, hi = demand.wilson(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo and hi < 0.61
    posts = [{"t": i, "skills": ["a"] if i >= 100 else ["b"]} for i in range(200)]
    d = demand.demand_index(posts, min_count=5)
    tr = {s["uri"]: s["trend"] for s in d["skills"]}
    assert tr == {"a": "rising", "b": "falling"}
    assert demand.demand_index([{"t": 1, "skills": ["a", "a"]}], min_count=1)["skills"][0]["postings"] == 1

# skill-linker

Map free-text job-ad skills to [ESCO](https://esco.ec.europa.eu) (the EU skills taxonomy) and publish a skill demand index.

**Live UI:** https://rishikeshn-eng.github.io/skill-linker/ (try the linker in your browser, accuracy tables, demand index) · **Labelling tool:** https://rishikeshn-eng.github.io/skill-linker/label.html

## What is real here and what is not

- **Real:** an ESCO subset of **10,356 English skills with alternative labels**, harvested from ESCO's public web service (the bulk CSV download sits behind a form). Coverage is partial: ESCO's classification has roughly 14k skills, and the harvest saturated before reaching them all. Run `scripts/harvest_esco.py` to extend it. ESCO is © European Union; check its licence terms and cite it if you reuse `data/esco_skills_en.json.gz`.
- **Real:** 1,051 postings fetched from [Arbeitnow's](https://www.arbeitnow.com/api/job-board-api) free public API (2026-08-12 to 2026-10-02); 569 are English. A small, Europe-skewed sample from one board. Postings are not redistributed (git-ignored).
- **Not done: the 1.3M-posting Kaggle set and the 500 human labels.** The Kaggle file needs a login. The labelled set has **227 strings labelled by a single LLM annotator (me, Claude), not humans**, reading candidates from this linker and from ESCO's own search API. They may anchor on the linker's suggestions; 106 are "not in ESCO" because ESCO lacks most modern tool names (Docker, Kubernetes, React, AWS...). The labelling tool lets you label more in the browser, see the linker's accuracy against *your* labels, and export CSV.

## How the linker works (`skilllink/linker.py`)

1. **Normalise** (lowercase, ASCII-fold, plural stemming for plain words only; `node.js`, `c++`, `.net` stay intact).
2. **Exact** match on any ESCO preferred or alternative label (parenthetical qualifiers also dropped: "Python (computer programming)" matches "python").
3. **Fuzzy**: character n-gram TF-IDF cosine over ~46k labels, demoted when the match shares no *informative* word with the query (IDF-weighted coverage gate: stops "Microsoft Excel" linking to "Microsoft Access" on the word "Microsoft"); plus an edit-distance path for single-word typos ("pyton").
4. **Abstain** below a score threshold instead of forcing a wrong skill.
5. **Optional Gemini rerank** of the ambiguous middle band (candidates in, validated index out). Unit-tested against a fake transport; **not run live** (no API key).

## Results (227 labelled strings; 121 linkable, 106 not in ESCO)

The abstention threshold (0.8) was chosen on a deterministic half of the strings and evaluated on the other half:

| | held-out half (113) | all 227 |
|---|---|---|
| End-to-end accuracy (right skill, or correctly abstain) | **70.8%** | n/a |
| Top-1 on linkable strings | 62.3% | 62.8% |
| Top-5 recall on linkable strings | 81.1% | 85.1% |
| Correct abstention on not-in-ESCO strings | 78.3% | 73.6% |
| Precision of exact-label hits | 76.9% | 81.7% |

Reading it honestly: the linker is decent, not good. The gap between top-1 (63%) and top-5 (85%) is what a reranker should recover, which is where Gemini would go, but that is untested here. A first version without the coverage gate scored 72% top-1 but abstained correctly on only 31% of not-in-ESCO strings (it linked "Docker" to "dock operations"); the gate trades some top-1 for far fewer false links. Exact hits are not always right either: an alternative label can belong to a different skill ("Data analysis" is an alt label of ESCO's "statistics"). Gold labels were assigned by one annotator on one pass; with 113 held-out strings, a 95% interval on 70.8% is roughly ±8 points.

## Skill demand index (`skilllink/extract.py`, `demand.py`)

Skills are found in English posting text by longest-match against normalised ESCO labels; one-word generic labels and single-word alternative labels are excluded. Each skill counts once per posting; shares carry Wilson 95% intervals and an early-vs-late-half trend flag.

- 569 English postings, 532 with at least one skill (mean 4.2 each). Top skills: Python, SaaS, computer science, data protection, project management.
- **Extraction precision:** a manual review of 120 extracted spans found **68%** were real skill mentions (false positives: other word senses like "ecosystem", "source", benefits boilerplate like "pregnancy"). I added a stoplist from that review, then reviewed a *fresh* sample of 100: **79%**. Remaining errors are loose alternative-label matches ("help other" to "guide others"). Both review files are in `gold/`; one reviewer.
- Trend flags on ~7 weeks of data are leads, not findings.

## Run

```bash
pip install numpy scikit-learn pytest
python -m pytest -q                                    # 9 tests
python scripts/harvest_esco.py --max-requests 800      # optional: extend the ESCO subset (~40 min)
python scripts/fetch_postings.py --pages 6             # real postings -> data/ (git-ignored)
python scripts/make_candidates.py && python scripts/resolve_labels.py   # rebuild gold/labels.csv from gold/label_spec.txt
python -m skilllink.run && python scripts/build_site.py
```
For the Kaggle data: link the `job_skills` strings with `Linker.link(text)` and feed `{"t": time, "skills": [uris]}` rows to `demand.demand_index`.

"""Evaluate the Gemini rerank on the labelled strings: python scripts/eval_gemini.py   (needs GEMINI_API_KEY)

Compares the linker with and without the LLM on the same held-out half used for the threshold, plus all strings.
Writes docs/gemini_eval.json.
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skilllink import esco, evaluate
from skilllink.linker import GeminiRerank, Linker

ix = esco.Index(esco.load())
gold = evaluate.load_gold("gold/labels.csv")
test = [g for g in gold if int(hashlib.md5(g["text"].encode()).hexdigest(), 16) % 2 == 1]
res = {}
for th in (0.8, 0.55):                       # 0.8 = threshold chosen on the dev half; 0.55 = more permissive
    base = Linker(ix, threshold=th)
    llm = GeminiRerank()
    withllm = Linker(ix, threshold=th, llm=llm, llm_band=(0.3, 0.95))
    keep = lambda e: {k: v for k, v in e.items() if k not in ("errors", "sweep")}
    res[str(th)] = {"no_llm_test": keep(evaluate.evaluate(base, test)), "llm_test": keep(evaluate.evaluate(withllm, test)),
                    "no_llm_all": keep(evaluate.evaluate(base, gold)), "llm_all": keep(evaluate.evaluate(withllm, gold)),
                    "llm_calls": llm.calls, "llm_rejected": llm.rejected, "model": llm.model}
    print(th, {k: round(v["end_to_end_accuracy"], 3) for k, v in res[str(th)].items() if isinstance(v, dict)}, "calls", llm.calls)
json.dump(res, open("docs/gemini_eval.json", "w"), indent=1)

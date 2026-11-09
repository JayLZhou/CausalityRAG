"""Score the Table IV transfer runs: Ans-FR/CFR and EM-FR/CFR per model, dataset and method.

Answers are parsed and compared with the Table 3 reader functions. A query counts
when the model's own clean answer is valid; EM metrics use queries the model
answered correctly. A control with no eligible position counts as no flip.
Usage: score_tableiv_transfer.py META ANSWERS [ANSWERS ...]
"""
import collections
import json
import sys

sys.path.insert(0, "/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810")
from causalityrag.reader import (  # noqa: E402
    answers_exact_match,
    canonicalize_reader_answer,
    parse_json_object,
    reader_answers_exact_match,
)

meta = {(r["dataset"], r["id"]): r for r in map(json.loads, open(sys.argv[1]))}


def parse(question, text):
    obj = parse_json_object(text)
    ans = str(obj["answer"]).strip() if isinstance(obj, dict) and obj.get("answer") is not None else str(text).strip()
    return canonicalize_reader_answer(question, ans, reader_mode="short_answer")


def golds(m):
    g = m.get("gold_answers")
    return [str(x) for x in (g if isinstance(g, list) else [g]) if x]


result = {}
for path in sys.argv[2:]:
    answers = collections.defaultdict(dict)
    model = None
    for r in map(json.loads, open(path)):
        model = r["model"]
        answers[(r["dataset"], r["id"])][r["method"]] = r["response"]
    stats = collections.defaultdict(lambda: collections.Counter())
    for (ds, q), got in answers.items():
        m = meta.get((ds, q))
        if m is None or "clean" not in got:
            continue
        question = m["question"]
        clean = parse(question, got["clean"])
        if not clean:
            continue
        correct = any(answers_exact_match(clean, canonicalize_reader_answer(question, g, reader_mode="short_answer"))
                      for g in golds(m))
        for method, info in m["methods"].items():
            if info.get("edited") and method not in got:
                continue  # not answered yet
            if info.get("control") and f"{method}_ctrl" not in got:
                continue
            s = stats[(ds, method)]
            s["n"] += 1
            s["tokens"] += info["n_tokens"]
            s["em_n"] += correct
            for kind, key in (("fact", method), ("ctrl", f"{method}_ctrl")):
                flip = em_flip = False
                if key in got:
                    edited = parse(question, got[key])
                    flip = not reader_answers_exact_match(question, edited, clean, reader_mode="short_answer")
                    em_flip = correct and not any(
                        answers_exact_match(edited, canonicalize_reader_answer(question, g, reader_mode="short_answer"))
                        for g in golds(m))
                s[f"{kind}_flip"] += flip
                s[f"{kind}_em_flip"] += em_flip
    for (ds, method), s in sorted(stats.items()):
        n, en = s["n"], max(s["em_n"], 1)
        row = {"queries": n, "em_queries": s["em_n"],
               "Ans-FR": 100 * s["fact_flip"] / n, "Ans-CFR": max(0.0, 100 * (s["fact_flip"] - s["ctrl_flip"]) / n),
               "EM-FR": 100 * s["fact_em_flip"] / en, "EM-CFR": max(0.0, 100 * (s["fact_em_flip"] - s["ctrl_em_flip"]) / en),
               "tokens": s["tokens"] / n}
        result.setdefault(model, {}).setdefault(ds, {})[method] = row
        print(f"{model:14s} {ds:9s} {method:9s} n={n:4d} Ans-FR {row['Ans-FR']:5.1f} Ans-CFR {row['Ans-CFR']:5.1f} "
              f"EM-FR {row['EM-FR']:5.1f} EM-CFR {row['EM-CFR']:5.1f} tokens {row['tokens']:.2f}")
json.dump(result, open("tableiv_transfer_scores.json", "w"), indent=1)

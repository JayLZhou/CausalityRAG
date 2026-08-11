import json
import sys

from scripts.audit_popqa_alias_metrics import main


def test_alias_audit_uses_the_best_gold_alias(monkeypatch, tmp_path):
    results = tmp_path / "results.jsonl"
    output = tmp_path / "audit.json"
    rows = [
        {
            "id": "q1",
            "gold_answer": "J-pop",
            "gold_answers": ["J-pop", "Japanese pop", "jpop"],
            "clean_answer": "J-pop",
            "edited_answer": "Japanese pop",
            "clean_correct": True,
            "evaluation_status": "verified_flip",
        },
        {
            "id": "q2",
            "gold_answer": "J-pop",
            "gold_answers": ["J-pop", "Japanese pop", "jpop"],
            "clean_answer": "jpop",
            "edited_answer": "noise",
            "clean_correct": True,
            "evaluation_status": "verified_flip",
        },
        {
            "id": "q3",
            "gold_answer": "J-pop",
            "gold_answers": ["J-pop", "Japanese pop", "jpop"],
            "clean_answer": "rock",
            "edited_answer": "noise",
            "clean_correct": False,
            "evaluation_status": "verified_flip",
        },
        {
            "id": "q4",
            "gold_answer": "Japanese pop",
            "gold_answers": ["Japanese pop"],
            "clean_answer": "Japanese pop",
            "edited_answer": "Japanese",
            "clean_correct": True,
            "evaluation_status": "verified_flip",
        },
    ]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_popqa_alias_metrics.py",
            "--results",
            str(results),
            "--out",
            str(output),
            "--expected-rows",
            "4",
        ],
    )

    main()

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["clean_exact_any_alias"] == 3
    assert audit["clean_f1_one_any_alias"] == 3
    assert audit["stored_clean_correct_mismatches"] == []
    assert audit["acc_cfr_numerator"] == 2
    assert audit["acc_cfr_denominator"] == 3
    assert audit["acc_cfr"] == 2.0 / 3.0
    assert audit["f1_cfr_numerator"] == 2
    assert audit["f1_cfr_denominator"] == 3
    assert audit["edited_remains_correct_any_alias"] == 1
    assert audit["edited_correct_only_via_noncanonical_alias"] == 1
    assert audit["canonical_only_would_false_count_as_incorrect"] == 1
    assert audit["edited_f1_on_clean_correct"] == {
        "zero": 1,
        "partial": 1,
        "one": 1,
        "mean": (0.0 + 2.0 / 3.0 + 1.0) / 3.0,
    }
    assert "clean_f1_population_ids" not in audit
    assert "f1_decreased_ids" not in audit

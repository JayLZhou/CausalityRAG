import json

import pytest

from scripts.run_flow_ablation_full import audit_rows, reuse_pilot, summarize_full


def test_full_summary_same_denominator_and_invalid_edited_failure():
    records = {q: {'id': q, 'question': 'Where?', 'answer': 'Paris'} for q in ['a', 'b', 'c']}
    old = {q: {'clean_answer': 'Paris', 'edited_answer': 'London', 'reader_calls': 2,
               'evaluation_status': 'verified_flip'} for q in records}
    old['c']['clean_answer'] = '[]'
    rows = [
        {'id': 'a', 'methods': {'direct_score': {'reader_called': True, 'edited_answer': 'London', 'status': 'verified_flip'},
                               'individual_flow': {'reader_called': True, 'edited_answer': '[]', 'status': 'verified_flip'}}},
        {'id': 'b', 'methods': {}, 'evaluation_status': 'failed'},
        {'id': 'c', 'methods': {}, 'evaluation_status': 'invalid_clean_answer'},
    ]
    summary = summarize_full(rows, records, old, 'hotpotqa', 'short_answer')
    assert summary['queries'] == 3
    assert summary['invalid_clean'] == 1
    assert summary['denominators']['em'] == 2
    assert summary['rates']['direct_score']['em'] == 0.5
    assert summary['rates']['individual_flow']['answer'] == 0
    assert summary['rates']['reflow_cached']['answer'] == 1
    assert summary['reader_calls']['reflow_cached'] == 4
    assert summary['statuses']['individual_flow']['invalid_edited_answer'] == 1


def test_quartz_label_is_canonicalized_for_scoring():
    records = {'q': {'question': 'Choices: (A) more; (B) less.', 'answer': 'less'}}
    old = {'q': {'clean_answer': 'B', 'edited_answer': 'A', 'reader_calls': 1}}
    rows = [{'id': 'q', 'methods': {'direct_score': {'reader_called': True, 'edited_answer': 'A'}}}]
    summary = summarize_full(rows, records, old, 'quartz', 'quartz')
    assert summary['denominators']['em'] == 1
    assert summary['rates']['direct_score']['em'] == 1


def test_pilot_reuse_rejects_changed_artifacts(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'artifact_sha256': {'pool': 'old'}}))
    with pytest.raises(ValueError, match='artifact SHA mismatch'):
        reuse_pilot(tmp_path, {'expected_sha256': {'pool': 'new'}}, ['a'])


def test_audit_rejects_wrong_budget():
    records = {'q': {'question': 'Where?'}}
    old = {'q': {'clean_answer': 'Paris', 'n_modified_tokens': 2}}
    rows = [{'id': 'q', 'clean_answer': 'Paris', 'reader_calls': 1, 'methods': {
        'direct_score': {'reader_called': True, 'selected_ids': ['t'], 'n_modified_tokens': 1}}}]
    with pytest.raises(ValueError, match='token budget mismatch'):
        audit_rows(rows, records, old, 'short_answer')

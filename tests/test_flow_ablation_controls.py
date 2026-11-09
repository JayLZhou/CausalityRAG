import json

import pytest

from scripts.run_flow_ablation_controls import (
    checkpoints, eligible_clean, pair_key, reuse_control, summarize_jobs,
)


def result(answer, called=True):
    return dict(status='evaluated', reader_called=called, edited_answer=answer)


def job(name='direct_score', answer='London'):
    return dict(id='q', method=name, clean_answer='Paris', factual=result(answer))


def checkpoint(name='direct_score', answer='London'):
    return dict(id='q', method=name, clean_answer='Paris', result=result(answer),
                new_reader_calls=1, reused_from=None)


def summarize(jobs, rows):
    return summarize_jobs(jobs, {pair_key(r['id'], r['method']): r for r in rows},
                          {'q': {'question': 'Capital?', 'answer': 'Paris'}}, 'hotpotqa', 'short_answer')


def test_cfr_subtracts_control_on_same_denominator():
    s = summarize([job()], [checkpoint()])
    for v in s['metrics']['direct_score'].values():
        assert v['factual_flips'] == v['control_flips'] == v['denominator'] == 1
        assert v['cfr'] == 0


def test_negative_cfr_is_not_clamped():
    s = summarize([job(answer='Paris')], [checkpoint()])
    assert s['metrics']['direct_score']['answer']['cfr'] == -1


def test_missing_control_is_not_a_zero_flip():
    s = summarize([job()], [])
    assert s['metrics']['direct_score']['answer']['cfr'] is None
    assert s['diagnostics']['direct_score']['missing_control'] == 1


def test_invalid_factual_pair_excluded_and_disclosed():
    j = job(answer='')
    s = summarize([j], [checkpoint(answer='Paris')])
    assert s['metrics']['direct_score']['answer']['denominator'] == 0
    assert s['diagnostics']['direct_score']['invalid_pair'] == 1


def test_invalid_control_is_not_successful_pair():
    s = summarize([job()], [checkpoint(answer='')])
    assert s['metrics']['direct_score']['answer']['denominator'] == 0
    assert s['diagnostics']['direct_score']['invalid_pair'] == 1


def test_numeric_exclusion_has_unchanged_control():
    c = checkpoint()
    c['result'] = dict(status='no_paraphrase_eligible_tokens', reader_called=False)
    c['new_reader_calls'] = 0
    s = summarize([job()], [c])
    assert s['metrics']['direct_score']['answer']['cfr'] == 1


def test_invalid_clean_excluded():
    record = {'question': 'Capital?', 'answer': 'Paris'}
    assert eligible_clean(record, {'clean_answer': ''}, 'hotpotqa', 'short_answer') is None


def test_duplicate_checkpoint_rejected(tmp_path):
    p = tmp_path / 'c.jsonl'
    row = checkpoint()
    p.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicate'):
        checkpoints(p)


def test_reuse_requires_same_clean_selection_and_replacement():
    units = {'u': {'text': 'city', 'pos': 'NOUN', 'type': 'token'}}
    pool = {'u': {'candidates': [{'new': 'town'}]}}
    r = dict(result('Paris'), selected_ids=['u'], control_selected_ids=['u'],
             n_modified_tokens=1, edits=[{'unit_id': 'u', 'ok': True, 'new': 'town'}])
    old = {'clean_answer': 'Paris', 'methods': {'reflow': r}}
    f = {'selected_ids': ['u']}
    assert reuse_control(old, f, 'Paris', 'short_answer', 'Capital?', units, pool, 0) == r
    assert reuse_control(old, f, 'London', 'short_answer', 'Capital?', units, pool, 0) is None
    pool['u']['candidates'][0]['new'] = 'CITY'
    assert reuse_control(old, f, 'Paris', 'short_answer', 'Capital?', units, pool, 0) is None

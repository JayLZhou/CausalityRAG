import pytest

from scripts.run_flow_ablation_paired_reader import METHOD, check_frozen_selection, paired_summary


def test_no_replacement_of_missing_frozen_selection():
    with pytest.raises(ValueError, match='outside frozen pool'):
        check_frozen_selection({'selected_ids': ['a'], 'n_modified_tokens': 1}, {'b'})


def test_summary_uses_current_clean_and_reverified_response():
    records = {'q': {'question': 'Where?', 'answer': 'Paris'}}
    old = {'q': {'clean_answer': 'London', 'edited_answer': 'Rome', 'reader_calls': 9}}
    rows = [{'id': 'q', 'clean_answer': 'Paris', 'cached_clean_answer': 'London',
             'evaluation_status': 'evaluated', 'clean_reader_calls': 1, 'clean_matches_cached': False,
             'methods': {METHOD: {'edited_answer': 'Paris', 'reader_called': True, 'status': 'verified_no_flip'},
                         'direct_score': {'edited_answer': 'Rome', 'reader_called': True, 'status': 'verified_flip'}}}]
    summary = paired_summary(rows, records, old)
    assert summary['denominators']['em'] == 1
    assert summary['rates'][METHOD]['answer'] == 0
    assert summary['rates']['direct_score']['em'] == 1
    assert summary['reader_calls'][METHOD] == 1
    assert summary['changed_clean_answers'] == 1
    assert 'reflow_cached' not in summary['rates']

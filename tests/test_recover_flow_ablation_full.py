import pytest

from scripts.recover_flow_ablation_full import partition_clean


def test_unmappable_quartz_clean_excluded_without_guessing():
    records = {q: {'question': 'Choices: (A) higher; (B) lower.'} for q in ('a', 'b')}
    old = {'a': {'clean_answer': 'stronger'}, 'b': {'clean_answer': 'higher'}}
    rows = [{'id': 'a', 'reader_calls': 0}, {'id': 'b', 'reader_calls': 2}]
    valid, excluded = partition_clean(rows, records, old, 'quartz')
    assert [r['id'] for r in valid] == ['b']
    assert [r['id'] for r in excluded] == ['a']
    assert excluded[0]['cached_clean'] == 'stronger'


def test_excluded_clean_cannot_hide_reader_calls():
    with pytest.raises(ValueError, match='unexpectedly has reader calls'):
        partition_clean([{'id': 'q', 'reader_calls': 1}], {'q': {'question': '?'}},
                        {'q': {'clean_answer': '[]'}}, 'short_answer')

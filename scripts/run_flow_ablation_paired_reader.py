#!/usr/bin/env python3
"""Revalidate frozen TriviaQA interventions against a shared current reader."""
from __future__ import annotations

from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_flow_ablation_full as full
from scripts.recover_flow_ablation_full import OLD_ROOT, NEW_ROOT, require_free_device, verify_inputs
from causalityrag.evaluation_metrics import valid_answer

METHOD = 'reflow_frozen_set_reverified'
OUT = NEW_ROOT / 'triviaqa_paired_reader'


def check_frozen_selection(old, eligible):
    selected = old.get('selected_ids', [])
    budget = int(old.get('n_modified_tokens', 0))
    if len(selected) != len(set(selected)) or len(selected) != budget:
        raise ValueError('cached ReFlow selection/budget mismatch')
    if not set(selected) <= eligible:
        raise ValueError('cached ReFlow selection outside frozen pool/domain')
    return list(selected)


def paired_summary(rows, records, reflow):
    references = {}
    for row in rows:
        q = row['id']
        result = row.get('methods', {}).get(METHOD, {})
        references[q] = dict(reflow[q], clean_answer=row['clean_answer'],
                             edited_answer=result.get('edited_answer', ''),
                             reader_calls=int(bool(result.get('reader_called'))),
                             evaluation_status=result.get('status', row['evaluation_status']))
    summary = full.summarize_full(rows, records, references, 'triviaqa', 'short_answer')
    for field in ('numerators', 'rates', 'statuses', 'reader_calls'):
        summary[field][METHOD] = summary[field].pop('reflow_cached')
    summary.update(
        protocol='paired_current_reader_frozen_graphs_and_frozen_reflow_terminal_sets',
        reflow_search='not_rerun; historical terminal selections and token budgets frozen',
        clean_reader_calls=sum(r.get('clean_reader_calls', 0) for r in rows),
        changed_clean_answers=sum(r.get('clean_matches_cached') is False for r in rows),
        query_statuses=dict(Counter(r['evaluation_status'] for r in rows)),
        cohort='historical-valid-clean AND current-valid-clean; no filtering on answer agreement or intervention outcome',
        budget='same historical terminal token budget and one intervention reader call per method; shared clean call',
        graph_target='historical_cached_answer; graphs are not reconstructed for changed current answers',
    )
    return summary


def audit_paired(rows, records, reflow):
    for row in rows:
        if not valid_answer(row['clean_answer']):
            if row.get('reader_calls', 0):
                raise ValueError('invalid clean has intervention calls')
            continue
        methods = row['methods']
        if set(methods) != {*full.METHODS, METHOD}:
            raise ValueError('missing method in paired output')
        if sum(bool(r.get('reader_called')) for r in methods.values()) != row['reader_calls']:
            raise ValueError('intervention call accounting mismatch')
        old = reflow[row['id']]
        if methods[METHOD].get('selected_ids') != old.get('selected_ids'):
            raise ValueError('ReFlow frozen selection changed')
        for result in methods.values():
            selected = result.get('selected_ids', [])
            if len(set(selected)) != len(selected) or result.get('ranking_completion_used'):
                raise ValueError('duplicate or fallback selection')
            if result.get('reader_called') and result.get('n_modified_tokens') != old['n_modified_tokens']:
                raise ValueError('token budget mismatch')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((OLD_ROOT / 'configs/triviaqa.json').read_text())
        verify_inputs(config)
        records, units, graphs, reflow = (full.keyed(config[k]) for k in ('input', 'units', 'graphs', 'reflow'))
        ids = sorted(records, key=lambda q: (hashlib.sha256(f'20261023:{q}'.encode()).hexdigest(), q))
        if len(ids) != 1000 or any(set(t) != set(ids) for t in (units, graphs, reflow)):
            raise ValueError('expected 1000 unique aligned IDs')
        files = [*Path('causalityrag').glob('*.py'), Path(__file__),
                 Path('scripts/run_flow_ablation_full.py'), Path('scripts/run_flow_ablation_pilot.py'),
                 Path('scripts/recover_flow_ablation_full.py'),
                 Path('scripts/evaluate_matched_budget_baselines.py'), Path('scripts/evaluate_reflow.py')]
        full.freeze_json(OUT / 'manifest.json', dict(
            config=config, ids=ids, code_sha256={str(p): full.file_sha256(str(p)) for p in sorted(files)},
            protocol='same_reader_clean_and_three_frozen_selection_methods',
            clean_cohort='historical valid intersect current valid; retain mismatching valid clean',
            graph_target='historical clean answer; unchanged', graph_domain='editable_projection',
            reflow_selection='cached terminal set, not new search',
        ))
        path = OUT / 'factual.jsonl'
        done = full.keyed(str(path)) if path.exists() else {}
        if not set(done) <= set(ids):
            raise ValueError('unexpected resume IDs')
        for q in ids:
            if q not in done and not valid_answer(reflow[q].get('clean_answer', '')):
                row = dict(id=q, clean_answer=reflow[q].get('clean_answer', ''),
                           cached_clean_answer=reflow[q].get('clean_answer', ''), methods={},
                           reader_calls=0, clean_reader_calls=0, evaluation_status='historical_invalid_clean')
                full.append_row(path, row)
                done[q] = row
        if len(done) < 1000:
            require_free_device()
            pool = full.FrozenSharedReplacementPool(config['pool'])
            # Validate every historical selection before spending reader calls.
            for q in ids:
                if q in done:
                    continue
                all_units = full.units_from_cache_row(records[q], units[q], k=config['k'])
                full.require_complete_graph_domain(graphs[q], all_units)
                eligible = {str(u['unit_id']) for u in all_units if pool.is_eligible(str(u['unit_id']))}
                check_frozen_selection(reflow[q], eligible)
            print(f'PREFLIGHT passed, ready={len(done)}/1000; all frozen ReFlow selections validated', flush=True)
            reader = full.ProtocolAdapter(config['model'], 'short_answer')
            for index, q in enumerate(ids):
                if q in done:
                    continue
                started = time.perf_counter()
                clean_file = OUT / f'clean_{index:04d}.json'
                if clean_file.exists():
                    clean = json.loads(clean_file.read_text())['answer']
                else:
                    clean = reader.answer(records[q]['question'], full.retrieved_contexts(records[q])[:config['k']])
                    full.freeze_json(clean_file, dict(id=q, answer=clean, reader_calls=1))
                matched = full.reader_answers_exact_match(records[q]['question'], clean, reflow[q]['clean_answer'], reader_mode='short_answer')
                if not valid_answer(clean):
                    row = dict(id=q, clean_answer=clean, methods={}, reader_calls=0,
                               evaluation_status='current_invalid_clean')
                else:
                    cache = OUT / f'ranking_{index:04d}.json'
                    if cache.exists():
                        rankings = json.loads(cache.read_text())
                    else:
                        all_units = full.units_from_cache_row(records[q], units[q], k=config['k'])
                        eligible = {str(u['unit_id']) for u in all_units if pool.is_eligible(str(u['unit_id']))}
                        rankings = {m: full.rank_graph_tokens(graphs[q], eligible, method=m, graph_domain='editable_projection') for m in full.METHODS}
                        rankings[METHOD] = dict(ranked_ids=check_frozen_selection(reflow[q], eligible))
                        full.freeze_json(cache, rankings)
                    row = full.evaluate_query(records[q], units[q], dict(reflow[q], clean_answer=clean),
                                              rankings, pool, reader, random_seeds=[],
                                              replacement_seed=config['replacement_seed'], k=config['k'])
                    for result in row.get('methods', {}).values():
                        if result.get('reader_called') and not valid_answer(result.get('edited_answer', '')):
                            result.update(status='invalid_edited_answer', verified_flip=False)
                    row['rankings_file'] = str(cache)
                row.update(cached_clean_answer=reflow[q]['clean_answer'], clean_reader_calls=1,
                           clean_matches_cached=matched, elapsed_seconds=time.perf_counter() - started)
                full.append_row(path, row)
                done[q] = row
                print(f'PAIRED VERIFIED {len(done)}/1000 {q} clean_matches={matched} intervention_calls={row["reader_calls"]}', flush=True)
        if set(done) != set(ids):
            raise ValueError('incomplete ID set')
        audit_paired(list(done.values()), records, reflow)
        verify_inputs(config)
        summary = paired_summary(list(done.values()), records, reflow)
        summary.update(factual_sha256=full.file_sha256(str(path)),
                       manifest_sha256=full.file_sha256(str(OUT / 'manifest.json')),
                       audit='1000 IDs aligned; actual edit counts and frozen ReFlow sets verified')
        full.freeze_json(OUT / 'summary.json', summary)
        csv_path = OUT / 'comparison.csv'
        if not csv_path.exists():
            with csv_path.open('x') as stream:
                writer = csv.writer(stream)
                writer.writerow(['dataset', 'method', 'metric', 'numerator', 'denominator', 'rate'])
                for name, rates in summary['rates'].items():
                    for metric, rate in rates.items():
                        writer.writerow(['triviaqa', name, metric, summary['numerators'][name][metric], summary['denominators'][metric], rate])
        print('TRIVIA PAIRED COMPLETE 1000/1000; frozen-set replay, not new ReFlow search', flush=True)


if __name__ == '__main__':
    run()

#!/usr/bin/env python3
"""Full-cohort flow ablations, reusing frozen graphs and completed pilot rows."""
from __future__ import annotations

import argparse
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
from scripts.run_flow_ablation_pilot import freeze_json, keyed
from scripts.evaluate_matched_budget_baselines import LocalReaderAdapter, evaluate_query
from causalityrag.evaluation_metrics import (
    answer_changed, clean_correct, correctness_lost, gold_answer_spec, valid_answer,
)
from causalityrag.flow_ablation import METHODS, rank_graph_tokens
from causalityrag.flow_contract import require_complete_graph_domain
from causalityrag.io import retrieved_contexts
from causalityrag.reader import (
    LocalHFReader, ReaderProtocolError, canonicalize_reader_answer,
    reader_answers_exact_match, reader_prompt,
)
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool, file_sha256
from causalityrag.token_units import units_from_cache_row

DATASETS = ('hotpotqa', 'timeqa', 'finqa', 'musique', 'quartz', 'triviaqa', '2wiki', 'popqa')
SOURCE = Path('/data1/yujia/CausalityRAG')
METRICS = ('answer', 'f1', 'em', 'acc')


class ProtocolHFReader(LocalHFReader):
    def _prompt_ids(self, question, contexts):
        system, user = reader_prompt(question, list(contexts), reader_mode=self.reader_mode)
        prompt = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(prompt, add_special_tokens=False)['input_ids']


class ProtocolAdapter(LocalReaderAdapter):
    def __init__(self, model, mode):
        self.reader_mode = mode
        self.reader = ProtocolHFReader(model, device='cuda', dtype='bfloat16', attn_implementation='sdpa')
        self.reader.reader_mode = mode


def append_row(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(root):
    audit_path = SOURCE / 'out_table3_final8_20260811/audit.json'
    audit = json.loads(audit_path.read_text())
    for entry in audit['datasets']:
        ds = entry['dataset']
        src = SOURCE / ('out_popqa_20260810' if ds == 'popqa' else 'out') / ds
        paths = {
            'input': str(src / 'retrieval/top10_1000.jsonl'),
            'units': str(src / 'inputs' / ('context_units_top10_1000.jsonl' if ds == 'triviaqa' else 'token_units_top10_1000.jsonl')),
            'graphs': str(Path(entry['paths']['reflow']).parent / 'graphs.jsonl'),
            'reflow': entry['paths']['reflow'],
            'pool': str(src / 'replacements/shared_pool_top10_v1' / ('shared_pool_top10_final.jsonl' if ds in {'hotpotqa', 'triviaqa'} else 'shared_pool.jsonl')),
        }
        shas = {name: file_sha256(path) for name, path in paths.items()}
        if shas['reflow'] != entry['shas']['reflow']:
            raise ValueError(f'{ds}: audited ReFlow SHA mismatch')
        tables = {name: keyed(paths[name]) for name in ('input', 'units', 'graphs', 'reflow')}
        ids = set(tables['input'])
        if len(ids) != 1000 or any(set(t) != ids for t in tables.values()):
            raise ValueError(f'{ds}: expected 1000 unique aligned IDs')
        valid = [q for q in ids if valid_answer(tables['reflow'][q].get('clean_answer', ''))]
        if len(valid) != entry['valid_clean_queries']:
            raise ValueError(f'{ds}: valid-clean count changed')
        for q in valid:
            graph = tables['graphs'][q]
            units = units_from_cache_row(tables['input'][q], tables['units'][q], k=5)
            require_complete_graph_domain(graph, units)
        config = dict(paths, dataset=ds, expected_sha256=shas, k=5, replacement_seed=0,
                      model='/data1/yujia/models/Qwen2.5-7B-Instruct',
                      reader_mode='quartz' if ds == 'quartz' else 'short_answer',
                      expected_valid=len(valid), audit_sha256=file_sha256(str(audit_path)))
        (root / 'configs').mkdir(parents=True, exist_ok=True)
        freeze_json(root / 'configs' / f'{ds}.json', config)
        print(f'PREFLIGHT {ds}: 1000 aligned; valid={len(valid)}; frozen SHA and graph domain OK', flush=True)


def reuse_pilot(source, config, ids):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['artifact_sha256'] != config['expected_sha256']:
        raise ValueError('pilot artifact SHA mismatch')
    for name in ('dataset', 'model', 'k', 'replacement_seed'):
        if manifest['config'][name] != config[name]:
            raise ValueError(f'pilot config mismatch: {name}')
    if manifest['graph_domain'] != 'editable_projection':
        raise ValueError('pilot graph domain mismatch')
    for path, sha in manifest['code_sha256'].items():
        if file_sha256(path) != sha:
            raise ValueError(f'pilot code changed: {path}')
    rows = keyed(str(source / 'factual.jsonl'))
    if set(rows) != set(manifest['ids']) or not set(rows) <= set(ids):
        raise ValueError('incomplete or unaligned pilot')
    for row in rows.values():
        row['reused_from'] = str(source / 'factual.jsonl')
        if 'rankings_file' in row:
            row['rankings_file'] = str(source / row['rankings_file'])
    return rows


def summarize_full(rows, records, reflow, dataset, mode):
    denominators = dict.fromkeys(METRICS, 0)
    names = (*METHODS, 'reflow_cached')
    numerators = {name: dict.fromkeys(METRICS, 0) for name in names}
    statuses = {name: Counter() for name in names}
    calls = dict.fromkeys(names, 0)
    for row in rows:
        q = row['id']
        old = reflow[q]
        clean = old.get('clean_answer', '')
        if not valid_answer(clean):
            continue
        clean = canonicalize_reader_answer(records[q]['question'], clean, reader_mode=mode)
        spec = gold_answer_spec(records[q], dataset)
        results = dict(row.get('methods', {}))
        results['reflow_cached'] = dict(old, reader_called=old.get('reader_calls', 0) > 0,
                                       status=old.get('evaluation_status', 'unknown'))
        eligible = {m: m == 'answer' or clean_correct(m, clean, spec, reader_mode=mode) for m in METRICS}
        for m in METRICS:
            denominators[m] += int(eligible[m])
        for name in names:
            result = results.get(name, {})
            status = result.get('status', row.get('evaluation_status', 'missing'))
            edited = result.get('edited_answer', '')
            calls[name] += old.get('reader_calls', 0) if name == 'reflow_cached' else int(bool(result.get('reader_called')))
            executed = result.get('reader_called') and not str(status).startswith('protocol_violation')
            if executed and valid_answer(edited):
                try:
                    edited = canonicalize_reader_answer(records[q]['question'], edited, reader_mode=mode)
                except ReaderProtocolError:
                    executed = False
                    status = 'invalid_edited_answer'
            else:
                if result.get('reader_called') and not valid_answer(edited):
                    status = 'invalid_edited_answer'
                executed = False
            statuses[name][status] += 1
            if not executed:
                continue
            for m in METRICS:
                if eligible[m]:
                    numerators[name][m] += int(answer_changed(clean, edited) if m == 'answer' else
                                              correctness_lost(m, clean, edited, spec, reader_mode=mode))
    return dict(dataset=dataset, queries=len(rows), valid_clean=denominators['answer'],
                invalid_clean=sum(not valid_answer(reflow[r['id']].get('clean_answer', '')) for r in rows),
                denominators=denominators, numerators=numerators,
                rates={name: {m: numerators[name][m] / denominators[m] if denominators[m] else None for m in METRICS} for name in names},
                statuses={name: dict(value) for name, value in statuses.items()}, reader_calls=calls,
                controls_status='not_run_no_cfr_claim', budget='matched_reflow_terminal_tokens_not_reader_calls',
                method_failures_retained_in_denominator=True)


def audit_rows(rows, records, reflow, mode):
    for row in rows:
        q = row['id']
        clean = reflow[q].get('clean_answer', '')
        if not valid_answer(clean):
            if row.get('reader_calls', 0) != 0:
                raise ValueError(f'{q}: invalid clean incurred reader calls')
            continue
        if not reader_answers_exact_match(records[q]['question'], clean, row['clean_answer'], reader_mode=mode):
            raise ValueError(f'{q}: clean answer changed')
        methods = row.get('methods', {})
        if row.get('evaluation_status') == 'evaluated' and set(methods) != set(METHODS):
            raise ValueError(f'{q}: missing method output')
        if sum(bool(r.get('reader_called')) for r in methods.values()) != row.get('reader_calls', 0):
            raise ValueError(f'{q}: reader call accounting mismatch')
        budget = int(reflow[q].get('n_modified_tokens', 0))
        for result in methods.values():
            selected = result.get('selected_ids', [])
            if len(selected) != len(set(selected)) or result.get('ranking_completion_used'):
                raise ValueError(f'{q}: duplicate selection or fallback ranking')
            if result.get('reader_called') and (len(selected) != budget or result.get('n_modified_tokens') != budget):
                raise ValueError(f'{q}: token budget mismatch')


def run_dataset(config_path, out, reuse=None):
    config = json.loads(config_path.read_text())
    ds, mode = config['dataset'], config['reader_mode']
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, sha in config['expected_sha256'].items():
            if file_sha256(config[name]) != sha:
                raise ValueError(f'{ds}: frozen artifact changed: {name}')
        records, units, graphs, reflow = (keyed(config[n]) for n in ('input', 'units', 'graphs', 'reflow'))
        ids = sorted(records, key=lambda q: (hashlib.sha256(f'20261023:{q}'.encode()).hexdigest(), q))
        if len(ids) != 1000 or any(set(t) != set(ids) for t in (units, graphs, reflow)):
            raise ValueError('expected 1000 aligned IDs')
        code = [*Path('causalityrag').glob('*.py'), Path(__file__),
                Path('scripts/run_flow_ablation_pilot.py'), Path('scripts/evaluate_matched_budget_baselines.py'),
                Path('scripts/evaluate_reflow.py')]
        manifest = dict(config=config, ids=ids, graph_domain='editable_projection',
                        code_sha256={str(p): file_sha256(str(p)) for p in sorted(code)},
                        reuse=str(reuse) if reuse else None,
                        reuse_sha256=file_sha256(str(reuse / 'factual.jsonl')) if reuse else None)
        freeze_json(out / 'manifest.json', manifest)
        result_path = out / 'factual.jsonl'
        done = keyed(str(result_path)) if result_path.exists() else {}
        if not set(done) <= set(ids):
            raise ValueError('unexpected output IDs')
        if reuse:
            for q, row in reuse_pilot(reuse, config, ids).items():
                if q not in done:
                    append_row(result_path, row)
                    done[q] = row
        for q in ids:
            if q not in done and not valid_answer(reflow[q].get('clean_answer', '')):
                row = dict(id=q, clean_answer=reflow[q].get('clean_answer', ''), methods={},
                           reader_calls=0, evaluation_status='invalid_clean_answer')
                append_row(result_path, row)
                done[q] = row
        print(f'START {ds} complete={len(done)}/1000 valid={config["expected_valid"]}', flush=True)
        if len(done) < 1000:
            reader = ProtocolAdapter(config['model'], mode)
            checks_path = out / 'reader_compatibility.json'
            if not checks_path.exists():
                checks = []
                valid_ids = [q for q in ids if valid_answer(reflow[q].get('clean_answer', ''))]
                for q in valid_ids[:3]:
                    answer = reader.answer(records[q]['question'], retrieved_contexts(records[q])[:5])
                    cached = reflow[q]['clean_answer']
                    checks.append(dict(id=q, current=answer, cached=cached,
                                       matches=reader_answers_exact_match(records[q]['question'], answer, cached, reader_mode=mode)))
                freeze_json(checks_path, dict(rows=checks, reader_calls=len(checks), reader_mode=mode))
            if not all(r['matches'] for r in json.loads(checks_path.read_text())['rows']):
                raise ValueError(f'{ds}: reader compatibility mismatch; stopped without intervention calls')
            print(f'READER COMPATIBLE {ds}: 3/3', flush=True)
            pool = FrozenSharedReplacementPool(config['pool'])
            for index, q in enumerate(ids):
                if q in done:
                    continue
                started = time.perf_counter()
                cache = out / f'ranking_{index:04d}.json'
                if cache.exists():
                    rankings = json.loads(cache.read_text())
                else:
                    all_units = units_from_cache_row(records[q], units[q], k=5)
                    require_complete_graph_domain(graphs[q], all_units)
                    eligible = {str(u['unit_id']) for u in all_units if pool.is_eligible(str(u['unit_id']))}
                    rankings = {m: rank_graph_tokens(graphs[q], eligible, method=m, graph_domain='editable_projection') for m in METHODS}
                    freeze_json(cache, rankings)
                row = evaluate_query(records[q], units[q], reflow[q], rankings, pool, reader,
                                     random_seeds=[], replacement_seed=config['replacement_seed'], k=5)
                for result in row.get('methods', {}).values():
                    if result.get('reader_called') and not valid_answer(result.get('edited_answer', '')):
                        result.update(status='invalid_edited_answer', verified_flip=False)
                row.update(elapsed_seconds=time.perf_counter() - started, rankings_file=str(cache))
                append_row(result_path, row)
                done[q] = row
                print(f'VERIFIED {ds} {len(done)}/1000 {q} calls={row["reader_calls"]}', flush=True)
        if set(done) != set(ids):
            raise ValueError('incomplete final ID set')
        audit_rows(list(done.values()), records, reflow, mode)
        for name, sha in config['expected_sha256'].items():
            if file_sha256(config[name]) != sha:
                raise ValueError(f'{ds}: artifact changed during execution: {name}')
        summary = summarize_full(list(done.values()), records, reflow, ds, mode)
        summary['audit'] = '1000_unique_aligned_ids_clean_answers_calls_and_token_budgets_checked'
        summary['factual_sha256'] = file_sha256(str(result_path))
        summary['manifest_sha256'] = file_sha256(str(out / 'manifest.json'))
        freeze_json(out / 'summary.json', summary)
        csv_path = out / 'comparison.csv'
        if not csv_path.exists():
            with csv_path.open('x') as stream:
                writer = csv.writer(stream)
                writer.writerow(['dataset', 'method', 'metric', 'numerator', 'denominator', 'rate'])
                for name in summary['rates']:
                    for m in METRICS:
                        writer.writerow([ds, name, m, summary['numerators'][name][m], summary['denominators'][m], summary['rates'][name][m]])
        print(f'COMPLETE {ds}: 1000/1000; factual metrics only; controls pending', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--reuse-pilot', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    elif args.dataset:
        run_dataset(args.root / 'configs' / f'{args.dataset}.json', args.root / args.dataset, args.reuse_pilot)
    else:
        parser.error('choose --prepare or --dataset')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Resume same-position controls without changing frozen factual artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_flow_ablation_full as full
from scripts.evaluate_paraphrase_controls import evaluate_selected, stable_candidate
from scripts.build_paraphrase_control_pool import generate_batch, target_from_unit
from causalityrag.paraphrase_control import PARAPHRASE_POLICY, excluded_from_paraphrase_control
from causalityrag.reader import ReaderProtocolError, canonicalize_reader_answer
from causalityrag.evaluation_metrics import valid_answer, gold_answer_spec, clean_correct, correctness_lost, answer_changed

SOURCE = Path('/data1/yujia/CausalityRAG')
FULL = Path('out/flow_ablation_full8_20261024')
RECOVERY = Path('out/flow_ablation_recovery_20261025')
DEFAULT_OUT = Path('out/flow_ablation_cfr_20261025')
ORDER = ('triviaqa', 'hotpotqa', 'timeqa', 'finqa', 'musique', 'quartz', '2wiki', 'popqa')
REF = 'reflow_frozen_set_reverified'
METRICS = full.METRICS


def load(path):
    return json.loads(Path(path).read_text())


def available_gpu(uuid):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = {a.strip(): int(b.strip()) for a, b in (line.split(',') for line in rows.splitlines())}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    other = any(a.strip() == uuid and int(b.strip()) != os.getpid()
                for a, b in (line.split(',') for line in apps.splitlines() if ',' in line))
    if uuid not in memory or memory[uuid] >= 2048 or other:
        raise RuntimeError(f'GPU not free: {uuid}; no services will be interrupted')


def eligible_clean(record, row, dataset, mode):
    clean = row.get('clean_answer', '')
    if not valid_answer(clean) or str(row.get('evaluation_status', '')).startswith('protocol_violation_invalid_clean'):
        return None
    try:
        clean = canonicalize_reader_answer(record['question'], clean, reader_mode=mode)
    except ReaderProtocolError:
        return None
    return clean if gold_answer_spec(record, dataset).is_valid else None


def pair_key(q, method):
    return json.dumps([q, method], separators=(',', ':'))


def checkpoints(path):
    done = {}
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            key = pair_key(row['id'], row['method'])
            if key in done:
                raise ValueError(f'duplicate control checkpoint: {key}')
            done[key] = row
    return done


def usable_control(result):
    status = result.get('status', '')
    if status in {'no_selected_tokens', 'no_paraphrase_eligible_tokens'}:
        return True
    return status == 'evaluated' and result.get('reader_called') and valid_answer(result.get('edited_answer', ''))


def reuse_control(old_row, factual, clean, mode, question, units, pool, seed):
    if not old_row:
        return None
    try:
        previous = canonicalize_reader_answer(question, old_row.get('clean_answer', ''), reader_mode=mode)
    except ReaderProtocolError:
        return None
    result = old_row.get('methods', {}).get('reflow', {})
    selected = factual.get('selected_ids', [])
    expected = [u for u in selected if not excluded_from_paraphrase_control(units[u])]
    if previous != clean or result.get('selected_ids') != selected or not usable_control(result):
        return None
    if result.get('control_selected_ids', []) != expected:
        return None
    if result.get('reader_called') and result.get('n_modified_tokens') != len(expected):
        return None
    edits = {e['unit_id']: e for e in result.get('edits', []) if e.get('ok')}
    if result.get('reader_called') and any(u not in pool or u not in edits or
            edits[u].get('new') != stable_candidate(pool[u]['candidates'], u, seed)['new'] for u in expected):
        return None
    return result


def prepare_dataset(dataset, out, audit):
    dest = out / dataset
    dest.mkdir(parents=True, exist_ok=True)
    config_path = FULL / 'configs' / f'{dataset}.json'
    config = load(config_path)
    fact_path = RECOVERY / 'triviaqa_paired_reader/factual.jsonl' if dataset == 'triviaqa' else FULL / dataset / 'factual.jsonl'
    records, units, reflow, factual = (full.keyed(str(p)) for p in (config['input'], config['units'], config['reflow'], fact_path))
    ids = set(records)
    if len(ids) != 1000 or any(set(t) != ids for t in (units, reflow, factual)):
        raise ValueError(f'{dataset}: expected 1000 unique aligned IDs')
    for key, sha in config['expected_sha256'].items():
        if full.file_sha256(config[key]) != sha:
            raise ValueError(f'{dataset}: frozen {key} SHA changed')
    entry = next(e for e in audit['datasets'] if e['dataset'] == dataset)
    old_controls = full.keyed(entry['paths']['controls'])
    for key in ('controls', 'control_summary'):
        if full.file_sha256(entry['paths'][key]) != entry['shas'][key]:
            raise ValueError(f'historical {key} SHA mismatch')
    control_dir = Path(entry['paths']['controls']).parent
    pool_paths = [control_dir / 'paraphrase_pool.jsonl']
    pool_paths += sorted(control_dir.glob('*pool*.jsonl'), reverse=True)
    pool_paths += sorted((SOURCE / 'out' / dataset / 'controls').glob('*pool*.jsonl'), reverse=True)
    pool_paths += [SOURCE / 'out' / dataset / 'analysis_v1/prefix_n1_10/paraphrase_pool.jsonl',
                   SOURCE / 'out/appendix_sensitivity_cfr_v1' / dataset / 'paraphrase_pool.jsonl']
    pool_paths = list(dict.fromkeys(p for p in pool_paths if p.exists()))
    provenance = {str(p): full.file_sha256(str(p)) for p in [config_path, fact_path, Path(entry['paths']['controls']), Path(entry['paths']['control_summary']), *pool_paths]}
    pool = {}
    for p in pool_paths:
        for line in p.open():
            row = json.loads(line)
            if row.get('candidates') and str(row.get('policy', '')).startswith('llm_wordnet_paraphrase_control_v'):
                pool.setdefault(row['unit_id'], row)
    jobs, targets, excluded = [], {}, []
    for q in sorted(ids):
        f = factual[q]
        clean = eligible_clean(records[q], f, dataset, config['reader_mode'])
        if clean is None:
            excluded.append(q)
            continue
        selections = dict(f['methods'])
        if dataset != 'triviaqa':
            selections['reflow_cached'] = dict(reflow[q], reader_called=reflow[q].get('reader_calls', 0) > 0,
                                                status=reflow[q].get('evaluation_status', 'unknown'))
        expected_names = {*full.METHODS, REF if dataset == 'triviaqa' else 'reflow_cached'}
        if set(selections) != expected_names:
            raise ValueError(f'{dataset}/{q}: unexpected methods')
        by_id = {u['unit_id']: u for u in full.units_from_cache_row(records[q], units[q], k=config['k'])}
        sentences = {s['sentence_id']: s for s in units[q]['sentences']}
        for name, result in selections.items():
            selected = result.get('selected_ids', [])
            if len(set(selected)) != len(selected) or not set(selected) <= set(by_id):
                raise ValueError('invalid frozen selection')
            if result.get('reader_called') and result.get('n_modified_tokens') != len(selected):
                raise ValueError('factual token budget mismatch')
            reused = reuse_control(old_controls.get(q), result, clean, config['reader_mode'], records[q]['question'], by_id, pool, config['replacement_seed']) if name == 'reflow_cached' else None
            jobs.append(dict(id=q, method=name, clean_answer=clean, factual=result,
                             reused_result=reused, reused_from=entry['paths']['controls'] if reused else None))
            if reused is None:
                for u in selected:
                    if not excluded_from_paraphrase_control(by_id[u]):
                        target = target_from_unit(by_id[u], sentences)
                        if u in targets and targets[u] != target:
                            raise ValueError('same unit ID with inconsistent context')
                        targets[u] = target
    selected_pool = {}
    for u, target in targets.items():
        if u in pool and pool[u].get('original') == target['token']:
            selected_pool[u] = pool[u]
    missing = [targets[u] for u in sorted(set(targets) - set(selected_pool))]
    manifest = dict(dataset=dataset, config=config, sources_sha256=provenance,
                    code_sha256={str(p): full.file_sha256(str(p)) for p in sorted([Path(__file__),
                        *Path('causalityrag').glob('*.py'), Path('scripts/evaluate_paraphrase_controls.py'),
                        Path('scripts/build_paraphrase_control_pool.py'), Path('scripts/build_paraphrase_pool_for_selected_units.py'),
                        Path('scripts/run_flow_ablation_full.py')])},
                    queries=1000, invalid_clean_ids=excluded, valid_clean=1000-len(excluded),
                    expected_pairs=len(jobs), reused_pairs=sum(j['reused_result'] is not None for j in jobs),
                    missing_pool_tokens=len(missing), replacement_seed=config['replacement_seed'],
                    pool_policy=PARAPHRASE_POLICY, numeric_date_policy='exclude per original control protocol',
                    factual_protocol='current_reader_frozen_graph_terminal_replay' if dataset == 'triviaqa' else 'cached_clean_and_reflow; new_ablation_factual',
                    cfr_policy='paired factual-minus-control events; invalid clean excluded; invalid outputs excluded from paired denominator and counted explicitly; no clamping',
                    gpu_calls='control only; existing factual and graphs are not recomputed')
    for name, value in [('manifest.json', manifest), ('jobs.json', jobs), ('base_pool.json', selected_pool), ('missing_targets.json', missing)]:
        full.freeze_json(dest / name, value)
    print(f'PREFLIGHT {dataset}: valid={manifest["valid_clean"]}/1000 pairs={len(jobs)} reused={manifest["reused_pairs"]} missing_pool_tokens={len(missing)}', flush=True)


def read_pool(dest):
    pool = load(dest / 'base_pool.json')
    for path in sorted((dest / 'pool_batches').glob('*.json')):
        for row in load(path)['rows']:
            if row.get('candidates'):
                if row['unit_id'] in pool and row != pool[row['unit_id']]:
                    raise ValueError('pool checkpoint conflict')
                pool[row['unit_id']] = row
    return pool


def evaluate_pass(out, datasets, gpu):
    reader = None
    for dataset in datasets:
        dest = out / dataset
        manifest = load(dest / 'manifest.json')
        config = manifest['config']
        records, unit_rows = (full.keyed(config[k]) for k in ('input', 'units'))
        pool = read_pool(dest)
        path = dest / 'controls.jsonl'
        done = checkpoints(path)
        jobs = load(dest / 'jobs.json')
        for job in jobs:
            key = pair_key(job['id'], job['method'])
            if key in done:
                if done[key]['clean_answer'] != job['clean_answer'] or done[key]['factual_selected_ids'] != job['factual'].get('selected_ids', []):
                    raise ValueError('checkpoint no longer matches frozen job')
                continue
            q = job['id']
            selected = job['factual'].get('selected_ids', [])
            units = {u['unit_id']: u for u in full.units_from_cache_row(records[q], unit_rows[q], k=config['k'])}
            needed = [u for u in selected if not excluded_from_paraphrase_control(units[u])]
            result = job['reused_result']
            if result is None:
                if any(u not in pool for u in needed):
                    continue
                if needed and reader is None:
                    available_gpu(gpu)
                    reader = full.ProtocolAdapter(config['model'], config['reader_mode'])
                if reader is not None:
                    reader.reader_mode = config['reader_mode']
                    reader.reader.reader_mode = config['reader_mode']
                else:
                    # Empty selections return without a reader call.
                    reader_stub = type('NoCallReader', (), {'reader_mode': config['reader_mode']})()
                result = evaluate_selected(record=records[q], units=units, selected=selected,
                    clean_answer=job['clean_answer'], gold_answer=str(records[q].get('answer', '')),
                    pool=pool, reader=reader or reader_stub, seed=config['replacement_seed'], k=config['k'], reader_attempts=1)
                if result.get('status') == 'missing_paraphrase':
                    raise ValueError('preflight failed to cover selected positions')
            row = dict(id=q, method=job['method'], clean_answer=job['clean_answer'],
                       factual_selected_ids=selected, result=result, reused_from=job['reused_from'],
                       new_reader_calls=int(bool(result.get('reader_called')) and job['reused_result'] is None))
            full.append_row(path, row)
            done[key] = row
            if len(done) % 25 == 0 or len(done) == len(jobs):
                print(f'CONTROL {dataset} {len(done)}/{len(jobs)} method-pairs', flush=True)
        print(f'PASS {dataset} {len(done)}/{len(jobs)} method-pairs', flush=True)
    if reader is not None:
        del reader
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def generate_missing(out, datasets, gpu):
    client = None
    for dataset in datasets:
        dest = out / dataset
        targets = load(dest / 'missing_targets.json')
        batches = dest / 'pool_batches'
        batches.mkdir(exist_ok=True)
        for i in range(0, len(targets), 16):
            path = batches / f'{i:06d}.json'
            if path.exists():
                continue
            if client is None:
                available_gpu(gpu)
                from causalityrag.semantic_filter import ensure_wordnet_available
                from scripts.build_paraphrase_pool_for_selected_units import LocalSentenceParaphraseClient
                ensure_wordnet_available()
                client = LocalSentenceParaphraseClient(load(dest / 'manifest.json')['config']['model'], device='cuda', dtype='bfloat16')
            before = client.calls
            rows = generate_batch(targets[i:i+16], client=client, attempts=6, max_candidates=3)
            full.freeze_json(path, dict(rows=rows, generation_and_judgment_calls=client.calls-before))
            print(f'POOL {dataset} {min(i+16,len(targets))}/{len(targets)} tokens unresolved={sum(not r["candidates"] for r in rows)}', flush=True)
    if client is not None:
        del client
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def output_valid(result):
    if str(result.get('status', '')).startswith('protocol_violation'):
        return False
    return not result.get('reader_called') or valid_answer(result.get('edited_answer', ''))


def metric_event(result, clean, spec, metric, record, mode):
    if not result.get('reader_called'):
        return False
    edited = canonicalize_reader_answer(record['question'], result.get('edited_answer', ''), reader_mode=mode)
    return answer_changed(clean, edited) if metric == 'answer' else correctness_lost(metric, clean, edited, spec, reader_mode=mode)


def summarize_jobs(jobs, done, records, dataset, mode):
    counts = {}
    diagnostics = {}
    for job in jobs:
        name, q, clean = job['method'], job['id'], job['clean_answer']
        counts.setdefault(name, {m: dict(denominator=0, factual_flips=0, control_flips=0) for m in METRICS})
        diagnostics.setdefault(name, Counter())
        diag = diagnostics[name]
        row = done.get(pair_key(q, name))
        if row is None:
            diag['missing_control'] += 1
            continue
        c, f = row['result'], job['factual']
        diag['new_reader_calls'] += row['new_reader_calls']
        diag['reused_controls'] += int(bool(row['reused_from']))
        diag['control_status:' + c.get('status', 'unknown')] += 1
        if not usable_control(c) or not output_valid(f):
            diag['invalid_pair'] += 1
            continue
        try:
            spec = gold_answer_spec(records[q], dataset)
            events = {}
            for metric in METRICS:
                if metric == 'answer' or clean_correct(metric, clean, spec, reader_mode=mode):
                    events[metric] = (metric_event(f, clean, spec, metric, records[q], mode),
                                      metric_event(c, clean, spec, metric, records[q], mode))
        except ReaderProtocolError:
            diag['invalid_pair'] += 1
            continue
        for metric, (ff, cf) in events.items():
            item = counts[name][metric]
            item['denominator'] += 1
            item['factual_flips'] += int(ff)
            item['control_flips'] += int(cf)
    for metrics in counts.values():
        for item in metrics.values():
            n = item['denominator']
            item['delta_sum'] = item['factual_flips'] - item['control_flips']
            item['cfr'] = item['delta_sum'] / n if n else None
            item['paired_fr'] = item['factual_flips'] / n if n else None
            item['control_fr'] = item['control_flips'] / n if n else None
    return dict(metrics=counts, diagnostics={n: dict(d) for n, d in diagnostics.items()})


def finalize(out, datasets):
    all_summaries = []
    for dataset in datasets:
        dest = out / dataset
        manifest = load(dest / 'manifest.json')
        for p, sha in {**manifest['sources_sha256'], **manifest['code_sha256']}.items():
            if full.file_sha256(p) != sha:
                raise ValueError(f'provenance changed: {p}')
        config = manifest['config']
        for key, sha in config['expected_sha256'].items():
            if full.file_sha256(config[key]) != sha:
                raise ValueError('frozen input changed during controls')
        jobs, done = load(dest / 'jobs.json'), checkpoints(dest / 'controls.jsonl')
        if set(done) - {pair_key(j['id'], j['method']) for j in jobs}:
            raise ValueError('unexpected control keys')
        summary = summarize_jobs(jobs, done, full.keyed(config['input']), dataset, config['reader_mode'])
        complete = len(done) == len(jobs)
        pool = read_pool(dest)
        full.freeze_json(dest / 'effective_pool.json', pool)
        summary.update(dataset=dataset, queries=1000, valid_clean=manifest['valid_clean'],
                       invalid_clean_ids=manifest['invalid_clean_ids'], completed_pairs=len(done), expected_pairs=len(jobs),
                       status='complete' if complete else 'incomplete_missing_controls',
                       protocol=manifest['factual_protocol'], cfr_policy=manifest['cfr_policy'],
                       manifest_sha256=full.file_sha256(str(dest/'manifest.json')),
                       controls_sha256=full.file_sha256(str(dest/'controls.jsonl')),
                       effective_pool_sha256=full.file_sha256(str(dest/'effective_pool.json')),
                       pool_batch_sha256={str(p): full.file_sha256(str(p)) for p in sorted((dest/'pool_batches').glob('*.json'))},
                       pool_fallback_tokens=sum(any(c.get('fallback') for c in r['candidates']) for r in pool.values()))
        full.freeze_json(dest / ('summary.json' if complete else 'incomplete_summary.json'), summary)
        all_summaries.append(summary)
        print(f'CFR {dataset}: {summary["status"]} {len(done)}/{len(jobs)}', flush=True)
    if all(s['status'] == 'complete' for s in all_summaries):
        full.freeze_json(out / 'summary.json', all_summaries)
        path = out / 'comparison.csv'
        if not path.exists():
            with path.open('x') as stream:
                writer = csv.writer(stream)
                writer.writerow(['dataset', 'method', 'metric', 'denominator', 'factual_flips', 'control_flips', 'cfr'])
                for s in all_summaries:
                    for name, metrics in s['metrics'].items():
                        for metric, v in metrics.items():
                            writer.writerow([s['dataset'], name, metric, v['denominator'], v['factual_flips'], v['control_flips'], v['cfr']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with (args.out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = load(SOURCE/'out_table3_final8_20260811/audit.json')
        for dataset in ORDER:
            prepare_dataset(dataset, args.out, audit)
        if args.prepare_only:
            return
        evaluate_pass(args.out, ORDER, args.gpu)
        generate_missing(args.out, ORDER, args.gpu)
        evaluate_pass(args.out, ORDER, args.gpu)
        finalize(args.out, ORDER)


if __name__ == '__main__':
    main()

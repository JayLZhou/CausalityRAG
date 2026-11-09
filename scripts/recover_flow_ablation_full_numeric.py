#!/usr/bin/env python3
"""Targeted recovery without changing frozen sources or successful datasets."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_flow_ablation_full as full
from causalityrag.reader import ReaderClient, ReaderProtocolError, canonicalize_reader_answer
from causalityrag.evaluation_metrics import valid_answer

OLD_ROOT = Path('/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/out/flow_ablation_full8_20261024')
NEW_ROOT = OLD_ROOT.parent / 'flow_ablation_recovery_20261025'
GPU = 'GPU-4a836dad-0a67-0358-2529-5185dce33356'
PORT = 8018


def verify_inputs(config):
    for key, sha in config['expected_sha256'].items():
        if full.file_sha256(config[key]) != sha:
            raise ValueError(f'frozen input changed: {key}')


def partition_clean(rows, records, reflow, mode):
    valid, excluded = [], []
    for row in rows:
        q = row['id']
        answer = reflow[q].get('clean_answer', '')
        reason = None
        if not valid_answer(answer):
            reason = 'invalid_clean_answer'
        else:
            try:
                canonicalize_reader_answer(records[q]['question'], answer, reader_mode=mode)
            except ReaderProtocolError as exc:
                reason = str(exc)
        if reason:
            if row.get('reader_calls', 0):
                raise ValueError(f'{q}: excluded clean unexpectedly has reader calls')
            excluded.append(dict(id=q, cached_clean=answer, reason=reason))
        else:
            valid.append(row)
    return valid, excluded


def finish_quartz():
    config = json.loads((OLD_ROOT / 'configs/quartz.json').read_text())
    verify_inputs(config)
    records, reflow = (full.keyed(config[k]) for k in ('input', 'reflow'))
    source = OLD_ROOT / 'quartz/factual.jsonl'
    rows = full.keyed(str(source))
    if len(rows) != 1000 or set(rows) != set(records) or set(rows) != set(reflow):
        raise ValueError('expected 1000 unique aligned IDs')
    valid, excluded = partition_clean(list(rows.values()), records, reflow, 'quartz')
    full.audit_rows(valid, records, reflow, 'quartz')
    summary = full.summarize_full(valid, records, reflow, 'quartz', 'quartz')
    summary.update(queries=1000, invalid_clean=len(excluded), excluded_clean=excluded,
                   clean_validity='valid_text_and_unambiguous_quartz_choice',
                   factual_source=str(source), factual_sha256=full.file_sha256(str(source)),
                   audit='1000_unique_aligned_ids_protocol_valid_clean_calls_and_budgets_checked',
                   recovery_reader_calls=0)
    out = NEW_ROOT / 'quartz'
    out.mkdir(parents=True, exist_ok=True)
    full.freeze_json(out / 'manifest.json', dict(
        config=config, source_manifest_sha256=full.file_sha256(str(OLD_ROOT / 'quartz/manifest.json')),
        source_factual_sha256=summary['factual_sha256'],
        recovery_code_sha256=full.file_sha256(__file__),
        rule='exclude unmappable clean from every method numerator and denominator; never infer choice',
    ))
    full.freeze_json(out / 'summary.json', summary)
    path = out / 'comparison.csv'
    if not path.exists():
        with path.open('x') as stream:
            writer = csv.writer(stream)
            writer.writerow(['dataset', 'method', 'metric', 'numerator', 'denominator', 'rate'])
            for name, rates in summary['rates'].items():
                for metric, rate in rates.items():
                    writer.writerow(['quartz', name, metric, summary['numerators'][name][metric], summary['denominators'][metric], rate])
    print(f'QUARTZ COMPLETE: {len(valid)} evaluated, {len(excluded)} excluded clean; no new reader calls', flush=True)


def require_free_device():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    devices = {r[1].strip(): (r[0].strip(), int(r[2])) for r in csv.reader(raw.splitlines())}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    if devices[GPU][1] >= 2048 or GPU in apps.splitlines():
        raise RuntimeError('GPU occupied; refusing to compete or stop another task')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    return devices[GPU][0]


def stop_owned_service(process):
    # This session was created by this invocation, never discovered by name.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_trivia():
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    with (NEW_ROOT / 'trivia_service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((OLD_ROOT / 'configs/triviaqa.json').read_text())
        verify_inputs(config)
        (NEW_ROOT / 'configs').mkdir(exist_ok=True)
        config_path = NEW_ROOT / 'configs/triviaqa.json'
        full.freeze_json(config_path, config)
        out = NEW_ROOT / 'triviaqa'
        out.mkdir(exist_ok=True)
        if (out / 'summary.json').exists():
            print('TRIVIA ALREADY COMPLETE', flush=True)
            return
        device_index = require_free_device()
        compat = '/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452'
        command = ['/data1/yujia/envs/graphrag/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
                   '--model', config['model'], '--served-model-name', 'qwen2.5-7b',
                   '--dtype', 'bfloat16', '--trust-remote-code', '--max-model-len', '16384',
                   '--enable-chunked-prefill', '--max-num-seqs', '64', '--enforce-eager',
                   '--disable-log-requests', '--gpu-memory-utilization', '0.90',
                   '--host', '127.0.0.1', '--port', str(PORT)]
        full.freeze_json(out / 'backend_manifest_numeric_device.json', dict(
            command=command, gpu_uuid=GPU, cuda_visible_devices=device_index,
            temperature=0, max_tokens='legacy API default',
            historical_log='/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810/table3_resume_logs/vllm_8002.log',
            recovery_code_sha256=full.file_sha256(__file__),
            compat_sha256=full.file_sha256(compat + '/sitecustomize.py'),
            prior_failed_check=str(OLD_ROOT / 'triviaqa/reader_compatibility.json'),
            prior_failed_check_sha256=full.file_sha256(str(OLD_ROOT / 'triviaqa/reader_compatibility.json')),
        ))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device_index, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=compat,
                   OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        os.environ.pop('CAUSALITYRAG_READER_MAX_TOKENS', None)
        endpoint = f'http://127.0.0.1:{PORT}/v1'
        with (out / 'vllm.log').open('a') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f'START owned vLLM PID={process.pid} GPU={GPU}', flush=True)
            try:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('vLLM startup failed; inspect vllm.log')
                    try:
                        with urllib.request.urlopen(endpoint + '/models', timeout=3) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(5)
                else:
                    raise RuntimeError('vLLM startup timeout')
                print('LEGACY VLLM READY; starting full 1000-query runner with strict compatibility gate', flush=True)
                full.ProtocolAdapter = lambda model, mode: ReaderClient(
                    base_url=endpoint, model='qwen2.5-7b', timeout=300, reader_mode=mode,
                )
                full.run_dataset(config_path, out)
            finally:
                stop_owned_service(process)
                print('Owned vLLM service stopped; other services untouched', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['quartz', 'triviaqa'])
    args = parser.parse_args()
    if args.mode == 'quartz':
        finish_quartz()
    else:
        run_trivia()


if __name__ == '__main__':
    main()

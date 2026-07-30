"""Build the remaining frozen top-5 contribution graphs unattended.

The runner is intentionally server-specific: it manages the two local vLLM
replicas for reader-heavy stages, releases both GPUs for graph construction,
and resumes from validated artifacts after interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, record_id
from scripts.run_seven_dataset_pools import (
    build_inventory,
    freeze_pool,
    generate_until_complete,
)


DATASETS = (
    (
        "timeqa",
        "/data1/yujia/RAGData/timeqa/questions/timeqa.json",
        "/data1/yujia/RAGData/timeqa/corpus/timeqa_corpus.json",
    ),
    (
        "musique",
        "/data1/yujia/RAGData/musique/questions/musique.json",
        "/data1/yujia/RAGData/musique/corpus/musique_corpus.json",
    ),
    (
        "finqa",
        "/data1/yujia/RAGData/finqa/questions/finqa.json",
        "/data1/yujia/RAGData/finqa/corpus/finqa_corpus.json",
    ),
    (
        "qasper",
        "/data1/yujia/RAGData/qasper/questions/qasper.json",
        "/data1/yujia/RAGData/qasper/corpus/qasper_corpus.json",
    ),
    (
        "cuad",
        "/data1/yujia/RAGData/cuad/questions/cuad.json",
        "/data1/yujia/RAGData/cuad/corpus/cuad_corpus.json",
    ),
)

EXPECTED_ROWS = 1000
MODEL_PATH = Path("/data1/yujia/models/Qwen2.5-7B-Instruct")
EMBED_MODEL_PATH = Path("/data1/yujia/models/Qwen3-Embedding-0.6B")
GRAPH_PYTHON = "/data1/yujia/envs/graphrag/bin/python"
SPACY_PYTHON = "/data1/yujia/envs/spacyner/bin/python"
SERVICE_LOG_ROOT = Path("/data1/yujia/vllm_logs")


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def update_status(
    status_path: Path,
    *,
    dataset: str,
    stage: str,
    state: str,
    detail: str = "",
) -> None:
    atomic_json(
        status_path,
        {
            "updated_at": timestamp(),
            "dataset": dataset,
            "stage": stage,
            "state": state,
            "detail": detail,
        },
    )
    print(
        f"[automation] dataset={dataset} stage={stage} "
        f"state={state} detail={detail}",
        flush=True,
    )


def artifact_complete(
    path: Path,
    summary_path: Path,
    *,
    summary_field: str,
    summary_value: int = EXPECTED_ROWS,
) -> bool:
    return (
        jsonl_count(path) == EXPECTED_ROWS
        and int(read_json(summary_path).get(summary_field, -1)) == summary_value
    )


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    attempts: int = 3,
) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env:
        merged_env.update(env)
    for attempt in range(1, attempts + 1):
        print(
            f"[command attempt={attempt}/{attempts}] "
            + " ".join(command),
            flush=True,
        )
        completed = subprocess.run(command, cwd=cwd, env=merged_env)
        if completed.returncode == 0:
            return
        if attempt < attempts:
            time.sleep(15 * attempt)
    raise subprocess.CalledProcessError(completed.returncode, command)


def endpoint_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def wait_for_endpoint(url: str, *, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if endpoint_healthy(url):
            return
        time.sleep(5)
    raise TimeoutError(f"service did not become healthy: {url}")


def launch_detached(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    pid_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    output.close()


def matching_pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        text=True,
        capture_output=True,
        check=False,
    )
    return [
        int(value)
        for value in result.stdout.split()
        if value.isdigit() and int(value) != os.getpid()
    ]


def terminate_process_groups(pattern: str) -> None:
    groups = set()
    for pid in matching_pids(pattern):
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if groups:
        time.sleep(8)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Services:
    def __init__(self, repository: Path) -> None:
        self.repository = repository
        SERVICE_LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def ensure_spacy(self) -> None:
        if endpoint_healthy("http://127.0.0.1:8021/health"):
            return
        launch_detached(
            [
                SPACY_PYTHON,
                "scripts/spacy_annotation_server.py",
                "--port",
                "8021",
            ],
            cwd=self.repository,
            env={},
            log_path=SERVICE_LOG_ROOT / "spacy_8021_automation.log",
            pid_path=SERVICE_LOG_ROOT / "spacy_8021_automation.pid",
        )
        wait_for_endpoint("http://127.0.0.1:8021/health", timeout=180)

    def ensure_embedding(self) -> None:
        if endpoint_healthy("http://127.0.0.1:8017/v1/models"):
            return
        launch_detached(
            [
                GRAPH_PYTHON,
                "/data1/yujia/YVETTE/scripts/local_embedding_server.py",
                "--model-path",
                str(EMBED_MODEL_PATH),
                "--served-model-name",
                "Qwen3-Embedding-0.6B",
                "--port",
                "8017",
                "--device",
                "cuda",
                "--max-seq-len",
                "512",
                "--encode-batch-size",
                "16",
            ],
            cwd=self.repository,
            env={"CUDA_VISIBLE_DEVICES": "1"},
            log_path=SERVICE_LOG_ROOT / "embedding_8017_automation.log",
            pid_path=SERVICE_LOG_ROOT / "embedding_8017_automation.pid",
        )
        wait_for_endpoint(
            "http://127.0.0.1:8017/v1/models",
            timeout=300,
        )

    def ensure_reader(self) -> None:
        replicas = (
            (0, 8002, "0.92", "256"),
            (1, 8003, "0.82", "128"),
        )
        for gpu, port, utilization, sequences in replicas:
            url = f"http://127.0.0.1:{port}/v1/models"
            if endpoint_healthy(url):
                continue
            launch_detached(
                [
                    GRAPH_PYTHON,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    str(MODEL_PATH),
                    "--served-model-name",
                    "qwen2.5-7b",
                    "--dtype",
                    "bfloat16",
                    "--trust-remote-code",
                    "--max-model-len",
                    "32768",
                    "--enable-chunked-prefill",
                    "--max-num-seqs",
                    sequences,
                    "--disable-log-requests",
                    "--gpu-memory-utilization",
                    utilization,
                    "--port",
                    str(port),
                ],
                cwd=self.repository,
                env={"CUDA_VISIBLE_DEVICES": str(gpu)},
                log_path=SERVICE_LOG_ROOT / f"vllm_{port}_automation.log",
                pid_path=SERVICE_LOG_ROOT / f"vllm_{port}_automation.pid",
            )
        for _, port, _, _ in replicas:
            wait_for_endpoint(
                f"http://127.0.0.1:{port}/v1/models",
                timeout=600,
            )
        if not endpoint_healthy("http://127.0.0.1:8000/v1/models"):
            launch_detached(
                [
                    GRAPH_PYTHON,
                    "/data1/yujia/lb_proxy_healthy_2gpu.py",
                ],
                cwd=self.repository,
                env={},
                log_path=SERVICE_LOG_ROOT / "lb_proxy_automation.log",
                pid_path=SERVICE_LOG_ROOT / "lb_proxy_automation.pid",
            )
            wait_for_endpoint(
                "http://127.0.0.1:8000/v1/models",
                timeout=120,
            )

    def stop_reader(self) -> None:
        terminate_process_groups("lb_proxy_healthy_2gpu.py")
        terminate_process_groups(
            "vllm.entrypoints.openai.api_server.*--port 8002"
        )
        terminate_process_groups(
            "vllm.entrypoints.openai.api_server.*--port 8003"
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not endpoint_healthy(
                "http://127.0.0.1:8002/v1/models"
            ) and not endpoint_healthy("http://127.0.0.1:8003/v1/models"):
                return
            time.sleep(3)
        raise TimeoutError("reader replicas did not stop")


def ensure_retrieval(
    repository: Path,
    services: Services,
    *,
    dataset: str,
    questions: str,
    corpus: str,
    root: Path,
) -> None:
    retrieval = root / "retrieval" / "top10_1000.jsonl"
    summary = root / "retrieval" / "top10_1000.summary.json"
    if artifact_complete(
        retrieval,
        summary,
        summary_field="queries",
    ) and int(read_json(summary).get("retrieval_top_k", -1)) == 10:
        return
    services.stop_reader()
    services.ensure_embedding()
    run(
        [
            GRAPH_PYTHON,
            "scripts/prepare_dataset_retrieval.py",
            "--dataset",
            dataset,
            "--questions",
            questions,
            "--corpus",
            corpus,
            "--out-root",
            str(root),
            "--tokenizer-path",
            str(EMBED_MODEL_PATH),
            "--n",
            str(EXPECTED_ROWS),
            "--top-k",
            "10",
            "--chunk-size",
            "384",
            "--overlap",
            "64",
            "--embedding-base-url",
            "http://127.0.0.1:8017/v1",
            "--embedding-model",
            "Qwen3-Embedding-0.6B",
            "--embedding-batch-size",
            "64",
        ],
        cwd=repository,
    )


def ensure_units(
    repository: Path,
    services: Services,
    *,
    root: Path,
) -> None:
    units = root / "inputs" / "token_units_top10_1000.jsonl"
    summary = root / "inputs" / "token_units_top10_1000.summary.json"
    if artifact_complete(units, summary, summary_field="queries"):
        return
    services.ensure_spacy()
    run(
        [
            SPACY_PYTHON,
            "scripts/build_context_units.py",
            "--input",
            str(root / "retrieval" / "top10_1000.jsonl"),
            "--out",
            str(units),
            "--summary-out",
            str(summary),
            "--n",
            str(EXPECTED_ROWS),
            "--k",
            "10",
            "--workers",
            "8",
            "--backend",
            "service",
            "--spacy-base-url",
            "http://127.0.0.1:8021",
        ],
        cwd=repository,
    )


def pool_complete(pool_dir: Path) -> bool:
    pool = pool_dir / "shared_pool.jsonl"
    manifest = read_json(pool_dir / "shared_pool.manifest.json")
    return (
        pool.exists()
        and int(manifest.get("unresolved_typed_positions", -1)) == 0
        and int(manifest.get("eligible_positions", -1))
        == int(manifest.get("positions", -2))
    )


def ensure_pool(
    repository: Path,
    services: Services,
    *,
    root: Path,
) -> None:
    formal = root / "replacements" / "shared_pool_top10_v1"
    if pool_complete(formal):
        return
    services.ensure_spacy()
    services.ensure_reader()
    retrieval = root / "retrieval" / "top10_1000.jsonl"
    units = root / "inputs" / "token_units_top10_1000.jsonl"
    smoke = root / "replacements" / "smoke10"
    build_inventory(
        python=SPACY_PYTHON,
        repository=repository,
        retrieval=retrieval,
        units=units,
        pool_dir=smoke,
        n=10,
    )
    generate_until_complete(
        python=SPACY_PYTHON,
        repository=repository,
        typed_keys=smoke / "typed_keys.jsonl",
        seed=smoke / "typed_candidates.jsonl",
        pool_dir=smoke,
        max_passes=100,
    )
    freeze_pool(
        python=SPACY_PYTHON,
        repository=repository,
        pool_dir=smoke,
    )
    build_inventory(
        python=SPACY_PYTHON,
        repository=repository,
        retrieval=retrieval,
        units=units,
        pool_dir=formal,
        n=EXPECTED_ROWS,
    )
    generate_until_complete(
        python=SPACY_PYTHON,
        repository=repository,
        typed_keys=formal / "typed_keys.jsonl",
        seed=smoke / "typed_candidates.jsonl",
        pool_dir=formal,
        max_passes=100,
    )
    freeze_pool(
        python=SPACY_PYTHON,
        repository=repository,
        pool_dir=formal,
    )
    if not pool_complete(formal):
        raise RuntimeError(f"replacement pool failed validation: {formal}")


def ensure_clean_targets(
    repository: Path,
    services: Services,
    *,
    root: Path,
) -> None:
    target = root / "inputs" / "clean_targets_top5_1000.jsonl"
    summary = root / "inputs" / "clean_targets_top5_1000.summary.json"
    if artifact_complete(target, summary, summary_field="records"):
        return
    services.ensure_reader()
    run(
        [
            GRAPH_PYTHON,
            "scripts/generate_reader_targets.py",
            "--input",
            str(root / "retrieval" / "top10_1000.jsonl"),
            "--out",
            str(target),
            "--summary-out",
            str(summary),
            "--n",
            str(EXPECTED_ROWS),
            "--k",
            "5",
            "--workers",
            "16",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--served-model",
            "qwen2.5-7b",
        ],
        cwd=repository,
    )


def merge_graph_shards(
    retrieval_path: Path,
    shard_paths: list[Path],
    output_path: Path,
    summary_path: Path,
) -> None:
    expected_ids = [record_id(row) for row in iter_records(retrieval_path)]
    if len(expected_ids) != EXPECTED_ROWS or len(set(expected_ids)) != EXPECTED_ROWS:
        raise ValueError("retrieval does not contain 1,000 unique IDs")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in shard_paths:
        for row in iter_records(str(path)):
            identifier = record_id(row)
            if identifier in rows_by_id:
                raise ValueError(f"duplicate graph row: {identifier}")
            rows_by_id[identifier] = row
    missing = [identifier for identifier in expected_ids if identifier not in rows_by_id]
    extra = sorted(set(rows_by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(
            f"graph alignment failure: missing={missing[:5]} extra={extra[:5]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for identifier in expected_ids:
            output.write(
                json.dumps(rows_by_id[identifier], ensure_ascii=False) + "\n"
            )
    os.replace(temporary, output_path)
    statuses = Counter(
        str(rows_by_id[identifier].get("status", "")) for identifier in expected_ids
    )
    atomic_json(
        summary_path,
        {
            "records": EXPECTED_ROWS,
            "ok": statuses.get("ok", 0),
            "status_histogram": dict(sorted(statuses.items())),
            "method": "closed_flow_token_contribution_graph",
            "target_source": "frozen_vllm_results",
            "receiver_beam": 48,
            "parts": [str(path) for path in shard_paths],
            "out": str(output_path),
        },
    )


def ensure_graph(
    repository: Path,
    services: Services,
    *,
    root: Path,
) -> None:
    graph = root / "graphs" / "contribution_graph_top5_1000.jsonl"
    summary = root / "graphs" / "contribution_graph_top5_1000.summary.json"
    if artifact_complete(graph, summary, summary_field="records"):
        return
    services.stop_reader()
    shards = root / "graphs" / "shards"
    logs = root / "logs"
    shards.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    shard_specs = (
        (0, 0, 500),
        (1, 500, 500),
    )
    processes: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    shard_paths = []
    for gpu, start, count in shard_specs:
        shard = shards / f"contribution_graph_top5_{start:04d}_{start + count - 1:04d}.jsonl"
        shard_summary = shard.with_suffix(".summary.json")
        shard_paths.append(shard)
        if artifact_complete(
            shard,
            shard_summary,
            summary_field="records",
            summary_value=count,
        ):
            continue
        log_handle = (
            logs / f"build_contribution_graph_top5_{start:04d}_{start + count - 1:04d}.log"
        ).open("ab")
        command = [
            GRAPH_PYTHON,
            "scripts/build_contribution_graph.py",
            "--input",
            str(root / "retrieval" / "top10_1000.jsonl"),
            "--out",
            str(shard),
            "--summary-out",
            str(shard_summary),
            "--model-path",
            str(MODEL_PATH),
            "--start",
            str(start),
            "--n",
            str(count),
            "--k",
            "5",
            "--target",
            "results",
            "--target-results",
            str(root / "inputs" / "clean_targets_top5_1000.jsonl"),
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--max-receivers-per-layer",
            "48",
        ]
        process = subprocess.Popen(
            command,
            cwd=repository,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": str(gpu),
            },
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle, shard))
    failures = []
    for process, log_handle, shard in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append((str(shard), return_code))
    if failures:
        raise RuntimeError(f"graph shard failures: {failures}")
    merge_graph_shards(
        root / "retrieval" / "top10_1000.jsonl",
        shard_paths,
        graph,
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default="/data1/yujia/CausalityRAG/out",
    )
    parser.add_argument(
        "--status-out",
        default="/data1/yujia/CausalityRAG/out/top5_graph_automation_status.json",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    status_path = Path(args.status_out)
    services = Services(repository)
    services.ensure_spacy()
    services.ensure_embedding()
    stages = (
        ("retrieval", ensure_retrieval),
        ("token_units", ensure_units),
        ("replacement_pool", ensure_pool),
        ("clean_targets", ensure_clean_targets),
        ("contribution_graph", ensure_graph),
    )
    for dataset, questions, corpus in DATASETS:
        root = out_root / dataset
        for stage_name, stage in stages:
            update_status(
                status_path,
                dataset=dataset,
                stage=stage_name,
                state="running",
            )
            if stage_name == "retrieval":
                stage(
                    repository,
                    services,
                    dataset=dataset,
                    questions=questions,
                    corpus=corpus,
                    root=root,
                )
            else:
                stage(repository, services, root=root)
            update_status(
                status_path,
                dataset=dataset,
                stage=stage_name,
                state="complete",
            )
    services.stop_reader()
    update_status(
        status_path,
        dataset="all",
        stage="contribution_graph",
        state="complete",
        detail="timeqa,musique,finqa,qasper,cuad",
    )


if __name__ == "__main__":
    main()

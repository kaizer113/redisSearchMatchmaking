import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from matchmaking_data.benchmark import BenchmarkResult, percentile
from matchmaking_data.config import PipelineConfig
from matchmaking_data.redis_loader import get_redis_client

_THREAD_LOCAL = threading.local()


def _vector_bytes(vector: List[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def vset_element(player_id: int) -> str:
    return f"player:{player_id}"


def vector_set_progress_key(config: PipelineConfig) -> str:
    safe_key = config.vector_set_key.replace(":", "|")
    return f"{config.vector_set_progress_key}:{safe_key}:{config.dataset_version}"


def write_vector_set_checkpoint(client, config: PipelineConfig, next_player_id: int, status: str) -> None:
    client.hset(
        vector_set_progress_key(config),
        mapping={
            "dataset_version": config.dataset_version,
            "total_players": str(config.total_players),
            "duplication_factor_max": str(config.duplication_factor_max),
            "random_seed": str(config.random_seed),
            "next_player_id": str(next_player_id),
            "status": status,
        },
    )


def read_vector_set_checkpoint(client, config: PipelineConfig) -> Optional[Dict[str, str]]:
    data = client.hgetall(vector_set_progress_key(config))
    if not data:
        return None
    decoded = {}
    for key, value in data.items():
        decoded[key.decode("utf-8")] = value.decode("utf-8")
    return decoded


def vector_set_cardinality(client, key: str) -> int:
    return int(client.execute_command("VCARD", key))


def load_vector_set_batch(
    client,
    config: PipelineConfig,
    players: List[Dict[str, object]],
    use_cas: bool = True,
) -> int:
    if not players:
        return 0
    pipeline = client.pipeline(transaction=False)
    for player in players:
        command = ["VADD", config.vector_set_key]
        command.extend(
            [
                "FP32",
                _vector_bytes(player["embedding"]),
                vset_element(int(player["player_id"])),
            ]
        )
        if use_cas:
            command.append("CAS")
        attrs = json.dumps(
            {"field1": str(player["field1"]), "field2": str(player["field2"])},
            separators=(",", ":"),
        )
        command.extend(["SETATTR", attrs])
        pipeline.execute_command(*command)
    pipeline.execute()
    return len(players)


def _thread_client(redis_url: str):
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = get_redis_client(redis_url, health_check_interval=0, socket_keepalive=True)
        _THREAD_LOCAL.client = client
    return client


def vsim_query(
    client,
    config: PipelineConfig,
    element: str,
    k: int,
    ef_runtime: Optional[int] = None,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_ef: Optional[int] = None,
) -> int:
    command = [
        "VSIM",
        config.vector_set_key,
        "ELE",
        element,
        "COUNT",
        str(k),
    ]
    if ef_runtime is not None:
        command.extend(["EF", str(ef_runtime)])
    if filter_field is not None and filter_value is not None:
        command.extend(["FILTER", f'.{filter_field} == "{filter_value}"'])
        if filter_ef is not None:
            command.extend(["FILTER-EF", str(filter_ef)])
    result = client.execute_command(*command)
    return len(result) if result else 0


def run_single_vset_query(
    redis_url: str,
    config: PipelineConfig,
    element: str,
    k: int,
    mode: str,
    filter_field: Optional[str],
    filter_value: Optional[str],
    ef_runtime: Optional[int],
    filter_ef: Optional[int],
    min_results: int,
) -> float:
    client = _thread_client(redis_url)
    started = time.perf_counter()
    count = vsim_query(
        client,
        config,
        element=element,
        k=k,
        ef_runtime=ef_runtime,
        filter_field=mode if mode != "none" else None,
        filter_value=filter_value if mode != "none" else None,
        filter_ef=filter_ef,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if count < min_results:
        raise RuntimeError(f"Expected at least {min_results} results, received {count}")
    return elapsed_ms


def preload_vset_queries(
    config: PipelineConfig,
    max_player_id: int,
    query_pool_size: int,
    duplication_factor_max: int,
    seed: int,
) -> List[Tuple[str, Dict[str, str]]]:
    import random

    randomizer = random.Random(seed)
    seen = set()
    client = get_redis_client(config.redis_url, health_check_interval=0, socket_keepalive=True)
    queries: List[Tuple[str, Dict[str, str]]] = []
    while len(queries) < query_pool_size:
        player_id = randomizer.randrange(0, max_player_id)
        if player_id in seen:
            continue
        seen.add(player_id)
        key = f"{config.key_prefix}{player_id}"
        field1, field2 = client.hmget(key, "field1", "field2")
        if field1 is None or field2 is None:
            continue
        queries.append(
            (
                vset_element(player_id),
                {
                    "field1": field1.decode("utf-8") if isinstance(field1, bytes) else str(field1),
                    "field2": field2.decode("utf-8") if isinstance(field2, bytes) else str(field2),
                },
            )
        )
    return queries


def run_vset_benchmark(
    config: PipelineConfig,
    qps: int,
    duration_seconds: int,
    concurrency: int,
    max_player_id: int,
    k: int,
    query_pool_size: int,
    mode: str,
    ef_runtime: Optional[int],
    filter_ef: Optional[int],
    seed: int,
) -> BenchmarkResult:
    import random

    randomizer = random.Random(seed)
    query_pool = preload_vset_queries(
        config=config,
        max_player_id=max_player_id,
        query_pool_size=query_pool_size,
        duplication_factor_max=config.duplication_factor_max,
        seed=seed,
    )
    latencies_ms: List[float] = []
    failed_requests = 0
    lock = threading.Lock()
    started_at = time.perf_counter()
    total_requests = qps * duration_seconds
    submitted = 0
    in_flight = set()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while submitted < total_requests or in_flight:
            now = time.perf_counter()
            while submitted < total_requests and len(in_flight) < concurrency:
                target_time = started_at + (submitted / float(qps))
                if now < target_time:
                    break
                element, fields = query_pool[randomizer.randrange(0, len(query_pool))]
                future = executor.submit(
                    run_single_vset_query,
                    config.redis_url,
                    config,
                    element,
                    k,
                    mode,
                    mode if mode != "none" else None,
                    fields.get(mode) if mode != "none" else None,
                    ef_runtime,
                    filter_ef,
                    k,
                )
                in_flight.add(future)
                submitted += 1
                now = time.perf_counter()

            if not in_flight:
                sleep_for = max(0.0, (started_at + (submitted / float(qps))) - now)
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.01))
                continue

            done, pending = wait(in_flight, timeout=0.01, return_when=FIRST_COMPLETED)
            in_flight = set(pending)
            for future in done:
                try:
                    latency = future.result()
                    with lock:
                        latencies_ms.append(latency)
                except Exception:
                    with lock:
                        failed_requests += 1

    actual_duration = max(time.perf_counter() - started_at, 0.001)
    success_count = len(latencies_ms)
    achieved_qps = success_count / actual_duration if actual_duration else 0.0
    return BenchmarkResult(
        requested_qps=qps,
        achieved_qps=achieved_qps,
        duration_seconds=actual_duration,
        total_requests=total_requests,
        successful_requests=success_count,
        failed_requests=failed_requests,
        p50_ms=percentile(latencies_ms, 0.50),
        p95_ms=percentile(latencies_ms, 0.95),
        p99_ms=percentile(latencies_ms, 0.99),
        min_ms=min(latencies_ms) if latencies_ms else 0.0,
        max_ms=max(latencies_ms) if latencies_ms else 0.0,
    )

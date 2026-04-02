import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import List, Optional, Tuple

from matchmaking_data.config import PipelineConfig
from matchmaking_data.redis_loader import get_redis_client, player_key

_THREAD_LOCAL = threading.local()


@dataclass
class BenchmarkResult:
    requested_qps: int
    achieved_qps: float
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return ordered[index]


def fetch_player_embedding(client, config: PipelineConfig, player_id: int) -> bytes:
    value = client.hget(player_key(config.key_prefix, player_id), "embedding")
    if value is None:
        raise KeyError(f"Missing embedding for player:{player_id}")
    return value


def fetch_player_binary(client, config: PipelineConfig, player_id: int) -> str:
    value = client.hget(player_key(config.key_prefix, player_id), "binary")
    if value is None:
        raise KeyError(f"Missing binary for player:{player_id}")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def escape_tag_value(value: str) -> str:
    special = set('{}[]()|-=><~"\'@:;,./+*&!$%^\\ ')
    escaped = []
    for char in value:
        if char in special:
            escaped.append("\\" + char)
        else:
            escaped.append(char)
    return "".join(escaped)


def preload_query_vectors(
    config: PipelineConfig,
    max_player_id: int,
    query_pool_size: int,
    seed: int,
) -> List[Tuple[bytes, str]]:
    randomizer = random.Random(seed)
    client = get_redis_client(
        config.redis_url,
        health_check_interval=0,
        socket_keepalive=True,
    )
    vectors: List[Tuple[bytes, str]] = []
    seen = set()
    while len(vectors) < query_pool_size:
        player_id = randomizer.randrange(0, max_player_id)
        if player_id in seen:
            continue
        seen.add(player_id)
        vector = fetch_player_embedding(client, config, player_id)
        binary = fetch_player_binary(client, config, player_id)
        vectors.append((vector, binary))
    return vectors


def knn_query_from_bytes(
    client,
    config: PipelineConfig,
    query_vector: bytes,
    k: int = 50,
    binary_value: Optional[str] = None,
) -> int:
    query = f"*=>[KNN {k} @embedding $vector AS score]"
    if binary_value is not None:
        query = (
            f"@binary:{{{escape_tag_value(binary_value)}}}"
            f"=>[KNN {k} @embedding $vector AS score]"
        )
    result = client.execute_command(
        "FT.SEARCH",
        config.index_name,
        query,
        "PARAMS",
        "2",
        "vector",
        query_vector,
        "SORTBY",
        "score",
        "ASC",
        "NOCONTENT",
        "DIALECT",
        "2",
    )
    return int(result[0]) if result else 0


def run_single_query(
    redis_url: str,
    config: PipelineConfig,
    query_vector: bytes,
    expected_k: int,
    binary_value: Optional[str] = None,
    min_results: int = 1,
) -> float:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = get_redis_client(
            redis_url,
            health_check_interval=0,
            socket_keepalive=True,
        )
        _THREAD_LOCAL.client = client
    started = time.perf_counter()
    count = knn_query_from_bytes(
        client,
        config,
        query_vector,
        k=expected_k,
        binary_value=binary_value,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if count < min_results:
        raise RuntimeError(f"Expected at least {min_results} matches, received {count}")
    return elapsed_ms


def run_benchmark(
    config: PipelineConfig,
    qps: int,
    duration_seconds: int,
    concurrency: int,
    max_player_id: int,
    k: int = 50,
    query_pool_size: int = 20,
    prefilter_field: str = "none",
    seed: int = 1337,
) -> BenchmarkResult:
    randomizer = random.Random(seed)
    query_vectors = preload_query_vectors(
        config=config,
        max_player_id=max_player_id,
        query_pool_size=query_pool_size,
        seed=seed,
    )
    latencies_ms: List[float] = []
    failed_requests = 0
    lock = threading.Lock()
    started_at = time.perf_counter()
    stop_at = started_at + duration_seconds
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
                query_vector, binary_value = query_vectors[randomizer.randrange(0, len(query_vectors))]
                filtered_binary = binary_value if prefilter_field == "binary" else None
                min_results = 1 if filtered_binary is not None else k
                future = executor.submit(
                    run_single_query,
                    config.redis_url,
                    config,
                    query_vector,
                    k,
                    filtered_binary,
                    min_results,
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

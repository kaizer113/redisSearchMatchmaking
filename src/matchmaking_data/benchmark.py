import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from matchmaking_data.config import PipelineConfig
from matchmaking_data.redis_loader import get_redis_client, player_key

_THREAD_LOCAL = threading.local()
WRITE_BATCHES_PER_SECOND = 30
WRITE_BATCH_SIZE = 100


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
    sample_command: str = ""
    requested_write_qps: int = 0
    achieved_write_qps: float = 0.0
    total_writes: int = 0
    successful_writes: int = 0
    failed_writes: int = 0
    sample_write_command: str = ""


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return ordered[index]


def shell_escape_binary(value: bytes) -> str:
    escaped = []
    for byte in value:
        if byte == 0x5C:
            escaped.append(r"\\")
        elif byte == 0x27:
            escaped.append(r"\'")
        elif byte == 0x0A:
            escaped.append(r"\n")
        elif byte == 0x0D:
            escaped.append(r"\r")
        elif byte == 0x09:
            escaped.append(r"\t")
        elif 32 <= byte <= 126:
            escaped.append(chr(byte))
        else:
            escaped.append(f"\\x{byte:02x}")
    return "$'" + "".join(escaped) + "'"


def escape_tag_value(value: str) -> str:
    special = set('{}[]()|-=><~"\'@:;,./+*&!$%^\\ ')
    escaped = []
    for char in value:
        if char in special:
            escaped.append("\\" + char)
        else:
            escaped.append(char)
    return "".join(escaped)


def escape_aggregate_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def fetch_player_embedding(client, config: PipelineConfig, player_id: int) -> bytes:
    value = client.hget(player_key(config.key_prefix, player_id), "embedding")
    if value is None:
        raise KeyError(f"Missing embedding for player:{player_id}")
    return value


def fetch_player_fields(client, config: PipelineConfig, player_id: int) -> Dict[str, str]:
    raw_values = client.hmget(player_key(config.key_prefix, player_id), "field1", "field2")
    if raw_values is None or raw_values[0] is None or raw_values[1] is None:
        raise KeyError(f"Missing indexed fields for player:{player_id}")
    decoded = []
    for value in raw_values:
        decoded.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return {"field1": decoded[0], "field2": decoded[1]}


def preload_query_vectors(
    config: PipelineConfig,
    max_player_id: int,
    query_pool_size: int,
    seed: int,
    filter_field: str = "none",
    filter_value: Optional[str] = None,
) -> List[Tuple[bytes, Dict[str, str]]]:
    if max_player_id <= 0:
        raise ValueError("max_player_id must be positive")
    randomizer = random.Random(seed)
    client = get_redis_client(
        config.redis_url,
        health_check_interval=0,
        socket_keepalive=True,
    )
    vectors: List[Tuple[bytes, Dict[str, str]]] = []
    seen = set()
    target_size = min(query_pool_size, max_player_id)
    while len(vectors) < target_size:
        player_id = randomizer.randrange(0, max_player_id)
        if player_id in seen:
            continue
        seen.add(player_id)
        fields = fetch_player_fields(client, config, player_id)
        if filter_field != "none" and filter_value is not None and fields[filter_field] != filter_value:
            continue
        vectors.append((fetch_player_embedding(client, config, player_id), fields))
    return vectors


def preload_write_batches(
    config: PipelineConfig,
    max_player_id: int,
    write_pool_size: int,
    seed: int,
    batch_size: int,
) -> List[List[Tuple[str, Dict[bytes, bytes]]]]:
    if max_player_id < batch_size:
        raise ValueError(
            f"max_player_id must be at least the write batch size ({batch_size}) to run write benchmarks"
        )
    randomizer = random.Random(seed + 17)
    client = get_redis_client(
        config.redis_url,
        health_check_interval=0,
        socket_keepalive=True,
    )
    pool: List[Tuple[str, Dict[bytes, bytes]]] = []
    seen = set()
    target_records = min(max(write_pool_size * batch_size, batch_size), max_player_id)
    while len(pool) < target_records:
        player_id = randomizer.randrange(0, max_player_id)
        if player_id in seen:
            continue
        seen.add(player_id)
        key = player_key(config.key_prefix, player_id)
        mapping = client.hgetall(key)
        if mapping:
            pool.append((key, mapping))

    batches: List[List[Tuple[str, Dict[bytes, bytes]]]] = []
    for start in range(0, len(pool), batch_size):
        batch = pool[start : start + batch_size]
        if len(batch) == batch_size:
            batches.append(batch)
    if not batches:
        raise RuntimeError("Unable to build any full write batches from the existing dataset")
    return batches


def _runtime_clause(config: PipelineConfig, runtime_value: Optional[int]) -> str:
    if runtime_value is None:
        return ""
    if config.vector_algorithm.upper() == "SVS-VAMANA":
        return f" SEARCH_WINDOW_SIZE {runtime_value}"
    return f" EF_RUNTIME {runtime_value}"


def build_sample_command(
    config: PipelineConfig,
    query_vector: bytes,
    k: int,
    filter_field: str,
    filter_value: Optional[str],
    ef_runtime: Optional[int],
    filter_mode: str = "prefilter",
    aggregate_limit: int = 10_000,
) -> str:
    runtime_clause = _runtime_clause(config, ef_runtime)
    vector_arg = shell_escape_binary(query_vector)
    if filter_mode == "postfilter" and filter_field != "none" and filter_value is not None:
        query = f"*=>[KNN {aggregate_limit} @embedding $vector{runtime_clause} AS score]"
        filter_expr = f"@{filter_field}=='{escape_aggregate_string(filter_value)}'"
        return (
            f"redis-cli FT.AGGREGATE {config.index_name} '{query}' "
            f"PARAMS 2 vector {vector_arg} LOAD 3 __key @{filter_field} @score "
            f"FILTER '{filter_expr}' SORTBY 2 @score ASC LIMIT 0 {k} DIALECT 2"
        )
    query = f"*=>[KNN {k} @embedding $vector{runtime_clause} AS score]"
    if filter_field != "none" and filter_value is not None:
        query = (
            f"@{filter_field}:{{{escape_tag_value(filter_value)}}}"
            f"=>[KNN {k} @embedding $vector{runtime_clause} AS score]"
        )
    return (
        f"redis-cli FT.SEARCH {config.index_name} '{query}' "
        f"PARAMS 2 vector {vector_arg} SORTBY score ASC RETURN 4 "
        f"player_id last_login field1 field2 LIMIT 0 {k} DIALECT 2"
    )


def build_sample_write_command(batch: Sequence[Tuple[str, Dict[bytes, bytes]]]) -> str:
    preview = []
    for key, mapping in batch[:2]:
        parts = [f"HSET {key}"]
        for field, value in mapping.items():
            field_text = field.decode("utf-8") if isinstance(field, bytes) else str(field)
            value_bytes = value if isinstance(value, bytes) else str(value).encode("utf-8")
            parts.append(field_text)
            parts.append(shell_escape_binary(value_bytes))
        preview.append("redis-cli " + " ".join(parts))
    return "\n".join(preview)


def knn_query_from_bytes(
    client,
    config: PipelineConfig,
    query_vector: bytes,
    k: int = 50,
    ef_runtime: Optional[int] = None,
    filter_field: str = "none",
    filter_value: Optional[str] = None,
) -> int:
    runtime_clause = _runtime_clause(config, ef_runtime)
    query = f"*=>[KNN {k} @embedding $vector{runtime_clause} AS score]"
    if filter_field != "none" and filter_value is not None:
        query = (
            f"@{filter_field}:{{{escape_tag_value(filter_value)}}}"
            f"=>[KNN {k} @embedding $vector{runtime_clause} AS score]"
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
        "LIMIT",
        "0",
        str(k),
        "DIALECT",
        "2",
    )
    return int(result[0]) if result else 0


def aggregate_postfilter_query_from_bytes(
    client,
    config: PipelineConfig,
    query_vector: bytes,
    k: int,
    filter_field: str,
    filter_value: str,
    aggregate_limit: int,
    ef_runtime: Optional[int],
) -> int:
    runtime_clause = _runtime_clause(config, ef_runtime)
    query = f"*=>[KNN {aggregate_limit} @embedding $vector{runtime_clause} AS score]"
    filter_expr = f"@{filter_field}=='{escape_aggregate_string(filter_value)}'"
    result = client.execute_command(
        "FT.AGGREGATE",
        config.index_name,
        query,
        "PARAMS",
        "2",
        "vector",
        query_vector,
        "LOAD",
        "3",
        "__key",
        f"@{filter_field}",
        "@score",
        "FILTER",
        filter_expr,
        "SORTBY",
        "2",
        "@score",
        "ASC",
        "LIMIT",
        "0",
        str(k),
        "DIALECT",
        "2",
    )
    return int(result[0]) if result else 0


def run_single_query(
    redis_url: str,
    config: PipelineConfig,
    query_vector: bytes,
    expected_k: int,
    filter_field: str = "none",
    filter_value: Optional[str] = None,
    ef_runtime: Optional[int] = None,
    filter_mode: str = "prefilter",
    aggregate_limit: int = 10_000,
    min_results: int = 1,
) -> float:
    client = getattr(_THREAD_LOCAL, "client", None)
    client_url = getattr(_THREAD_LOCAL, "redis_url", None)
    if client is None or client_url != redis_url:
        client = get_redis_client(
            redis_url,
            health_check_interval=0,
            socket_keepalive=True,
        )
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.redis_url = redis_url
    started = time.perf_counter()
    if filter_mode == "postfilter" and filter_field != "none" and filter_value is not None:
        count = aggregate_postfilter_query_from_bytes(
            client,
            config,
            query_vector,
            k=expected_k,
            filter_field=filter_field,
            filter_value=filter_value,
            aggregate_limit=aggregate_limit,
            ef_runtime=ef_runtime,
        )
    else:
        count = knn_query_from_bytes(
            client,
            config,
            query_vector,
            k=expected_k,
            ef_runtime=ef_runtime,
            filter_field=filter_field,
            filter_value=filter_value,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if count < min_results:
        raise RuntimeError(f"Expected at least {min_results} matches, received {count}")
    return elapsed_ms


def run_single_write_batch(
    redis_url: str,
    batch: Sequence[Tuple[str, Dict[bytes, bytes]]],
) -> None:
    client = getattr(_THREAD_LOCAL, "write_client", None)
    client_url = getattr(_THREAD_LOCAL, "write_redis_url", None)
    if client is None or client_url != redis_url:
        client = get_redis_client(
            redis_url,
            health_check_interval=0,
            socket_keepalive=True,
        )
        _THREAD_LOCAL.write_client = client
        _THREAD_LOCAL.write_redis_url = redis_url
    pipeline = client.pipeline(transaction=False)
    for key, mapping in batch:
        pipeline.hset(key, mapping=mapping)
    pipeline.execute()


def run_benchmark(
    config: PipelineConfig,
    qps: int,
    duration_seconds: int,
    concurrency: int,
    max_player_id: int,
    k: int = 50,
    query_pool_size: int = 20,
    filter_field: str = "none",
    filter_value: Optional[str] = None,
    ef_runtime: Optional[int] = None,
    filter_mode: str = "prefilter",
    aggregate_limit: int = 10_000,
    write_qps: int = WRITE_BATCHES_PER_SECOND,
    write_pool_size: int = 30,
    seed: int = 1337,
) -> BenchmarkResult:
    randomizer = random.Random(seed)
    query_vectors = preload_query_vectors(
        config=config,
        max_player_id=max_player_id,
        query_pool_size=query_pool_size,
        seed=seed,
        filter_field=filter_field,
        filter_value=filter_value,
    )
    write_batches = (
        preload_write_batches(
            config=config,
            max_player_id=max_player_id,
            write_pool_size=write_pool_size,
            seed=seed,
            batch_size=WRITE_BATCH_SIZE,
        )
        if write_qps > 0
        else []
    )
    latencies_ms: List[float] = []
    failed_requests = 0
    failed_writes = 0
    lock = threading.Lock()
    started_at = time.perf_counter()
    total_requests = qps * duration_seconds
    total_writes = write_qps * duration_seconds
    submitted = 0
    submitted_writes = 0
    in_flight = set()
    sample_command = ""
    sample_write_command = ""

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while submitted < total_requests or submitted_writes < total_writes or in_flight:
            now = time.perf_counter()
            while submitted < total_requests and len(in_flight) < concurrency:
                target_time = started_at + (submitted / float(qps))
                if now < target_time:
                    break
                query_vector, query_fields = query_vectors[randomizer.randrange(0, len(query_vectors))]
                active_filter_value = query_fields[filter_field] if filter_field != "none" else None
                if filter_value is not None:
                    active_filter_value = filter_value
                if not sample_command:
                    sample_command = build_sample_command(
                        config=config,
                        query_vector=query_vector,
                        k=k,
                        filter_field=filter_field,
                        filter_value=active_filter_value,
                        ef_runtime=ef_runtime,
                        filter_mode=filter_mode,
                        aggregate_limit=aggregate_limit,
                    )
                min_results = 1 if filter_field != "none" else k
                future = executor.submit(
                    run_single_query,
                    config.redis_url,
                    config,
                    query_vector,
                    k,
                    filter_field,
                    active_filter_value,
                    ef_runtime,
                    filter_mode,
                    aggregate_limit,
                    min_results,
                )
                in_flight.add(future)
                submitted += 1
                now = time.perf_counter()

            while write_qps > 0 and submitted_writes < total_writes and len(in_flight) < concurrency:
                target_time = started_at + (submitted_writes / float(write_qps))
                if now < target_time:
                    break
                batch = write_batches[randomizer.randrange(0, len(write_batches))]
                if not sample_write_command:
                    sample_write_command = build_sample_write_command(batch)
                future = executor.submit(run_single_write_batch, config.redis_url, batch)
                future._benchmark_kind = "write"
                in_flight.add(future)
                submitted_writes += 1
                now = time.perf_counter()

            if not in_flight:
                next_read_time = started_at + (submitted / float(qps)) if submitted < total_requests else None
                next_write_time = (
                    started_at + (submitted_writes / float(write_qps))
                    if write_qps > 0 and submitted_writes < total_writes
                    else None
                )
                target_candidates = [value for value in [next_read_time, next_write_time] if value is not None]
                sleep_for = max(0.0, min(target_candidates) - now) if target_candidates else 0.0
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.01))
                continue

            done, pending = wait(in_flight, timeout=0.01, return_when=FIRST_COMPLETED)
            in_flight = set(pending)
            for future in done:
                try:
                    kind = getattr(future, "_benchmark_kind", "read")
                    result = future.result()
                    with lock:
                        if kind == "read":
                            latencies_ms.append(result)
                except Exception:
                    with lock:
                        if getattr(future, "_benchmark_kind", "read") == "write":
                            failed_writes += 1
                        else:
                            failed_requests += 1

    actual_duration = max(time.perf_counter() - started_at, 0.001)
    success_count = len(latencies_ms)
    successful_writes = total_writes - failed_writes
    achieved_qps = success_count / actual_duration if actual_duration else 0.0
    achieved_write_qps = successful_writes / actual_duration if actual_duration else 0.0
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
        sample_command=sample_command,
        requested_write_qps=write_qps,
        achieved_write_qps=achieved_write_qps,
        total_writes=total_writes,
        successful_writes=successful_writes,
        failed_writes=failed_writes,
        sample_write_command=sample_write_command,
    )

import argparse
import math
import os
import sys
from typing import Dict, Iterator, List, Optional

from matchmaking_data.benchmark import run_benchmark
from matchmaking_data.config import PipelineConfig
from matchmaking_data.embedder import SentenceTransformerBackend, attach_embeddings, embed_profiles
from matchmaking_data.generator import (
    chunked,
    generate_canonical_profiles,
    iter_expanded_players,
)
from matchmaking_data.redis_loader import (
    create_index,
    get_redis_client,
    knn_query,
    load_batch,
    verify_redis_stack,
)
from matchmaking_data.threaded_loader import load_dataset_threaded
from matchmaking_data.vector_set import (
    load_vector_set_batch,
    read_vector_set_checkpoint,
    run_vset_benchmark,
    vector_set_cardinality,
    write_vector_set_checkpoint,
)


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        redis_url=args.redis_url,
        index_name=args.index_name,
        total_players=args.total_players,
        canonical_profile_count=args.canonical_profile_count,
        duplication_factor_max=args.duplication_factor_max,
        batch_size=args.batch_size,
        random_seed=args.seed,
        embedding_model_name=args.model_name,
        vector_set_key=args.vector_set_key,
        vector_algorithm=args.vector_algorithm,
    )


def _embedded_canonical_stream(
    start_profile_id: int,
    profile_count: int,
    batch_size: int,
    seed: int,
    dimensions: int,
    model_name: str,
) -> Iterator[dict]:
    backend = SentenceTransformerBackend(model_name=model_name)
    for canonical_batch in chunked(
        generate_canonical_profiles(start_id=start_profile_id, count=profile_count, seed=seed),
        batch_size,
    ):
        yield from attach_embeddings(
            canonical_batch,
            dimensions=dimensions,
            backend=backend,
            model_name=model_name,
        )


def _decode_hash(data: Dict[bytes, bytes]) -> Dict[str, str]:
    decoded = {}
    for key, value in data.items():
        decoded[key.decode("utf-8")] = value.decode("utf-8")
    return decoded


def _load_progress_key(config: PipelineConfig, suffix: Optional[str] = None) -> str:
    key = f"{config.load_progress_key}:{config.dataset_version}"
    if suffix:
        return f"{key}:{suffix}"
    return key


def _read_load_checkpoint(
    client,
    config: PipelineConfig,
    suffix: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    data = client.hgetall(_load_progress_key(config, suffix=suffix))
    if not data:
        return None
    return _decode_hash(data)


def _write_load_checkpoint(
    client,
    config: PipelineConfig,
    next_player_id: int,
    status: str,
    suffix: Optional[str] = None,
) -> None:
    client.hset(
        _load_progress_key(config, suffix=suffix),
        mapping={
            "dataset_version": config.dataset_version,
            "total_players": str(config.total_players),
            "duplication_factor_max": str(config.duplication_factor_max),
            "random_seed": str(config.random_seed),
            "next_player_id": str(next_player_id),
            "status": status,
        },
    )


def cmd_create_index(args: argparse.Namespace) -> int:
    config = _build_config(args)
    client = get_redis_client(config.redis_url)
    verify_redis_stack(client)
    create_index(client, config)
    print(f"Index {config.index_name} is ready on {config.redis_url}")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    config = _build_config(args)
    config.validate()
    client = get_redis_client(config.redis_url)
    verify_redis_stack(client)
    create_index(client, config)

    checkpoint = _read_load_checkpoint(client, config) if args.resume else None
    if checkpoint:
        if checkpoint.get("dataset_version") != config.dataset_version:
            raise SystemExit("Checkpoint dataset_version does not match current dataset version")
        if checkpoint.get("total_players") != str(config.total_players):
            raise SystemExit("Checkpoint total_players does not match current load arguments")
        if checkpoint.get("duplication_factor_max") != str(config.duplication_factor_max):
            raise SystemExit("Checkpoint duplication_factor_max does not match current load arguments")
        if checkpoint.get("random_seed") != str(config.random_seed):
            raise SystemExit("Checkpoint random_seed does not match current load arguments")
        effective_start_player_id = int(checkpoint["next_player_id"])
    else:
        effective_start_player_id = args.start_player_id

    if effective_start_player_id >= config.total_players:
        print(f"Dataset load already complete at player {effective_start_player_id}")
        return 0

    remaining_players = config.total_players - effective_start_player_id
    start_profile_id = effective_start_player_id // config.duplication_factor_max
    start_variant_offset = effective_start_player_id % config.duplication_factor_max
    profile_count = int(
        math.ceil((remaining_players + start_variant_offset) / float(config.duplication_factor_max))
    )

    embedded_canonicals = _embedded_canonical_stream(
        start_profile_id=start_profile_id,
        profile_count=profile_count,
        batch_size=config.batch_size,
        seed=config.random_seed,
        dimensions=config.embedding_dimensions,
        model_name=config.embedding_model_name,
    )
    expanded_players = iter_expanded_players(
        canonical_profiles=embedded_canonicals,
        start_player_id=effective_start_player_id,
        total_players=remaining_players,
        duplication_factor_max=config.duplication_factor_max,
        start_variant_offset=start_variant_offset,
    )

    loaded = 0
    _write_load_checkpoint(client, config, effective_start_player_id, "running")
    for batch in chunked(expanded_players, config.batch_size):
        loaded += load_batch(client, config, batch)
        next_player_id = effective_start_player_id + loaded
        _write_load_checkpoint(client, config, next_player_id, "running")
        if loaded % max(config.batch_size, 1000) == 0 or next_player_id == config.total_players:
            print(f"Loaded {next_player_id}/{config.total_players} players", file=sys.stderr)
    _write_load_checkpoint(client, config, config.total_players, "completed")
    print(f"Finished loading {config.total_players} players")
    return 0


def cmd_load_threaded(args: argparse.Namespace) -> int:
    config = _build_config(args)
    config.validate()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.profile_batch_size <= 0:
        raise SystemExit("--profile-batch-size must be positive")
    client = get_redis_client(config.redis_url)
    verify_redis_stack(client)
    create_index(client, config)

    checkpoint_suffix = "threaded"
    checkpoint = _read_load_checkpoint(client, config, suffix=checkpoint_suffix) if args.resume else None
    if checkpoint:
        if checkpoint.get("dataset_version") != config.dataset_version:
            raise SystemExit("Checkpoint dataset_version does not match current dataset version")
        if checkpoint.get("total_players") != str(config.total_players):
            raise SystemExit("Checkpoint total_players does not match current load arguments")
        if checkpoint.get("duplication_factor_max") != str(config.duplication_factor_max):
            raise SystemExit("Checkpoint duplication_factor_max does not match current load arguments")
        if checkpoint.get("random_seed") != str(config.random_seed):
            raise SystemExit("Checkpoint random_seed does not match current load arguments")
        effective_start_player_id = int(checkpoint["next_player_id"])
    else:
        effective_start_player_id = args.start_player_id

    if effective_start_player_id >= config.total_players:
        print(f"Threaded dataset load already complete at player {effective_start_player_id}")
        return 0

    remaining_players = config.total_players - effective_start_player_id
    start_profile_id = effective_start_player_id // config.duplication_factor_max
    start_variant_offset = effective_start_player_id % config.duplication_factor_max
    profile_count = int(
        math.ceil((remaining_players + start_variant_offset) / float(config.duplication_factor_max))
    )

    _write_load_checkpoint(
        client,
        config,
        effective_start_player_id,
        "running",
        suffix=checkpoint_suffix,
    )
    load_dataset_threaded(
        config=config,
        effective_start_player_id=effective_start_player_id,
        profile_count=profile_count,
        start_profile_id=start_profile_id,
        profile_batch_size=args.profile_batch_size,
        workers=args.workers,
        write_checkpoint=lambda next_player_id, status: _write_load_checkpoint(
            client,
            config,
            next_player_id,
            status,
            suffix=checkpoint_suffix,
        ),
        progress_stream=sys.stderr,
    )
    _write_load_checkpoint(
        client,
        config,
        config.total_players,
        "completed",
        suffix=checkpoint_suffix,
    )
    print(f"Finished threaded loading of {config.total_players} players")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = _build_config(args)
    client = get_redis_client(config.redis_url)
    verify_redis_stack(client)

    vector = embed_profiles(
        texts=[args.text],
        dimensions=config.embedding_dimensions,
        model_name=config.embedding_model_name,
    )[0].tolist()

    filters: Dict[str, str] = {}
    if args.field1 is not None:
        filters["field1"] = str(args.field1)
    if args.field2 is not None:
        filters["field2"] = str(args.field2)

    results = knn_query(client, config, query_vector=vector, k=args.k, filters=filters)
    print(results)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = _build_config(args)
    client = get_redis_client(config.redis_url)
    verify_redis_stack(client)
    result = run_benchmark(
        config=config,
        qps=args.qps,
        duration_seconds=args.duration_seconds,
        concurrency=args.concurrency,
        max_player_id=args.max_player_id,
        k=args.k,
        query_pool_size=args.query_pool_size,
        filter_field=args.filter_field,
        filter_value=(str(args.filter_value) if args.filter_value is not None else None),
        ef_runtime=args.ef_runtime,
        write_qps=args.write_qps,
        write_pool_size=args.write_pool_size,
        seed=args.seed,
    )
    print("sample_command=")
    print(result.sample_command)
    if result.requested_write_qps > 0:
        print("sample_write_command=")
        print(result.sample_write_command)
    print(f"requested_qps={result.requested_qps}")
    print(f"achieved_qps={result.achieved_qps:.2f}")
    print(f"requested_write_qps={result.requested_write_qps}")
    print(f"achieved_write_qps={result.achieved_write_qps:.2f}")
    print(f"duration_seconds={result.duration_seconds:.2f}")
    print(f"total_requests={result.total_requests}")
    print(f"successful_requests={result.successful_requests}")
    print(f"failed_requests={result.failed_requests}")
    print(f"total_writes={result.total_writes}")
    print(f"successful_writes={result.successful_writes}")
    print(f"failed_writes={result.failed_writes}")
    print(f"min_ms={result.min_ms:.2f}")
    print(f"p50_ms={result.p50_ms:.2f}")
    print(f"p95_ms={result.p95_ms:.2f}")
    print(f"p99_ms={result.p99_ms:.2f}")
    print(f"max_ms={result.max_ms:.2f}")
    return 0


def cmd_load_vset(args: argparse.Namespace) -> int:
    config = _build_config(args)
    config.validate()
    client = get_redis_client(config.redis_url)

    checkpoint = read_vector_set_checkpoint(client, config) if args.resume else None
    if checkpoint:
        if checkpoint.get("dataset_version") != config.dataset_version:
            raise SystemExit("Checkpoint dataset_version does not match current dataset version")
        if checkpoint.get("total_players") != str(config.total_players):
            raise SystemExit("Checkpoint total_players does not match current load arguments")
        if checkpoint.get("duplication_factor_max") != str(config.duplication_factor_max):
            raise SystemExit("Checkpoint duplication_factor_max does not match current load arguments")
        if checkpoint.get("random_seed") != str(config.random_seed):
            raise SystemExit("Checkpoint random_seed does not match current load arguments")
        effective_start_player_id = int(checkpoint["next_player_id"])
    else:
        effective_start_player_id = args.start_player_id

    if effective_start_player_id >= config.total_players:
        print(f"Vector set load already complete at player {effective_start_player_id}")
        return 0

    remaining_players = config.total_players - effective_start_player_id
    start_profile_id = effective_start_player_id // config.duplication_factor_max
    start_variant_offset = effective_start_player_id % config.duplication_factor_max
    profile_count = int(
        math.ceil((remaining_players + start_variant_offset) / float(config.duplication_factor_max))
    )

    embedded_canonicals = _embedded_canonical_stream(
        start_profile_id=start_profile_id,
        profile_count=profile_count,
        batch_size=config.batch_size,
        seed=config.random_seed,
        dimensions=config.embedding_dimensions,
        model_name=config.embedding_model_name,
    )
    expanded_players = iter_expanded_players(
        canonical_profiles=embedded_canonicals,
        start_player_id=effective_start_player_id,
        total_players=remaining_players,
        duplication_factor_max=config.duplication_factor_max,
        start_variant_offset=start_variant_offset,
    )

    loaded = 0
    write_vector_set_checkpoint(client, config, effective_start_player_id, "running")
    for batch in chunked(expanded_players, config.batch_size):
        loaded += load_vector_set_batch(client, config, batch, use_cas=not args.no_cas)
        next_player_id = effective_start_player_id + loaded
        write_vector_set_checkpoint(client, config, next_player_id, "running")
        if loaded % max(config.batch_size, 1000) == 0 or next_player_id == config.total_players:
            current_size = vector_set_cardinality(client, config.vector_set_key)
            print(
                f"Loaded {next_player_id}/{config.total_players} vector-set entries (vcard={current_size})",
                file=sys.stderr,
            )
    write_vector_set_checkpoint(client, config, config.total_players, "completed")
    print(f"Finished loading {config.total_players} vector-set entries")
    return 0


def cmd_benchmark_vset(args: argparse.Namespace) -> int:
    config = _build_config(args)
    client = get_redis_client(config.redis_url)
    card = vector_set_cardinality(client, config.vector_set_key)
    if card == 0:
        raise SystemExit(f"Vector set {config.vector_set_key} is empty")
    result = run_vset_benchmark(
        config=config,
        qps=args.qps,
        duration_seconds=args.duration_seconds,
        concurrency=args.concurrency,
        max_player_id=args.max_player_id,
        k=args.k,
        query_pool_size=args.query_pool_size,
        mode=args.mode,
        ef_runtime=args.ef_runtime,
        filter_ef=args.filter_ef,
        seed=args.seed,
    )
    print(f"requested_qps={result.requested_qps}")
    print(f"achieved_qps={result.achieved_qps:.2f}")
    print(f"duration_seconds={result.duration_seconds:.2f}")
    print(f"total_requests={result.total_requests}")
    print(f"successful_requests={result.successful_requests}")
    print(f"failed_requests={result.failed_requests}")
    print(f"min_ms={result.min_ms:.2f}")
    print(f"p50_ms={result.p50_ms:.2f}")
    print(f"p95_ms={result.p95_ms:.2f}")
    print(f"p99_ms={result.p99_ms:.2f}")
    print(f"max_ms={result.max_ms:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic matchmaking dataset pipeline")
    parser.set_defaults(func=None)
    defaults = PipelineConfig()

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--redis-url", default=os.getenv("REDIS_URL", defaults.redis_url))
    common.add_argument("--index-name", default=os.getenv("INDEX_NAME", defaults.index_name))
    common.add_argument("--total-players", type=int, default=defaults.total_players)
    common.add_argument("--canonical-profile-count", type=int, default=defaults.canonical_profile_count)
    common.add_argument("--duplication-factor-max", type=int, default=defaults.duplication_factor_max)
    common.add_argument("--batch-size", type=int, default=defaults.batch_size)
    common.add_argument("--seed", type=int, default=defaults.random_seed)
    common.add_argument("--vector-set-key", default=os.getenv("VECTOR_SET_KEY", defaults.vector_set_key))
    common.add_argument(
        "--vector-algorithm",
        choices=["HNSW", "SVS-VAMANA"],
        default="HNSW",
    )
    common.add_argument(
        "--model-name",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="SentenceTransformers model to use for local embeddings",
    )

    subparsers = parser.add_subparsers(dest="command")

    create_index_parser = subparsers.add_parser("create-index", parents=[common])
    create_index_parser.set_defaults(func=cmd_create_index)

    load_parser = subparsers.add_parser("load", parents=[common])
    load_parser.add_argument("--start-player-id", type=int, default=0)
    load_parser.add_argument("--resume", action="store_true")
    load_parser.set_defaults(func=cmd_load)

    threaded_load_parser = subparsers.add_parser("load-threaded", parents=[common])
    threaded_load_parser.add_argument("--start-player-id", type=int, default=0)
    threaded_load_parser.add_argument("--resume", action="store_true")
    threaded_load_parser.add_argument(
        "--workers",
        type=int,
        default=max((os.cpu_count() or 2) // 2, 2),
        help="Number of worker threads. Each worker keeps its own Redis client and embedding backend.",
    )
    threaded_load_parser.add_argument(
        "--profile-batch-size",
        type=int,
        default=64,
        help="Canonical profiles processed per threaded task.",
    )
    threaded_load_parser.set_defaults(func=cmd_load_threaded)

    load_vset_parser = subparsers.add_parser("load-vset", parents=[common])
    load_vset_parser.add_argument("--start-player-id", type=int, default=0)
    load_vset_parser.add_argument("--resume", action="store_true")
    load_vset_parser.add_argument("--no-cas", action="store_true")
    load_vset_parser.set_defaults(func=cmd_load_vset)

    query_parser = subparsers.add_parser("query", parents=[common])
    query_parser.add_argument("--text", required=True)
    query_parser.add_argument("--k", type=int, default=10)
    query_parser.add_argument("--field1", choices=[0, 1], type=int)
    query_parser.add_argument("--field2", choices=[0, 1], type=int)
    query_parser.set_defaults(func=cmd_query)

    benchmark_parser = subparsers.add_parser("benchmark", parents=[common])
    benchmark_parser.add_argument("--qps", type=int, default=1000)
    benchmark_parser.add_argument("--duration-seconds", type=int, default=30)
    benchmark_parser.add_argument("--concurrency", type=int, default=128)
    benchmark_parser.add_argument("--max-player-id", type=int, default=defaults.total_players)
    benchmark_parser.add_argument("--k", type=int, default=50)
    benchmark_parser.add_argument("--query-pool-size", type=int, default=20)
    benchmark_parser.add_argument(
        "--filter-field",
        choices=["none", "field1", "field2"],
        default="none",
    )
    benchmark_parser.add_argument(
        "--filter-value",
        choices=[0, 1],
        type=int,
        default=None,
        help="Optional fixed value to use when filtering by field1 or field2.",
    )
    benchmark_parser.add_argument(
        "--ef-runtime",
        type=int,
        default=None,
        help="Override HNSW EF_RUNTIME or SVS-VAMANA SEARCH_WINDOW_SIZE at query time.",
    )
    benchmark_parser.add_argument(
        "--write-qps",
        type=int,
        default=30,
        help="Background write batch rate. Use 0 for reads-only benchmarks.",
    )
    benchmark_parser.add_argument(
        "--write-pool-size",
        type=int,
        default=30,
        help="Number of preloaded 100-record write batches to cycle through.",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    benchmark_vset_parser = subparsers.add_parser("benchmark-vset", parents=[common])
    benchmark_vset_parser.add_argument("--qps", type=int, default=1000)
    benchmark_vset_parser.add_argument("--duration-seconds", type=int, default=30)
    benchmark_vset_parser.add_argument("--concurrency", type=int, default=128)
    benchmark_vset_parser.add_argument("--max-player-id", type=int, default=defaults.total_players)
    benchmark_vset_parser.add_argument("--k", type=int, default=50)
    benchmark_vset_parser.add_argument("--query-pool-size", type=int, default=20)
    benchmark_vset_parser.add_argument("--ef-runtime", type=int, default=64)
    benchmark_vset_parser.add_argument("--filter-ef", type=int, default=1000)
    benchmark_vset_parser.add_argument("--mode", choices=["none", "field1", "field2"], default="none")
    benchmark_vset_parser.set_defaults(func=cmd_benchmark_vset)

    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.func:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

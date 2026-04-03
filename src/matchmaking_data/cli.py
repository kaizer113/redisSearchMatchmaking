import argparse
import math
import os
import sys
from typing import Dict, Iterator, List, Optional

from matchmaking_data.benchmark import run_benchmark
from matchmaking_data.config import PipelineConfig
from matchmaking_data.embedder import SentenceTransformerBackend, attach_embeddings, embed_profiles
from matchmaking_data.generator import (
    binary_value_for_profile,
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


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        redis_url=args.redis_url,
        total_players=args.total_players,
        canonical_profile_count=args.canonical_profile_count,
        duplication_factor_max=args.duplication_factor_max,
        batch_size=args.batch_size,
        random_seed=args.seed,
        embedding_model_name=args.model_name,
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
    if args.game:
        filters["game"] = args.game
    if args.platform:
        filters["platform"] = args.platform
    if args.region:
        filters["region"] = args.region
    if args.rank_tier:
        filters["rank_tier"] = args.rank_tier

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
        prefilter_field=args.prefilter_field,
        aggregate_limit=args.aggregate_limit,
        ef_runtime=args.ef_runtime,
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


def cmd_rewrite_binary(args: argparse.Namespace) -> int:
    config = _build_config(args)
    client = get_redis_client(config.redis_url)
    verify_redis_stack(client)

    updated = 0
    batch_size = args.batch_size
    end_player_id = args.start_player_id + config.total_players

    for batch_start in range(args.start_player_id, end_player_id, batch_size):
        batch_end = min(batch_start + batch_size, end_player_id)
        pipe = client.pipeline(transaction=False)
        for player_id in range(batch_start, batch_end):
            player_key = f"{config.key_prefix}{player_id}"
            profile_id = player_id // args.duplication_factor_max
            binary_value = binary_value_for_profile(profile_id)
            pipe.hset(player_key, "binary", binary_value)
        pipe.execute()
        updated += batch_end - batch_start
        if updated % max(100_000, batch_size) == 0 or updated == config.total_players:
            print(f"Rewrote {updated} binary fields", file=sys.stderr)

    print(f"Finished rewriting binary field for {updated} players")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic matchmaking dataset pipeline")
    parser.set_defaults(func=None)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    common.add_argument("--total-players", type=int, default=10_000_000)
    common.add_argument("--canonical-profile-count", type=int, default=1_000_000)
    common.add_argument("--duplication-factor-max", type=int, default=10)
    common.add_argument("--batch-size", type=int, default=500)
    common.add_argument("--seed", type=int, default=1337)
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

    query_parser = subparsers.add_parser("query", parents=[common])
    query_parser.add_argument("--text", required=True)
    query_parser.add_argument("--k", type=int, default=10)
    query_parser.add_argument("--game")
    query_parser.add_argument("--platform")
    query_parser.add_argument("--region")
    query_parser.add_argument("--rank-tier")
    query_parser.set_defaults(func=cmd_query)

    benchmark_parser = subparsers.add_parser("benchmark", parents=[common])
    benchmark_parser.add_argument("--qps", type=int, default=1000)
    benchmark_parser.add_argument("--duration-seconds", type=int, default=30)
    benchmark_parser.add_argument("--concurrency", type=int, default=128)
    benchmark_parser.add_argument("--max-player-id", type=int, default=10_000_000)
    benchmark_parser.add_argument("--k", type=int, default=50)
    benchmark_parser.add_argument("--query-pool-size", type=int, default=20)
    benchmark_parser.add_argument(
        "--prefilter-field",
        choices=["none", "binary", "postfilter"],
        default="none",
    )
    benchmark_parser.add_argument(
        "--aggregate-limit",
        type=int,
        default=10_000,
        help="KNN candidate pool for postfilter aggregate benchmarks.",
    )
    benchmark_parser.add_argument(
        "--ef-runtime",
        type=int,
        default=None,
        help="Override HNSW EF_RUNTIME at query time.",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    rewrite_binary_parser = subparsers.add_parser("rewrite-binary", parents=[common])
    rewrite_binary_parser.add_argument("--start-player-id", type=int, default=0)
    rewrite_binary_parser.set_defaults(func=cmd_rewrite_binary)

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

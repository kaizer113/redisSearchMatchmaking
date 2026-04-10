from typing import Dict, Iterable, List, Optional

import numpy as np

from matchmaking_data.config import PipelineConfig


def get_redis_client(redis_url: str, **kwargs):
    try:
        from redis import Redis
    except ImportError as exc:
        raise RuntimeError(
            "redis-py is required for Redis loading. Install dependencies from requirements.txt first."
        ) from exc
    options = {"decode_responses": False}
    options.update(kwargs)
    return Redis.from_url(redis_url, **options)


def player_key(prefix: str, player_id: int) -> str:
    return f"{prefix}{player_id}"


def _vector_bytes(vector: List[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def create_index(client, config: PipelineConfig) -> None:
    try:
        existing = client.execute_command("FT._LIST")
        if config.index_name.encode("utf-8") in existing or config.index_name in existing:
            return
    except Exception:
        pass

    command = [
        "FT.CREATE",
        config.index_name,
        "ON",
        "HASH",
        "PREFIX",
        "1",
        config.key_prefix,
        "SCHEMA",
        "field1",
        "TAG",
        "field2",
        "TAG",
        "embedding",
        "VECTOR",
        config.vector_algorithm,
        "12",
        "TYPE",
        "FLOAT32",
        "DIM",
        str(config.embedding_dimensions),
        "DISTANCE_METRIC",
        config.distance_metric,
    ]
    if config.vector_algorithm.upper() == "SVS-VAMANA":
        command.extend(
            [
                "GRAPH_MAX_DEGREE",
                str(config.vamana_graph_max_degree),
                "CONSTRUCTION_WINDOW_SIZE",
                str(config.vamana_construction_window_size),
                "SEARCH_WINDOW_SIZE",
                str(config.vamana_search_window_size),
            ]
        )
    else:
        command.extend(
            [
                "M",
                str(config.hnsw_m),
                "EF_CONSTRUCTION",
                str(config.hnsw_ef_construction),
                "EF_RUNTIME",
                str(config.hnsw_ef_runtime),
            ]
        )
    client.execute_command(*command)


def load_batch(client, config: PipelineConfig, players: List[Dict[str, object]]) -> int:
    pipeline = client.pipeline(transaction=False)
    for player in players:
        mapping = {}
        for key, value in player.items():
            if key == "profile_text":
                continue
            if key == "embedding":
                mapping[key] = _vector_bytes(value)
            elif isinstance(value, (int, float)):
                mapping[key] = str(value)
            else:
                mapping[key] = value
        pipeline.hset(player_key(config.key_prefix, int(player["player_id"])), mapping=mapping)
    pipeline.execute()
    return len(players)


def knn_query(
    client,
    config: PipelineConfig,
    query_vector: List[float],
    k: int = 10,
    filters: Optional[Dict[str, str]] = None,
):
    filters = filters or {}
    clauses = []
    for field, value in sorted(filters.items()):
        clauses.append(f"@{field}:{{{value}}}")
    filter_query = " ".join(clauses) if clauses else "*"
    query = f"{filter_query}=>[KNN {k} @embedding $vector AS score]"
    return client.execute_command(
        "FT.SEARCH",
        config.index_name,
        query,
        "PARAMS",
        "2",
        "vector",
        _vector_bytes(query_vector),
        "SORTBY",
        "score",
        "ASC",
        "RETURN",
        "4",
        "player_id",
        "last_login",
        "field1",
        "field2",
        "DIALECT",
        "2",
    )


def verify_redis_stack(client) -> None:
    client.execute_command("PING")
    modules = client.execute_command("MODULE", "LIST")
    lowered = repr(modules).lower()
    if "search" not in lowered:
        module_names = []
        for module in modules:
            if isinstance(module, list):
                for index in range(0, len(module) - 1, 2):
                    if module[index] in (b"name", "name"):
                        raw_name = module[index + 1]
                        if isinstance(raw_name, bytes):
                            module_names.append(raw_name.decode("utf-8", errors="replace"))
                        else:
                            module_names.append(str(raw_name))
        raise RuntimeError(
            f"Required Redis modules not detected on {client.connection_pool.connection_kwargs.get('host', 'the configured Redis server')}. "
            "Expected RediSearch, found: "
            + (", ".join(module_names) if module_names else "none")
        )

from __future__ import annotations

import math
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from matchmaking_data.config import PipelineConfig
from matchmaking_data.embedder import SentenceTransformerBackend, attach_embeddings
from matchmaking_data.generator import expand_profile, generate_canonical_profiles
from matchmaking_data.redis_loader import get_redis_client, load_batch


_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class BatchResult:
    range_start: int
    range_end: int
    loaded_players: int


def player_range_for_profile_batch(
    batch_start_profile_id: int,
    profile_count: int,
    duplication_factor_max: int,
    effective_start_player_id: int,
    total_players: int,
) -> Tuple[int, int]:
    range_start = max(effective_start_player_id, batch_start_profile_id * duplication_factor_max)
    range_end = min(total_players, (batch_start_profile_id + profile_count) * duplication_factor_max)
    if range_start >= range_end:
        return total_players, total_players
    return range_start, range_end


class CompletedRanges:
    def __init__(self, initial_offset: int) -> None:
        self._next_offset = initial_offset
        self._completed: Dict[int, int] = {}

    @property
    def next_offset(self) -> int:
        return self._next_offset

    def mark_completed(self, range_start: int, range_end: int) -> int:
        if range_end <= range_start:
            return self._next_offset
        self._completed[range_start] = range_end
        while self._next_offset in self._completed:
            self._next_offset = self._completed.pop(self._next_offset)
        return self._next_offset


def _worker_backend(model_name: str) -> SentenceTransformerBackend:
    backend = getattr(_THREAD_LOCAL, "embedding_backend", None)
    backend_model = getattr(_THREAD_LOCAL, "embedding_model_name", None)
    if backend is None or backend_model != model_name:
        backend = SentenceTransformerBackend(model_name=model_name)
        _THREAD_LOCAL.embedding_backend = backend
        _THREAD_LOCAL.embedding_model_name = model_name
    return backend


def _worker_client(redis_url: str):
    client = getattr(_THREAD_LOCAL, "redis_client", None)
    client_url = getattr(_THREAD_LOCAL, "redis_url", None)
    if client is None or client_url != redis_url:
        client = get_redis_client(redis_url)
        _THREAD_LOCAL.redis_client = client
        _THREAD_LOCAL.redis_url = redis_url
    return client


def build_expanded_players_for_batch(
    embedded_profiles: Iterable[dict],
    batch_start_profile_id: int,
    duplication_factor_max: int,
    effective_start_player_id: int,
    total_players: int,
) -> List[dict]:
    players: List[dict] = []
    for offset, profile in enumerate(embedded_profiles):
        profile_id = batch_start_profile_id + offset
        base_player_id = profile_id * duplication_factor_max
        for variant_index in range(duplication_factor_max):
            player_id = base_player_id + variant_index
            if player_id < effective_start_player_id:
                continue
            if player_id >= total_players:
                return players
            player = expand_profile(profile, variant_index)
            player["player_id"] = player_id
            players.append(player)
    return players


def process_profile_batch(
    config: PipelineConfig,
    batch_start_profile_id: int,
    profile_count: int,
    effective_start_player_id: int,
) -> BatchResult:
    range_start, range_end = player_range_for_profile_batch(
        batch_start_profile_id=batch_start_profile_id,
        profile_count=profile_count,
        duplication_factor_max=config.duplication_factor_max,
        effective_start_player_id=effective_start_player_id,
        total_players=config.total_players,
    )
    if range_start >= range_end:
        return BatchResult(range_start=range_start, range_end=range_end, loaded_players=0)

    backend = _worker_backend(config.embedding_model_name)
    client = _worker_client(config.redis_url)
    canonical_profiles = list(
        generate_canonical_profiles(
            start_id=batch_start_profile_id,
            count=profile_count,
            seed=config.random_seed,
        )
    )
    embedded_profiles = attach_embeddings(
        canonical_profiles,
        dimensions=config.embedding_dimensions,
        backend=backend,
        model_name=config.embedding_model_name,
    )
    players = build_expanded_players_for_batch(
        embedded_profiles=embedded_profiles,
        batch_start_profile_id=batch_start_profile_id,
        duplication_factor_max=config.duplication_factor_max,
        effective_start_player_id=effective_start_player_id,
        total_players=config.total_players,
    )
    loaded = load_batch(client, config, players) if players else 0
    return BatchResult(range_start=range_start, range_end=range_end, loaded_players=loaded)


def load_dataset_threaded(
    *,
    config: PipelineConfig,
    effective_start_player_id: int,
    profile_count: int,
    start_profile_id: int,
    profile_batch_size: int,
    workers: int,
    write_checkpoint,
    progress_stream,
) -> None:
    max_pending = max(workers * 2, 1)
    completed = CompletedRanges(initial_offset=effective_start_player_id)
    submitted_profile_id = start_profile_id
    remaining_profiles = profile_count
    pending: Dict[Future, Tuple[int, int]] = {}

    def submit_one(executor: ThreadPoolExecutor) -> bool:
        nonlocal submitted_profile_id, remaining_profiles
        if remaining_profiles <= 0:
            return False
        this_batch_profiles = min(profile_batch_size, remaining_profiles)
        future = executor.submit(
            process_profile_batch,
            config,
            submitted_profile_id,
            this_batch_profiles,
            effective_start_player_id,
        )
        pending[future] = (submitted_profile_id, this_batch_profiles)
        submitted_profile_id += this_batch_profiles
        remaining_profiles -= this_batch_profiles
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(pending) < max_pending and submit_one(executor):
            pass

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                next_player_id = completed.mark_completed(result.range_start, result.range_end)
                write_checkpoint(next_player_id, "running")
                if next_player_id % max(config.batch_size, 1000) == 0 or next_player_id == config.total_players:
                    print(
                        f"Loaded {next_player_id}/{config.total_players} players",
                        file=progress_stream,
                    )
                while len(pending) < max_pending and submit_one(executor):
                    pass

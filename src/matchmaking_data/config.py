from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    redis_url: str = "redis://localhost:6379"
    index_name: str = "idx:players"
    key_prefix: str = "player:"
    total_players: int = 10_000_000
    canonical_profile_count: int = 1_000_000
    duplication_factor_max: int = 10
    batch_size: int = 500
    embedding_dimensions: int = 64
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    distance_metric: str = "L2"
    vector_algorithm: str = "HNSW"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 64
    random_seed: int = 1337
    dataset_version: str = "games-2026-04-v1"
    load_progress_key: str = "load_progress:players"

    def validate(self) -> None:
        if self.total_players <= 0:
            raise ValueError("total_players must be positive")
        if self.canonical_profile_count <= 0:
            raise ValueError("canonical_profile_count must be positive")
        if self.duplication_factor_max <= 0:
            raise ValueError("duplication_factor_max must be positive")
        max_possible = self.canonical_profile_count * self.duplication_factor_max
        if max_possible < self.total_players:
            raise ValueError(
                "canonical_profile_count * duplication_factor_max must cover total_players"
            )

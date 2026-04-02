from typing import Iterable, List, Optional

import numpy as np


class EmbeddingBackend:
    def encode(self, texts: List[str], dimensions: int) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install dependencies from requirements.txt first."
                ) from exc
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def encode(self, texts: List[str], dimensions: int) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
            truncate_dim=dimensions,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != dimensions:
            raise ValueError(
                f"Expected embedding shape (N, {dimensions}), received {array.shape}"
            )
        return array


def embed_profiles(
    texts: List[str],
    dimensions: int = 64,
    backend: Optional[EmbeddingBackend] = None,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
) -> np.ndarray:
    if not texts:
        return np.empty((0, dimensions), dtype=np.float32)
    active_backend = backend or SentenceTransformerBackend(model_name=model_name)
    vectors = active_backend.encode(texts, dimensions=dimensions)
    return np.asarray(vectors, dtype=np.float32)


def attach_embeddings(
    profiles: Iterable[dict],
    dimensions: int = 64,
    backend: Optional[EmbeddingBackend] = None,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
) -> List[dict]:
    materialized = list(profiles)
    texts = [profile["profile_text"] for profile in materialized]
    vectors = embed_profiles(
        texts=texts,
        dimensions=dimensions,
        backend=backend,
        model_name=model_name,
    )
    for profile, vector in zip(materialized, vectors):
        profile["embedding"] = vector.astype(np.float32).tolist()
    return materialized

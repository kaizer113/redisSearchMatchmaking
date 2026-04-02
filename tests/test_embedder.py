import unittest

import numpy as np

from matchmaking_data.embedder import EmbeddingBackend, attach_embeddings, embed_profiles


class FakeBackend(EmbeddingBackend):
    def encode(self, texts, dimensions):
        values = np.arange(len(texts) * dimensions, dtype=np.float32)
        return values.reshape(len(texts), dimensions)


class EmbedderTests(unittest.TestCase):
    def test_embed_profiles_shape_and_dtype(self):
        vectors = embed_profiles(
            ["one", "two"],
            dimensions=64,
            backend=FakeBackend(),
        )
        self.assertEqual((2, 64), vectors.shape)
        self.assertEqual(np.float32, vectors.dtype)

    def test_attach_embeddings_sets_list_values(self):
        profiles = [
            {"profile_text": "alpha", "canonical_profile_id": 1},
            {"profile_text": "beta", "canonical_profile_id": 2},
        ]
        updated = attach_embeddings(
            profiles=profiles,
            dimensions=4,
            backend=FakeBackend(),
        )
        self.assertEqual([0.0, 1.0, 2.0, 3.0], updated[0]["embedding"])
        self.assertEqual([4.0, 5.0, 6.0, 7.0], updated[1]["embedding"])


if __name__ == "__main__":
    unittest.main()

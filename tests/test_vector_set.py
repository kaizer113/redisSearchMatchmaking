import unittest

from matchmaking_data.config import PipelineConfig
from matchmaking_data.vector_set import vset_element, vector_set_progress_key


class VectorSetTests(unittest.TestCase):
    def test_vset_element_format(self):
        self.assertEqual("player:42", vset_element(42))

    def test_vector_set_progress_key_uses_dataset_version(self):
        config = PipelineConfig(dataset_version="games-2026-04-v1", vector_set_key="vset:players:exp1")
        self.assertEqual(
            "load_progress:vset:vset|players|exp1:games-2026-04-v1",
            vector_set_progress_key(config),
        )


if __name__ == "__main__":
    unittest.main()

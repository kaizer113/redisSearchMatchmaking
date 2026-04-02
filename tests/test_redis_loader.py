import os
import unittest

from matchmaking_data.config import PipelineConfig
from matchmaking_data.redis_loader import create_index, load_batch, player_key


class FakeRedis:
    def __init__(self):
        self.commands = []
        self.last_pipeline = None

    def execute_command(self, *args):
        self.commands.append(args)
        if args == ("FT._LIST",):
            return []
        if args[:2] == ("MODULE", "LIST"):
            return [[b"name", b"search"], [b"name", b"ReJSON"]]
        if args == ("PING",):
            return b"PONG"
        return b"OK"

    def pipeline(self, transaction=False):
        self.last_pipeline = FakePipeline()
        return self.last_pipeline


class FakePipeline:
    def __init__(self):
        self.commands = []

    def execute_command(self, *args):
        self.commands.append(args)

    def hset(self, name, mapping):
        self.commands.append(("HSET", name, mapping))

    def execute(self):
        return [b"OK" for _ in self.commands]


class RedisLoaderUnitTests(unittest.TestCase):
    def test_player_key_uses_prefix(self):
        self.assertEqual("player:12", player_key("player:", 12))

    def test_create_index_uses_hnsw_l2(self):
        config = PipelineConfig()
        client = FakeRedis()
        create_index(client, config)
        joined = " ".join(str(part) for part in client.commands[-1])
        self.assertIn("VECTOR HNSW", joined)
        self.assertIn("DISTANCE_METRIC L2", joined)
        self.assertIn("DIM 64", joined)
        self.assertIn("ON HASH", joined)
        self.assertIn("binary TAG", joined)
        self.assertNotIn("$.player_id", joined)

    def test_load_batch_omits_player_id_from_json_payload(self):
        config = PipelineConfig()
        client = FakeRedis()
        players = [
            {
                "player_id": 7,
                "username": "Nova",
                "game": "valorant",
                "embedding": [0.1, 0.2],
            }
        ]
        load_batch(client, config, players)
        self.assertIsNotNone(client.last_pipeline)
        command = client.last_pipeline.commands[0]
        self.assertEqual("HSET", command[0])
        self.assertEqual("player:7", command[1])
        self.assertNotIn("player_id", command[2])
        self.assertIsInstance(command[2]["embedding"], bytes)


@unittest.skipUnless(os.getenv("RUN_REDIS_INTEGRATION") == "1", "Redis integration not enabled")
class RedisLoaderIntegrationTests(unittest.TestCase):
    def test_create_index_on_local_redis_stack(self):
        from matchmaking_data.redis_loader import get_redis_client, verify_redis_stack

        config = PipelineConfig(index_name="idx:players:test")
        client = get_redis_client(config.redis_url)
        verify_redis_stack(client)
        create_index(client, config)
        info = client.execute_command("FT.INFO", config.index_name)
        blob = str(info)
        self.assertIn("HNSW", blob)
        self.assertIn("L2", blob)


if __name__ == "__main__":
    unittest.main()

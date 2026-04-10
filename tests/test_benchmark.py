import unittest

from matchmaking_data.benchmark import (
    BenchmarkResult,
    WRITE_BATCH_SIZE,
    WRITE_BATCHES_PER_SECOND,
    _runtime_clause,
    build_sample_command,
    build_sample_write_command,
    escape_tag_value,
    percentile,
)
from matchmaking_data.config import PipelineConfig


class BenchmarkTests(unittest.TestCase):
    def test_percentile_empty(self):
        self.assertEqual(0.0, percentile([], 0.95))

    def test_percentile_values(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(30.0, percentile(values, 0.50))
        self.assertEqual(50.0, percentile(values, 0.95))

    def test_result_shape(self):
        result = BenchmarkResult(
            requested_qps=1000,
            achieved_qps=980.5,
            duration_seconds=30.0,
            total_requests=30000,
            successful_requests=29900,
            failed_requests=100,
            p50_ms=8.1,
            p95_ms=14.2,
            p99_ms=21.7,
            min_ms=2.1,
            max_ms=44.9,
            sample_command="redis-cli FT.SEARCH idx:players ...",
            requested_write_qps=WRITE_BATCHES_PER_SECOND,
            achieved_write_qps=29.1,
            total_writes=900,
            successful_writes=900,
            failed_writes=0,
            sample_write_command="redis-cli HSET player:1 ...",
        )
        self.assertEqual(1000, result.requested_qps)
        self.assertEqual(29900, result.successful_requests)
        self.assertEqual(21.7, result.p99_ms)
        self.assertEqual(900, result.total_writes)

    def test_escape_tag_value(self):
        self.assertEqual(r"abc\+\/\=", escape_tag_value("abc+/="))

    def test_runtime_clause_for_hnsw(self):
        config = PipelineConfig(vector_algorithm="HNSW")
        self.assertEqual(" EF_RUNTIME 16", _runtime_clause(config, 16))

    def test_runtime_clause_for_vamana(self):
        config = PipelineConfig(vector_algorithm="SVS-VAMANA")
        self.assertEqual(" SEARCH_WINDOW_SIZE 16", _runtime_clause(config, 16))

    def test_build_sample_command_for_prefilter(self):
        config = PipelineConfig(index_name="idx:players")
        command = build_sample_command(
            config=config,
            query_vector=b"\x01\x02",
            k=50,
            filter_field="field1",
            filter_value="1",
            ef_runtime=64,
        )
        self.assertIn("FT.SEARCH idx:players", command)
        self.assertIn("@field1:{1}", command)
        self.assertIn("EF_RUNTIME 64", command)
        self.assertIn("LIMIT 0 50", command)

    def test_build_sample_command_without_filter(self):
        config = PipelineConfig(index_name="idx:players:v2")
        command = build_sample_command(
            config=config,
            query_vector=b"\x01\x02",
            k=50,
            filter_field="none",
            filter_value=None,
            ef_runtime=64,
        )
        self.assertIn("FT.SEARCH idx:players:v2", command)
        self.assertNotIn("@field1:{", command)
        self.assertNotIn("@field2:{", command)

    def test_build_sample_write_command(self):
        command = build_sample_write_command(
            [
                ("player:42", {b"player_id": b"42", b"field1": b"1", b"embedding": b"\x01\x02"}),
                ("player:43", {b"player_id": b"43", b"field2": b"0", b"embedding": b"\x03\x04"}),
            ]
        )
        self.assertIn("redis-cli HSET player:42", command)
        self.assertIn("player_id", command)
        self.assertIn("embedding", command)
        self.assertEqual(WRITE_BATCH_SIZE, 100)
        self.assertEqual(WRITE_BATCHES_PER_SECOND, 30)


if __name__ == "__main__":
    unittest.main()

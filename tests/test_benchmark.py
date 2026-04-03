import unittest

from matchmaking_data.benchmark import (
    BenchmarkResult,
    _runtime_clause,
    build_sample_command,
    escape_aggregate_string,
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
        )
        self.assertEqual(1000, result.requested_qps)
        self.assertEqual(29900, result.successful_requests)
        self.assertEqual(21.7, result.p99_ms)

    def test_escape_tag_value(self):
        self.assertEqual(r"abc\+\/\=", escape_tag_value("abc+/="))

    def test_escape_aggregate_string(self):
        self.assertEqual(r"a\\b\'c", escape_aggregate_string("a\\b'c"))

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
            mode="binary",
            binary_value="abc+/=",
            aggregate_limit=10000,
            ef_runtime=64,
        )
        self.assertIn("FT.SEARCH idx:players", command)
        self.assertIn(r"@binary:{abc\+\/\=}", command)
        self.assertIn("EF_RUNTIME 64", command)

    def test_build_sample_command_for_vamana_postfilter(self):
        config = PipelineConfig(index_name="idx:players:vamana", vector_algorithm="SVS-VAMANA")
        command = build_sample_command(
            config=config,
            query_vector=b"\x01\x02",
            k=50,
            mode="postfilter",
            binary_value="abc",
            aggregate_limit=10000,
            ef_runtime=64,
        )
        self.assertIn("FT.AGGREGATE idx:players:vamana", command)
        self.assertIn("SEARCH_WINDOW_SIZE 64", command)
        self.assertIn("FILTER '@binary=='", command)


if __name__ == "__main__":
    unittest.main()

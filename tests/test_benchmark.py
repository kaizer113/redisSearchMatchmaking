import unittest

from matchmaking_data.benchmark import BenchmarkResult, escape_tag_value, percentile


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
        )
        self.assertEqual(1000, result.requested_qps)
        self.assertEqual(29900, result.successful_requests)
        self.assertEqual(21.7, result.p99_ms)

    def test_escape_tag_value(self):
        self.assertEqual(r"abc\+\/\=", escape_tag_value("abc+/="))


if __name__ == "__main__":
    unittest.main()

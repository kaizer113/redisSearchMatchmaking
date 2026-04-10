import unittest

from matchmaking_data.generator import generate_canonical_profile
from matchmaking_data.threaded_loader import (
    CompletedRanges,
    build_expanded_players_for_batch,
    player_range_for_profile_batch,
)


class ThreadedLoaderTests(unittest.TestCase):
    def test_player_range_for_profile_batch_respects_resume_offset(self):
        range_start, range_end = player_range_for_profile_batch(
            batch_start_profile_id=10,
            profile_count=3,
            duplication_factor_max=10,
            effective_start_player_id=107,
            total_players=1_000,
        )
        self.assertEqual(107, range_start)
        self.assertEqual(130, range_end)

    def test_completed_ranges_advances_only_when_contiguous(self):
        completed = CompletedRanges(initial_offset=100)
        self.assertEqual(100, completed.mark_completed(120, 130))
        self.assertEqual(130, completed.mark_completed(100, 120))
        self.assertEqual(130, completed.next_offset)

    def test_build_expanded_players_for_batch_honors_limits(self):
        profiles = [generate_canonical_profile(3, seed=1337), generate_canonical_profile(4, seed=1337)]
        for profile in profiles:
            profile["embedding"] = [0.1, 0.2]

        players = build_expanded_players_for_batch(
            embedded_profiles=profiles,
            batch_start_profile_id=3,
            duplication_factor_max=10,
            effective_start_player_id=34,
            total_players=38,
        )
        self.assertEqual([34, 35, 36, 37], [player["player_id"] for player in players])
        self.assertTrue(all("field1" in player and "field2" in player for player in players))


if __name__ == "__main__":
    unittest.main()

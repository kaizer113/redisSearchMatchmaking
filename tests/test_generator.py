import unittest

from matchmaking_data.generator import (
    GAMES,
    LAST_LOGIN_END_EPOCH,
    LAST_LOGIN_START_EPOCH,
    PLATFORMS,
    expand_profile,
    generate_canonical_profile,
    iter_expanded_players,
)


class GeneratorTests(unittest.TestCase):
    def test_generate_canonical_profile_is_deterministic(self):
        left = generate_canonical_profile(42, seed=1337)
        right = generate_canonical_profile(42, seed=1337)
        self.assertEqual(left, right)

    def test_generate_canonical_profile_has_expected_fields(self):
        profile = generate_canonical_profile(1, seed=1337)
        self.assertIn("profile_text", profile)
        self.assertIsNone(profile["embedding"])
        self.assertIsInstance(profile["last_login"], int)
        self.assertIn(profile["field1"], (0, 1))
        self.assertIn(profile["field2"], (0, 1))
        self.assertGreaterEqual(profile["last_login"], LAST_LOGIN_START_EPOCH)
        self.assertLessEqual(profile["last_login"], LAST_LOGIN_END_EPOCH)

    def test_expand_profile_keeps_only_v2_runtime_fields(self):
        profile = generate_canonical_profile(7, seed=99)
        profile["embedding"] = [0.1, 0.2, 0.3]
        expanded = expand_profile(profile, variant_index=3)
        self.assertEqual({"last_login", "field1", "field2", "embedding"}, set(expanded.keys()))
        self.assertEqual(profile["embedding"], expanded["embedding"])
        self.assertIn(expanded["field1"], (0, 1))
        self.assertIn(expanded["field2"], (0, 1))

    def test_iter_expanded_players_respects_limit(self):
        canonical_profiles = [generate_canonical_profile(i, seed=10) for i in range(3)]
        players = list(
            iter_expanded_players(
                canonical_profiles=canonical_profiles,
                start_player_id=100,
                total_players=5,
                duplication_factor_max=10,
            )
        )
        self.assertEqual(5, len(players))
        self.assertEqual(100, players[0]["player_id"])
        self.assertEqual(104, players[-1]["player_id"])

    def test_games_and_platforms_are_valid(self):
        profile = generate_canonical_profile(17, seed=1337)
        self.assertIn(profile["game"], GAMES)
        self.assertIn(profile["platform"], PLATFORMS)


if __name__ == "__main__":
    unittest.main()

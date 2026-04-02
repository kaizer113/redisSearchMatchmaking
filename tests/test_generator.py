import unittest

from matchmaking_data.generator import (
    BINARY_BUCKET_COUNT,
    GAMES,
    binary_value_for_bucket,
    binary_value_for_profile,
    build_username,
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
        self.assertIn("game", profile)
        self.assertIn("platform", profile)
        self.assertIn("rank_tier", profile)
        self.assertIn("profile_text", profile)
        self.assertIsNone(profile["embedding"])
        self.assertIsInstance(profile["availability"], str)
        self.assertIsInstance(profile["language"], str)
        self.assertIsInstance(profile["role"], str)
        self.assertIsInstance(profile["binary"], str)
        self.assertEqual(28, len(profile["binary"]))
        self.assertNotIn("canonical_profile_id", profile)
        self.assertNotIn("recent_progress", profile)
        self.assertNotIn("party_preferences", profile)
        self.assertNotIn("skill_tag", profile)

    def test_expand_profile_changes_identity_fields_only(self):
        profile = generate_canonical_profile(7, seed=99)
        profile["embedding"] = [0.1, 0.2, 0.3]
        expanded = expand_profile(profile, variant_index=3)
        self.assertNotEqual(expanded["username"], build_username(profile, 0))
        self.assertEqual(profile["embedding"], expanded["embedding"])
        self.assertEqual(profile["game"], expanded["game"])
        self.assertNotIn("profile_text", expanded)
        self.assertNotIn("variant_index", expanded)
        self.assertNotIn("avatar_seed", expanded)
        self.assertNotIn("account_age_days", expanded)

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

    def test_binary_cycles_over_256_buckets(self):
        self.assertEqual(binary_value_for_profile(0), binary_value_for_profile(BINARY_BUCKET_COUNT))
        self.assertNotEqual(binary_value_for_profile(0), binary_value_for_profile(1))
        self.assertEqual(binary_value_for_bucket(12), binary_value_for_profile(12))

    def test_games_and_platforms_are_valid(self):
        profile = generate_canonical_profile(17, seed=1337)
        self.assertIn(profile["game"], GAMES)
        self.assertIn(profile["platform"], GAMES[profile["game"]]["platforms"])


if __name__ == "__main__":
    unittest.main()

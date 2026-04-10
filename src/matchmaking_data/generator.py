import copy
import hashlib
import random
from typing import Dict, Iterable, Iterator, List


GAMES = [
    "apex_legends",
    "battlefield_2042",
    "college_football_25",
    "fc_25",
    "f1_24",
    "madden_nfl_25",
    "need_for_speed_unbound",
    "nhl_25",
]

PLATFORMS = ["pc", "playstation", "xbox", "switch"]
REGIONS = ["na", "eu", "apac", "latam"]
PLAY_STYLES = ["aggressive", "balanced", "defensive", "strategic", "supportive"]
SESSION_WINDOWS = ["late_night", "afternoon", "weekday_evening", "weekend", "early_morning"]
SKILL_ARCHETYPES = ["clutch", "anchor", "entry", "controller", "support", "igl"]

LAST_LOGIN_START_EPOCH = 1_704_067_200  # 2024-01-01T00:00:00Z
LAST_LOGIN_END_EPOCH = 1_775_174_400  # 2026-04-01T00:00:00Z


def _rng(seed: int, entity_id: int) -> random.Random:
    return random.Random(f"{seed}:{entity_id}")


def _stable_token(text: str, length: int = 8) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return digest[:length]


def _profile_text(profile: Dict[str, object]) -> str:
    return (
        f"Player {_stable_token(str(profile['canonical_profile_id']))} prefers {profile['game']} on "
        f"{profile['platform']} in {profile['region']}. "
        f"Style {profile['play_style']}, session window {profile['session_window']}, "
        f"skill focus {profile['skill_archetype']}, squad size {profile['preferred_party_size']}. "
        f"Rank score {profile['rank_score']}, voice chat {profile['voice_chat']}, "
        f"controller {profile['controller_input']}, progression lane {profile['progression_lane']}."
    )


def generate_canonical_profile(profile_id: int, seed: int) -> Dict[str, object]:
    rng = _rng(seed, profile_id)
    profile = {
        "canonical_profile_id": profile_id,
        "game": rng.choice(GAMES),
        "platform": rng.choice(PLATFORMS),
        "region": rng.choice(REGIONS),
        "play_style": rng.choice(PLAY_STYLES),
        "session_window": rng.choice(SESSION_WINDOWS),
        "skill_archetype": rng.choice(SKILL_ARCHETYPES),
        "preferred_party_size": rng.randint(1, 5),
        "rank_score": rng.randint(0, 5000),
        "voice_chat": rng.choice(["enabled", "disabled"]),
        "controller_input": rng.choice(["controller", "mouse_keyboard"]),
        "progression_lane": rng.choice(["ranked", "casual", "co_op"]),
        "last_login": rng.randint(LAST_LOGIN_START_EPOCH, LAST_LOGIN_END_EPOCH),
        "field1": rng.randint(0, 1),
        "field2": rng.randint(0, 1),
        "embedding": None,
    }
    profile["profile_text"] = _profile_text(profile)
    return profile


def generate_canonical_profiles(start_id: int, count: int, seed: int) -> Iterator[Dict[str, object]]:
    for profile_id in range(start_id, start_id + count):
        yield generate_canonical_profile(profile_id, seed)


def expand_profile(profile: Dict[str, object], variant_index: int) -> Dict[str, object]:
    expanded = copy.deepcopy(profile)
    expanded["last_login"] = min(LAST_LOGIN_END_EPOCH, int(profile["last_login"]) + (variant_index * 86_400))
    expanded["field1"] = (int(profile["field1"]) + variant_index) % 2
    expanded["field2"] = (int(profile["field2"]) + (variant_index // 2)) % 2
    for transient_key in (
        "profile_text",
        "canonical_profile_id",
        "game",
        "platform",
        "region",
        "play_style",
        "session_window",
        "skill_archetype",
        "preferred_party_size",
        "rank_score",
        "voice_chat",
        "controller_input",
        "progression_lane",
    ):
        expanded.pop(transient_key, None)
    return expanded


def iter_expanded_players(
    canonical_profiles: Iterable[Dict[str, object]],
    start_player_id: int,
    total_players: int,
    duplication_factor_max: int,
    start_variant_offset: int = 0,
) -> Iterator[Dict[str, object]]:
    player_id = start_player_id
    produced = 0
    first_profile = True
    for profile in canonical_profiles:
        variant_start = start_variant_offset if first_profile else 0
        for variant_index in range(variant_start, duplication_factor_max):
            if produced >= total_players:
                return
            player = expand_profile(profile, variant_index)
            player["player_id"] = player_id
            yield player
            player_id += 1
            produced += 1
        first_profile = False


def chunked(iterable: Iterable[Dict[str, object]], chunk_size: int) -> Iterator[List[Dict[str, object]]]:
    batch: List[Dict[str, object]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

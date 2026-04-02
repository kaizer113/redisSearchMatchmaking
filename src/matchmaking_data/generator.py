import base64
import copy
import hashlib
import random
from typing import Dict, Iterable, Iterator, List


GAMES = {
    "apex_legends": {
        "platforms": ["pc", "playstation", "xbox", "switch"],
        "roles": ["fragger", "skirmisher", "support", "igl"],
        "skills": ["movement", "tracking", "positioning", "teamplay", "survivability"],
    },
    "battlefield_2042": {
        "platforms": ["pc", "playstation", "xbox"],
        "roles": ["assault", "engineer", "recon", "support"],
        "skills": ["vehicle control", "squad play", "objective control", "long range aim", "flanking"],
    },
    "college_football_25": {
        "platforms": ["playstation", "xbox"],
        "roles": ["quarterback", "running_back", "wide_receiver", "linebacker"],
        "skills": ["play calling", "route timing", "coverage reads", "ball security", "clock management"],
    },
    "fc_25": {
        "platforms": ["pc", "playstation", "xbox", "switch"],
        "roles": ["striker", "midfielder", "defender", "goalkeeper"],
        "skills": ["passing", "finishing", "defending", "build-up play", "set pieces"],
    },
    "f1_24": {
        "platforms": ["pc", "playstation", "xbox"],
        "roles": ["qualifying_specialist", "race_pace_driver", "wet_weather_driver", "strategy_driver"],
        "skills": ["tire management", "consistency", "cornering", "overtaking", "race starts"],
    },
    "it_takes_two": {
        "platforms": ["pc", "playstation", "xbox", "switch"],
        "roles": ["cody_main", "may_main", "puzzle_solver", "platforming_specialist"],
        "skills": ["co_op_timing", "puzzle solving", "communication", "platforming", "boss execution"],
    },
    "madden_nfl_25": {
        "platforms": ["pc", "playstation", "xbox"],
        "roles": ["quarterback", "running_back", "wide_receiver", "cornerback"],
        "skills": ["audibles", "coverage reads", "user defense", "play action", "red zone execution"],
    },
    "need_for_speed_unbound": {
        "platforms": ["pc", "playstation", "xbox"],
        "roles": ["grip_driver", "drift_driver", "sprinter", "cop_escape_specialist"],
        "skills": ["corner exits", "nitrous timing", "drifting", "traffic weaving", "route knowledge"],
    },
    "nhl_25": {
        "platforms": ["playstation", "xbox"],
        "roles": ["center", "wing", "defenseman", "goalie"],
        "skills": ["puck control", "defensive positioning", "one timers", "forechecking", "crease play"],
    },
}

REGIONS = {
    "na": ["US", "CA", "MX"],
    "eu": ["DE", "FR", "GB", "ES", "IT", "SE"],
    "apac": ["JP", "KR", "SG", "AU", "PH"],
    "latam": ["BR", "AR", "CL", "CO"],
}

LANGUAGES = {
    "US": ["english", "spanish"],
    "CA": ["english", "french"],
    "MX": ["spanish", "english"],
    "DE": ["german", "english"],
    "FR": ["french", "english"],
    "GB": ["english"],
    "ES": ["spanish", "english"],
    "IT": ["italian", "english"],
    "SE": ["swedish", "english"],
    "JP": ["japanese", "english"],
    "KR": ["korean", "english"],
    "SG": ["english", "mandarin"],
    "AU": ["english"],
    "PH": ["english", "tagalog"],
    "BR": ["portuguese", "english"],
    "AR": ["spanish", "english"],
    "CL": ["spanish", "english"],
    "CO": ["spanish", "english"],
}

RANKS = [
    "bronze",
    "silver",
    "gold",
    "platinum",
    "diamond",
    "master",
    "grandmaster",
]

PLAY_STYLES = ["aggressive", "balanced", "defensive", "strategic", "supportive"]

AVAILABILITY_WINDOWS = [
    "weekday_evenings",
    "late_nights",
    "weekends",
    "early_mornings",
    "afternoons",
]

NAME_PREFIXES = [
    "Nova",
    "Ghost",
    "Pixel",
    "Drift",
    "Echo",
    "Cipher",
    "Aero",
    "Zen",
    "Pulse",
    "Blitz",
]

NAME_SUFFIXES = [
    "Wolf",
    "Spark",
    "Storm",
    "Blade",
    "Fox",
    "Shade",
    "Rider",
    "Knight",
    "Viper",
    "Byte",
]

BINARY_BUCKET_COUNT = 256


def _rng(seed: int, entity_id: int) -> random.Random:
    return random.Random(f"{seed}:{entity_id}")


def _stable_token(text: str, length: int = 6) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return digest[:length]


def _stable_binary_blob(text: str) -> str:
    raw = hashlib.sha1(text.encode("utf-8")).digest()[:20]
    return base64.b64encode(raw).decode("ascii")


def binary_value_for_bucket(bucket: int) -> str:
    normalized_bucket = bucket % BINARY_BUCKET_COUNT
    return _stable_binary_blob(f"binary-bucket:{normalized_bucket}")


def binary_value_for_profile(profile_id: int) -> str:
    return binary_value_for_bucket(profile_id)


def _build_profile_text(profile: Dict[str, object]) -> str:
    return (
        f"Player profile for {profile['game']} on {profile['platform']} in {profile['region']}. "
        f"Country {profile['country']}. Rank {profile['rank_tier']} with score {profile['rank_score']}. "
        f"Primary role: {profile['role']}. "
        f"Play style: {profile['play_style']}. "
        f"Availability: {profile['availability']}. "
        f"Language: {profile['language']}. "
        f"Hours played: {profile['hours_played']}."
    )


def generate_canonical_profile(profile_id: int, seed: int) -> Dict[str, object]:
    rng = _rng(seed, profile_id)
    game = rng.choice(sorted(GAMES.keys()))
    game_info = GAMES[game]
    platform = rng.choice(game_info["platforms"])
    region = rng.choice(sorted(REGIONS.keys()))
    country = rng.choice(REGIONS[region])
    languages = LANGUAGES[country][:]
    rng.shuffle(languages)
    rank_tier = rng.choices(RANKS, weights=[18, 20, 22, 18, 12, 7, 3], k=1)[0]
    rank_score = int(rng.uniform(0, 1000))
    role = rng.choice(game_info["roles"])
    profile = {
        "game": game,
        "platform": platform,
        "region": region,
        "country": country,
        "language": languages[0],
        "rank_tier": rank_tier,
        "rank_score": rank_score,
        "play_style": rng.choice(PLAY_STYLES),
        "role": role,
        "hours_played": rng.randint(10, 5000),
        "availability": rng.choice(AVAILABILITY_WINDOWS),
        "binary": binary_value_for_profile(profile_id),
        "embedding": None,
    }
    profile["profile_text"] = _build_profile_text(profile)
    return profile


def generate_canonical_profiles(start_id: int, count: int, seed: int) -> Iterator[Dict[str, object]]:
    for profile_id in range(start_id, start_id + count):
        yield generate_canonical_profile(profile_id, seed)


def build_username(profile: Dict[str, object], variant_index: int) -> str:
    token = _stable_token(
        f"{profile['game']}:{profile['rank_tier']}:{profile['region']}:{variant_index}",
        length=5,
    )
    name_basis = f"{profile['game']}:{profile['platform']}:{profile['country']}"
    basis_int = int(hashlib.sha1(name_basis.encode("utf-8")).hexdigest()[:8], 16)
    prefix = NAME_PREFIXES[(basis_int + variant_index) % len(NAME_PREFIXES)]
    suffix = NAME_SUFFIXES[(basis_int * 3 + variant_index) % len(NAME_SUFFIXES)]
    return f"{prefix}{suffix}{token}"


def expand_profile(profile: Dict[str, object], variant_index: int) -> Dict[str, object]:
    expanded = copy.deepcopy(profile)
    expanded["username"] = build_username(profile, variant_index)
    expanded.pop("profile_text", None)
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

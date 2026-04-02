#!/Applications/Xcode.app/Contents/Developer/usr/bin/python3
import argparse
import os
import random
import statistics
import time

from redis import Redis


def escape_tag_value(value: str) -> str:
    special = set('{}[]()|-=><~"\'@:;,./+*&!$%^\\ ')
    escaped = []
    for char in value:
        if char in special:
            escaped.append("\\" + char)
        else:
            escaped.append(char)
    return "".join(escaped)


def escape_aggregate_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def run_search_query(client: Redis, index_name: str, query: str, vector: bytes, limit: int):
    started = time.perf_counter()
    result = client.execute_command(
        "FT.SEARCH",
        index_name,
        query,
        "PARAMS",
        "2",
        "vector",
        vector,
        "SORTBY",
        "score",
        "ASC",
        "NOCONTENT",
        "LIMIT",
        "0",
        str(limit),
        "DIALECT",
        "2",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    matches = int(result[0]) if result else 0
    ids = []
    if result:
        for item in result[1:]:
            if isinstance(item, bytes):
                ids.append(item.decode("utf-8"))
            else:
                ids.append(str(item))
    return matches, elapsed_ms, ids


def run_aggregate_query(
    client: Redis,
    index_name: str,
    query: str,
    vector: bytes,
    limit: int,
    binary_value: str,
):
    filter_expr = "@binary=='{}'".format(escape_aggregate_string(binary_value))
    started = time.perf_counter()
    result = client.execute_command(
        "FT.AGGREGATE",
        index_name,
        query,
        "PARAMS",
        "2",
        "vector",
        vector,
        "LOAD",
        "3",
        "__key",
        "@binary",
        "@score",
        "FILTER",
        filter_expr,
        "SORTBY",
        "2",
        "@score",
        "ASC",
        "LIMIT",
        "0",
        str(limit),
        "DIALECT",
        "2",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    matches = int(result[0]) if result else 0
    ids = []
    rows = []
    if result:
        for row in result[1:]:
            if isinstance(row, list) and row:
                parsed = {}
                for index in range(0, len(row), 2):
                    key = row[index]
                    value = row[index + 1] if index + 1 < len(row) else None
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    else:
                        key = str(key)
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    elif value is not None:
                        value = str(value)
                    parsed[key] = value
                rows.append(parsed)
                raw_id = parsed.get("__key")
                if raw_id:
                    ids.append(raw_id)
    return matches, elapsed_ms, ids, filter_expr, rows


def overlap(reference_ids, candidate_ids):
    if not reference_ids:
        return 0, 0.0
    reference_set = set(reference_ids)
    candidate_set = set(candidate_ids)
    shared = len(reference_set & candidate_set)
    recall = shared / float(len(reference_set))
    return shared, recall


def summarize(values):
    if not values:
        return {"avg": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p50 = ordered[int(round((len(ordered) - 1) * 0.50))]
    p95 = ordered[int(round((len(ordered) - 1) * 0.95))]
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "p50": p50,
        "p95": p95,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare unfiltered and binary-filtered vector queries")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--index-name", default="idx:players")
    parser.add_argument("--player-id", type=int)
    parser.add_argument("--max-player-id", type=int, default=10_000_000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--aggregate-limit", type=int, default=None)
    args = parser.parse_args()

    randomizer = random.Random(args.seed)
    client = Redis.from_url(args.redis_url, decode_responses=False)
    ef_values = [128, 64, 32, 16]
    aggregate = {
        "unfiltered": {ef: {"times": [], "recalls": []} for ef in ef_values},
        "filtered_pre": {ef: {"times": [], "recalls": []} for ef in ef_values},
        "filtered_post": {ef: {"times": [], "recalls": []} for ef in ef_values},
    }

    sample_player_ids = []
    if args.player_id is not None:
        sample_player_ids = [args.player_id]
    else:
        while len(sample_player_ids) < args.samples:
            sample_player_ids.append(randomizer.randrange(0, args.max_player_id))

    for sample_index, player_id in enumerate(sample_player_ids, start=1):
        key = f"player:{player_id}"
        vector = client.hget(key, "embedding")
        binary = client.hget(key, "binary")

        if vector is None:
            raise SystemExit(f"Missing embedding for {key}")
        if binary is None:
            raise SystemExit(f"Missing binary for {key}")

        binary_text = binary.decode("utf-8")
        escaped_binary = escape_tag_value(binary_text)

        print(f"sample={sample_index} player_id={player_id} binary={binary_text}")
        reference_unfiltered_ids = None
        reference_filtered_pre_ids = None
        reference_filtered_post_ids = None

        for ef_runtime in ef_values:
            limit = args.k
            aggregate_limit = args.aggregate_limit or args.k
            unfiltered_query = (
                f"*=>[KNN {args.k} @embedding $vector EF_RUNTIME {ef_runtime} AS score]"
            )
            filtered_pre_query = (
                f"@binary:{{{escaped_binary}}}"
                f"=>[KNN {args.k} @embedding $vector EF_RUNTIME {ef_runtime} AS score]"
            )
            filtered_post_query = (
                f"*=>[KNN {aggregate_limit} @embedding $vector EF_RUNTIME {ef_runtime} AS score]"
            )

            matches, elapsed_ms, ids = run_search_query(
                client, args.index_name, unfiltered_query, vector, limit
            )
            aggregate["unfiltered"][ef_runtime]["times"].append(elapsed_ms)
            if reference_unfiltered_ids is None:
                reference_unfiltered_ids = ids
                recall = 1.0
            else:
                _, recall = overlap(reference_unfiltered_ids, ids)
            aggregate["unfiltered"][ef_runtime]["recalls"].append(recall)
            print(
                f"  unfiltered ef={ef_runtime} matches={matches} time_ms={elapsed_ms:.2f} recall_vs_128={recall:.2%}"
            )
            print(
                f"    FT.SEARCH {args.index_name} \"{unfiltered_query}\" "
                f"PARAMS 2 vector <256-byte-vector> SORTBY score ASC NOCONTENT LIMIT 0 {limit} DIALECT 2"
            )

            matches, elapsed_ms, ids = run_search_query(
                client, args.index_name, filtered_pre_query, vector, limit
            )
            aggregate["filtered_pre"][ef_runtime]["times"].append(elapsed_ms)
            if reference_filtered_pre_ids is None:
                reference_filtered_pre_ids = ids
                recall = 1.0
            else:
                _, recall = overlap(reference_filtered_pre_ids, ids)
            aggregate["filtered_pre"][ef_runtime]["recalls"].append(recall)
            print(
                f"  filtered_pre  ef={ef_runtime} matches={matches} time_ms={elapsed_ms:.2f} recall_vs_128={recall:.2%}"
            )
            print(
                f"    FT.SEARCH {args.index_name} \"{filtered_pre_query}\" "
                f"PARAMS 2 vector <256-byte-vector> SORTBY score ASC NOCONTENT LIMIT 0 {limit} DIALECT 2"
            )

            matches, elapsed_ms, ids, filter_expr, rows = run_aggregate_query(
                client,
                args.index_name,
                filtered_post_query,
                vector,
                limit,
                binary_text,
            )
            aggregate["filtered_post"][ef_runtime]["times"].append(elapsed_ms)
            if reference_filtered_post_ids is None:
                reference_filtered_post_ids = ids
                recall = 1.0
            else:
                _, recall = overlap(reference_filtered_post_ids, ids)
            aggregate["filtered_post"][ef_runtime]["recalls"].append(recall)
            print(
                f"  filtered_post ef={ef_runtime} matches={matches} time_ms={elapsed_ms:.2f} recall_vs_128={recall:.2%}"
            )
            print(
                f"    FT.AGGREGATE {args.index_name} \"{filtered_post_query}\" "
                f"PARAMS 2 vector <256-byte-vector> LOAD 3 __key @binary @score FILTER \"{filter_expr}\" "
                f"SORTBY 2 @score ASC "
                f"LIMIT 0 {limit} DIALECT 2"
            )
            if rows:
                preview = rows[: min(3, len(rows))]
                print(f"    aggregate_preview={preview}")
        print()

    print("summary")
    for mode in ["unfiltered", "filtered_pre", "filtered_post"]:
        print(mode)
        for ef_runtime in ef_values:
            time_summary = summarize(aggregate[mode][ef_runtime]["times"])
            recall_summary = summarize(aggregate[mode][ef_runtime]["recalls"])
            print(
                f"  ef={ef_runtime} "
                f"time_avg_ms={time_summary['avg']:.2f} "
                f"time_p50_ms={time_summary['p50']:.2f} "
                f"time_p95_ms={time_summary['p95']:.2f} "
                f"recall_avg={recall_summary['avg']:.2%} "
                f"recall_min={recall_summary['min']:.2%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

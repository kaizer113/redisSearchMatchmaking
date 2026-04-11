#!/Applications/Xcode.app/Contents/Developer/usr/bin/python3
import argparse
import os
from pathlib import Path
from pprint import pformat

from redis import Redis
from redis.exceptions import ResponseError


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


def build_queries(
    index_name: str,
    filters: list[tuple[str, str]],
    k: int,
    ef_runtime: int,
):
    filter_prefix = " ".join(
        f"@{field}:{{{escape_tag_value(value)}}}" for field, value in filters
    )
    if len(filters) > 1:
        filter_prefix = f"({filter_prefix})"
    filtered_query = f"{filter_prefix}=>[KNN {k} @embedding $vector EF_RUNTIME {ef_runtime} AS score]"
    return {
        "unfiltered": {
            "type": "SEARCH",
            "query": f"*=>[KNN {k} @embedding $vector EF_RUNTIME {ef_runtime} AS score]",
            "command_tail": [
                "PARAMS", "2", "vector", "<256-byte-vector>",
                "SORTBY", "score", "ASC",
                "NOCONTENT",
                "LIMIT", "0", str(k),
                "DIALECT", "2",
            ],
        },
        "prefilter": {
            "type": "SEARCH",
            "query": filtered_query,
            "command_tail": [
                "PARAMS", "2", "vector", "<256-byte-vector>",
                "SORTBY", "score", "ASC",
                "NOCONTENT",
                "LIMIT", "0", str(k),
                "DIALECT", "2",
            ],
        },
    }


def command_string(index_name: str, query_type: str, query: str, command_tail: list) -> str:
    command = [f"FT.{query_type}", index_name, f'"{query}"']
    command.extend(command_tail)
    return " ".join(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture FT.PROFILE and FT.EXPLAIN outputs")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--index-name", default="idx:players")
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--ef-runtime", type=int, default=64)
    parser.add_argument("--field1-value", choices=["0", "1"], default=None)
    parser.add_argument("--field2-value", choices=["0", "1"], default=None)
    parser.add_argument(
        "--output",
        default="artifacts/ft_profile_explain_ef64.txt",
        help="Path to the output text file",
    )
    args = parser.parse_args()

    client = Redis.from_url(args.redis_url, decode_responses=False)
    key = f"player:{args.player_id}"
    vector = client.hget(key, "embedding")
    raw_field1 = client.hget(key, "field1")
    raw_field2 = client.hget(key, "field2")

    if vector is None:
        raise SystemExit(f"Missing embedding for {key}")
    if raw_field1 is None and args.field1_value is None:
        raise SystemExit(f"Missing field1 for {key}")
    if raw_field2 is None and args.field2_value is None:
        raise SystemExit(f"Missing field2 for {key}")

    explain_vector = client.hget(key, "embedding")
    if explain_vector is None:
        raise SystemExit(f"Missing embedding for {key} (used for FT.EXPLAIN PARAMS)")

    field1_value = args.field1_value if args.field1_value is not None else raw_field1.decode("utf-8")
    field2_value = args.field2_value if args.field2_value is not None else raw_field2.decode("utf-8")
    filters = [("field1", field1_value), ("field2", field2_value)]
    queries = build_queries(
        index_name=args.index_name,
        filters=filters,
        k=args.k,
        ef_runtime=args.ef_runtime,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("Redis query capture")
    lines.append("===================")
    lines.append(f"redis_url={args.redis_url}")
    lines.append(f"index_name={args.index_name}")
    lines.append(f"player_id={args.player_id}")
    lines.append(f"field1={field1_value}")
    lines.append(f"field2={field2_value}")
    lines.append(f"k={args.k}")
    lines.append(f"ef_runtime={args.ef_runtime}")
    lines.append("")

    for label, query_info in queries.items():
        query_type = query_info["type"]
        query = query_info["query"]
        lines.append(label)
        lines.append("-" * len(label))
        lines.append("command")
        lines.append(command_string(args.index_name, query_type, query, query_info["command_tail"]))
        lines.append("")

        if query_type == "SEARCH":
            try:
                explain = client.execute_command("FT.EXPLAIN", args.index_name, query, "PARAMS", "2", "vector", explain_vector, "DIALECT", "2")
            except ResponseError as exc:
                explain = f"FT.EXPLAIN error:\n{exc}"
            profile = client.execute_command(
                "FT.PROFILE",
                args.index_name,
                "SEARCH",
                "QUERY",
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
                str(args.k),
                "DIALECT",
                "2",
            )

        if isinstance(explain, bytes):
            explain = explain.decode("utf-8", errors="replace")

        lines.append("ft.explain")
        lines.append(explain)
        lines.append("")
        lines.append("ft.profile")
        lines.append(pformat(profile, width=120, sort_dicts=False))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def build_queries(index_name: str, binary_text: str, k: int, aggregate_limit: int, ef_runtime: int):
    escaped_binary = escape_tag_value(binary_text)
    filter_expr = "@binary=='{}'".format(escape_aggregate_string(binary_text))
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
            "query": f"@binary:{{{escaped_binary}}}=>[KNN {k} @embedding $vector EF_RUNTIME {ef_runtime} AS score]",
            "command_tail": [
                "PARAMS", "2", "vector", "<256-byte-vector>",
                "SORTBY", "score", "ASC",
                "NOCONTENT",
                "LIMIT", "0", str(k),
                "DIALECT", "2",
            ],
        },
        "postfilter": {
            "type": "AGGREGATE",
            "query": f"*=>[KNN {aggregate_limit} @embedding $vector EF_RUNTIME {ef_runtime} AS score]",
            "command_tail": [
                "PARAMS", "2", "vector", "<256-byte-vector>",
                "LOAD", "3", "__key", "@binary", "@score",
                "FILTER", filter_expr,
                "SORTBY", "2", "@score", "ASC",
                "LIMIT", "0", str(k),
                "DIALECT", "2",
            ],
            "filter_expr": filter_expr,
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
    parser.add_argument("--aggregate-limit", type=int, default=10000)
    parser.add_argument("--ef-runtime", type=int, default=64)
    parser.add_argument(
        "--output",
        default="artifacts/ft_profile_explain_ef64.txt",
        help="Path to the output text file",
    )
    args = parser.parse_args()

    client = Redis.from_url(args.redis_url, decode_responses=False)
    key = f"player:{args.player_id}"
    vector = client.hget(key, "embedding")
    binary = client.hget(key, "binary")

    if vector is None:
        raise SystemExit(f"Missing embedding for {key}")
    if binary is None:
        raise SystemExit(f"Missing binary for {key}")

    explain_vector = client.hget("player:1000437", "embedding")
    if explain_vector is None:
        raise SystemExit("Missing embedding for player:1000437 (used for FT.EXPLAIN PARAMS)")

    binary_text = binary.decode("utf-8")
    queries = build_queries(
        index_name=args.index_name,
        binary_text=binary_text,
        k=args.k,
        aggregate_limit=args.aggregate_limit,
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
    lines.append(f"binary={binary_text}")
    lines.append(f"k={args.k}")
    lines.append(f"aggregate_limit={args.aggregate_limit}")
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
        else:
            try:
                aggregate_explain = client.execute_command("FT.EXPLAIN", args.index_name, query, "PARAMS", "2", "vector", explain_vector, "DIALECT", "2")
                if isinstance(aggregate_explain, bytes):
                    aggregate_explain = aggregate_explain.decode("utf-8", errors="replace")
                explain = (
                    "FT.EXPLAIN only describes the KNN query string. "
                    "For AGGREGATE, the full pipeline is represented in FT.PROFILE below.\n\n"
                    + aggregate_explain
                )
            except ResponseError as exc:
                explain = f"FT.EXPLAIN error:\n{exc}"
            profile = client.execute_command(
                "FT.PROFILE",
                args.index_name,
                "AGGREGATE",
                "QUERY",
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
                query_info["filter_expr"],
                "SORTBY",
                "2",
                "@score",
                "ASC",
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

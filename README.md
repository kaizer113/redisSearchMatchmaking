# Vector Search For Matchmaking

Synthetic player dataset generator, Redis loader, and benchmark toolkit for large-scale matchmaking experiments.

## Description

This project creates a synthetic player dataset at large scale, stores it in Redis as hashes, and benchmarks Redis vector search workloads.

It supports:
- deterministic dataset generation
- 64-dimensional `FLOAT32` embeddings
- up to 10 duplicate players per vector by varying only username
- low-cardinality binary-tag filters
- resumable long-running loads
- vector benchmark scenarios with and without filters

## Layout

- `src/matchmaking_data/generator.py`: canonical profile generation and profile expansion
- `src/matchmaking_data/embedder.py`: local Hugging Face embedding backend and helpers
- `src/matchmaking_data/redis_loader.py`: Redis JSON load, index creation, and query helpers
- `src/matchmaking_data/cli.py`: command-line entrypoint
- `scripts/compare_vector_queries.py`: ad hoc query timing and recall comparison script
- `tests/`: unit and optional integration tests

## Ubuntu Setup

Requirements:
- Ubuntu 22.04 or newer
- Python 3.9+
- A Redis Cloud database with vector search enabled

Example Redis endpoint:

```text
redis-15027.internal.c58799.us-west-2-mz.ec2.cloud.rlrcp.com:15027
```

Set the connection string before running the commands below:

```bash
export REDIS_URL=redis://redis-15027.internal.c58799.us-west-2-mz.ec2.cloud.rlrcp.com:15027
```

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip git
```

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run commands from the repository root.

## Create Dataset

Create the minimal index used by the current loader and benchmarks:

```bash
python3 -m matchmaking_data.cli create-index
```

Load a small sample:

```bash
python3 -m matchmaking_data.cli load \
  --total-players 10000 \
  --canonical-profile-count 1000 \
  --duplication-factor-max 10 \
  --batch-size 200
```

Run the full load:

```bash
python3 -m matchmaking_data.cli load \
  --total-players 10000000 \
  --canonical-profile-count 1000000 \
  --duplication-factor-max 10 \
  --batch-size 1000 \
  --start-player-id 0
```

Resume an interrupted load:

```bash
python3 -m matchmaking_data.cli load \
  --total-players 10000000 \
  --canonical-profile-count 1000000 \
  --duplication-factor-max 10 \
  --batch-size 1000 \
  --start-player-id 0 \
  --resume
```

## Run Benchmarks

Search-only benchmark without a filter:

```bash
python3 -m matchmaking_data.cli benchmark \
  --qps 1000 \
  --duration-seconds 5 \
  --concurrency 128 \
  --max-player-id 10000000 \
  --k 50 \
  --query-pool-size 20 \
  --prefilter-field none
```

Search-only benchmark with the binary filter:

```bash
python3 -m matchmaking_data.cli benchmark \
  --qps 1000 \
  --duration-seconds 5 \
  --concurrency 128 \
  --max-player-id 10000000 \
  --k 50 \
  --query-pool-size 20 \
  --prefilter-field binary
```

Compare one query across multiple `EF_RUNTIME` values:

```bash
python3 scripts/compare_vector_queries.py --samples 10
```

## Notes

- storage uses Redis hashes, not RedisJSON
- the current minimal index contains `binary` and `embedding`
- the default embedding model is `nomic-ai/nomic-embed-text-v1.5`
- embeddings are truncated to 64 dimensions and stored as `float32`
- vector distance metric is `L2`
- dataset load progress is checkpointed in Redis so long runs can resume

## Testing

Run unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Run Redis integration tests when Redis Stack is available:

```bash
RUN_REDIS_INTEGRATION=1 python3 -m unittest discover -s tests -v
```

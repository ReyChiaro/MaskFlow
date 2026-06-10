import argparse
import json
from pathlib import Path


def load_records(data_file: Path) -> list[dict]:
    suffix = data_file.suffix.lower()
    if suffix == ".jsonl":
        with data_file.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if suffix == ".json":
        with data_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {data_file}, got {type(data).__name__}.")
        return data

    raise ValueError(f"Unsupported data format: {suffix}. Only .jsonl and .json are supported.")


def split_records(records: list[dict], num_shards: int) -> list[list[dict]]:
    total = len(records)
    base_size, remainder = divmod(total, num_shards)
    shards = []
    start = 0

    for rank in range(num_shards):
        shard_size = base_size + (1 if rank < remainder else 0)
        end = start + shard_size
        shards.append(records[start:end])
        start = end

    return shards


def write_jsonl(records: list[dict], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split an evaluation data file into balanced JSONL shards.")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="eval_rank")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be greater than 0.")
    if not args.data_file.exists():
        raise FileNotFoundError(f"{args.data_file} not found.")

    args.output_dir.mkdir(exist_ok=True, parents=True)
    records = load_records(args.data_file)
    shards = split_records(records, args.num_shards)

    for rank, shard in enumerate(shards):
        output_file = args.output_dir / f"{args.prefix}_{rank}.jsonl"
        write_jsonl(shard, output_file)
        print(f"rank={rank} samples={len(shard)} file={output_file}")


if __name__ == "__main__":
    main()

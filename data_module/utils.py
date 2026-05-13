import os
import json
from loguru import logger

ASPECT_RATIOS = ["1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]


def is_bucketed_dataset(data_file: str, sample_size: int = 1):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found.")
    ext = os.path.splitext(data_file)[-1].lower()
    assert ext in [".json", ".jsonl"], f"Unsupported data format: {ext}, only JSON or JSONL is supported."

    def _is_bucket_format(data):
        if not isinstance(data, dict):
            return False

        if len(data) == 0:
            return False

        first_key = next(iter(data))
        first_value = data[first_key]

        if "prompt" in data:
            return False

        if isinstance(first_value, list) and len(first_value) > 0:
            if isinstance(first_value[0], dict) and "prompt" in first_value[0]:
                return True

        return False

    if ext == ".jsonl":
        with open(data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                line_data = json.loads(line.strip())
                if _is_bucket_format(line_data):
                    return True
        return False

    elif ext == ".json":
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return _is_bucket_format(data)

    else:
        logger.warning(f"Unsupported data format: {ext}, only JSON or JSONL is supported.")
        return False

from functools import reduce
from typing import Any


def get_nested_attr(obj, path: str) -> Any:
    return reduce(getattr, path.split("."), obj)


def set_nested_attr(obj, path: str, value):
    parts = path.split(".")
    parents = reduce(getattr, parts[:-1], obj) if len(parts) > 1 else obj
    setattr(parents, parts[-1], value)

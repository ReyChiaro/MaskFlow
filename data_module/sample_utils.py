from pathlib import PurePosixPath


def image_name(sample: dict) -> str:
    """Preserve target stems; use the prompt-specific mask path without a target."""
    if sample.get("target") is not None:
        return PurePosixPath(sample["target"]).stem
    conditions = sample.get("conditions")
    if not isinstance(conditions, dict) or not conditions.get("source") or not conditions.get("mask"):
        raise ValueError("Samples without targets require conditions.source and conditions.mask.")
    path = PurePosixPath(conditions["mask"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative mask path: {path}")
    if path.parts[0] == "mask":
        path = PurePosixPath(*path.parts[1:])
    return str(path.with_suffix(""))

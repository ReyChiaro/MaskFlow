"""Build a validated metric pipeline from JSONL or Hugging Face MaskEdit data.

Unavailable metrics are reported and skipped; valid metrics still run. Pairwise
metrics use target images when requested/declared, otherwise source images.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torchvision.transforms import functional as TF

from data_module.mask_edit_dataset import HFMaskEditDataset
from data_module.sample_utils import image_name
from data_module.mask_edit_dataset import HFMaskEditDataset
from data_module.utils import (
    center_crop_to_aspect_ratio,
    crop_image_to_aspect_ratio,
    reshape_to_divisible_max_resolution,
)
from evaluator.register import get_metrics, initialize_metrics

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_METRICS = ["CLIP-TEXT", "PSNR", "SSIM", "DISTS"]
REGION_SUFFIX = {"whole": "", "foreground": "-FG", "background": "-BG"}


class ImageDirectory:
    """Resolve relative paths first, allowing an unambiguous flat-name fallback."""

    def __init__(self, root):
        self.root = Path(root) if root is not None else None
        self.relative, self.stems = defaultdict(list), defaultdict(list)
        self.error = None
        if self.root is None:
            self.error = "directory not supplied"
        elif not self.root.is_dir():
            self.error = f"directory does not exist: {self.root}"
        else:
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.relative[path.relative_to(self.root).with_suffix("").as_posix()].append(path)
                    self.stems[path.stem].append(path)

    def resolve(self, value, role, fallback=None, rebase=False):
        if self.error:
            raise ValueError(self.error)
        if not isinstance(value, str) or not value:
            if fallback is None:
                raise ValueError(f"missing {role} path in JSONL")
            value = fallback
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{role} path must be relative: {value}")
        # A supplied role directory can replace source/, mask/ or target/ in JSONL.
        keys = [path.with_suffix("").as_posix()]
        if rebase and len(path.parts) > 1 and path.parts[0] in {role, "sources", "masks", "targets"}:
            keys.append(Path(*path.parts[1:]).with_suffix("").as_posix())
        for key in keys:
            candidates = self.relative.get(key, [])
            if candidates:
                if len(candidates) != 1:
                    raise ValueError(f"ambiguous {role} path {value}: {candidates}")
                return candidates[0]
        candidates = self.stems.get(path.stem, []) if rebase else []
        if len(candidates) != 1:
            raise ValueError(f"{role} {value}: found {len(candidates)} matching images under {self.root}")
        return candidates[0]


def image_index(folder):
    """Compatibility helper for the original folder-only API."""
    directory = ImageDirectory(folder)
    if directory.error:
        raise ValueError(directory.error)
    if not directory.relative:
        raise ValueError(f"No images found in {folder}")
    if any(len(paths) != 1 for paths in directory.relative.values()):
        raise ValueError(f"Duplicate image names in {folder}")
    return {key: paths[0] for key, paths in directory.relative.items()}


def read_records(data_file):
    records = []
    with Path(data_file).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {data_file}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {data_file}:{line_number}")
            records.append(row)
    if not records:
        raise ValueError(f"No records found in {data_file}")
    return records


class HFImageFiles(Sequence):
    """Expose embedded images as file objects without extracting or retaining them."""

    def __init__(self, dataset, role):
        self.dataset, self.role = dataset, role

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return BytesIO(self.dataset.load_image_bytes(index, self.role))


class ImageFiles(Sequence):
    """Lazily load RGB/mask tensors on a reference grid without retaining the dataset."""

    def __init__(self, paths, sizes=None, device="cpu", is_mask=False, crop=False):
        self.paths, self.sizes, self.device = paths, sizes, device
        self.is_mask, self.crop = is_mask, crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = TF.pil_to_tensor(image.convert("L" if self.is_mask else "RGB")).float() / 255
        if self.sizes is not None:
            width, height = self.sizes[index]
            if self.crop:
                tensor = center_crop_to_aspect_ratio(tensor, width / height)
            tensor = TF.resize(
                tensor,
                [height, width],
                interpolation=TF.InterpolationMode.NEAREST if self.is_mask else TF.InterpolationMode.BICUBIC,
                antialias=not self.is_mask,
            ).clamp(0, 1)
        return tensor.to(self.device)


class ReferenceImages(ImageFiles):
    def __init__(self, paths, device, preprocess, divisible_by):
        super().__init__(paths, device="cpu")
        self.output_device = device
        self.preprocess, self.divisible_by = preprocess, divisible_by
        self.cached_index, self.cached_image = None, None

    def __getitem__(self, index):
        if index != self.cached_index:
            image = super().__getitem__(index)
            if self.preprocess == "mask-edit":
                image, aspect = crop_image_to_aspect_ratio(image)
                image = reshape_to_divisible_max_resolution(image, aspect, divisible_by=self.divisible_by)
            self.cached_index, self.cached_image = index, image.to(self.output_device)
        return self.cached_image


class ReferenceSizes(Sequence):
    def __init__(self, references):
        self.references = references

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        image = self.references[index]
        return image.shape[-1], image.shape[-2]


class PixelBlendedImages(Sequence):
    def __init__(self, predictions, originals, masks):
        self.predictions, self.originals, self.masks = predictions, originals, masks

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        return torch.where(self.masks[index] > 0, self.predictions[index], self.originals[index])


@dataclass
class MetricStep:
    name: str
    function: object
    needs_reference: bool


class EvaluationData:
    def __init__(
        self,
        rows,
        pred_dir,
        source_dir,
        mask_dir,
        target_dir,
        image_root,
        reference,
        prompt_key,
        device,
        preprocess,
        divisible_by,
        dataset=None,
    ):
        self.rows, self.device = rows, device
        self.dataset = dataset
        self.preprocess, self.divisible_by = preprocess, divisible_by
        self.paths, self.errors = {}, {}
        self._views, self._checked_images = {}, {}
        declared_target = any(row.get("target") is not None for row in rows)
        self.reference = (
            ("target" if target_dir is not None or declared_target else "source") if reference == "auto" else reference
        )
        self.directories = {"prediction": ImageDirectory(pred_dir)}
        for role, override in [("source", source_dir), ("mask", mask_dir), ("target", target_dir)]:
            self.directories[role] = ImageDirectory(override if override is not None else image_root)
        self.overrides = {"source": source_dir, "mask": mask_dir, "target": target_dir}
        self.keys, self.key_errors = [], []
        source_counts = Counter(
            (row.get("conditions") or {}).get("source") for row in rows if isinstance(row.get("conditions"), dict)
        )
        prompt_counts = Counter()
        for index, row in enumerate(rows):
            try:
                # Optional explicit output name also supports custom test-set formats.
                conditions = row.get("conditions") or {}
                if row.get("image_name"):
                    key = row["image_name"]
                elif row.get("target") is not None:
                    key = image_name(row)
                elif isinstance(conditions, dict) and conditions.get("mask"):
                    path = Path(conditions["mask"])
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"mask path must be relative: {path}")
                    if path.parts[0] == "mask":
                        path = Path(*path.parts[1:])
                    key = path.with_suffix("").as_posix()
                else:
                    source = conditions.get("source") if isinstance(conditions, dict) else None
                    if not isinstance(source, str) or not source:
                        raise ValueError("prediction matching needs image_name, target, mask or source path")
                    path = Path(source)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"source path must be relative: {source}")
                    if path.parts[0] == "source":
                        path = Path(*path.parts[1:])
                    key = path.with_suffix("").as_posix()
                    if source_counts[source] > 1:
                        key += f"_{prompt_counts[source]}"
                    prompt_counts[source] += 1
                if not isinstance(key, str) or Path(key).is_absolute() or ".." in Path(key).parts:
                    raise ValueError("image_name must be a relative path")
                self.keys.append(key)
            except (ValueError, TypeError, IndexError) as error:
                self.keys.append(None)
                self.key_errors.append(f"row {index + 1}: {error}")
        duplicates = [key for key, count in Counter(self.keys).items() if key is not None and count > 1]
        if duplicates:
            self.key_errors.append(f"duplicate prediction names: {duplicates[:5]}")
        self.prompts = [row.get(prompt_key) for row in rows]
        self.prompt_errors = [
            f"row {index + 1}: missing/non-string/empty prompt key {prompt_key!r}"
            for index, value in enumerate(self.prompts)
            if not isinstance(value, str) or not value.strip()
        ]

    def require(self, role):
        if role in self.errors:
            return self.errors[role]
        if self.dataset is not None and role != "prediction" and self.overrides[role] is None:
            images = HFImageFiles(self.dataset, role)
            errors = []
            for index in range(len(images)):
                try:
                    with Image.open(images[index]) as image:
                        image.verify()
                except (OSError, ValueError, TypeError) as error:
                    errors.append(f"row {index + 1}: unreadable embedded {role}: {error}")
            self.paths[role], self.errors[role] = images, errors
            return errors
        errors, paths = [], []
        if role == "prediction":
            errors.extend(self.key_errors)
        for index, row in enumerate(self.rows):
            conditions = row.get("conditions") or {}
            if not isinstance(conditions, dict):
                conditions = {}
            value = (
                self.keys[index]
                if role == "prediction"
                else row.get("target")
                if role == "target"
                else conditions.get(role)
            )
            # Prediction keys have no suffix. Add a synthetic suffix so dotted IDs survive resolution.
            if role == "prediction" and value:
                value += ".png"
            # Evaluation saves masks under the target stem, not the original mask path.
            if self.dataset is not None and role == "mask" and self.overrides[role] is not None:
                value = self.keys[index] + ".png" if self.keys[index] else None
            fallback = None
            if role != "prediction" and self.overrides[role] is not None and self.keys[index]:
                fallback = self.keys[index] + ".png"
            try:
                path = self.directories[role].resolve(
                    value, role, fallback, rebase=role == "prediction" or self.overrides[role] is not None
                )
                if path not in self._checked_images:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        self._checked_images[path] = None
                    except (OSError, ValueError) as error:
                        self._checked_images[path] = f"unreadable image {path}: {error}"
                if self._checked_images[path]:
                    raise ValueError(self._checked_images[path])
                paths.append(path)
            except (ValueError, OSError) as error:
                paths.append(None)
                errors.append(f"row {index + 1}: {error}")
        if role == "prediction":
            reused = [str(path) for path, count in Counter(paths).items() if path is not None and count > 1]
            if reused:
                errors.append(f"multiple records resolve to the same prediction: {reused[:5]}")
        self.paths[role], self.errors[role] = paths, errors
        return errors

    def views(self, needs_reference, blend, region):
        # Text-image CLIP alone does not require a reference or its dimensions.
        grid = self.reference if needs_reference else "source" if blend else None
        cache_key = grid, blend, region
        if cache_key in self._views:
            return self._views[cache_key]
        references = (
            ReferenceImages(self.paths[grid], self.device, self.preprocess, self.divisible_by) if grid else None
        )
        sizes = ReferenceSizes(references) if references is not None else None
        predictions = ImageFiles(self.paths["prediction"], sizes, self.device)
        if sizes is None and region != "whole":
            # Regional text CLIP aligns its mask to predictions without consulting target/source.
            sizes = ReferenceSizes(predictions)
        masks = (
            ImageFiles(self.paths["mask"], sizes, self.device, is_mask=True, crop=self.preprocess == "mask-edit")
            if blend or region != "whole"
            else None
        )
        if blend:
            originals = ImageFiles(self.paths["source"], sizes, self.device, crop=self.preprocess == "mask-edit")
            predictions = PixelBlendedImages(predictions, originals, masks)
        result = predictions, references, masks
        self._views[cache_key] = result
        return result


def describe_errors(errors):
    return "; ".join(errors[:3]) + (f"; ... ({len(errors)} problems total)" if len(errors) > 3 else "")


def build_pipeline(
    data,
    metrics,
    registry,
    region,
    enable_pixel_blend,
    device,
    clip_model_id,
    clip_batch_size,
    preprocess,
    divisible_by,
):
    pipeline, skipped = [], {}
    common = []
    if region not in REGION_SUFFIX:
        common.append(f"invalid region={region!r}; choose whole, foreground or background")
    if preprocess not in {"mask-edit", "resize"}:
        common.append(f"invalid preprocess={preprocess!r}")
    if divisible_by <= 0:
        common.append("divisible_by must be positive")
    try:
        parsed_device = torch.device(device)
        if parsed_device.type == "cuda" and (
            not torch.cuda.is_available()
            or parsed_device.index is not None
            and parsed_device.index >= torch.cuda.device_count()
        ):
            common.append(f"CUDA device is unavailable: {device}")
        elif parsed_device.type not in {"cpu", "cuda"}:
            common.append(f"unsupported metric device: {device}; use cpu or cuda")
    except (RuntimeError, ValueError, TypeError) as error:
        common.append(f"invalid device {device!r}: {error}")
    common.extend(data.require("prediction"))
    if enable_pixel_blend:
        common.extend(f"pixel-blend requires source: {error}" for error in data.require("source"))
        common.extend(f"pixel-blend requires mask: {error}" for error in data.require("mask"))
    for requested in dict.fromkeys(metrics):
        requested = requested.upper()
        base = requested[:-3] if requested.endswith(("-FG", "-BG")) else requested
        name = base + REGION_SUFFIX.get(region, "")
        reasons = list(common)
        if requested != base and requested != name:
            reasons.append(f"{requested} conflicts with region={region}")
        if name not in registry:
            reasons.append(f"metric/region is not registered: {name}; available: {', '.join(sorted(registry))}")
        needs_reference = base != "CLIP-TEXT"
        if needs_reference:
            reasons.extend(f"{data.reference} reference: {error}" for error in data.require(data.reference))
        else:
            reasons.extend(data.prompt_errors)
            if not clip_model_id or Path(clip_model_id).is_absolute() and not Path(clip_model_id).is_dir():
                reasons.append(f"CLIP model directory is missing/invalid: {clip_model_id}")
        if base in {"CLIP", "CLIP-TEXT"} and clip_batch_size < 1:
            reasons.append("clip_batch_size must be positive")
        if region != "whole":
            reasons.extend(f"regional metric requires mask: {error}" for error in data.require("mask"))
        if base == "FID" and len(data.rows) < 2:
            reasons.append("FID requires at least two image pairs")
        if not reasons and region != "whole":
            # Empty regions do not have a meaningful per-image preservation score.
            try:
                _, _, masks = data.views(needs_reference, enable_pixel_blend, region)
                for index in range(len(masks)):
                    selected = masks[index] > 0 if region == "foreground" else masks[index] <= 0
                    if not selected.any():
                        reasons.append(f"row {index + 1}: empty {region} after mask alignment")
            except (ValueError, OSError, RuntimeError) as error:
                reasons.append(f"mask alignment failed: {error}")
        if reasons:
            skipped[requested] = describe_errors(reasons)
            logger.warning(f"Skip {requested}: {skipped[requested]}")
        elif not any(step.name == name for step in pipeline):
            pipeline.append(MetricStep(name, registry[name], needs_reference))
    return pipeline, skipped


def print_results(result):
    rows = [
        [name, "OK", format(value, ".6g") if isinstance(value, (int, float)) else value]
        for name, value in result["metrics"].items()
    ]
    rows += [[name, "SKIPPED", reason] for name, reason in result["skipped_metrics"].items()]
    rows += [[name, "FAILED", reason] for name, reason in result["failed_metrics"].items()]
    header = ["Metric", "Status", "Score / reason"]
    # Keep the table readable in a normal terminal; full reasons remain in JSON/warnings.
    rows = [[str(value).replace("\n", " ") for value in row] for row in rows]
    rows = [[row[0], row[1], row[2][:117] + "..." if len(row[2]) > 120 else row[2]] for row in rows]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(3)]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(
        f"\nImages: {result['num_images']} | region: {result['region']} | "
        f"pair reference: {result['reference']} | pixel-blend: {result['enable_pixel_blend']}"
    )
    print(border)
    for index, row in enumerate([header] + rows):
        print("| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |")
        if index == 0:
            print(border)
    print(border)


def save_results(result, output):
    # Perfect-match PSNR is infinite; emit strict JSON rather than a nonstandard Infinity token.
    serializable = dict(
        result, metrics={key: value if math.isfinite(value) else str(value) for key, value in result["metrics"].items()}
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


@torch.inference_mode()
def calculate_metrics(
    pred_dir,
    gt_dir=None,
    metrics=None,
    mask_dir=None,
    region="whole",
    device="cpu",
    enable_pixel_blend=False,
    source_dir=None,
    *,
    data_file=None,
    image_root=None,
    target_dir=None,
    reference="auto",
    prompt_key="prompt",
    preprocess=None,
    divisible_by=32,
    clip_model_id="/root/models/clip-vit-large-patch14-336",
    clip_batch_size=8,
    output=None,
    data_root=None,
    subsets=None,
    split=None,
):
    """Validate -> select metric steps -> execute -> report. Legacy folder arguments still work."""
    if reference not in {"auto", "source", "target"}:
        raise ValueError("reference must be auto, source or target")
    if target_dir is not None and gt_dir is not None and Path(target_dir) != Path(gt_dir):
        raise ValueError("target_dir and its legacy alias gt_dir disagree")
    target_dir = target_dir if target_dir is not None else gt_dir
    if data_root is not None and (data_file is not None or image_root is not None):
        raise ValueError("data_root cannot be combined with data_file or image_root.")
    if data_root is None and (subsets is not None or split is not None):
        raise ValueError("subsets and split require data_root.")
    dataset = None
    if data_root is not None:
        dataset = HFMaskEditDataset(
            data_root=data_root,
            subsets=list(HFMaskEditDataset.SUBSETS) if subsets is None else subsets,
            split="test" if split is None else split,
            divisible_by=divisible_by,
        )
        rows = dataset.samples
    elif data_file is not None:
        rows = read_records(data_file)
        image_root = Path(image_root) if image_root is not None else Path(data_file).parent
    else:
        # Adapt old folder-only callers to the same record-driven pipeline.
        rows = [
            {
                "image_name": key,
                "conditions": {"source": key + ".png", "mask": key + ".png"},
                "target": key + ".png" if target_dir is not None else None,
            }
            for key in image_index(Path(pred_dir))
        ]
    preprocess = preprocess or ("mask-edit" if data_file is not None or dataset is not None else "resize")
    initialize_metrics()
    registry = get_metrics()
    if metrics is None:
        metrics = (
            DEFAULT_METRICS
            if data_file is not None or dataset is not None
            else [name for name in registry if name != "CLIP-TEXT" and not name.endswith(("-FG", "-BG"))]
        )
    data = EvaluationData(
        rows,
        pred_dir,
        source_dir,
        mask_dir,
        target_dir,
        image_root,
        reference,
        prompt_key,
        device,
        preprocess,
        divisible_by,
        dataset=dataset,
    )
    pipeline, skipped = build_pipeline(
        data,
        metrics,
        registry,
        region,
        enable_pixel_blend,
        device,
        clip_model_id,
        clip_batch_size,
        preprocess,
        divisible_by,
    )
    result = {
        "num_images": len(rows),
        "region": region,
        "reference": data.reference,
        "enable_pixel_blend": enable_pixel_blend,
        "preprocess": preprocess,
        "divisible_by": divisible_by,
        "prompt_key": prompt_key,
        "clip_model_id": clip_model_id,
        "clip_score_definition": "mean(max(cosine(image, prompt), 0))",
        "data_file": str(data_file) if data_file is not None else None,
        "data_root": str(data_root) if data_root is not None else None,
        "subsets": dataset.subsets if dataset is not None else None,
        "split": dataset.split if dataset is not None else None,
        "image_root": str(image_root) if image_root is not None else None,
        "pred_dir": str(pred_dir),
        "source_dir": str(source_dir) if source_dir is not None else None,
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "target_dir": str(target_dir) if target_dir is not None else None,
        "requested_metrics": list(metrics),
        "pipeline": [step.name for step in pipeline],
        "metric_references": {step.name: data.reference if step.needs_reference else "prompt" for step in pipeline},
        "metrics": {},
        "skipped_metrics": skipped,
        "failed_metrics": {},
    }
    if not pipeline:
        logger.warning("No requested metrics passed validation; saving an empty report with reasons.")
    for step in pipeline:
        try:
            predictions, references, masks = data.views(step.needs_reference, enable_pixel_blend, region)
            kwargs = {"prompts": data.prompts, "clip_model_id": clip_model_id, "clip_batch_size": clip_batch_size}
            if region != "whole":
                kwargs["mask"] = masks
            logger.info(f"Computing {step.name} on {len(rows)} images")
            score = float(step.function(predictions, references, **kwargs))
            if math.isnan(score):
                raise ValueError("metric returned NaN")
            result["metrics"][step.name] = score
        except Exception as error:
            # A failed model load or one metric implementation must not discard other results.
            result["failed_metrics"][step.name] = f"{type(error).__name__}: {error}"
            logger.warning(f"Failed {step.name}: {result['failed_metrics'][step.name]}")
    print_results(result)
    if output is not None:
        save_results(result, output)
    return result


def calculate_target_free_metrics(
    pred_dir,
    data_file,
    image_root,
    metrics=None,
    device="cpu",
    divisible_by=32,
    clip_model_id="/root/models/clip-vit-large-patch14-336",
    clip_batch_size=8,
    prompt_key="prompt",
):
    """Compatibility wrapper for the previous target-free API."""
    return calculate_metrics(
        pred_dir,
        data_file=data_file,
        image_root=image_root,
        metrics=metrics,
        device=device,
        reference="source",
        divisible_by=divisible_by,
        clip_model_id=clip_model_id,
        clip_batch_size=clip_batch_size,
        prompt_key=prompt_key,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data-file", type=Path, help="JSONL test set; the filename/extension is arbitrary")
    inputs.add_argument("--data-root", type=Path, help="Downloaded MaskEdit-10k repository containing data/<subset>")
    parser.add_argument(
        "--subsets", nargs="+", choices=HFMaskEditDataset.SUBSETS, help="HF subsets; default: all three"
    )
    parser.add_argument("--split", choices=["train", "test"], help="HF split; default: test")
    parser.add_argument("--source-dir", type=Path, help="Original-image directory; replaces conditions.source root")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Root for JSONL paths; default: JSONL parent directory")
    parser.add_argument("--mask-dir", type=Path, help="Optional mask directory; HF masks match prediction filenames")
    parser.add_argument("--target-dir", "--gt-dir", dest="target_dir", type=Path, help="Optional target directory")
    parser.add_argument("--reference", choices=["auto", "source", "target"], default="auto")
    parser.add_argument("--region", choices=list(REGION_SUFFIX), default="whole")
    parser.add_argument("--enable-pixel-blend", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--prompt-key", default="prompt", help="Top-level text field in JSONL or HF metadata")
    parser.add_argument("--preprocess", choices=["mask-edit", "resize"], default="mask-edit")
    parser.add_argument("--divisible-by", type=int, default=32)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Default: predictions' parent / metrics_<region>.json")
    args = parser.parse_args()
    options = vars(args)
    options["output"] = args.output or args.pred_dir.parent / f"metrics_{args.region}.json"
    try:
        result = calculate_metrics(**options)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0 if result["metrics"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

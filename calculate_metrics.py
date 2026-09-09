"""Evaluate matching image folders with metrics registered in evaluator.

Images are matched by filename stem. Predictions are resized to ground-truth
resolution with bicubic interpolation; masks use nearest-neighbor interpolation.
Regional scores use evaluator's existing -FG/-BG definitions (outside is zeroed).
Optional pixel blending keeps predictions where mask > 0 and uses source elsewhere.
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from evaluator.register import get_metrics, initialize_metrics


EVAL_DIR = Path("outputs/evaluation/eval_QwenImage-MaskFlow-r256/20260909-043912/evaluations")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def image_index(folder: Path) -> dict[str, Path]:
    images = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            raise ValueError(f"Duplicate image stem in {folder}: {path.stem}")
        images[path.stem] = path
    if not images:
        raise ValueError(f"No images found in {folder}")
    return images


class ImageFiles(Sequence):
    """Load tensors on demand so each metric holds only the current image pair."""

    def __init__(self, paths, sizes, device, is_mask=False):
        self.paths, self.sizes = paths, sizes
        self.device, self.is_mask = device, is_mask

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L" if self.is_mask else "RGB")
            if image.size != self.sizes[index]:
                method = Image.Resampling.NEAREST if self.is_mask else Image.Resampling.BICUBIC
                image = image.resize(self.sizes[index], method)
            return pil_to_tensor(image).to(device=self.device, dtype=torch.float32) / 255.0


class PixelBlendedImages(Sequence):
    def __init__(self, predictions, originals, masks):
        self.predictions, self.originals, self.masks = predictions, originals, masks

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        return torch.where(self.masks[index] > 0, self.predictions[index], self.originals[index])


@torch.inference_mode()
def calculate_metrics(
    pred_dir, gt_dir, metrics=None, mask_dir=None, region="whole", device="cpu",
    enable_pixel_blend=False, source_dir=None,
):
    if region not in {"whole", "foreground", "background"}:
        raise ValueError(f"Unknown region: {region}")
    if region != "whole" and mask_dir is None:
        raise ValueError("--mask-dir is required for foreground/background evaluation.")
    if enable_pixel_blend and (source_dir is None or mask_dir is None):
        raise ValueError("--enable-pixel-blend requires both --source-dir and --mask-dir.")

    initialize_metrics()
    registry = get_metrics()
    suffix = {"whole": "", "foreground": "-FG", "background": "-BG"}[region]
    if metrics is None:
        names = [name for name in registry if (
            name.endswith(suffix) if suffix else not name.endswith(("-FG", "-BG"))
        )]
    else:
        names = []
        for name in metrics:
            name = name.upper()
            base = name[:-3] if name.endswith(("-FG", "-BG")) else name
            if name != base and name != base + suffix:
                raise ValueError(f"Metric {name} conflicts with region={region}.")
            names.append(base + suffix)
    names = list(dict.fromkeys(names))
    unknown = set(names) - registry.keys()
    if unknown or not names:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}. Registered: {sorted(registry)}")

    predictions, targets = image_index(Path(pred_dir)), image_index(Path(gt_dir))
    masks = image_index(Path(mask_dir)) if suffix or enable_pixel_blend else None
    originals = image_index(Path(source_dir)) if enable_pixel_blend else None
    keys = sorted(predictions)
    for label, index in [("ground truth", targets), ("mask", masks), ("source", originals)]:
        if index is not None:
            missing = set(keys) - index.keys()
            if missing:
                raise ValueError(f"Missing {label} images for {len(missing)} predictions: {sorted(missing)[:5]}")
    if "FID" in names and len(keys) < 2:
        raise ValueError("FID requires at least two image pairs.")

    sizes = []
    for key in keys:
        with Image.open(targets[key]) as image:
            sizes.append(image.size)
    sources = ImageFiles([predictions[key] for key in keys], sizes, device)
    references = ImageFiles([targets[key] for key in keys], sizes, device)
    mask_images = ImageFiles([masks[key] for key in keys], sizes, device, is_mask=True) if masks else None
    if enable_pixel_blend:
        original_images = ImageFiles([originals[key] for key in keys], sizes, device)
        sources = PixelBlendedImages(sources, original_images, mask_images)
    kwargs = {"mask": mask_images} if suffix else {}

    print(f"Evaluating {len(keys)} pairs, region={region}, pixel_blend={enable_pixel_blend}, device={device}", flush=True)
    results = {}
    for name in names:
        print(f"Computing {name}...", flush=True)
        # Pass the complete dataset once, including for distribution metrics such as FID.
        results[name] = float(registry[name](sources, references, **kwargs))
        print(f"{name}: {results[name]:.6g}", flush=True)
    return {"num_images": len(keys), "region": region, "enable_pixel_blend": enable_pixel_blend, "metrics": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=EVAL_DIR / "predictions")
    parser.add_argument("--gt-dir", type=Path, default=Path("dataset/testset/target"))
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--source-dir", type=Path, help="Original images, e.g. dataset/testset/source")
    parser.add_argument("--enable-pixel-blend", action="store_true", help="Use source pixels where mask is zero before evaluation")
    parser.add_argument("--region", choices=["whole", "foreground", "background"], default="whole")
    parser.add_argument("--metrics", nargs="+", help="Registered names, e.g. MSE PSNR SSIM LPIPS; default: all supported for region")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Default: predictions' parent / metrics_<region>.json")
    args = parser.parse_args()

    result = calculate_metrics(
        args.pred_dir, args.gt_dir, args.metrics, args.mask_dir, args.region, args.device,
        enable_pixel_blend=args.enable_pixel_blend, source_dir=args.source_dir,
    )
    result.update(pred_dir=str(args.pred_dir), gt_dir=str(args.gt_dir), mask_dir=str(args.mask_dir) if args.mask_dir else None)
    result["source_dir"] = str(args.source_dir) if args.source_dir else None
    # Preserve perfect-match PSNR as a string instead of emitting invalid JSON Infinity.
    result["metrics"] = {k: v if math.isfinite(v) else str(v) for k, v in result["metrics"].items()}
    output = args.output or args.pred_dir.parent / f"metrics_{args.region}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()

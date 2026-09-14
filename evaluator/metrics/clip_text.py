import torch
from transformers import CLIPModel, CLIPProcessor

from evaluator.register import REGISTER_METRIC
from .mask_utils import RegionalImages, image_device, image_list, mean_metric

CLIP_MODEL_ID = "/root/models/intern-xr/clip-vit-large-patch14-336"
_MODEL_CACHE = {}


def _load_model(model_id, device):
    key = (model_id, str(device))
    if key not in _MODEL_CACHE:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval().requires_grad_(False)
        _MODEL_CACHE[key] = processor, model
    return _MODEL_CACHE[key]


@REGISTER_METRIC("CLIP-TEXT")
@torch.inference_mode()
def CLIP_TEXT(source, target=None, prompts=None, clip_model_id=CLIP_MODEL_ID, clip_batch_size=8, **kwargs):
    """Mean max(cosine(edited image, instruction), 0); higher is better."""
    images = image_list(source)
    if isinstance(prompts, str):
        prompts = [prompts]
    if prompts is None or len(images) != len(prompts) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("CLIP-TEXT requires one prompt string per edited image.")
    if clip_batch_size < 1:
        raise ValueError("clip_batch_size must be positive.")
    device = image_device(images)
    processor, model = _load_model(clip_model_id, device)
    scores = []
    for start in range(0, len(images), clip_batch_size):
        inputs = processor(
            text=list(prompts[start:start + clip_batch_size]),
            images=[images[i].detach().cpu().float() for i in range(start, min(start + clip_batch_size, len(images)))],
            return_tensors="pt", padding=True, truncation=True,
            max_length=model.config.text_config.max_position_embeddings, do_rescale=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        # CLIPModel returns normalized embeddings, independent of its learned logit scale.
        values = ((outputs.image_embeds * outputs.text_embeds).sum(dim=-1)).clamp_min(0)
        scores.extend(values.float().tolist())
    return mean_metric(scores)


@REGISTER_METRIC("CLIP-TEXT-FG")
@torch.inference_mode()
def CLIP_TEXT_FG(source, target=None, mask=None, **kwargs):
    """Text alignment of foreground pixels; background pixels are zeroed."""
    return CLIP_TEXT(RegionalImages(source, mask, foreground=True), **kwargs)


@REGISTER_METRIC("CLIP-TEXT-BG")
@torch.inference_mode()
def CLIP_TEXT_BG(source, target=None, mask=None, **kwargs):
    """Text alignment of background pixels; foreground pixels are zeroed."""
    return CLIP_TEXT(RegionalImages(source, mask, foreground=False), **kwargs)

"""Condition selection and velocity combination shared by MaskFlow pipelines."""

import math

import torch


BRANCHES = ("pm", "pn", "nm", "nn")


def branch_conditions(branch: str) -> tuple[bool, bool]:
    """Return whether text and mask are visible; the source is always retained."""
    if branch not in BRANCHES:
        raise ValueError(f"Unknown CFG branch {branch!r}; expected one of {BRANCHES}.")
    return branch[0] == "p", branch[1] == "m"


def training_probabilities(probabilities=None, text_dropout=0.1, mask_dropout=0.0):
    """Explicit branch probabilities take precedence over legacy dropout rates."""
    if probabilities is None:
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (text_dropout, mask_dropout)):
            raise ValueError("CFG dropout probabilities must be between 0 and 1.")
        probabilities = dict(
            pm=(1 - text_dropout) * (1 - mask_dropout),
            pn=(1 - text_dropout) * mask_dropout,
            nm=text_dropout * (1 - mask_dropout),
            nn=text_dropout * mask_dropout,
        )
    if set(probabilities) != set(BRANCHES):
        raise ValueError("cfg_branch_probabilities must contain pm, pn, nm and nn.")
    result = {name: float(probabilities[name]) for name in BRANCHES}
    if not all(math.isfinite(p) and p >= 0 for p in result.values()):
        raise ValueError("CFG branch probabilities must be finite and nonnegative.")
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-6):
        raise ValueError("CFG branch probabilities must sum to 1.")
    return result


def cfg_coefficients(text_scale=1.0, mask_scale=1.0, interaction_scale=None):
    """Default: nn + mask*(nm-nn) + text*(pm-nm).

    An explicit interaction scale enables the general four-branch formula:
    nn + text*(pn-nn) + mask*(nm-nn) + interaction*(pm-pn-nm+nn).
    """
    interaction = text_scale if interaction_scale is None else interaction_scale
    if not all(math.isfinite(s) for s in (text_scale, mask_scale, interaction)):
        raise ValueError("CFG scales must be finite.")
    return {
        "pm": interaction,
        "pn": text_scale - interaction,
        "nm": mask_scale - interaction,
        "nn": 1 - text_scale - mask_scale + interaction,
    }


def required_branches(text_scale=1.0, mask_scale=1.0, interaction_scale=None, rescale=True):
    coefficients = cfg_coefficients(text_scale, mask_scale, interaction_scale)
    return [name for name in BRANCHES if coefficients[name] != 0 or (rescale and name == "pm")]


def combine_predictions(predictions, text_scale=1.0, mask_scale=1.0, interaction_scale=None, rescale=True):
    coefficients = cfg_coefficients(text_scale, mask_scale, interaction_scale)
    needed = required_branches(text_scale, mask_scale, interaction_scale, rescale)
    missing = set(needed) - predictions.keys()
    if missing:
        raise ValueError(f"Missing CFG predictions: {sorted(missing)}")
    # Preserve the existing two-branch arithmetic for text-only guidance.
    if mask_scale == 1 and coefficients["pn"] == 0 and text_scale not in (0, 1):
        combined = predictions["nm"] + text_scale * (predictions["pm"] - predictions["nm"])
    else:
        combined = sum(coefficients[name] * predictions[name] for name in needed if coefficients[name] != 0)
    if rescale:
        positive_norm = torch.linalg.vector_norm(predictions["pm"], dim=-1, keepdim=True).clamp_min(1e-6)
        combined_norm = torch.linalg.vector_norm(combined, dim=-1, keepdim=True).clamp_min(1e-6)
        combined = combined * (positive_norm / combined_norm)
    return combined

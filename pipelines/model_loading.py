"""Construct empty Hugging Face components and materialize weights after placement."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from diffusers import ModelMixin
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from transformers import PreTrainedModel

from pipelines.utils import set_nested_attr

ModelClass = type[ModelMixin] | type[PreTrainedModel]
ModelTransform = Callable[[torch.nn.Module], object]


@dataclass
class ModelLoader:
    """A component's checkpoint source and ordered CPU weight transformations.

    Parameters are constructed on meta; buffers retain their constructor values
    so nonpersistent buffers (such as rotary frequencies) survive materialization.
    """

    model_class: ModelClass
    pretrained_model: str
    subfolder: str
    dtype: torch.dtype
    transforms: list[ModelTransform] = field(default_factory=list)

    def build(self) -> torch.nn.Module:
        """Read configuration only, without loading checkpoint tensors."""
        if issubclass(self.model_class, ModelMixin):
            config = self.model_class.load_config(self.pretrained_model, subfolder=self.subfolder)
            with init_empty_weights(include_buffers=False):
                model = self.model_class.from_config(config)
            # Keep constructor buffers at their native precision (e.g. RoPE in fp32).
            for parameter in model.parameters():
                parameter.data = parameter.to(dtype=self.dtype)
        else:
            config = self.model_class.config_class.from_pretrained(self.pretrained_model, subfolder=self.subfolder)
            with init_empty_weights(include_buffers=False):
                model = self.model_class._from_config(config, torch_dtype=self.dtype)
            model.tie_weights()
        return model.requires_grad_(False).eval()

    def load(self, model: torch.nn.Module, device: torch.device, *, broadcast: bool) -> None:
        """Load on rank 0 CPU, then broadcast into the model's existing shard layout.

        DCP assigns real tensors to meta parameters; optimizers and parameter
        snapshots must be created afterwards. Only one component's full CPU
        weights are held at a time. Unsharded inference uses the same CPU loader.
        """
        state: dict[str, torch.Tensor] = {}
        buffers: dict[str, torch.Tensor] = {}
        if not broadcast or dist.get_rank() == 0:
            with torch.device("cpu"):
                reference = self.model_class.from_pretrained(
                    self.pretrained_model, subfolder=self.subfolder, torch_dtype=self.dtype
                ).requires_grad_(False)
                for transform in self.transforms:
                    transform(reference)
                state = reference.state_dict()
                buffers = dict(reference.named_buffers(remove_duplicate=False))
                del reference

        # Nonpersistent buffers are absent from state_dict; synchronize them too.
        # Real device buffers also tell DCP where to materialize meta parameters.
        persistent_names = set(model.state_dict())
        for name, buffer in model.named_buffers(remove_duplicate=False):
            value = buffers.get(name, buffer).to(device)
            if broadcast and name not in persistent_names:
                dist.broadcast(value, src=0)
            set_nested_attr(model, name, value)

        if broadcast:
            set_model_state_dict(
                model,
                state,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
            )
        else:
            model.load_state_dict(state, strict=True, assign=True)
            model.to(device)
        # DCP transports scalar state on CPU, including scalar persistent buffers.
        for name, buffer in model.named_buffers(remove_duplicate=False):
            set_nested_attr(model, name, buffer.to(device))

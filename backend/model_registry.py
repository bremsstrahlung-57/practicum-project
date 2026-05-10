from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from model_def import get_pruned_architecture

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    short_name: str
    path: str
    pruning_ratio: float | None
    precision: str
    tags: tuple[str, ...]
    stats: tuple[tuple[str, str], ...]

    @property
    def resolved_path(self) -> Path:
        return (ROOT_DIR / self.path).resolve()

    def to_api(self, available: bool = True, error: str | None = None) -> dict:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["stats"] = [{"label": key, "value": value} for key, value in self.stats]
        data["available"] = available
        if error:
            data["error"] = error
        return data


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="pruned_50_fp32",
        name="Structured 50% Distilled",
        short_name="50%",
        path="models/distillation/structured_pruned_50pct_distilled.pth",
        pruning_ratio=0.5,
        precision="FP32",
        tags=("pruned", "distilled", "fp32"),
        stats=(
            ("parameters", "~2.8M"),
            ("model size", "~11.0 MB"),
            ("pruning", "50% structured"),
            ("distilled", "ResNet-50 teacher"),
            ("precision", "FP32"),
        ),
    ),
    ModelSpec(
        id="pruned_50_distilled_int8",
        name="Structured 50% Distilled INT8",
        short_name="50%",
        path="models/distillation/structured_pruned_50pct_distil_int8.pt",
        pruning_ratio=0.5,
        precision="INT8",
        tags=("pruned", "distilled", "quantized", "int8"),
        stats=(
            ("parameters", "~2.8M"),
            ("model size", "~2.8 MB"),
            ("pruning", "50% structured"),
            ("distilled", "ResNet-50 teacher"),
            ("precision", "INT8 static"),
        ),
    ),
    ModelSpec(
        id="pruned_50_finetuned_fp32",
        name="Structured 50% Fine-tuned",
        short_name="50%",
        path="models/final_finetuned/structured_pruned_50pct_fp32_finetuned40.pth",
        pruning_ratio=0.5,
        precision="FP32",
        tags=("pruned", "fine-tuned", "fp32"),
        stats=(
            ("parameters", "~2.8M"),
            ("model size", "~11.0 MB"),
            ("pruning", "50% structured"),
            ("fine-tune", "40 epochs"),
            ("precision", "FP32"),
        ),
    ),
    ModelSpec(
        id="pruned_70_fp32",
        name="Structured 70% Distilled",
        short_name="70%",
        path="models/distillation/resnet_pruned70_distilled.pth",
        pruning_ratio=0.7,
        precision="FP32",
        tags=("pruned", "distilled", "fp32"),
        stats=(
            ("parameters", "~1.1M"),
            ("model size", "~4.0 MB"),
            ("pruning", "70% structured"),
            ("distilled", "ResNet-50 teacher"),
            ("precision", "FP32"),
        ),
    ),
)


MODEL_SPEC_BY_ID = {spec.id: spec for spec in MODEL_SPECS}

MODULE_ATTR_DEFAULTS = {
    "_parameters": dict,
    "_buffers": dict,
    "_non_persistent_buffers_set": set,
    "_backward_pre_hooks": OrderedDict,
    "_backward_hooks": OrderedDict,
    "_is_full_backward_hook": lambda: None,
    "_forward_hooks": OrderedDict,
    "_forward_hooks_with_kwargs": OrderedDict,
    "_forward_hooks_always_called": OrderedDict,
    "_forward_pre_hooks": OrderedDict,
    "_forward_pre_hooks_with_kwargs": OrderedDict,
    "_state_dict_hooks": OrderedDict,
    "_state_dict_pre_hooks": OrderedDict,
    "_load_state_dict_pre_hooks": OrderedDict,
    "_load_state_dict_post_hooks": OrderedDict,
    "_modules": dict,
}


def _configure_quantized_engine() -> None:
    supported = set(torch.backends.quantized.supported_engines)
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return


def _repair_legacy_module_attrs(module: nn.Module) -> None:
    for attr, factory in MODULE_ATTR_DEFAULTS.items():
        if not hasattr(module, attr):
            setattr(module, attr, factory())


@contextmanager
def _torch_load_compat():
    original_named_modules = nn.Module.named_modules

    def named_modules_with_legacy_state(
        self,
        memo=None,
        prefix="",
        remove_duplicate=True,
    ):
        _repair_legacy_module_attrs(self)
        return original_named_modules(
            self,
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

    nn.Module.named_modules = named_modules_with_legacy_state
    try:
        yield
    finally:
        nn.Module.named_modules = original_named_modules


def _repair_legacy_module_tree(model: nn.Module) -> None:
    with _torch_load_compat():
        for module in model.modules():
            _repair_legacy_module_attrs(module)


def _checkpoint_state_dict(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _strip_module_prefix(state_dict: dict) -> dict:
    if not all(isinstance(key, str) for key in state_dict):
        return state_dict

    if not any(key.startswith("module.") for key in state_dict):
        return state_dict

    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _load_fp32_state_dict(checkpoint: dict, pruning_ratio: float) -> nn.Module:
    model = get_pruned_architecture(pruning_ratio=pruning_ratio)
    state_dict = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
    model.load_state_dict(state_dict)
    return model


def load_model(spec: ModelSpec) -> nn.Module:
    if not spec.resolved_path.exists():
        raise FileNotFoundError(f"model file not found: {spec.path}")

    _configure_quantized_engine()
    with _torch_load_compat():
        checkpoint = torch.load(
            spec.resolved_path,
            map_location="cpu",
            weights_only=False,
        )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model"), nn.Module
    ):
        model = checkpoint["model"]
    elif isinstance(checkpoint, dict) and spec.pruning_ratio is not None:
        model = _load_fp32_state_dict(checkpoint, spec.pruning_ratio)
    else:
        raise ValueError(
            f"{spec.id} is not a loadable module and has no FP32 pruning ratio"
        )

    _repair_legacy_module_tree(model)
    model.eval()
    return model

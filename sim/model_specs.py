from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    qkv_dim: int       # num_heads * head_dim
    hidden: int
    intermediate: int
    layers: int
    num_heads: int
    kv_heads: int
    head_dim: int = 128
    model_type: str = "llm"
    cv_trace_module: str | None = None

    def __getitem__(self, idx: int):
        """Preserve legacy tuple access: (qkv, hidden, intermediate, layers, num_heads, kv_heads)."""
        return (self.qkv_dim, self.hidden, self.intermediate,
                self.layers, self.num_heads, self.kv_heads)[idx]

    def __len__(self) -> int:
        return 6


MODELS = {
    "qwen2.5-1.5b": ModelSpec("qwen2.5-1.5b", 1536, 1536, 8960, 28, 12, 2, head_dim=128),
    "qwen2.5-3b":   ModelSpec("qwen2.5-3b",   2048, 2048, 11008, 36, 16, 16, head_dim=128),
    "qwen2.5-7b":   ModelSpec("qwen2.5-7b",   3584, 3584, 18944, 28, 28, 4, head_dim=128),
    "qwen3-8b":     ModelSpec("qwen3-8b",     4096, 4096, 12288, 32, 32, 4, head_dim=128),
    "gemma-4-12b":  ModelSpec("gemma-4-12b",  4096, 4096, 16384, 40, 16, 8, head_dim=256),
    "mobilenetv3":  ModelSpec("mobilenetv3",  0, 0, 0, 0, 0, 0, head_dim=0,
                              model_type="cv", cv_trace_module="sim.cv.cv_trace"),
    "resnet18":     ModelSpec("resnet18",     0, 0, 0, 0, 0, 0, head_dim=0,
                              model_type="cv", cv_trace_module="sim.cv.traces.resnet18_trace"),
    "resnet50":     ModelSpec("resnet50",     0, 0, 0, 0, 0, 0, head_dim=0,
                              model_type="cv", cv_trace_module="sim.cv.traces.resnet50_trace"),
    "vit-b16":      ModelSpec("vit-b16",      0, 0, 0, 0, 0, 0, head_dim=0,
                              model_type="cv", cv_trace_module="sim.cv.traces.vit_trace"),
    "yolov8n":      ModelSpec("yolov8n",      0, 0, 0, 0, 0, 0, head_dim=0,
                              model_type="cv", cv_trace_module="sim.cv.traces.yolov8n_trace"),
}


def get_spec(alias: str) -> ModelSpec:
    if alias not in MODELS:
        raise ValueError(f"Unknown model alias: {alias}. Known: {list(MODELS.keys())}")
    return MODELS[alias]


def all_aliases() -> list:
    return list(MODELS.keys())


def llm_aliases() -> list:
    return [alias for alias, spec in MODELS.items() if spec.model_type == "llm"]


def cv_aliases() -> list:
    return [alias for alias, spec in MODELS.items() if spec.model_type == "cv"]

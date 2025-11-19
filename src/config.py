"""Configuration settings for the project."""

from dataclasses import dataclass

# Model registry for comparison feature
MODEL_REGISTRY = {
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass
class Config:
    # HuggingFace model
    HF_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Device preference
    DEVICE: str = "auto"  # "auto", "cuda", or "cpu"

    # Sampling settings
    MAX_NEW_TOKENS: int = 20
    NUM_SAMPLES: int = 64  # start with 64 futures per prompt, can tune
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.95
    TOP_K: int = 50

    # Landscape / projection (keep what we had before)
    GRID_SIZE: int = 80
    GRID_MARGIN: float = 0.1
    KERNEL_SIGMA: float = 0.6
    NORMALIZE_FIELD: bool = True

    # Projection
    PROJECTION_METHOD: str = "umap"

    # Seed
    SEED: int = 42

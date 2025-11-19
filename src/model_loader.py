"""Model loading utilities."""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import Config, MODEL_REGISTRY
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_device: Optional[torch.device] = None


def get_device() -> torch.device:
    """
    Resolve and cache the torch.device based on Config.DEVICE.
    """
    global _device
    if _device is not None:
        return _device

    preferred = Config.DEVICE.lower()
    if preferred == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preferred == "cpu":
        device = torch.device("cpu")
    else:
        # "auto"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    _device = device
    return device


def get_tokenizer() -> AutoTokenizer:
    """
    Load and cache the AutoTokenizer for HF_MODEL_NAME.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    logger.info(f"Loading tokenizer: {Config.HF_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.HF_MODEL_NAME,
    )
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback: create a pad token id
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    _tokenizer = tokenizer
    return tokenizer


def get_model(override_name: str | None = None) -> AutoModelForCausalLM:
    """
    Load and cache the AutoModelForCausalLM for HF_MODEL_NAME.

    Uses float16 on CUDA for efficiency, except for OpenAI OSS-20 models which use bfloat16.
    
    Args:
        override_name: If provided, load this model instead of Config.HF_MODEL_NAME.
                      Does not cache when override_name is provided.
    """
    global _model
    if _model is not None and override_name is None:
        return _model

    model_name = override_name or Config.HF_MODEL_NAME
    device = get_device()

    # Decide dtype per model
    if "gpt-oss" in model_name.lower() or "oss-20" in model_name.lower():
        # OpenAI OSS models prefer bfloat16
        torch_dtype = torch.bfloat16
    else:
        # Default for other models
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info(f"Loading model: {model_name} on device: {device} with dtype={torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,  # we move manually
    )

    # If we added a pad token to tokenizer, we should also resize embeddings
    tokenizer = get_tokenizer()
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        logger.info("Resizing token embeddings to match tokenizer length")
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    
    if override_name is None:
        _model = model
    
    return model


def load_model_by_name(name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a model and tokenizer by name from MODEL_REGISTRY.
    
    Args:
        name: Model name from MODEL_REGISTRY
        
    Returns:
        Tuple of (tokenizer, model)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in MODEL_REGISTRY. Available: {list(MODEL_REGISTRY.keys())}")
    
    hf_id = MODEL_REGISTRY[name]
    device = get_device()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    # Decide dtype per model
    if "gpt-oss" in hf_id.lower() or "oss-20" in hf_id.lower():
        # OpenAI OSS models prefer bfloat16
        torch_dtype = torch.bfloat16
    else:
        # Default for other models
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    logger.info(f"Loading model '{name}' ({hf_id}) on device: {device} with dtype={torch_dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        device_map=None,  # we move manually
    )
    
    # Resize embeddings if needed
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        logger.info("Resizing token embeddings to match tokenizer length")
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    model.eval()
    
    return tokenizer, model

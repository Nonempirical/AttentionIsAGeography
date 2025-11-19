"""Sampling utilities."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.model_loader import get_device, get_model, get_tokenizer
from src.types import SampledSequence
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _truncate_to_completion(text: str) -> str:
    """
    Heuristic: cut the continuation at the first 'completion-like' boundary.

    We look for:
      - double newline (paragraph break)
      - sentence-ending punctuation followed by a space: ". ", "? ", "! "

    If none found, return the text as-is.
    """
    text = text.strip()
    if not text:
        return text

    # Priority: paragraph break
    para_idx = text.find("\n\n")
    if para_idx != -1:
        return text[:para_idx].strip()

    # Sentence-like boundaries
    for boundary in [". ", "? ", "! "]:
        idx = text.find(boundary)
        if idx != -1:
            # keep the punctuation
            cut = idx + len(boundary.strip())
            return text[:cut].strip()

    return text


def _compute_logprobs_for_generated(
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    prompt_length: int,
) -> List[float]:
    """
    Given the full generated sequence (prompt + new tokens),
    compute log P(token_t | previous tokens) for each newly generated token.

    Args:
        input_ids: (1, L_prompt) original prompt ids
        generated_ids: (1, L_total) ids including prompt + new tokens
        prompt_length: length of original prompt (L_prompt)

    Returns:
        List[float] of length (L_total - L_prompt) with log probs for each new token.
    """
    device = generated_ids.device
    model = get_model()

    # We run the model on the full sequence (prompt + new tokens).
    with torch.no_grad():
        outputs = model(generated_ids)
        logits = outputs.logits  # shape (1, L_total, vocab_size)

    # Compute log-softmax over vocab
    log_probs = F.log_softmax(logits, dim=-1)  # (1, L_total, vocab)

    # For each position t >= prompt_length, we take the logprob
    # of token generated_ids[0, t] conditioned on prefix up to t-1.
    # In standard causal LM, logits at position t-1 predict token at position t.
    new_token_logprobs: List[float] = []
    total_len = generated_ids.size(1)
    for t in range(prompt_length, total_len):
        # logits at position t-1 predict token at position t
        prev_index = t - 1
        token_id = generated_ids[0, t]
        lp = log_probs[0, prev_index, token_id].item()
        new_token_logprobs.append(lp)

    return new_token_logprobs


def sample_futures(
    prompt: str,
    max_new_tokens: int | None = None,
    num_samples: int | None = None,
    completion_mode: bool = False,
    override_tokenizer: Optional[AutoTokenizer] = None,
    override_model: Optional[AutoModelForCausalLM] = None,
) -> List[SampledSequence]:
    """
    Sample num_samples futures of length max_new_tokens from the HF model.
    If max_new_tokens/num_samples are None, fall back to Config defaults.

    Uses nucleus + temperature sampling via model.generate, then computes
    token-aligned logprobs for the generated continuation.
    
    Args:
        override_tokenizer: If provided, use this tokenizer instead of the default
        override_model: If provided, use this model instead of the default
    """
    device = get_device()
    tokenizer = override_tokenizer if override_tokenizer is not None else get_tokenizer()
    model = override_model if override_model is not None else get_model()

    logger.info(f"Sampling futures for prompt: {prompt!r}")

    # Encode prompt
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_length = input_ids.size(1)

    if max_new_tokens is None:
        max_new_tokens = Config.MAX_NEW_TOKENS

    if num_samples is None:
        num_samples = Config.NUM_SAMPLES

    # Generate multiple sequences
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        top_k=Config.TOP_K,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            **gen_kwargs,
        )

    # generated shape: (num_samples, L_total)
    sequences: List[SampledSequence] = []

    for seq_idx in range(num_samples):
        seq_ids = generated[seq_idx : seq_idx + 1]  # shape (1, L_total)

        full_ids = seq_ids[0].tolist()
        prompt_ids = full_ids[:prompt_length]
        completion_ids = full_ids[prompt_length:]

        # Decode texts
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        if completion_mode:
            completion_text = _truncate_to_completion(completion_text)

        # Compute logprobs for generated tokens
        logprobs = _compute_logprobs_for_generated(
            input_ids=input_ids,
            generated_ids=seq_ids,
            prompt_length=prompt_length,
        )
        total_logprob = float(sum(logprobs))

        sequences.append(
            SampledSequence(
                sequence_id=seq_idx,
                prompt=prompt,
                completion=completion_text,
                full_text=full_text,
                logprobs=logprobs,
                total_logprob=total_logprob,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
            )
        )

    return sequences

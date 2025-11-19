"""Sampling utilities."""

from typing import List

import torch
import torch.nn.functional as F

from src.config import Config
from src.model_loader import get_device, get_model, get_tokenizer
from src.types import SampledSequence
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


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


def sample_futures(prompt: str) -> List[SampledSequence]:
    """
    Sample NUM_SAMPLES futures of length MAX_NEW_TOKENS from the HF model.

    Uses nucleus + temperature sampling via model.generate, then computes
    token-aligned logprobs for the generated continuation.
    """
    device = get_device()
    tokenizer = get_tokenizer()
    model = get_model()

    logger.info(f"Sampling futures for prompt: {prompt!r}")

    # Encode prompt
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_length = input_ids.size(1)

    # Generate multiple sequences
    gen_kwargs = dict(
        max_new_tokens=Config.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        top_k=Config.TOP_K,
        num_return_sequences=Config.NUM_SAMPLES,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            **gen_kwargs,
        )

    # generated shape: (NUM_SAMPLES, L_total)
    sequences: List[SampledSequence] = []

    for seq_idx in range(Config.NUM_SAMPLES):
        seq_ids = generated[seq_idx : seq_idx + 1]  # shape (1, L_total)

        full_ids = seq_ids[0].tolist()
        prompt_ids = full_ids[:prompt_length]
        completion_ids = full_ids[prompt_length:]

        # Decode texts
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

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

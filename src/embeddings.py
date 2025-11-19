"""Embedding utilities for sequence representations."""

from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.model_loader import get_tokenizer  # IMPORTANT: reuse LM tokenizer
from src.types import SampledSequence
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model for sequence embeddings.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    # Choose device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info(f"Loading embedding model on device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    _embedding_model = model
    return _embedding_model


def embed_full_sequences(samples: List[SampledSequence]) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full_text of each SampledSequence into a high-dimensional vector.

    Returns:
        embeddings: np.ndarray of shape (N, D)
        seq_ids: list of sequence_ids in the same order
    """
    model = get_embedding_model()
    texts = [s.full_text for s in samples]
    seq_ids = [s.sequence_id for s in samples]

    # model.encode returns a numpy array by default, we can ensure that:
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings, seq_ids


def embed_prefixes(
    samples: List[SampledSequence],
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Compute embeddings for tokenizer-aligned prefixes along each sampled sequence.

    For each SampledSequence:
        - We use prompt_token_ids + first k completion_token_ids as the prefix.
        - We decode those ids with the SAME tokenizer used by the LM.
        - The resulting string is fed into the SentenceTransformer embedding model.

    Returns:
        prefix_embeddings: np.ndarray of shape (M, D), where M = total number of prefixes
        seq_ids: List[int] length M, sequence_id for each prefix
        positions: List[int] length M, position index within the completion (1..len(completion_token_ids))
    """
    model = get_embedding_model()
    tokenizer = get_tokenizer()

    texts: List[str] = []
    seq_ids: List[int] = []
    positions: List[int] = []

    for sample in samples:
        if not sample.completion_token_ids:
            continue

        # We will build prefixes purely via token IDs, no .split() hacks.
        prompt_ids = sample.prompt_token_ids
        completion_ids = sample.completion_token_ids

        for pos in range(1, len(completion_ids) + 1):
            prefix_ids = prompt_ids + completion_ids[:pos]
            # Decode with the LM tokenizer so prefix boundaries reflect exactly what the model sees
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

            texts.append(prefix_text)
            seq_ids.append(sample.sequence_id)
            positions.append(pos)

    if not texts:
        logger.warning("embed_prefixes: no prefixes to embed (empty texts).")
        return np.zeros((0, 0), dtype=np.float32), [], []

    logger.info(f"Embedding {len(texts)} prefixes with SentenceTransformer")
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings, seq_ids, positions


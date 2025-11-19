"""Type definitions for the project."""

from dataclasses import dataclass
from typing import List


@dataclass
class SampledSequence:
    sequence_id: int
    prompt: str
    completion: str  # generated text only
    full_text: str  # prompt + completion
    logprobs: List[float]  # per-generated-token logprobs
    total_logprob: float  # sum(logprobs)

    prompt_token_ids: List[int]
    completion_token_ids: List[int]


@dataclass
class RiverPoint:
    """
    Represents a prefix location in the 2D semantic plane.

    Used to draw semantic rivers (paths) per sequence.
    """
    sequence_id: int
    position: int  # token index in the completion (1..MAX_NEW_TOKENS)
    x: float
    y: float
    weight: float  # e.g. normalized prefix probability or reuse sequence weight


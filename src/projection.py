"""2D projection utilities for embedding visualization."""

from typing import Any, List, Tuple

import numpy as np
import umap
from sklearn.decomposition import PCA

from src.config import Config
from src.types import RiverPoint, SampledSequence
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _compute_normalized_weights(samples: List[SampledSequence]) -> np.ndarray:
    """
    Compute normalized probability weights from total_logprob for each sample.

    Uses log-sum-exp trick for numerical stability.
    """
    logprobs = np.array([s.total_logprob for s in samples], dtype=np.float64)
    max_lp = np.max(logprobs)
    exp_scores = np.exp(logprobs - max_lp)
    weights = exp_scores / np.sum(exp_scores)
    return weights


def project_to_2d(
    embeddings: np.ndarray,
    samples: List[SampledSequence],
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Project high-dimensional embeddings to 2D semantic coordinates.

    Args:
        embeddings: (N, D) array
        samples: list of SampledSequence of length N

    Returns:
        coords: (N, 2) array of (x, y) positions
        weights: (N,) array of normalized probability weights. These are normalized
            probabilities derived from total_logprob using the log-sum-exp trick.
            They are intended to be used directly in build_probability_field.
        projector: fitted PCA or UMAP object with a .transform() method
    """
    assert embeddings.shape[0] == len(samples), "Mismatch N between embeddings and samples"

    # Compute weights from total_logprob
    weights = _compute_normalized_weights(samples)

    method = getattr(Config, "PROJECTION_METHOD", "umap").lower()
    logger.info(f"Projecting embeddings to 2D using method: {method}")

    if method == "pca":
        projector = PCA(n_components=2)
        coords = projector.fit_transform(embeddings)
    else:
        # Default: UMAP
        projector = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=Config.SEED,
        )
        coords = projector.fit_transform(embeddings)

    return coords, weights, projector


def project_prefixes_to_2d(
    prefix_embeddings: np.ndarray,
    seq_ids: List[int],
    positions: List[int],
    projector: Any,
    full_weights: np.ndarray,
) -> List[RiverPoint]:
    """
    Project prefix embeddings into the same 2D space using an already-fitted projector
    (PCA or UMAP).

    Args:
        prefix_embeddings: (M, D) array of prefix embeddings.
        seq_ids: list of length M, sequence_id for each prefix.
        positions: list of length M, prefix position index (1..len(completion)).
        projector: fitted PCA or UMAP object with a .transform() method.
        full_weights: (N,) array of normalized sequence-level weights for full sequences.

    Returns:
        List[RiverPoint] for all prefixes.
    """
    if prefix_embeddings.shape[0] == 0:
        return []

    logger.info(f"Projecting {prefix_embeddings.shape[0]} prefixes to 2D")

    coords = projector.transform(prefix_embeddings)  # (M, 2)

    # We need a mapping seq_id -> weight. Assume full_weights are in the same order
    # as the SampledSequence list used to create them and that seq_ids are 0..N-1.
    # For now, build a simple mapping seq_id -> weight index = seq_id.
    # (We can refine later if sequence_id is not contiguous.)
    seq_to_weight = {}
    for seq_id, w in enumerate(full_weights):
        seq_to_weight[seq_id] = float(w)

    river_points: List[RiverPoint] = []
    for i in range(len(seq_ids)):
        seq_id = seq_ids[i]
        pos = positions[i]
        x, y = float(coords[i, 0]), float(coords[i, 1])

        # Use the full sequence weight as the base weight for all its prefixes.
        # Later we can refine this to prefix-specific probabilities if needed.
        w = seq_to_weight.get(seq_id, 0.0)

        river_points.append(
            RiverPoint(
                sequence_id=seq_id,
                position=pos,
                x=x,
                y=y,
                weight=w,
            )
        )

    return river_points


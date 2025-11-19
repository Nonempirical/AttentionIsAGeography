"""Landscape grid and probability field computation."""

from typing import Tuple

import numpy as np

from src.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_grid(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given 2D coordinates of shape (N, 2), construct a regular grid (X, Y)
    covering the semantic plane with some margin.

    Returns:
        X, Y: meshgrid arrays of shape (G, G), where G = Config.GRID_SIZE.
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be (N, 2)"

    x = coords[:, 0]
    y = coords[:, 1]

    # Compute ranges with margin
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0.0:
        x_range = 1.0
    if y_range == 0.0:
        y_range = 1.0

    margin_x = x_range * Config.GRID_MARGIN
    margin_y = y_range * Config.GRID_MARGIN

    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    grid_size = Config.GRID_SIZE
    x_lin = np.linspace(x_min, x_max, grid_size)
    y_lin = np.linspace(y_min, y_max, grid_size)

    X, Y = np.meshgrid(x_lin, y_lin)
    logger.info(
        f"Grid created with size={grid_size} "
        f"and x in [{x_min:.3f}, {x_max:.3f}], y in [{y_min:.3f}, {y_max:.3f}]"
    )
    return X, Y


def build_probability_field(
    coords: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a continuous probability field Z(X, Y) over the semantic plane.

    Args:
        coords: (N, 2) array of (x_i, y_i) positions for each future.
        weights: (N,) array of normalized probability weights (sum ~ 1).

    Returns:
        X, Y, Z:
          - X, Y: meshgrid arrays of shape (G, G)
          - Z: probability density / field of shape (G, G)
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be (N, 2)"
    assert weights.ndim == 1 and weights.shape[0] == coords.shape[0], (
        "weights must be (N,) aligned with coords"
    )

    X, Y = compute_grid(coords)

    # Flatten grid for vectorized distance computation
    # X_flat, Y_flat have shape (G*G,)
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # coords_x, coords_y have shape (N, 1)
    coords_x = coords[:, 0][:, np.newaxis]
    coords_y = coords[:, 1][:, np.newaxis]

    # grid_x, grid_y have shape (1, G*G)
    grid_x = X_flat[np.newaxis, :]
    grid_y = Y_flat[np.newaxis, :]

    # Squared distance between each coord and each grid point: (N, G*G)
    dist2 = (coords_x - grid_x) ** 2 + (coords_y - grid_y) ** 2

    sigma = Config.KERNEL_SIGMA
    if sigma <= 0.0:
        raise ValueError("Config.KERNEL_SIGMA must be positive")

    # Gaussian kernel contribution: (N, G*G)
    kernel = np.exp(-dist2 / (2.0 * sigma ** 2))

    # Multiply each row by the corresponding weight
    # weights[:, None] -> (N, 1) for broadcasting
    weighted_kernel = weights[:, np.newaxis] * kernel

    # Sum over all points to get Z_flat of shape (G*G,)
    Z_flat = np.sum(weighted_kernel, axis=0)

    Z = Z_flat.reshape(X.shape)

    if Config.NORMALIZE_FIELD:
        total = np.sum(Z)
        if total > 0.0:
            Z = Z / total
            logger.info("Probability field normalized to sum to 1.0 over grid.")

    return X, Y, Z


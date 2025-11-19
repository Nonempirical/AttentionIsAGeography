"""Visualization utilities for geography plots."""

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go

from src.config import Config
from src.types import RiverPoint
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _interpolate_z_nearest(
    coords: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Approximate Z values for each (x, y) by taking the value
    of the nearest grid point in X, Y.

    Args:
        coords: (N, 2) array of (x, y) positions.
        X, Y, Z: meshgrid arrays of shape (G, G).

    Returns:
        z_vals: (N,) array of approximate Z values.
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be (N, 2)"
    x_grid = X[0, :]  # shape (G,)
    y_grid = Y[:, 0]  # shape (G,)

    z_vals = np.zeros(coords.shape[0], dtype=float)

    for i, (x, y) in enumerate(coords):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        z_vals[i] = float(Z[iy, ix])

    return z_vals


def make_geography_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    rivers: Dict[int, List[RiverPoint]],
) -> go.Figure:
    """
    Construct the 3D Plotly figure for "Attention Is a Geography":

      - Surface: probability landscape Z(X, Y)
      - Endpoints: one point per sampled future
      - Rivers: semantic prefix paths per sequence

    Args:
        X, Y, Z: meshgrid arrays from build_probability_field
        coords: (N, 2) array of full-sequence coordinates
        weights: (N,) array of normalized probabilities
        rivers: dict seq_id -> ordered list of RiverPoint

    Returns:
        Plotly Figure object.
    """
    logger.info("Building Plotly figure for geography visualization")

    # --- Surface ---
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        showscale=True,
        opacity=0.8,
        name="Probability landscape",
    )

    # --- Endpoint markers ---
    z_endpoints = _interpolate_z_nearest(coords, X, Y, Z)

    marker_sizes = 6.0 + 14.0 * (weights / (weights.max() + 1e-8))

    endpoints = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=z_endpoints,
        mode="markers",
        marker=dict(
            size=marker_sizes,
            opacity=0.9,
        ),
        name="Futures",
        text=[f"w={w:.4f}" for w in weights],
        hoverinfo="text",
    )

    # --- Rivers ---
    river_traces: List[go.Scatter3d] = []

    for seq_id, points in rivers.items():
        if len(points) < 2:
            continue

        xs = np.array([p.x for p in points], dtype=float)
        ys = np.array([p.y for p in points], dtype=float)
        coords_path = np.column_stack([xs, ys])
        zs = _interpolate_z_nearest(coords_path, X, Y, Z)

        river_traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=2),
                name=f"river_{seq_id}",
                hoverinfo="none",
                showlegend=False,
            )
        )

    # --- Figure assembly ---
    data = [surface, endpoints] + river_traces

    figure_title = getattr(Config, "FIGURE_TITLE", "Attention Is a Geography")
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=figure_title,
            scene=dict(
                xaxis_title="Semantic X",
                yaxis_title="Semantic Y",
                zaxis_title="Probability",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        ),
    )

    return fig


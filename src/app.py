"""Gradio application for Attention Is a Geography visualization."""

from typing import Any

import gradio as gr

from src.config import Config
from src.embeddings import embed_full_sequences, embed_prefixes
from src.landscape import build_probability_field
from src.projection import project_prefixes_to_2d, project_to_2d
from src.rivers import group_rivers_by_sequence
from src.sampling import sample_futures
from src.utils.logging_utils import get_logger
from src.viz import make_geography_figure

logger = get_logger(__name__)


def build_geography_for_prompt(prompt: str) -> Any:
    """
    High-level pipeline:

      1) Sample futures from the language model.
      2) Embed full sequences.
      3) Project to 2D and compute weights.
      4) Build probability field over 2D plane.
      5) Embed prefixes.
      6) Project prefixes to 2D using same projector.
      7) Group prefixes into semantic rivers.
      8) Build Plotly 3D figure.
    """
    logger.info(f"Building geography for prompt: {prompt!r}")

    # 1) Sample futures
    samples = sample_futures(prompt)
    if not samples:
        logger.warning("No samples returned from sampling.")
        return None

    # 2) Embed full sequences
    embeddings, seq_ids = embed_full_sequences(samples)

    # 3) Project to 2D
    coords, weights, projector = project_to_2d(embeddings, samples)

    # 4) Probability field
    X, Y, Z = build_probability_field(coords, weights)

    # 5) Embed prefixes
    prefix_embeddings, prefix_seq_ids, positions = embed_prefixes(samples)

    # 6) Project prefixes to 2D
    from src.types import RiverPoint  # ensure imported; not strictly needed here

    river_points = project_prefixes_to_2d(
        prefix_embeddings,
        prefix_seq_ids,
        positions,
        projector,
        weights,
    )

    # 7) Group into rivers
    rivers = group_rivers_by_sequence(river_points)

    # Build hover texts for futures: short preview + weight
    hover_texts = []
    for sample, w in zip(samples, weights):
        # Shorten long completions for hover
        snippet = sample.completion.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        hover_texts.append(
            f"weight={w:.4f}<br><br><b>Prompt:</b> {sample.prompt}<br><br><b>Completion:</b> {snippet}"
        )

    # 8) Build figure
    fig = make_geography_figure(X, Y, Z, coords, weights, rivers, hover_texts)
    return fig


def create_gradio_app() -> gr.Interface:
    """
    Create a Gradio interface that:
      - Takes a prompt.
      - Runs build_geography_for_prompt.
      - Shows the 3D Plotly figure.
    """
    def _wrapped(prompt: str):
        fig = build_geography_for_prompt(prompt)
        return fig

    iface = gr.Interface(
        fn=_wrapped,
        inputs=gr.Textbox(
            lines=2,
            label="Prompt",
            value="The dog",
            placeholder="Type a prompt for the language model...",
        ),
        outputs=gr.Plot(label="Attention Is a Geography"),
        title="Attention Is a Geography",
        description=(
            "Visualizing semantic futures and probability landscape ~20 tokens ahead "
            "for a given prompt."
        ),
    )
    return iface


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()


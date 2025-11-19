"""Gradio application for Attention Is a Geography visualization."""

from typing import Any, Optional, Tuple

import numpy as np
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config, MODEL_REGISTRY
from src.embeddings import embed_full_sequences, embed_prefixes
from src.landscape import build_probability_field
from src.projection import project_prefixes_to_2d, project_to_2d
from src.rivers import group_rivers_by_sequence
from src.model_loader import load_model_by_name
from src.sampling import sample_futures
from src.types import RiverPoint, SampledSequence
from src.utils.logging_utils import get_logger
from src.viz import make_geography_figure

logger = get_logger(__name__)


def _compute_geography_components(
    prompt: str,
    max_new_tokens: int | None = None,
    num_samples: int | None = None,
    completion_mode: bool = False,
    override_tokenizer: Optional[AutoTokenizer] = None,
    override_model: Optional[AutoModelForCausalLM] = None,
) -> Tuple[
    np.ndarray,  # X
    np.ndarray,  # Y
    np.ndarray,  # Z
    np.ndarray,  # coords
    np.ndarray,  # weights
    dict[int, list[RiverPoint]],  # rivers
    list[SampledSequence],  # samples
]:
    """
    Core pipeline without figure construction.
    Returns all components needed to build figures or do comparisons.
    """
    logger.info(f"Building geography components for prompt: {prompt!r}")

    # 1) Sample futures
    samples = sample_futures(
        prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        completion_mode=completion_mode,
        override_tokenizer=override_tokenizer,
        override_model=override_model,
    )
    if not samples:
        logger.warning("No samples returned from sampling.")
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            {},
            [],
        )

    # 2) Embed full sequences
    embeddings, seq_ids = embed_full_sequences(samples)

    # 3) Project to 2D
    coords, weights, projector = project_to_2d(embeddings, samples)

    # 4) Probability field
    X, Y, Z = build_probability_field(coords, weights)

    # 5) Embed prefixes
    prefix_embeddings, prefix_seq_ids, positions = embed_prefixes(samples)

    # 6) Project prefixes to 2D
    river_points = project_prefixes_to_2d(
        prefix_embeddings,
        prefix_seq_ids,
        positions,
        projector,
        weights,
    )

    # 7) Group rivers
    rivers = group_rivers_by_sequence(river_points)

    return X, Y, Z, coords, weights, rivers, samples


def build_geography_for_prompt(
    prompt: str,
    max_new_tokens: int | None = None,
    num_samples: int | None = None,
    completion_mode: bool = False,
    show_rivers: bool = True,
    override_tokenizer: Optional[AutoTokenizer] = None,
    override_model: Optional[AutoModelForCausalLM] = None,
) -> Any:
    """
    High-level pipeline for a single prompt.
    """
    (
        X,
        Y,
        Z,
        coords,
        weights,
        rivers,
        samples,
    ) = _compute_geography_components(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        completion_mode=completion_mode,
        override_tokenizer=override_tokenizer,
        override_model=override_model,
    )

    if X.size == 0:
        return None

    # Build hover texts
    hover_texts: list[str] = []
    for sample, w in zip(samples, weights):
        snippet = sample.completion.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        hover_texts.append(
            f"weight={w:.4f}<br><br><b>Prompt:</b> {sample.prompt}<br><br><b>Completion:</b> {snippet}"
        )

    fig = make_geography_figure(
        X,
        Y,
        Z,
        coords,
        weights,
        rivers,
        hover_texts,
        show_rivers=show_rivers,
    )
    return fig


def build_geography_compare(
    prompt: str,
    model_name_a: str,
    model_name_b: str,
    max_new_tokens: int | None = None,
    num_samples: int | None = None,
    completion_mode: bool = False,
    show_rivers: bool = True,
) -> tuple[Any, Any]:
    """
    Build comparative geographies for two models on the same prompt.

    Returns:
        fig_a: 3D figure for model A
        fig_b: 3D figure for model B
    """
    # Load both models
    tok_a, model_a = load_model_by_name(model_name_a)
    tok_b, model_b = load_model_by_name(model_name_b)

    # Compute both geographies
    fig_a = build_geography_for_prompt(
        prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        completion_mode=completion_mode,
        show_rivers=show_rivers,
        override_tokenizer=tok_a,
        override_model=model_a,
    )

    fig_b = build_geography_for_prompt(
        prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        completion_mode=completion_mode,
        show_rivers=show_rivers,
        override_tokenizer=tok_b,
        override_model=model_b,
    )

    return fig_a, fig_b


def build_geography_diff(
    prompt_a: str,
    prompt_b: str,
    max_new_tokens: int | None = None,
    num_samples: int | None = None,
) -> tuple[Any, Any, Any]:
    """
    Build comparative geographies for two prompts A and B.

    Returns:
        fig_a: 3D figure for prompt A
        fig_b: 3D figure for prompt B
        fig_diff: 3D figure of Z_B - Z_A on shared grid
    """
    # Compute components for both prompts
    (
        X_a,
        Y_a,
        Z_a,
        coords_a,
        weights_a,
        rivers_a,
        samples_a,
    ) = _compute_geography_components(
        prompt=prompt_a,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
    )

    (
        X_b,
        Y_b,
        Z_b,
        coords_b,
        weights_b,
        rivers_b,
        samples_b,
    ) = _compute_geography_components(
        prompt=prompt_b,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
    )

    if X_a.size == 0 or X_b.size == 0:
        return None, None, None

    # For simplicity, assume grids are compatible (same shape & ranges).
    # If not, we could resample, but for now we rely on similar sampling.
    if X_a.shape != X_b.shape:
        logger.warning("Grid shapes differ between A and B; cannot compute diff surface reliably.")
        X_a, Y_a, Z_a = X_b, Y_b, Z_b  # crude fallback

    # Hover texts
    hover_a: list[str] = []
    for sample, w in zip(samples_a, weights_a):
        snippet = sample.completion.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        hover_a.append(
            f"[A] weight={w:.4f}<br><br><b>Prompt:</b> {sample.prompt}<br><br><b>Completion:</b> {snippet}"
        )

    hover_b: list[str] = []
    for sample, w in zip(samples_b, weights_b):
        snippet = sample.completion.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        hover_b.append(
            f"[B] weight={w:.4f}<br><br><b>Prompt:</b> {sample.prompt}<br><br><b>Completion:</b> {snippet}"
        )

    # Figures for A and B
    fig_a = make_geography_figure(X_a, Y_a, Z_a, coords_a, weights_a, rivers_a, hover_a)
    fig_a.update_layout(title=f"Geography A: {prompt_a}")

    fig_b = make_geography_figure(X_b, Y_b, Z_b, coords_b, weights_b, rivers_b, hover_b)
    fig_b.update_layout(title=f"Geography B: {prompt_b}")

    # Difference surface Z_diff = Z_b - Z_a
    Z_diff = Z_b - Z_a

    import plotly.graph_objects as go

    surface_diff = go.Surface(
        x=X_a,
        y=Y_a,
        z=Z_diff,
        showscale=True,
        colorscale="RdBu",
        name="Z_B - Z_A",
    )

    fig_diff = go.Figure(
        data=[surface_diff],
        layout=go.Layout(
            title="Difference geography (B - A)",
            scene=dict(
                xaxis_title="Semantic X",
                yaxis_title="Semantic Y",
                zaxis_title="ΔProbability",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        ),
    )

    return fig_a, fig_b, fig_diff


def create_gradio_app() -> gr.Interface:
    """
    Create a Gradio interface that:
      - Takes a prompt.
      - Runs build_geography_for_prompt.
      - Shows the 3D Plotly figure.
    """
    def _wrapped(
        prompt: str,
        max_new_tokens: int,
        num_samples: int,
        completion_mode: bool,
        show_rivers: bool,
    ):
        fig = build_geography_for_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            completion_mode=completion_mode,
            show_rivers=show_rivers,
        )
        return fig

    iface = gr.Interface(
        fn=_wrapped,
        inputs=[
            gr.Textbox(
                lines=2,
                label="Prompt",
                value="The dog",
                placeholder="Type a prompt for the language model...",
            ),
            gr.Slider(
                minimum=5,
                maximum=60,
                step=5,
                value=Config.MAX_NEW_TOKENS,
                label="Max new tokens (horizon)",
            ),
            gr.Slider(
                minimum=8,
                maximum=256,
                step=8,
                value=Config.NUM_SAMPLES,
                label="Number of futures (samples)",
            ),
            gr.Checkbox(
                label="Completion mode (stop at first answer/sentence)",
                value=False,
            ),
            gr.Checkbox(
                label="Show semantic rivers",
                value=True,
            ),
        ],
        outputs=gr.Plot(label="Attention Is a Geography"),
        title="Attention Is a Geography",
        description=(
            "Visualizing semantic futures and probability landscape N tokens ahead "
            "for a given prompt."
        ),
    )
    return iface


def create_model_compare_app() -> gr.Interface:
    """
    Gradio interface to compare geographies for two models on the same prompt.
    Returns two plots: fig_A, fig_B.
    """
    def _wrapped_compare(
        prompt: str,
        model_name_a: str,
        model_name_b: str,
        max_new_tokens: int,
        num_samples: int,
        completion_mode: bool,
        show_rivers: bool,
    ):
        figs = build_geography_compare(
            prompt=prompt,
            model_name_a=model_name_a,
            model_name_b=model_name_b,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            completion_mode=completion_mode,
            show_rivers=show_rivers,
        )
        return figs

    iface = gr.Interface(
        fn=_wrapped_compare,
        inputs=[
            gr.Textbox(
                lines=2,
                label="Prompt",
                value="The dog",
                placeholder="Type a prompt for the language model...",
            ),
            gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                label="Model A",
                value=list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else None,
            ),
            gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                label="Model B",
                value=list(MODEL_REGISTRY.keys())[1] if len(MODEL_REGISTRY) > 1 else list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else None,
            ),
            gr.Slider(
                minimum=5,
                maximum=60,
                step=5,
                value=Config.MAX_NEW_TOKENS,
                label="Max new tokens (horizon)",
            ),
            gr.Slider(
                minimum=8,
                maximum=256,
                step=8,
                value=Config.NUM_SAMPLES,
                label="Number of futures (samples)",
            ),
            gr.Checkbox(
                label="Completion mode (stop at first answer/sentence)",
                value=False,
            ),
            gr.Checkbox(
                label="Show semantic rivers",
                value=True,
            ),
        ],
        outputs=[
            gr.Plot(label="Geography Model A"),
            gr.Plot(label="Geography Model B"),
        ],
        title="Attention Is a Geography — Model Comparison",
        description=(
            "Compare the semantic probability landscapes of two models on the same prompt. "
            "See how different models explore the semantic space differently."
        ),
    )
    return iface


def create_gradio_compare_app() -> gr.Interface:
    """
    Gradio interface to compare geographies for two prompts A and B.
    Returns three plots: fig_A, fig_B, fig_diff.
    """

    def _wrapped(prompt_a: str, prompt_b: str, max_new_tokens: int, num_samples: int):
        figs = build_geography_diff(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
        )
        return figs

    iface = gr.Interface(
        fn=_wrapped,
        inputs=[
            gr.Textbox(
                lines=2,
                label="Prompt A",
                value="The dog chased the ball",
            ),
            gr.Textbox(
                lines=2,
                label="Prompt B",
                value="The cat climbed the tree",
            ),
            gr.Slider(
                minimum=5,
                maximum=60,
                step=5,
                value=Config.MAX_NEW_TOKENS,
                label="Max new tokens (horizon)",
            ),
            gr.Slider(
                minimum=8,
                maximum=256,
                step=8,
                value=Config.NUM_SAMPLES,
                label="Number of futures (samples)",
            ),
        ],
        outputs=[
            gr.Plot(label="Geography A"),
            gr.Plot(label="Geography B"),
            gr.Plot(label="Difference (B - A)"),
        ],
        title="Attention Is a Geography — Prompt Comparison",
        description=(
            "Compare the semantic probability landscapes of two prompts. "
            "The difference plot shows where probability mass shifts from A to B."
        ),
    )
    return iface


def create_unified_app() -> gr.Blocks:
    """
    Create a unified Gradio Blocks app with tabs for single model and model comparison.
    """
    with gr.Blocks(title="Attention Is a Geography") as demo:
        gr.Markdown("# Attention Is a Geography")
        gr.Markdown("Visualize semantic futures and probability landscapes for language models.")
        
        with gr.Tab("Single Model"):
            with gr.Row():
                prompt_input = gr.Textbox(
                    lines=2,
                    label="Prompt",
                    value="The dog",
                    placeholder="Type a prompt for the language model...",
                )
            with gr.Row():
                max_tokens_input = gr.Slider(
                    minimum=5,
                    maximum=60,
                    step=5,
                    value=Config.MAX_NEW_TOKENS,
                    label="Max new tokens (horizon)",
                )
                num_samples_input = gr.Slider(
                    minimum=8,
                    maximum=256,
                    step=8,
                    value=Config.NUM_SAMPLES,
                    label="Number of futures (samples)",
                )
            with gr.Row():
                completion_mode_input = gr.Checkbox(
                    label="Completion mode (stop at first answer/sentence)",
                    value=False,
                )
                show_rivers_input = gr.Checkbox(
                    label="Show semantic rivers",
                    value=True,
                )
            single_output = gr.Plot(label="Attention Is a Geography")
            single_button = gr.Button("Generate Geography", variant="primary")
            
            def single_wrapped(prompt, max_tokens, num_samples, completion_mode, show_rivers):
                return build_geography_for_prompt(
                    prompt,
                    max_new_tokens=max_tokens,
                    num_samples=num_samples,
                    completion_mode=completion_mode,
                    show_rivers=show_rivers,
                )
            
            single_button.click(
                fn=single_wrapped,
                inputs=[prompt_input, max_tokens_input, num_samples_input, completion_mode_input, show_rivers_input],
                outputs=single_output,
            )
        
        with gr.Tab("Compare Models"):
            with gr.Row():
                compare_prompt_input = gr.Textbox(
                    lines=2,
                    label="Prompt",
                    value="The dog",
                    placeholder="Type a prompt for the language model...",
                )
            with gr.Row():
                model_a_input = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    label="Model A",
                    value=list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else None,
                )
                model_b_input = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    label="Model B",
                    value=list(MODEL_REGISTRY.keys())[1] if len(MODEL_REGISTRY) > 1 else list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else None,
                )
            with gr.Row():
                compare_max_tokens_input = gr.Slider(
                    minimum=5,
                    maximum=60,
                    step=5,
                    value=Config.MAX_NEW_TOKENS,
                    label="Max new tokens (horizon)",
                )
                compare_num_samples_input = gr.Slider(
                    minimum=8,
                    maximum=256,
                    step=8,
                    value=Config.NUM_SAMPLES,
                    label="Number of futures (samples)",
                )
            with gr.Row():
                compare_completion_mode_input = gr.Checkbox(
                    label="Completion mode (stop at first answer/sentence)",
                    value=False,
                )
                compare_show_rivers_input = gr.Checkbox(
                    label="Show semantic rivers",
                    value=True,
                )
            with gr.Row():
                compare_output_a = gr.Plot(label="Geography Model A")
                compare_output_b = gr.Plot(label="Geography Model B")
            compare_button = gr.Button("Compare Models", variant="primary")
            
            def compare_wrapped(prompt, model_a, model_b, max_tokens, num_samples, completion_mode, show_rivers):
                return build_geography_compare(
                    prompt=prompt,
                    model_name_a=model_a,
                    model_name_b=model_b,
                    max_new_tokens=max_tokens,
                    num_samples=num_samples,
                    completion_mode=completion_mode,
                    show_rivers=show_rivers,
                )
            
            compare_button.click(
                fn=compare_wrapped,
                inputs=[
                    compare_prompt_input,
                    model_a_input,
                    model_b_input,
                    compare_max_tokens_input,
                    compare_num_samples_input,
                    compare_completion_mode_input,
                    compare_show_rivers_input,
                ],
                outputs=[compare_output_a, compare_output_b],
            )
    
    return demo


if __name__ == "__main__":
    # Launch unified app with tabs
    demo = create_unified_app()
    demo.launch()


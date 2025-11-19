# Attention Is a Geography

**Visualizing how language models explore semantic space across multiple possible futures.**

This project creates interactive 3D visualizations showing how a language model's context branches into many semantic futures. It samples multiple completions from a prompt, embeds them in semantic space, and visualizes the probability landscape as a 3D geography with "rivers" showing how each sequence evolves.

## What It Does

When you give a language model a prompt, it has a probability distribution over many possible futures. **Attention Is a Geography** visualizes this as a 3D landscape:

- **X, Y axes**: Semantic coordinates (2D projection of embedding space)
- **Z axis**: Probability density
- **Surface**: Probability landscape showing where the model's probability mass is concentrated
- **Endpoints**: Individual sampled completions, sized and colored by probability weights
- **Rivers**: Semantic paths showing how each sequence evolves token-by-token

**Features:**
- Single model visualization
- Model comparison (compare different models on the same prompt)
- Interactive controls for sampling parameters and visualization options

## How It Works

1. **Sampling**: Generates multiple completions (default: 64) using nucleus sampling
2. **Embedding**: Converts completions to high-dimensional vectors using SentenceTransformer
3. **Projection**: Projects to 2D using UMAP (or PCA) to create semantic coordinates
4. **Probability Field**: Builds a continuous landscape using Gaussian kernels
5. **Rivers**: Traces semantic paths by embedding prefixes at each token position
6. **Visualization**: Creates an interactive 3D Plotly figure

## Installation

```bash
git clone YOUR_REPO_URL.git
cd attention-is-a-geography
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch, Transformers, Sentence Transformers, UMAP-learn, Plotly, Gradio

## Usage

### Interactive Web Interface

```bash
python -m src.app
```

Opens a web interface with two tabs:
- **Single Model**: Visualize one model's semantic exploration
- **Compare Models**: Side-by-side comparison of different models

### Programmatic Usage

```python
from src.app import build_geography_for_prompt, build_geography_compare

# Single model
fig = build_geography_for_prompt(
    prompt="The dog",
    max_new_tokens=20,
    num_samples=64,
    show_rivers=True,
)

# Compare models
fig_a, fig_b = build_geography_compare(
    prompt="The dog",
    model_name_a="LLaMA-3 8B",
    model_name_b="Phi-3",
)
```

## Configuration

Edit `src/config.py` to customize sampling parameters, projection method, landscape settings, and add models to `MODEL_REGISTRY`.

## Model Registry

Pre-configured models:
- **Phi-3**: `microsoft/Phi-3-mini-4k-instruct`
- **LLaMA-3 8B**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Mistral 7B**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Qwen2 7B**: `Qwen/Qwen2-7B-Instruct`

Add more models by editing `MODEL_REGISTRY` in `src/config.py`.

## Google Colab

See `notebooks/colab_entry.ipynb` for setup instructions.

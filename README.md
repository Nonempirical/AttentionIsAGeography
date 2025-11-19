# Attention Is a Geography

This project visualizes how a language model's context can branch into many semantic futures 20+ tokens ahead. It uses a local LLM (~5 GB, stored in Google Drive) from Google Colab to sample futures and analyze / visualize them.

## Quickstart (Colab)

1. Git clone this repo in Colab.
2. Mount Google Drive.
3. Set `MODEL_DIR` in `config.py` to point to the Drive path.
4. Run `python -m src.sampling` or launch the Gradio app (later).

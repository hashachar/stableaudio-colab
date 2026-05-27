# Stable Audio 3 — Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb)

Text-to-music generation and audio inpainting with Stability AI's open-weight Stable Audio 3 models, running entirely in Google Colab.

## Open the notebook

**→ https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb**

This link always opens the latest version of the notebook directly in Colab — no download needed.

## Models

| Model | Params | Max Duration | Hardware |
|---|---|---|---|
| `small-music` | 433 M | 120 s | CPU or GPU |
| `medium` | 1.4 B | 285 s | GPU required (A100 / L4) |

## Features

- **Text-to-music** — generate audio from a text prompt
- **Inpainting / audio editing** — replace a time segment with new content described by a prompt
- Google Drive cache — model weights persist across sessions
- Session audio registry — generated clips are available as inpainting sources

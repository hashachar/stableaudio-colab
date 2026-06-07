# Stable Audio 3 — Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb)

Make music from a text prompt, edit it, remix it, and morph clips into each other — all in your browser with Stability AI's open-weight Stable Audio 3 models. No install, no coding.

## Open the notebook

**→ https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb**

This link always opens the latest version in Colab. Run the short setup cells once (pick a GPU runtime if you can), then jump to whichever tool you want. Each tool has its own controls and a single run button.

## What you can do

### 🎵 Text-to-music
Describe what you want — *"warm lo-fi piano with vinyl crackle, 70 bpm"* — and get audio back. You set the length, how random each take is, and how closely it sticks to your words.

### ✂️ Inpainting (audio editing)
Load a clip, mark a start and end time, and replace just that slice with something new from a prompt. Everything outside the marked region stays exactly as it was — handy for fixing a rough section or dropping in a new phrase.

### 🔀 Audio-to-Audio (variations)
Feed in a clip and get a fresh take that keeps its overall musical feel. A single "how different" dial takes you from a subtle remix all the way to a loose reinterpretation.

### 🎚️ RF-Inversion (re-style)
Re-skin a clip in a new style while keeping its underlying structure — the rhythm and shape of the original stay, but the character changes. It does this by first working out the exact "noise fingerprint" the model would need to recreate your clip, then regenerating from that fingerprint with your new prompt. Two dials let you trade off how faithful the result stays versus how freely it follows the prompt.

### 🌅 Audio Transition (morph)
Blend two clips into one smooth A→B transition. Choose how long the crossover lasts and the shape of its curve. An optional repair pass cleans up the seam — either a quick smoothing, or (with a toggle) the more structure-preserving RF-Inversion method.

## Models

| Model | Size | Max length | Hardware |
|---|---|---|---|
| `small-music` | 433M | 120s | CPU or GPU |
| `medium` | 1.4B | 285s | GPU (A100 / L4) |

`small-music` runs anywhere and is great for quick sketches. `medium` sounds noticeably better and is the one to use for the highest-quality results.

## Good to know

- **One-time setup:** accept the model license on Hugging Face and add an `HF_TOKEN` — the notebook walks you through both steps.
- **Cached weights:** model files are saved to your Google Drive, so you don't re-download them every session (the first run takes 1–3 minutes).
- **Session clips:** anything you generate is added to a running list you can reuse as the input for the other tools.

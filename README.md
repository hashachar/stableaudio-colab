# Stable Audio 3 — Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb)

Make music from a text prompt, edit it, remix it, extend it, loop it, and morph clips into each other — all in your browser with Stability AI's open-weight Stable Audio 3 models. No install, no coding.

## Open the notebook

**→ https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb**

This link always opens the latest version in Colab. Run the short setup cells once (pick a GPU runtime if you can), then jump to whichever tool you want. Each tool has its own controls and a single run button.

**Full documentation:** [GUIDE.md](GUIDE.md) — quick start, prompt-engineering guide, every parameter explained, performance tips, FAQ, and troubleshooting. Design history lives in [DECISIONS.md](DECISIONS.md).

All six tools run on the same latent rectified-flow diffusion model: your audio
is encoded into a compact **latent** representation, the model denoises it, and
the result is decoded back to a waveform. Most tools share a few controls:

- **Prompt / negative prompt** — what to generate, and what to avoid.
- **Steps** — number of diffusion (denoising) steps; more steps = higher quality, slower.
- **CFG scale** (classifier-free guidance) — how strongly the output is pushed toward the prompt. Low = loose, high = literal.
- **Seed** — fix it to reproduce a result, or randomize for a new take.

### 🎵 Text-to-music
Generate audio from a text prompt — *"warm lo-fi piano with vinyl crackle, 70 bpm"*. The model denoises from pure random noise into a clip. Controls: prompt, duration, and (under Advanced options) negative prompt, steps, CFG scale, seed, and **takes** — generate up to four seeds of the same prompt in one click and keep the best. Every result shows the seed that made it.

### ✂️ Inpainting (audio editing)
Replace a chosen time region of a clip while keeping everything outside it untouched. The clip is encoded to latents, a mask covers the `[start, end]` seconds you pick, and only the masked region is regenerated from the prompt — the unmasked latents are held fixed at every denoising step, so the surrounding audio is preserved. Good for fixing a section or dropping in a new phrase.

### 🔀 Audio-to-Audio (variations)
Generate a variation of an existing clip. The source is encoded to latents and mixed with random noise at a chosen **noise level** before denoising: low noise stays close to the original (subtle remix), high noise drifts further (loose reinterpretation). Add a prompt and raise CFG to steer the variation.

### 🎚️ RF-Inversion (re-style)
Re-style a clip while preserving its structural identity — melody and rhythm stay, timbre and character change. Unlike audio-to-audio (which just adds random noise), **RF-Inversion** runs the rectified-flow ODE *backwards* to recover the specific "noise fingerprint" the model would denoise back into your clip, then regenerates from that fingerprint with your new prompt. Two controls shape the trade-off:

- **Gamma (γ)** — inversion strength. Low γ keeps the fingerprint close to the model's own representation of the audio (faithful); high γ pushes it toward generic Gaussian noise (more freedom to change).
- **Eta (η)** — structure preservation during regeneration. Higher η pulls each step back toward the original, locking in its structure while the prompt re-styles the surface.

### 🌅 Audio Transition (morph)
Blend two clips into a single A→B transition. Both are encoded to latents, and across a transition region the latent frames are crossfaded with **spherical interpolation (slerp)** — a blend that follows the curved geometry of the latent space instead of a straight line. You set the transition length and curve. An optional **re-coherence** pass repairs the seam by running the morph back through the model at a low noise level; a toggle switches that repair from the default audio-to-audio pass to the structure-preserving RF-Inversion method.

### 🔁 Extend & Loop
Outpainting, three ways: **continue** a clip past its end, compose a **lead-in** before it, or turn it into a **seamless loop** — the clip is rotated so its start/end joint sits in the middle, the joint is regenerated with the inpainting engine, and the clip is rotated back. A "seam check" player plays the loop twice back to back so you can judge the joint.

## Models

| Model | Size | Max length | Hardware |
|---|---|---|---|
| `small-music` | 433M | 120s | CPU or GPU |
| `medium` | 1.4B | 285s | GPU (A100 / L4) |

`small-music` runs anywhere and is great for quick sketches. `medium` sounds noticeably better and is the one to use for the highest-quality results.

## Good to know

- **One-time setup:** accept the model license on Hugging Face and add an `HF_TOKEN` — the notebook walks you through both steps.
- **Cached weights:** model files are saved to your Google Drive, so you don't re-download them every session (the first run takes 1–3 minutes).
- **Session clips:** every tool's output is added to a running list, and every tool's source dropdown updates instantly — generate, inpaint, remix, extend, and morph in any order without downloading and re-uploading anything.

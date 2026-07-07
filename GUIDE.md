# Stable Audio 3 Colab — User Guide

The complete guide to the notebook. For the short version, see the [README](README.md).

**Contents:** [Overview](#overview) · [Quick Start](#quick-start) · [Installation & Requirements](#installation--requirements) · [Notebook Structure](#notebook-structure) · [Prompt Engineering](#prompt-engineering-guide) · [Advanced Usage](#advanced-usage) · [Performance](#performance-tips) · [FAQ](#faq) · [Troubleshooting](#troubleshooting)

---

## Overview

**Stable Audio 3** is Stability AI's family of open-weight text-to-audio models. It is a *latent rectified-flow diffusion* model: audio is compressed by an autoencoder (VAE) into a compact latent sequence, a diffusion transformer (DiT) denoises latents under the guidance of a T5-encoded text prompt, and the VAE decodes the result back to 44.1 kHz stereo audio. The models are distilled for few-step sampling — the default is just **8 denoising steps**, which is why generation takes seconds rather than minutes.

**This notebook** wraps two model variants in six interactive tools, all running in your browser on a free or paid Google Colab runtime:

| Tool | Section | What it does |
|---|---|---|
| 🎵 Text-to-music | §5 | Generate a clip from a text prompt (up to 4 takes per click) |
| ✂️ Inpainting | §7 | Regenerate a chosen time region of a clip, keep the rest |
| 🔀 Audio-to-Audio | §8 | Generate a variation of an existing clip |
| 🎚️ RF-Inversion | §9 | Re-style a clip while preserving its melody and rhythm |
| 🌅 Audio Transition | §10 | Morph clip A into clip B with a latent-space crossfade |
| 🔁 Extend & Loop | §11 | Continue a clip, compose an intro, or make it loop seamlessly |

**Who it's for:** musicians and producers sketching ideas, sound designers, hobbyists, and anyone curious about audio diffusion — no coding or local install required.

---

## Quick Start

Under two minutes to first audio (after the one-time account setup):

1. **Open the notebook:** https://colab.research.google.com/github/hashachar/stableaudio-colab/blob/master/stable_audio_3_demo.ipynb
2. **Pick a GPU runtime:** *Runtime → Change runtime type → T4 GPU* (or better).
3. **Run the setup cells** (top of the notebook through section 4). The install takes ~1–2 minutes; model weights download on first generation and are cached to your Drive afterwards.
4. **Scroll to section 5**, type a prompt — try `warm lo-fi piano with vinyl crackle, soft rain in the background, 70 bpm` — set a duration, and click **Generate Audio**.
5. Listen inline, download, or keep the clip in the session list to feed into the other tools.

One-time account setup (the notebook walks you through both):

- **Accept the model license** on Hugging Face (the model pages are gated).
- **Create an `HF_TOKEN` Colab secret** (🔑 icon in Colab's left sidebar) containing a Hugging Face access token, with *Notebook access* enabled.

---

## Installation & Requirements

Everything installs inside the Colab runtime — nothing touches your machine.

- **`stable-audio-3`** — Stability AI's inference package, installed from GitHub (it is not on PyPI). Brings in `torch`, `torchaudio`, `safetensors`, and `huggingface_hub`.
- **Flash Attention 2** — installed automatically only on GPUs that support it (compute capability ≥ 8.0: A100, L4, H100). On a T4 the model still runs, just slower, using PyTorch's built-in SDPA attention.
- **Google Drive (optional but recommended)** — section 2 mounts your Drive and points the Hugging Face cache (`HF_HOME`) at `MyDrive/stable-audio-cache`, so the multi-GB weights download once, not once per session.
- **Hugging Face token** — the model repos are gated; section 3 reads the `HF_TOKEN` Colab secret into the environment.

Runtime requirements:

| Runtime | small-music (433 M) | medium (1.4 B) |
|---|---|---|
| CPU only | ✅ works (slow) | ❌ refused |
| T4 (free tier) | ✅ | ✅ works, no Flash Attention |
| L4 / A100 | ✅ | ✅ recommended |

---

## Notebook Structure

Cells run top to bottom. Sections 1–4 are one-time setup; each tool after that is self-contained.

| Cell | What it does | Run when |
|---|---|---|
| **Style / Intro** | Injects the notebook CSS and shows the header | Once |
| **GPU Check** | Reports which accelerator the runtime has | Once |
| **1 — Install** | Installs `stable-audio-3` (+ Flash Attention where supported) | Once per runtime |
| **2 — Drive Cache** | Mounts Drive, sets `HF_HOME` so weights persist | Once per runtime |
| **3 — HF Auth** | Loads the `HF_TOKEN` secret, verifies access | Once per runtime |
| **4 — Utilities** | Model registry, `get_model()` cache, loading patches, warmup | Once per runtime |
| **generate_audio()** | The shared text-to-music helper | Once per runtime |
| **Example Prompts** | Clickable prompt chips that fill the §5 prompt box | Optional |
| **5 — Generate** | Text-to-music UI | Every generation |
| **6 — Tips** | Prompting quick-reference | Optional |
| **7 — Inpainting** | Regenerate a masked time region | As needed |
| **8 — Audio-to-Audio** | Variations of an existing clip | As needed |
| **9 — RF-Inversion** | Structure-preserving re-styling | As needed |
| **10 — Audio Transition** | Morph clip A into clip B | As needed |
| **11 — Extend & Loop** | Continue / intro / seamless loop | As needed |

**Session clips:** every tool's result is auto-registered in a session list (renameable in §5), and every tool's source dropdown refreshes the moment a clip is added. All the audio-input tools (§7–§11) can pick from that list, from an upload, or from a Drive folder — so you can chain tools in any order: generate → inpaint a section → make variations → extend the best one → morph two takes together, with zero downloads in between.

---

## Prompt Engineering Guide

The text encoder is T5 — it reads plain natural language, not keyword magic. Describe the audio the way a producer would describe a reference track.

### What to include

- **Genre / style:** `lo-fi hip hop`, `progressive trance`, `delta blues`, `orchestral film score`
- **Instrumentation:** name the instruments and how they're played — `fingerpicked nylon guitar`, `punchy 808 drums`, `warm analog pad`, `muted trumpet`
- **Mood:** `melancholic`, `triumphant`, `dreamy`, `menacing`, `playful`
- **Tempo & pacing:** an explicit BPM helps — `70 bpm`, `slow build`, `driving rhythm`, `half-time drop`
- **Production character:** `vinyl crackle`, `tape saturation`, `clean studio recording`, `live room ambience`, `heavy reverb`
- **Structure cues:** the model honors coarse arrangement hints on longer clips — `starts sparse, builds to a full drop`, `intro then main groove`
- **Key (sometimes):** `in A minor` is understood more often than not

### Good prompts

Music:

- `warm lo-fi piano with vinyl crackle, soft rain in the background, 70 bpm, mellow and nostalgic`
- `energetic drum and bass, rolling breakbeats at 174 bpm, deep sub bass, atmospheric pads`
- `solo cello, slow and mournful, intimate close-mic recording, in D minor`
- `funky disco groove with slap bass, tight rhythm guitar, four-on-the-floor kick, 118 bpm`
- `epic orchestral trailer music, staccato strings, huge percussion hits, brass swells, builds to a climax`
- `minimal techno, hypnotic looping synth stab, crisp hi-hats, 128 bpm, dark warehouse atmosphere`
- `gentle acoustic folk, fingerpicked guitar and soft female humming, campfire warmth`
- `8-bit chiptune boss battle theme, fast arpeggios, square-wave lead, relentless energy`

Ambience & sound design:

- `heavy rainstorm on a tin roof, distant thunder, occasional wind gusts`
- `busy city street at night, traffic, footsteps, murmuring crowd, distant sirens`
- `calm ocean waves on a pebble beach, seagulls far away`
- `sci-fi spaceship engine room hum, deep mechanical drone, metallic creaks`

### Common mistakes

- **Too vague:** `nice music` gives the model nothing to work with. Add genre, instruments, mood, tempo.
- **Contradictions:** `fast relaxing aggressive ambient` pulls in opposite directions — pick a lane, or combine genres deliberately (`jazz-influenced drum and bass` works; word salad doesn't).
- **Overstuffing:** past ~6–8 distinct musical ideas, the model starts averaging them into mush. Two or three strong descriptors beat ten weak ones.
- **Expecting lyrics:** these models generate instrumental audio and vocal *textures* (humming, choir pads); they do not sing intelligible words.
- **Prompting duration:** `30 second clip` in the prompt does nothing — use the duration slider.
- **Negative prompt as a wish list:** the negative prompt is for steering *away* from qualities (`distortion, muddy low end, vocals`), not for adding content.

### Duration

Pick a duration that matches the material: 10–30 s for loops and sketches, 60 s+ for pieces with structure. Longer clips give the model room to arrange (intro/build/outro) but take proportionally longer to generate and decode. Maximum: 120 s (small-music), 285 s (medium).

---

## Advanced Usage

Parameters shared by several tools:

- **Steps** (default **8**) — how many denoising iterations the sampler runs. These models are distilled for few-step sampling, so 8 is a genuine sweet spot, not a compromise. Raising steps refines texture and transients with quickly diminishing returns; cost grows roughly linearly (16 steps ≈ 2× the time). Try 12–16 if a result sounds smeared; going past ~25 rarely changes anything audible in the standard tools.
- **CFG scale** (default **1.0**) — classifier-free guidance, i.e. how hard the output is pushed toward the prompt. 1.0 is the model's tuned default. 2–4 = noticeably stronger prompt adherence; 7+ = very literal but can sound forced or thin. 0 ignores the prompt entirely.
- **Seed** (default **−1**) — −1 draws a fresh random seed each run. Any fixed value reproduces the same output for identical settings — change one parameter at a time against a fixed seed to hear what that parameter does.
- **Negative prompt** — qualities to avoid (`distortion, vocals, muddy bass`). Most useful at CFG ≥ 2; at CFG 1.0 its influence is mild.

### §5 Text-to-music

Model, duration, prompt, plus the **Advanced options** accordion (negative prompt, steps, CFG, seed, takes). Defaults are correct for everyday use; open the accordion when you want reproducibility (fixed seed) or stronger prompt adherence (CFG 2–4).

- **Takes** (1–4): one click generates several takes of the same prompt, each from its own seed, shown side by side with per-take download buttons. With a fixed seed, takes use seed, seed+1, seed+2… Every result panel shows the seed that made it — enter that seed later to reproduce the take exactly.

### §7 Inpainting

Pick a source clip, set the **mask** (start/end seconds — the waveform view shows it in amber; *Preview mask* plays just that region), describe the replacement, generate. Everything outside the mask is held fixed at every denoising step, so the surroundings are bit-faithful.

- **CFG** (0–15, default 1.0): at 1.0 the fill blends seamlessly but follows the prompt loosely; 2–4 follows the prompt clearly with some risk of seam artifacts at the mask boundaries.
- Masks from ~0.5 s (a hit or fill) up to many seconds (a whole phrase) both work. The clip must fit within the model's max duration — longer files are truncated.
- The result offers a download with optional **click markers** at the mask edges, so you can find the seam in a DAW.

### §8 Audio-to-Audio (variations)

The source clip's latents are blended with random noise at the chosen **noise level** before denoising — that one dial sets how far the variation drifts:

| Noise | Result |
|---|---|
| 0.10–0.20 | Nearly identical; subtle timbre shift |
| 0.30–0.45 ★ | Sweet spot — harmony preserved, light drift |
| 0.50–0.65 | Creative variation; pair with a prompt |
| 0.70–0.85 | Loose reinterpretation; needs a strong prompt |
| 0.90–1.00 | Approaches pure generation |

**Interaction:** with no prompt, CFG is irrelevant and the model varies freely. With a prompt, noise sets how much *room* there is to change and CFG sets how that room is *used* — a common recipe is noise 0.5–0.65 + prompt + CFG 2–4 to push a clip toward a new style while keeping its bones. Steps behaves as in §5.

### §9 RF-Inversion (re-style)

Unlike A2A's random noise, RF-Inversion runs the rectified-flow ODE *backwards* to recover the specific noise "fingerprint" that denoises into your clip, then regenerates from that fingerprint under a new prompt. Structure (melody, rhythm) survives; surface (timbre, instrumentation) changes.

- **Gamma γ** (0–1, default 0.30) — inversion strength. Low = fingerprint stays close to the model's representation of your audio (faithful); high = pushed toward generic noise (more freedom).
- **Eta η** (0–1, default 0.0) and **η stop** (default 0.30) — structure lock during regeneration. η > 0 pulls each step back toward the original while `t ≥ η stop`; higher η preserves more structure at the cost of restyling freedom. At η = 0 the second phase is exactly the stock sampler.
- **Steps** (10–150, default 50) — inversion needs a finer ODE discretization than distilled generation; keep this well above 8.
- **CFG** defaults to 4.0 here: re-styling needs real prompt pressure.
- Untuned starting point for faithful edits: γ 0.3–0.5, η 0.3–0.6, η stop 0.3.

### §10 Audio Transition (morph)

Clips A and B are encoded to latents; across the transition region, latent frames are crossfaded with **slerp** (spherical interpolation — follows the curved geometry of latent space instead of a straight line, which keeps the blend from collapsing to mush).

- **Transition length** (0.5–60 s, capped by the combined clip length) and **curve**: ease-in-out (smooth, default), linear (constant), steep (decisive).
- **Re-coherence noise** (0–1, default 0.30) — a repair pass that runs the raw morph back through the model at low noise so the seam sounds *played* rather than crossfaded. 0 skips it; when active you get raw and re-cohered results side by side. Keep it ≤ 0.4 or the repair starts rewriting the clips themselves.
- **RF-Inversion re-coherence** (checkbox) — repairs the seam with the structure-preserving §9 method instead of blind A2A (run §9's cell once first). Slower, but changes the original material less.
- **CFG** (0–8) matters only when you give the re-coherence pass a prompt.

### §11 Extend & Loop

All three modes ride the inpainting engine: only the chosen region is regenerated and everything else is preserved exactly, with ~0.5 s of existing audio at each joint re-written so the splice is *composed*, not glued.

- **Continue end** — silence is appended after the clip and the model fills it, conditioned on what came before. **Extension** (1–60 s) sets how much is added; source + extension must fit the model's max duration.
- **Extend intro** — the mirror image: a lead-in is composed that arrives at your clip's downbeat.
- **Seamless loop** — the clip is rotated so its start/end joint sits mid-clip, a **seam window** (0.5–8 s) around the joint is regenerated, and the clip is rotated back. The result panel includes a *seam check* player (the loop twice back to back, the joint passing at the midpoint). Small windows make surgical fixes; large ones recompose the whole transition.
- **Prompt** describes only the regenerated region — `drum fill into a drop`, `soft outro, fading pads` — or leave it blank to let the surrounding context decide. CFG and steps behave as in §7/§8.

- **Model loading is the slow part, generation is fast.** First-ever load: 1–3 min download + ~60–90 s initialization (the notebook prints progress and a heartbeat). Subsequent sessions read weights from your Drive cache; within a session the model stays in RAM and re-generation starts instantly.
- **Generation speed:** with 8 steps on an A100/L4, expect a handful of seconds for a 30 s clip on `medium`; T4 and CPU are progressively slower. RF-Inversion at 50 steps is roughly an order of magnitude slower than a standard generate.
- **VRAM:** `medium` in fp16 fits comfortably on a 16 GB T4; `small-music` is light enough for CPU. The notebook keeps only one model in VRAM — loading a different one first frees the old.
- **What the notebook already does for you:** fp16 weights on CUDA (2–3× faster on tensor-core GPUs), CPU-side checkpoint loading (avoids a silent multi-minute stall), chunked VAE decode (caps decode VRAM on long clips), a warmup generation behind the loading wait (first real prompt runs at full speed), and Drive-cached weights.
- **`ENABLE_TORCH_COMPILE`** (Utilities cell, default `False`): JIT-compiles the DiT for ~20–40 % faster steps, but the compile itself stalls the first generation for minutes. Worth enabling only for long batch runs at a fixed duration; the inductor cache persists on Drive so the cost is paid at most once.
- **Free-tier reality:** T4 runs everything but without Flash Attention; sessions disconnect after idle timeouts, losing RAM (but not your Drive weight cache). Save clips you care about — the session list lives in `/tmp` and dies with the runtime.

---

## FAQ

**Do I need to pay for Colab?** No — the free T4 tier runs both models. Paid tiers (L4/A100) load and generate faster and allow longer sessions.

**Why do I need a Hugging Face account?** The model weights are gated behind a license acceptance. You accept once, create a token once, and the notebook handles the rest.

**Which model should I use?** Start with `small-music` for fast sketching and prompt iteration; switch to `medium` for the final take. `medium` is audibly better, especially in high frequencies and stereo image.

**Can it generate vocals or speech?** Vocal *textures* (humming, choirs, vowels), yes. Intelligible lyrics or speech, no.

**How do I get the same result twice?** Fix the seed (any value ≥ 0) and keep every other setting identical.

**Where do my files go?** Results live in the runtime's `/tmp` and appear in the session clip list; use the download buttons to keep them. Model weights (not audio) are cached to `MyDrive/stable-audio-cache`.

**Can I use outputs commercially?** That's governed by the Stability AI community license you accept on the model page — read it; terms depend on your revenue.

**Why is the first generation after loading fast, but the very first load so slow?** Loading pays for the download (once ever, thanks to the Drive cache), weight initialization, and a warmup generation. Everything after that is pure sampling.

**Can I make a track longer than the model's max duration?** Not in one pass — the total output of any tool is capped at the model max (120 s / 285 s). The workflow for longer pieces: generate sections separately (§5 or §11 Continue), then join them in a DAW, using §10 (Transition) to compose the joins.

**How do I make a clip loop?** §11, Seamless loop mode. Use the seam-check player (the loop twice in a row) to judge the joint before downloading.

**Can I run this locally?** The `stable-audio-3` package works anywhere with a suitable GPU, but this notebook's UI is Colab-specific (secrets, Drive mounting, upload widgets).

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `401 Unauthorized` / `GatedRepoError` | License not accepted or bad token. Accept the license on the model's Hugging Face page, check the `HF_TOKEN` secret exists with *Notebook access* on, re-run §3. |
| `HF_TOKEN not found in environment` | §3 wasn't run this session, or the secret toggle is off. Run §3 again. |
| `The 'medium' model requires a CUDA GPU` | Runtime → Change runtime type → GPU, then re-run from §1 (a new runtime is a blank machine). |
| Flash Attention warning on T4 | Informational — T4 (compute 7.5) can't run FA2. Everything still works, just slower. |
| Model re-downloads every session | §2 (Drive cache) wasn't run before the first load, so weights went to ephemeral disk. Run §2 → §4 in order. |
| Widgets/buttons do nothing | The runtime restarted and cell state is gone. Runtime → Run all of the setup sections, then the tool cell. |
| Out-of-memory (CUDA OOM) | Reduce duration, or use `small-music`. Loading the other model auto-frees the previous one; if OOM persists, Runtime → Restart. |
| Triton / `flex_attention` warnings during inpainting | Known upstream noise — harmless, the result is fine. |
| First generation stalls for minutes | If you enabled `ENABLE_TORCH_COMPILE`, that's the one-time JIT. Otherwise check the loading heartbeat — it's probably still in the 60–90 s init phase. |
| `No module named 'stable_audio_3'` | §1 (Install) hasn't run in this runtime. Run it and wait for it to finish before §4. |
| Session clips vanished | The runtime recycled; `/tmp` is ephemeral. Re-generate, and download anything you want to keep. |

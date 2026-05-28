#!/usr/bin/env python3
"""Restore matplotlib inpainting cell with zoom slider + BoundedFloatText mask inputs."""
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
NB   = REPO / "stable_audio_3_demo.ipynb"

NEW_SOURCE = '''#@title 7 — Inpainting
# ── Imports ───────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from IPython.display import Image
from google.colab import files as colab_files

# All other names (widgets, Layout, Audio, clear_output, HTML, display,
# torchaudio, SAMPLE_RATE, time, tempfile, get_model, SPINNER_HTML,
# _session_audio, _on_session_audio_updated) come from earlier cells.


# ── Waveform renderer ─────────────────────────────────────────────────────────
def _render_waveform(audio_tensor, mask_start, mask_end, view_start=None, view_end=None):
    """Render waveform PNG. view_start/view_end zoom the x-axis."""
    mono = audio_tensor.mean(dim=0).numpy()
    duration = len(mono) / SAMPLE_RATE
    step = max(1, len(mono) // 8000)
    mono_ds = mono[::step]
    times = np.linspace(0, duration, len(mono_ds))
    mask_start = max(0.0, float(mask_start))
    mask_end   = min(duration, float(mask_end))
    v0 = max(0.0, float(view_start)) if view_start is not None else 0.0
    v1 = min(duration, float(view_end)) if view_end is not None else duration
    if v0 >= v1:
        v0, v1 = 0.0, duration

    fig, ax = plt.subplots(figsize=(9, 2.4), facecolor=\'#F5F3FF\')
    ax.set_facecolor(\'#F5F3FF\')
    ax.plot(times, mono_ds, color=\'#7C3AED\', linewidth=0.4, alpha=0.9, rasterized=True)
    ax.fill_between(times, mono_ds, alpha=0.18, color=\'#7C3AED\')
    ax.axvspan(mask_start, mask_end, alpha=0.45, color=\'#FCD34D\', zorder=2)
    ax.axvline(mask_start, color=\'#D97706\', linewidth=2.0, zorder=3)
    ax.axvline(mask_end,   color=\'#D97706\', linewidth=2.0, zorder=3)
    mid = (mask_start + mask_end) / 2
    if v0 <= mid <= v1:
        y_top = float(np.abs(mono_ds).max()) * 0.88 if len(mono_ds) > 0 else 0.8
        ax.text(mid, y_top,
                f\'{mask_start:.2f}s\\u2013{mask_end:.2f}s  ({mask_end - mask_start:.2f}s)\',
                ha=\'center\', va=\'top\', fontsize=8, color=\'#92400E\',
                bbox=dict(boxstyle=\'round,pad=0.3\', facecolor=\'#FEF3C7\',
                          edgecolor=\'#FCD34D\', alpha=0.95), zorder=4)
    ax.set_xlim(v0, v1)
    ax.set_xlabel(\'Time (s)\', fontsize=9, color=\'#374151\')
    ax.tick_params(colors=\'#6B7280\', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(\'#DDD6FE\')
    plt.tight_layout(pad=0.4)
    buf = BytesIO()
    fig.savefig(buf, format=\'png\', dpi=110, bbox_inches=\'tight\', facecolor=\'#F5F3FF\')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── State ─────────────────────────────────────────────────────────────────────
_inpaint_audio = None   # torch.Tensor (channels, samples) at SAMPLE_RATE


# ── Widgets ───────────────────────────────────────────────────────────────────
_UPLOAD_OPT = \'\\u2014 Upload new file \\u2014\'

inpaint_model_dropdown = widgets.Dropdown(
    options=[(MODEL_CONFIGS[k]["label"], k) for k in MODEL_CONFIGS],
    value="small-music",
    description="Model:",
    style={"description_width": "52px"},
    layout=Layout(width="500px"),
)

inpaint_gpu_warning = widgets.HTML(
    value=\'<div class="sa3-warn">\'
          \'&#9888;&#65039;&nbsp; <b>medium</b> requires a GPU with compute capability \\u22658.0 \'
          \'(A100 or L4). No compatible GPU detected. \'
          \'Go to <b>Runtime &rarr; Change runtime type &rarr; GPU</b>.\'
          \'</div>\',
    layout=Layout(display="none", width="100%"),
)

source_dropdown = widgets.Dropdown(
    options=[_UPLOAD_OPT],
    description=\'Source:\',
    style={\'description_width\': \'60px\'},
    layout=Layout(width=\'440px\'),
)

upload_btn = widgets.Button(
    description=\'Upload Audio\', icon=\'upload\',
    layout=Layout(width=\'155px\', height=\'36px\'),
)
upload_btn.style.button_color = \'#EDE9FE\'

upload_status = widgets.HTML(
    value=\'<div class="sa3-info" style="margin:4px 0;">No file loaded yet.</div>\'
)
upload_row = widgets.HBox(
    [upload_btn, upload_status],
    layout=Layout(align_items=\'center\', gap=\'10px\', margin=\'4px 0 0\'),
)

waveform_out = widgets.Output()

zoom_slider = widgets.FloatRangeSlider(
    value=[0.0, 30.0], min=0.0, max=120.0, step=0.1,
    description=\'View (s):\',
    style={\'description_width\': \'60px\'},
    layout=Layout(width=\'540px\'),
    readout_format=\'.1f\',
    continuous_update=False,
)

mask_start_box = widgets.BoundedFloatText(
    value=5.0, min=0.0, max=120.0, step=0.01,
    description=\'Start (s):\',
    style={\'description_width\': \'60px\'},
    layout=Layout(width=\'175px\'),
)
mask_end_box = widgets.BoundedFloatText(
    value=15.0, min=0.0, max=120.0, step=0.01,
    description=\'End (s):\',
    style={\'description_width\': \'52px\'},
    layout=Layout(width=\'165px\'),
)
mask_row = widgets.HBox(
    [mask_start_box, mask_end_box],
    layout=Layout(gap=\'16px\', margin=\'2px 0\'),
)

preview_btn = widgets.Button(
    description=\'Preview mask\', icon=\'headphones\',
    layout=Layout(width=\'152px\', height=\'36px\'),
)
preview_btn.style.button_color = \'#EDE9FE\'
preview_out = widgets.Output()

inpaint_prompt = widgets.Textarea(
    value=\'Replace this region with a gentle acoustic guitar melody\',
    placeholder=\'Describe what should replace the masked region\\u2026\',
    layout=Layout(width=\'100%\', height=\'64px\'),
    style={\'description_width\': \'0px\'},
)

inpaint_btn = widgets.Button(
    description=\'  Inpaint\', icon=\'magic\', button_style=\'\',
    layout=Layout(width=\'150px\', height=\'42px\'),
)
inpaint_btn.style.button_color = \'#7C3AED\'
inpaint_btn.style.font_weight  = \'700\'

inpaint_out = widgets.Output(layout=Layout(min_height=\'60px\', margin=\'6px 0 0\'))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_audio_into_state(audio_tensor, label):
    global _inpaint_audio
    _inpaint_audio = audio_tensor
    duration = audio_tensor.shape[-1] / SAMPLE_RATE
    ms = round(min(5.0,  duration * 0.15), 2)
    me = round(min(15.0, duration * 0.40), 2)
    # Update mask boxes
    mask_start_box.max   = round(duration, 2)
    mask_end_box.max     = round(duration, 2)
    mask_start_box.value = ms
    mask_end_box.value   = me
    # Update zoom slider to show full audio
    zoom_slider.max   = round(duration, 1)
    zoom_slider.value = [0.0, round(duration, 1)]
    upload_status.value = (
        f\'<div class="sa3-info" style="margin:4px 0;">\
\\u2705&nbsp; <strong>{label}</strong>\
 &nbsp;|&nbsp; {duration:.1f}&thinsp;s\
 &nbsp;|&nbsp; {audio_tensor.shape[0]} ch\
</div>\'
    )
    _refresh_waveform()


def _refresh_waveform(*_):
    if _inpaint_audio is None:
        return
    start = mask_start_box.value
    end   = mask_end_box.value
    v0, v1 = zoom_slider.value
    png = _render_waveform(_inpaint_audio, start, end, v0, v1)
    with waveform_out:
        clear_output(wait=True)
        display(Image(data=png))


# ── Callbacks ─────────────────────────────────────────────────────────────────
def on_inpaint_model_change(change):
    new_model = change["new"]
    cfg = MODEL_CONFIGS[new_model]
    needs_warn = cfg["needs_gpu"] and not torch.cuda.is_available()
    inpaint_gpu_warning.layout.display = "" if needs_warn else "none"

inpaint_model_dropdown.observe(on_inpaint_model_change, names="value")


def on_source_change(change):
    title = change[\'new\']
    if title == _UPLOAD_OPT:
        upload_row.layout.display = \'\'
    else:
        upload_row.layout.display = \'none\'
        wav_path = _session_audio[title]
        audio, sr = torchaudio.load(wav_path)
        _load_audio_into_state(audio, title)


def on_upload(b):
    uploaded = colab_files.upload()
    if not uploaded:
        return
    for fname, data in uploaded.items():
        tmp = f\'/tmp/{fname}\'
        with open(tmp, \'wb\') as f:
            f.write(data)
        audio, sr = torchaudio.load(tmp)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        _load_audio_into_state(audio, fname)


def on_preview(b):
    if _inpaint_audio is None:
        return
    start = mask_start_box.value
    end   = mask_end_box.value
    segment = _inpaint_audio[:, int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
    tmp = tempfile.mktemp(suffix=\'.wav\', dir=\'/tmp\')
    torchaudio.save(tmp, segment.cpu(), SAMPLE_RATE)
    with preview_out:
        clear_output(wait=True)
        display(HTML(
            f\'<span class="sa3-stat" style="margin:4px 6px 4px 0;">\
Mask preview &nbsp;{start:.2f}s \\u2013 {end:.2f}s</span>\'
        ))
        display(Audio(tmp, autoplay=True))


def on_inpaint(b):
    if _inpaint_audio is None:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(\'<div class="sa3-error">Select or upload an audio file first.</div>\'))
        return
    prompt = inpaint_prompt.value.strip()
    if not prompt:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(\'<div class="sa3-error">Please enter a prompt.</div>\'))
        return

    model_name = inpaint_model_dropdown.value
    start = mask_start_box.value
    end   = mask_end_box.value
    if start >= end:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(\'<div class="sa3-error">Start must be less than End.</div>\'))
        return
    duration = _inpaint_audio.shape[-1] / SAMPLE_RATE
    inpaint_btn.disabled    = True
    inpaint_btn.description = \'  Inpainting\\u2026\'

    if model_name in _model_cache:
        _spin_status    = "Running inpainting diffusion\\u2026"
        _spin_substatus = f"&#x26A1;&nbsp; <strong>{model_name}</strong> already loaded in memory."
    elif _is_model_on_disk(model_name):
        _spin_status    = f"Loading <strong>{model_name}</strong> from Drive cache\\u2026"
        _spin_substatus = "&#x1F4BE;&nbsp; Weights found on Drive \\u2014 no download needed."
    else:
        _spin_status    = f"Downloading <strong>{model_name}</strong> weights\\u2026"
        _spin_substatus = "&#x2B07;&nbsp; First run \\u2014 may take 1&ndash;3&thinsp;min."
    with inpaint_out:
        clear_output(wait=True)
        display(HTML(SPINNER_HTML.format(status=_spin_status, substatus=_spin_substatus)))

    try:
        model = get_model(model_name)
        t0 = time.perf_counter()
        audio_out = model.generate(
            prompt=prompt,
            inpaint_audio=(SAMPLE_RATE, _inpaint_audio),
            inpaint_mask_start_seconds=start,
            inpaint_mask_end_seconds=end,
            duration=duration,
            steps=8, cfg_scale=1.0,
        )
        elapsed = time.perf_counter() - t0
        short = prompt[:90] + (\'\\u2026\' if len(prompt) > 90 else \'\')
        tmp = tempfile.mktemp(suffix=\'.wav\', dir=\'/tmp\')
        torchaudio.save(tmp, audio_out[0].cpu(), SAMPLE_RATE)
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(f"""
            <div class="sa3-result">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
                <span class="sa3-badge">{model_name}</span>
                <span class="sa3-stat">mask {start:.2f}s\\u2013{end:.2f}s</span>
                <span class="sa3-stat">generated in {elapsed:.1f}&thinsp;s</span>
              </div>
              <div style="font-size:13px;color:#374151;font-style:italic;line-height:1.5;">
                &ldquo;{short}&rdquo;
              </div>
            </div>
            """))
            display(Audio(tmp, autoplay=False))
    except RuntimeError as exc:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(f\'<div class="sa3-error"><strong>Error:</strong> {exc}</div>\'))
    finally:
        inpaint_btn.disabled    = False
        inpaint_btn.description = \'  Inpaint\'


# ── Connect observers ─────────────────────────────────────────────────────────
source_dropdown.observe(on_source_change, names=\'value\')
mask_start_box.observe(_refresh_waveform, names=\'value\')
mask_end_box.observe(_refresh_waveform, names=\'value\')
zoom_slider.observe(_refresh_waveform, names=\'value\')
upload_btn.on_click(on_upload)
preview_btn.on_click(on_preview)
inpaint_btn.on_click(on_inpaint)

def _on_session_audio_updated():
    prev = source_dropdown.value
    opts = [_UPLOAD_OPT] + list(_session_audio.keys())
    source_dropdown.options = opts
    if prev in opts:
        source_dropdown.value = prev


# ── Layout ────────────────────────────────────────────────────────────────────
display(widgets.VBox([
    widgets.HTML(\'<div style="font-size:12px;color:#6B7280;margin:0 0 3px;">Model:</div>\'),
    inpaint_model_dropdown,
    inpaint_gpu_warning,
    widgets.HTML(\'<div style="font-size:12px;color:#6B7280;margin:8px 0 3px;">Audio source:</div>\'),
    source_dropdown,
    upload_row,
    waveform_out,
    widgets.HTML(\'<div style="font-size:12px;color:#6B7280;margin:6px 0 2px;">Zoom (drag to set visible range):</div>\'),
    zoom_slider,
    widgets.HTML(\'<div style="font-size:12px;color:#6B7280;margin:8px 0 2px;">Mask region to replace:</div>\'),
    mask_row,
    widgets.HBox(
        [preview_btn, preview_out],
        layout=Layout(align_items=\'center\', gap=\'12px\'),
    ),
    widgets.HTML(\'<div style="font-size:12px;color:#6B7280;margin:10px 0 2px;">Prompt for the replacement region:</div>\'),
    inpaint_prompt,
    inpaint_btn,
    inpaint_out,
], layout=Layout(width=\'100%\', max_width=\'700px\')))
'''

nb = json.loads(NB.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("id") == "cell-11-inpainting-stub":
        cell["source"] = NEW_SOURCE
        break
else:
    raise RuntimeError("cell-11-inpainting-stub not found")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("patch_inpaint_classic: OK")

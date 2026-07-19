

import io
import base64

import numpy as np
import soundfile as sf
import scipy.signal as sg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.shortcuts import render

from .yin import pitchDetect

DEFAULTS = {"min_f0": 50, "max_f0": 1500, "W": 1024,
            "decimation": 1, "threshold": 0.30, "rms": 0.05}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_note(hz):
    midi = 69 + 12 * np.log2(hz / 440.0)
    r = int(round(midi))
    cents = int(round((midi - r) * 100))
    return NOTE_NAMES[r % 12], r // 12 - 1, cents, r


def get_notes(contour, step, fs):
    dt = step / fs
    notes, cur = [], None
    for i, f in enumerate(contour):
        if f is None or not np.isfinite(f):
            cur = None
            continue
        name, octave, cents, midi = hz_to_note(f)
        if cur and cur["midi"] == midi:
            cur["freqs"].append(f); cur["end"] = i
        else:
            cur = {"midi": midi, "freqs": [f], "start": i, "end": i}
            notes.append(cur)
    out = []
    for n in notes:
        if n["end"] - n["start"] < 2:
            continue
        med = float(np.median(n["freqs"]))
        name, octave, cents, _ = hz_to_note(med)
        out.append({"name": name, "octave": octave, "cents": cents,
                    "hz": round(med, 1),
                    "t0": round(n["start"] * dt, 2),
                    "t1": round((n["end"] + 1) * dt, 2)})
    return out

def smooth_contour(contour, kernel_size=5):

    if kernel_size % 2 == 0:
        kernel_size += 1
    vals = np.array([v if (v is not None and np.isfinite(v)) else np.nan
                      for v in contour], dtype=float)
    n = len(vals)
    half = kernel_size // 2

    # Pad with NaN so edge frames get smaller but valid windows
    padded = np.concatenate([np.full(half, np.nan), vals, np.full(half, np.nan)])

    # Build a 2D rolling window array (vectorized, no Python loop)
    # Shape: (n, kernel_size)
    indices = np.arange(kernel_size)[None, :] + np.arange(n)[:, None]
    windows = padded[indices]                       # (n, kernel_size)

    # nanmedian per row — ignores NaN (unvoiced neighbours)
    out = np.nanmedian(windows, axis=1)

    # Restore original None positions
    out[~np.isfinite(vals)] = np.nan
    return [None if np.isnan(v) else float(v) for v in out]

def make_plot(audio, fs, contour, W):
    step = W // 2
    dur = len(audio) / fs
    t_audio = np.linspace(0, dur, len(audio))
    t_pitch = np.arange(len(contour)) * step / fs
    pitch = np.array([np.nan if v is None else v for v in contour], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(t_audio, audio, color="b", linewidth=0.5)
    ax1.set_title("Recorded Audio Waveform"); ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, dur)
    ax2.plot(t_pitch, pitch, color="r", linewidth=2)
    ax2.set_title("Detected Pitch (F0)")
    ax2.set_xlabel("Time (Seconds)"); ax2.set_ylabel("Frequency (Hz)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _num(request, key, default, cast):
    try:
        return cast(request.POST.get(key, default))
    except (TypeError, ValueError):
        return default


def index(request):
    # Current parameter values (used to repopulate the form)
    params = dict(DEFAULTS)

    if request.method != "POST" or "audio" not in request.FILES:
        return render(request, "detector/index.html", {"params": params})

    # Read tweakable parameters from the form
    params = {
        "min_f0": _num(request, "min_f0", DEFAULTS["min_f0"], int),
        "max_f0": _num(request, "max_f0", DEFAULTS["max_f0"], int),
        "W": _num(request, "W", DEFAULTS["W"], int),
        "decimation": _num(request, "decimation", DEFAULTS["decimation"], int),
        "threshold": _num(request, "threshold", DEFAULTS["threshold"], float),
        "rms": _num(request, "rms", DEFAULTS["rms"], float),
    }

    # mic submissions ask for just the results fragment
    fragment = request.GET.get("fragment") == "1" or request.POST.get("fragment") == "1"
    template = "detector/_results.html" if fragment else "detector/index.html"

    try:
        raw = request.FILES["audio"].read()
        audio, fs = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        return render(request, template, {"params": params, "error": f"Could not read audio: {e}"})

    fs = int(fs)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) #normalization

        # 1. Get the raw mathematical output from the untouched YIN algorithm
    contour = list(pitchDetect(
        audio, fs,
        min_f0=params["min_f0"], max_f0=params["max_f0"],
        W=params["W"], decimation_factor=params["decimation"],
        cmndf_threshold=params["threshold"], rms_threshold=params["rms"],
    ))
    contour=smooth_contour(contour)

    ctx = {
        "params": params,
        "plot": make_plot(audio, fs, contour, params["W"]),
        "notes": get_notes(contour, params["W"] // 2, fs),
        "done": True,
    }
    return render(request, template, ctx)

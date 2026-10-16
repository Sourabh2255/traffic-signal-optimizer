"""
Generate a small, clearly documented SYNTHETIC siren/no_siren audio
dataset (Phase 10).

DATA SOURCE & LIMITATION (see project spec section 7, 36):
This project has no access to a licensed real-world siren-recording
dataset. Instead of leaving the audio pipeline untested, we synthesize:

  - "siren" clips: a sine wave whose frequency oscillates (sweeps) up
    and down periodically, similar in shape to a real wailing siren
    (a classic two-tone/wail siren cycles between ~500-1800 Hz).
  - "no_siren" clips: steady or randomly-varying tones plus filtered
    noise, representing generic road/engine/ambient sound WITHOUT the
    periodic frequency-sweep pattern that characterizes a siren.

This is an explicit simplification for a college project timeframe.
The resulting classifier learns to distinguish "periodic frequency
sweep" from "steady/noisy" audio using MFCC features — a genuine
supervised ML exercise — but it should NOT be presented as validated
against real ambulance/fire-truck siren recordings. This limitation is
stated again in the dashboard and README.
"""

import os
import numpy as np
import soundfile as sf

from utils.config import SIREN_SAMPLE_RATE, SIREN_CLIP_SECONDS, SIREN_DATA_DIR

RNG_SEED = 7


def _make_siren_clip(rng, duration=SIREN_CLIP_SECONDS, sr=SIREN_SAMPLE_RATE):
    """A frequency-sweeping tone, like a two-tone wail siren, + light noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    low_freq = rng.uniform(450, 650)
    high_freq = rng.uniform(1500, 1900)
    sweep_rate = rng.uniform(0.5, 1.5)  # sweeps per second

    freq_t = low_freq + (high_freq - low_freq) * (0.5 + 0.5 * np.sin(2 * np.pi * sweep_rate * t))
    phase = 2 * np.pi * np.cumsum(freq_t) / sr
    tone = 0.6 * np.sin(phase)

    noise = rng.normal(0, 0.02, len(t))
    return np.clip(tone + noise, -1.0, 1.0).astype(np.float32)


def _make_no_siren_clip(rng, duration=SIREN_CLIP_SECONDS, sr=SIREN_SAMPLE_RATE):
    """Steady-ish tone or noise, WITHOUT the periodic sweep pattern."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    kind = rng.choice(["steady_tone", "noise", "low_hum"])

    if kind == "steady_tone":
        freq = rng.uniform(200, 900)
        drift = rng.normal(0, 5, len(t)).cumsum() * 0.001  # slow random drift, not periodic
        signal = 0.4 * np.sin(2 * np.pi * freq * t + drift)
    elif kind == "low_hum":
        freq = rng.uniform(80, 180)
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
    else:
        signal = rng.normal(0, 0.3, len(t))  # engine/road noise proxy

    noise = rng.normal(0, 0.03, len(t))
    return np.clip(signal + noise, -1.0, 1.0).astype(np.float32)


def generate_siren_dataset(n_per_class: int = 40):
    rng = np.random.default_rng(RNG_SEED)

    siren_dir = os.path.join(SIREN_DATA_DIR, "siren")
    no_siren_dir = os.path.join(SIREN_DATA_DIR, "no_siren")
    os.makedirs(siren_dir, exist_ok=True)
    os.makedirs(no_siren_dir, exist_ok=True)

    for i in range(n_per_class):
        clip = _make_siren_clip(rng)
        sf.write(os.path.join(siren_dir, f"siren_{i:03d}.wav"), clip, SIREN_SAMPLE_RATE)

    for i in range(n_per_class):
        clip = _make_no_siren_clip(rng)
        sf.write(os.path.join(no_siren_dir, f"no_siren_{i:03d}.wav"), clip, SIREN_SAMPLE_RATE)

    print(f"Generated {n_per_class} siren + {n_per_class} no_siren synthetic clips.")


if __name__ == "__main__":
    generate_siren_dataset()

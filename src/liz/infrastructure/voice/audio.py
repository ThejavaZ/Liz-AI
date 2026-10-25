"""Microphone capture.

Recording shells out to `arecord` (ALSA) so no audio binding is needed;
reading the result uses the standard library `wave` module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from liz.core.voice import DEFAULT_SAMPLE_RATE, VoiceError

RECORDER = "arecord"
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def recorder_available() -> bool:
    return shutil.which(RECORDER) is not None


def record(
    seconds: float,
    out: Path | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Record mono 16-bit audio from the default microphone.

    Returns the path of a WAV file the caller is responsible for
    deleting.
    """
    if not recorder_available():
        raise VoiceError(f"{RECORDER} not found (install alsa-utils)")

    if out is None:
        handle, name = tempfile.mkstemp(prefix="liz-", suffix=".wav")
        os.close(handle)
        out = Path(name)

    duration = max(1, int(seconds))
    command = [
        RECORDER,
        "-q",
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", str(CHANNELS),
        "-d", str(duration),
        "-t", "wav",
        str(out),
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=duration + 15,
        )
    except subprocess.TimeoutExpired as error:
        raise VoiceError("recording timed out") from error

    if process.returncode != 0:
        out.unlink(missing_ok=True)
        raise VoiceError(process.stderr.strip() or "recording failed")

    return out


def read_pcm(wav: Path) -> tuple[bytes, int]:
    """Read a WAV file into raw 16-bit mono PCM and its sample rate."""
    try:
        with wave.open(str(wav), "rb") as handle:
            if handle.getsampwidth() != SAMPLE_WIDTH:
                raise VoiceError("expected 16-bit audio")
            pcm = handle.readframes(handle.getnframes())
            return pcm, handle.getframerate()
    except (wave.Error, EOFError) as error:
        raise VoiceError(f"unreadable audio: {error}") from error

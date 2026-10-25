"""Voice domain.

`SpeechToText` and `TextToSpeech` are the ports that any engine must
satisfy, so the voice pipeline never depends on a particular package or
binary. The policies here (energy-based voice activity detection and
wake word matching) are pure functions over bytes and strings and are
tested without a microphone or an engine.
"""

from __future__ import annotations

import array
import math
import re
import sys
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

DEFAULT_WAKE_WORDS = ("hola liz", "hey liz", "oye liz", "ei liz")
DEFAULT_SAMPLE_RATE = 16000
FRAME_MS = 30
DEFAULT_ENERGY_THRESHOLD = 500
DEFAULT_SPEECH_RATIO = 0.12


class VoiceError(RuntimeError):
    """An engine or capture failure the caller should report."""


class TextToSpeech(ABC):
    """Port: turn a reply into speech."""

    name: str = "tts"

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def speak(self, text: str) -> None:
        ...


class SpeechToText(ABC):
    """Port: turn recorded audio into text."""

    name: str = "stt"

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def transcribe(self, wav: Path) -> str:
        ...


def parse_wake_words(raw: str) -> tuple[str, ...]:
    """Split a comma separated setting into wake words."""
    words = tuple(part.strip() for part in raw.split(",") if part.strip())
    return words or DEFAULT_WAKE_WORDS


def normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation, squeeze whitespace."""
    lowered = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        char for char in lowered if not unicodedata.combining(char)
    )
    without_punctuation = re.sub(r"[^\w\s]", " ", without_accents)
    return " ".join(without_punctuation.split())


def detect_wake_word(
    text: str, wake_words: tuple[str, ...] = DEFAULT_WAKE_WORDS
) -> bool:
    """Whether a transcript asks for Liz's attention."""
    transcript = normalize(text)
    if not transcript:
        return False
    return any(normalize(word) in transcript for word in wake_words)


def strip_wake_word(
    text: str, wake_words: tuple[str, ...] = DEFAULT_WAKE_WORDS
) -> str:
    """Remove the wake word so the command reaches the model clean."""
    remainder = normalize(text)
    for word in (normalize(item) for item in wake_words):
        if word and word in remainder:
            remainder = remainder.replace(word, " ", 1)
            break
    return " ".join(remainder.split())


def frame_samples(sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    """Samples per VAD frame (about 30 ms of mono audio)."""
    return max(1, int(sample_rate * FRAME_MS / 1000))


def frame_energies(
    pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> list[int]:
    """RMS energy of each frame of 16-bit little-endian mono samples."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()

    width = frame_samples(sample_rate)
    energies: list[int] = []
    for start in range(0, len(samples) - width + 1, width):
        frame = samples[start:start + width]
        energies.append(int(math.sqrt(sum(s * s for s in frame) / width)))
    return energies


def is_speech(
    pcm: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    threshold: int = DEFAULT_ENERGY_THRESHOLD,
    min_ratio: float = DEFAULT_SPEECH_RATIO,
) -> bool:
    """Energy-based voice activity detection.

    A window counts as speech when enough frames are louder than the
    threshold, so quiet rooms and keyboard noise are ignored.
    """
    energies = frame_energies(pcm, sample_rate)
    if not energies:
        return False
    loud = sum(1 for energy in energies if energy >= threshold)
    return loud / len(energies) >= min_ratio

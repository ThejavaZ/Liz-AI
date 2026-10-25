"""Text-to-speech engines.

Engines are plain external commands, so no ML dependency is needed:
speech-dispatcher (`spd-say`) gives the most natural voice when its
daemon runs, `espeak-ng` is the dependable fallback. The factory picks
whatever exists on the machine and degrades to `NullTTS`.
"""

from __future__ import annotations

import shutil
import subprocess

from liz.core.voice import TextToSpeech, VoiceError

TIMEOUT_SECONDS = 120


class CommandTTS(TextToSpeech):
    """Base for engines that shell out to a binary."""

    def __init__(self, binary: str, language: str = "es") -> None:
        self.binary = binary
        self.language = language

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def command(self, text: str) -> list[str]:
        raise NotImplementedError

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        try:
            process = subprocess.run(
                self.command(text),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VoiceError(f"{self.name}: {error}") from error

        if process.returncode != 0:
            raise VoiceError(
                f"{self.name}: {process.stderr.strip() or 'failed'}"
            )


class SpeechDispatcherTTS(CommandTTS):
    """Uses the desktop speech-dispatcher (spd-say)."""

    name = "spd-say"

    def command(self, text: str) -> list[str]:
        return [self.binary, "-w", "-l", self.language, text]


class EspeakTTS(CommandTTS):
    """Uses espeak-ng (or the classic espeak)."""

    name = "espeak"

    def command(self, text: str) -> list[str]:
        return [self.binary, "-v", self.language, "-s", "170", text]


class NullTTS(TextToSpeech):
    """No engine installed; reports how to get one."""

    name = "none"

    def available(self) -> bool:
        return False

    def speak(self, text: str) -> None:
        raise VoiceError(
            "no TTS engine installed (try: sudo apt install espeak-ng)"
        )


class FallbackTTS(TextToSpeech):
    """Tries each engine until one actually speaks."""

    name = "fallback"

    def __init__(self, engines: list[TextToSpeech]) -> None:
        self.engines = engines

    def available(self) -> bool:
        return any(engine.available() for engine in self.engines)

    def speak(self, text: str) -> None:
        errors: list[str] = []
        for engine in self.engines:
            if not engine.available():
                continue
            try:
                engine.speak(text)
                return
            except VoiceError as error:
                errors.append(str(error))
        raise VoiceError(
            "no TTS engine worked: " + ("; ".join(errors) or "none available")
        )


def create_tts(language: str = "es") -> TextToSpeech:
    """Pick the best engine installed on this machine."""
    candidates: list[TextToSpeech] = [
        SpeechDispatcherTTS("spd-say", language),
        EspeakTTS("espeak-ng", language),
        EspeakTTS("espeak", language),
    ]
    available = [engine for engine in candidates if engine.available()]

    if not available:
        return NullTTS()
    if len(available) == 1:
        return available[0]
    return FallbackTTS(available)

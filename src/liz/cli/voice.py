"""Voice interaction loop.

This is the CLI-side orchestration: it records a window of audio, skips
silence, asks the `SpeechToText` port what was said, waits for a wake
word, sends the command to the agent and reads the answer aloud through
the `TextToSpeech` port. A background thread keeps listening until the
user says `/voice off`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.voice import (
    SpeechToText,
    TextToSpeech,
    VoiceError,
    detect_wake_word,
    is_speech,
    parse_wake_words,
    strip_wake_word,
)
from liz.infrastructure.voice import audio as audio_capture
from liz.infrastructure.voice.speech_to_text import create_stt
from liz.infrastructure.voice.text_to_speech import create_tts


@dataclass
class VoiceTurn:
    """What was heard and what Liz answered."""

    transcript: str
    reply: str


class VoiceLoop:
    """Records windows of audio and turns commands into spoken answers."""

    def __init__(
        self,
        agent: Agent,
        settings: Settings,
        tts: TextToSpeech | None = None,
        stt: SpeechToText | None = None,
        recorder: Callable[..., Path] | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.tts = (
            tts if tts is not None
            else create_tts(settings.voice_language)
        )
        self.stt = (
            stt if stt is not None
            else create_stt(
                settings.voice_language,
                settings.whisper_bin,
                settings.whisper_model,
            )
        )
        self.wake_words = parse_wake_words(settings.voice_wake_words)
        self._recorder = recorder or audio_capture.record
        self._on_event = on_event or (lambda message: print(message, flush=True))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- one iteration, testable without threads -----------------------

    def step(self, require_wake_word: bool = True) -> VoiceTurn | None:
        """Record one window and answer it.

        Returns None for silence or speech that was not meant for Liz.
        """
        wav = self._recorder(self.settings.voice_record_seconds)
        try:
            pcm, rate = audio_capture.read_pcm(wav)
            if not is_speech(pcm, rate):
                return None

            transcript = self.stt.transcribe(wav).strip()
            if not transcript:
                return None

            if require_wake_word:
                if not detect_wake_word(transcript, self.wake_words):
                    return None
                transcript = strip_wake_word(transcript, self.wake_words)
                if not transcript:
                    self._on_event("voice: wake word without a command")
                    return None

            reply = self.agent.chat(transcript)
            try:
                self.tts.speak(reply)
            except VoiceError as error:
                # the answer still counts, it just stays on screen
                self._on_event(f"voice: {error}")
            return VoiceTurn(transcript=transcript, reply=reply)
        finally:
            Path(wav).unlink(missing_ok=True)

    # --- background listener -------------------------------------------

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        """Start the background listener; False when already running."""
        if self.running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="liz-voice", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Ask the listener to stop and wait for it to finish."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            grace = self.settings.voice_record_seconds + 15
            thread.join(timeout=grace)

    def _run(self) -> None:
        self._on_event(
            f"voice: listening, say {self.wake_words[0]!r} "
            "(/voice off stops it)"
        )
        while not self._stop.is_set():
            try:
                turn = self.step()
            except VoiceError as error:
                self._on_event(f"voice error: {error}")
                self._stop.wait(2.0)
                continue
            except Exception as error:  # keep the listener alive
                self._on_event(f"voice error: {error}")
                self._stop.wait(2.0)
                continue

            if turn is not None:
                self._on_event(f'you: "{turn.transcript}"')
                self._on_event(f"Liz: {turn.reply}")
            self._stop.wait(0.2)


def create_voice_loop(
    agent: Agent,
    settings: Settings,
    on_event: Callable[[str], None] | None = None,
) -> VoiceLoop:
    """Build the voice loop with the engines found on this machine."""
    return VoiceLoop(agent, settings, on_event=on_event)

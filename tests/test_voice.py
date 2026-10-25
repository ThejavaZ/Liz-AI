"""Phase 7 voice tests.

Everything runs without a microphone or an engine: PCM is synthesized,
subprocesses are mocked and the loop talks to fake agents and engines.
"""

from __future__ import annotations

import array
import math
import subprocess
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from liz.cli.commands import CommandHandler
from liz.cli.voice import VoiceLoop, create_voice_loop
from liz.config.settings import Settings
from liz.core.context import Context
from liz.core.voice import (
    DEFAULT_WAKE_WORDS,
    VoiceError,
    detect_wake_word,
    frame_energies,
    is_speech,
    normalize,
    parse_wake_words,
    strip_wake_word,
)
from liz.infrastructure.voice import audio as audio_capture
from liz.infrastructure.voice.speech_to_text import (
    NullSTT,
    WhisperCppSTT,
    create_stt,
    find_binary,
    find_model,
)
from liz.infrastructure.voice.text_to_speech import (
    EspeakTTS,
    FallbackTTS,
    NullTTS,
    SpeechDispatcherTTS,
    create_tts,
)

RATE = 16000


# --- helpers ------------------------------------------------------------


def pcm_tone(seconds: float = 1.0, amplitude: int = 8000) -> bytes:
    samples = array.array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * 440 * i / RATE))
            for i in range(int(seconds * RATE))
        ),
    )
    return samples.tobytes()


def pcm_silence(seconds: float = 1.0) -> bytes:
    return bytes(2 * int(seconds * RATE))


def write_wav(path: Path, pcm: bytes, rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return path


def _settings(**overrides) -> Settings:
    return Settings(
        ai_provider="gemini",
        ai_model="test-model",
        gemini_api_key="test-key",
        **overrides,
    )


def _make_handler(voice=None, **kwargs) -> CommandHandler:
    settings = _settings(**kwargs.pop("settings_kwargs", {}))
    registry = MagicMock()
    return CommandHandler(
        settings,
        Context("system prompt"),
        registry,
        voice=voice,
        **kwargs,
    )


class FakeSTT:
    name = "fake-stt"

    def __init__(self, text: str = "hola liz dime la hora", ok: bool = True):
        self.text = text
        self.ok = ok

    def available(self) -> bool:
        return self.ok

    def transcribe(self, wav: Path) -> str:
        return self.text


class FakeTTS:
    name = "fake-tts"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.spoken: list[str] = []

    def available(self) -> bool:
        return not self.fail

    def speak(self, text: str) -> None:
        if self.fail:
            raise VoiceError("no engine worked")
        self.spoken.append(text)


class FakeAgent:
    def __init__(self, reply: str = "Son las diez."):
        self.reply = reply
        self.calls: list[str] = []

    def chat(self, text: str) -> str:
        self.calls.append(text)
        return self.reply


def make_loop(
    tmp_path: Path,
    pcm: bytes | None = None,
    transcript: str = "hola liz dime la hora",
    stt_ok: bool = True,
    tts: FakeTTS | None = None,
    agent: FakeAgent | None = None,
) -> tuple[VoiceLoop, FakeAgent, FakeTTS]:
    """Build a VoiceLoop whose recorder rewrites the same WAV every step."""
    pcm = pcm_tone() if pcm is None else pcm
    agent = agent or FakeAgent()
    tts = tts or FakeTTS()

    def recorder(seconds: float) -> Path:
        return write_wav(tmp_path / "capture.wav", pcm)

    loop = VoiceLoop(
        agent,  # type: ignore[arg-type]
        _settings(voice_record_seconds=1, voice_wake_words="hola liz"),
        tts=tts,  # type: ignore[arg-type]
        stt=FakeSTT(transcript, ok=stt_ok),  # type: ignore[arg-type]
        recorder=recorder,
        on_event=lambda message: None,
    )
    return loop, agent, tts


# --- VAD ----------------------------------------------------------------


class TestVoiceActivityDetection:
    def test_loud_tone_counts_as_speech(self) -> None:
        assert is_speech(pcm_tone()) is True

    def test_silence_is_not_speech(self) -> None:
        assert is_speech(pcm_silence()) is False

    def test_quiet_noise_is_not_speech(self) -> None:
        # ~amplitude 50, well under the energy threshold
        assert is_speech(pcm_tone(amplitude=50)) is False

    def test_empty_audio_is_not_speech(self) -> None:
        assert is_speech(b"") is False

    def test_frame_energies_length(self) -> None:
        pcm = pcm_tone(seconds=1.0)
        energies = frame_energies(pcm, RATE)
        # 1000 ms / 30 ms per frame
        assert len(energies) == 33
        assert all(energy > 0 for energy in energies)

    def test_frame_energies_tolerates_odd_byte(self) -> None:
        energies = frame_energies(pcm_tone(seconds=0.2) + b"\x00", RATE)
        assert energies  # odd trailing byte dropped, no crash

    def test_threshold_boundary(self) -> None:
        pcm = pcm_tone(amplitude=3000)  # RMS ~2121 > 500
        assert is_speech(pcm) is True
        loud = bytes(
            array.array("h", [400] * int(0.5 * RATE)).tobytes()
        )
        assert is_speech(loud) is False


# --- wake words ---------------------------------------------------------


class TestWakeWords:
    def test_normalize_strips_accents_and_punctuation(self) -> None:
        assert normalize("¡Hola, Liz! ¿Qué?") == "hola liz que"

    def test_normalize_squeezes_whitespace(self) -> None:
        assert normalize("  hola   liz \n") == "hola liz"

    def test_detect_wake_word_variants(self) -> None:
        assert detect_wake_word("hola liz, dime la hora")
        assert detect_wake_word("¡Oye, Liz!")
        assert detect_wake_word("HEY LIZ!!")
        assert not detect_wake_word("hola qué tal")
        assert not detect_wake_word("")

    def test_detect_custom_wake_words(self) -> None:
        words = ("asistente",)
        assert detect_wake_word("oye asistente", words)
        assert not detect_wake_word("hola liz", words)

    def test_strip_removes_only_wake_word(self) -> None:
        assert strip_wake_word("hola liz dime la hora") == "dime la hora"

    def test_strip_without_wake_word_keeps_text(self) -> None:
        assert strip_wake_word("dime la hora") == "dime la hora"

    def test_parse_defaults_when_empty(self) -> None:
        assert parse_wake_words(" , , ") == DEFAULT_WAKE_WORDS

    def test_parse_custom_list(self) -> None:
        assert parse_wake_words("hola liz,  computer ") == (
            "hola liz",
            "computer",
        )


# --- audio capture ------------------------------------------------------


class TestAudioCapture:
    def test_recorder_missing_binary(self, monkeypatch) -> None:
        monkeypatch.setattr(audio_capture.shutil, "which", lambda name: None)
        with pytest.raises(VoiceError, match="arecord"):
            audio_capture.record(1)

    def test_record_builds_expected_command(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(audio_capture, "recorder_available", lambda: True)
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            Path(command[-1]).write_bytes(b"RIFF")
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(audio_capture.subprocess, "run", fake_run)
        out = audio_capture.record(5, out=tmp_path / "x.wav")

        assert out == tmp_path / "x.wav"
        command = captured["command"]
        assert command[0] == "arecord"
        assert "-d" in command and command[command.index("-d") + 1] == "5"
        assert "-r" in command and command[command.index("-r") + 1] == str(RATE)

    def test_record_reports_stderr(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(audio_capture, "recorder_available", lambda: True)
        monkeypatch.setattr(
            audio_capture.subprocess,
            "run",
            lambda command, **kwargs: MagicMock(
                returncode=1, stderr="no soundcards found"
            ),
        )
        with pytest.raises(VoiceError, match="no soundcards"):
            audio_capture.record(1, out=tmp_path / "x.wav")
        assert not (tmp_path / "x.wav").exists()

    def test_record_timeout(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(audio_capture, "recorder_available", lambda: True)

        def expired(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 5)

        monkeypatch.setattr(audio_capture.subprocess, "run", expired)
        with pytest.raises(VoiceError, match="timed out"):
            audio_capture.record(1, out=tmp_path / "x.wav")

    def test_read_pcm_roundtrip(self, tmp_path) -> None:
        pcm = pcm_tone(seconds=0.5)
        wav = write_wav(tmp_path / "a.wav", pcm)

        read_pcm, rate = audio_capture.read_pcm(wav)

        assert read_pcm == pcm
        assert rate == RATE

    def test_read_pcm_rejects_bad_file(self, tmp_path) -> None:
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"not a wav")
        with pytest.raises(VoiceError):
            audio_capture.read_pcm(broken)


# --- text to speech -----------------------------------------------------


class TestTextToSpeech:
    def test_spd_say_command(self) -> None:
        engine = SpeechDispatcherTTS("spd-say", "es")
        assert engine.command("hola") == ["spd-say", "-w", "-l", "es", "hola"]

    def test_espeak_command(self) -> None:
        engine = EspeakTTS("espeak-ng", "es")
        assert engine.command("hola")[0] == "espeak-ng"
        assert "-v" in engine.command("hola")

    def test_available_follows_path(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/bin/x" if name == "spd-say" else None
        )
        assert SpeechDispatcherTTS("spd-say").available() is True
        assert EspeakTTS("espeak-ng").available() is False

    def test_speak_success(self, monkeypatch) -> None:
        calls = []

        monkeypatch.setattr(
            "subprocess.run",
            lambda command, **kwargs: (
                calls.append(command) or MagicMock(returncode=0, stderr="")
            ),
        )
        SpeechDispatcherTTS("spd-say").speak(" hola ")
        assert calls and calls[0][-1] == "hola"

    def test_speak_ignores_empty_text(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "subprocess.run",
            lambda command, **kwargs: pytest.fail("must not run"),
        )
        SpeechDispatcherTTS("spd-say").speak("   ")

    def test_speak_failure_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "subprocess.run",
            lambda command, **kwargs: MagicMock(returncode=1, stderr="boom"),
        )
        with pytest.raises(VoiceError, match="boom"):
            SpeechDispatcherTTS("spd-say").speak("hola")

    def test_null_tts_raises_install_hint(self) -> None:
        with pytest.raises(VoiceError, match="espeak-ng"):
            NullTTS().speak("hola")
        assert NullTTS().available() is False

    def test_fallback_tries_next_engine(self) -> None:
        broken, working = FakeTTS(fail=True), FakeTTS()
        fallback = FallbackTTS([broken, working])  # type: ignore[list-item]

        fallback.speak("hola")

        assert working.spoken == ["hola"]
        assert broken.spoken == []

    def test_fallback_reports_all_errors(self) -> None:
        fallback = FallbackTTS([FakeTTS(fail=True)])  # type: ignore[list-item]
        with pytest.raises(VoiceError, match="no TTS engine worked"):
            fallback.speak("hola")

    def test_create_tts_none_available(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        engine = create_tts()
        assert isinstance(engine, NullTTS)

    def test_create_tts_single_engine(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "shutil.which", lambda name: "/bin/espeak-ng" if name == "espeak-ng" else None
        )
        engine = create_tts()
        assert isinstance(engine, EspeakTTS)
        assert engine.binary == "espeak-ng"

    def test_create_tts_multiple_engines(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
        engine = create_tts()
        assert isinstance(engine, FallbackTTS)
        assert len(engine.engines) == 3  # spd-say + espeak-ng + espeak


# --- speech to text -----------------------------------------------------


class TestSpeechToText:
    def test_find_binary_explicit(self, tmp_path) -> None:
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        assert find_binary(str(binary)) == str(binary)

    def test_find_binary_missing_explicit_falls_through(self) -> None:
        found = find_binary("/nonexistent/whisper-cli")
        assert found != "/nonexistent/whisper-cli"

    def test_find_model_prefers_bigger(
        self, tmp_path, monkeypatch
    ) -> None:
        (tmp_path / "ggml-tiny.bin").write_bytes(b"x")
        (tmp_path / "ggml-base.bin").write_bytes(b"x")
        monkeypatch.setattr(
            "liz.infrastructure.voice.speech_to_text._MODEL_DIRS",
            (tmp_path,),
        )
        found = find_model()
        assert found is not None and "base" in found.name

    def test_find_model_explicit(self, tmp_path) -> None:
        model = tmp_path / "custom.bin"
        model.write_bytes(b"x")
        assert find_model(str(model)) == model

    def test_find_model_none_when_empty(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "liz.infrastructure.voice.speech_to_text._MODEL_DIRS",
            (tmp_path,),
        )
        assert find_model() is None

    def test_create_stt_null_without_pieces(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr(
            "liz.infrastructure.voice.speech_to_text._MODEL_DIRS",
            (tmp_path,),
        )
        assert isinstance(create_stt(), NullSTT)

    def test_create_stt_whisper_when_found(
        self, monkeypatch, tmp_path
    ) -> None:
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        model = tmp_path / "ggml-base.bin"
        model.write_bytes(b"x")
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr(
            "liz.infrastructure.voice.speech_to_text._MODEL_DIRS",
            (tmp_path,),
        )

        engine = create_stt(binary=str(binary))

        assert isinstance(engine, WhisperCppSTT)
        assert engine.model == model

    def _engine(self, tmp_path) -> WhisperCppSTT:
        binary = tmp_path / "whisper-cli"
        binary.write_text("")
        model = tmp_path / "ggml-base.bin"
        model.write_bytes(b"x")
        return WhisperCppSTT(str(binary), model)

    def test_available(self, tmp_path) -> None:
        assert self._engine(tmp_path).available() is True

    def test_command_flags(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        command = engine.command(Path("/tmp/a.wav"), Path("/tmp/out"), True)
        assert command[0] == str(engine.binary)
        assert "-m" in command and "-f" in command and "-l" in command
        assert command[command.index("-l") + 1] == "es"
        assert "-otxt" in command
        assert "--no-prints" in command

    def test_transcribe_reads_output_file(self, tmp_path, monkeypatch) -> None:
        engine = self._engine(tmp_path)
        wav = write_wav(tmp_path / "a.wav", pcm_tone(seconds=0.2))

        def fake_run(command, **kwargs):
            base = Path(command[command.index("-of") + 1])
            base.with_suffix(".txt").write_text(" hola liz \n")
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(engine_module().subprocess, "run", fake_run)

        assert engine.transcribe(wav) == "hola liz"

    def test_transcribe_retries_without_flag(self, tmp_path, monkeypatch) -> None:
        engine = self._engine(tmp_path)
        wav = write_wav(tmp_path / "a.wav", pcm_tone(seconds=0.2))
        attempts = []

        def fake_run(command, **kwargs):
            attempts.append("--no-prints" in command)
            if "--no-prints" in command:
                return MagicMock(returncode=1, stderr="unknown option")
            base = Path(command[command.index("-of") + 1])
            base.with_suffix(".txt").write_text("hola")
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(engine_module().subprocess, "run", fake_run)

        assert engine.transcribe(wav) == "hola"
        assert attempts == [True, False]

    def test_transcribe_missing_wav(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        with pytest.raises(VoiceError, match="no audio file"):
            engine.transcribe(tmp_path / "nope.wav")

    def test_transcribe_failure_raises(self, tmp_path, monkeypatch) -> None:
        engine = self._engine(tmp_path)
        wav = write_wav(tmp_path / "a.wav", pcm_tone(seconds=0.2))
        monkeypatch.setattr(
            engine_module().subprocess,
            "run",
            lambda command, **kwargs: MagicMock(
                returncode=1, stderr="", stdout="fatal: bad model"
            ),
        )
        with pytest.raises(VoiceError, match="whisper.cpp failed"):
            engine.transcribe(wav)

    def test_transcribe_no_output_file(self, tmp_path, monkeypatch) -> None:
        engine = self._engine(tmp_path)
        wav = write_wav(tmp_path / "a.wav", pcm_tone(seconds=0.2))
        monkeypatch.setattr(
            engine_module().subprocess,
            "run",
            lambda command, **kwargs: MagicMock(returncode=0, stderr=""),
        )
        with pytest.raises(VoiceError, match="no transcript"):
            engine.transcribe(wav)

    def test_null_stt_raises(self) -> None:
        with pytest.raises(VoiceError, match="whisper.cpp"):
            NullSTT().transcribe(Path("/tmp/x.wav"))


def engine_module():
    from liz.infrastructure.voice import speech_to_text

    return speech_to_text


# --- the loop -----------------------------------------------------------


class TestVoiceLoop:
    def test_silence_returns_none(self, tmp_path) -> None:
        loop, agent, _ = make_loop(tmp_path, pcm=pcm_silence())
        assert loop.step() is None
        assert agent.calls == []

    def test_wake_word_stripped_before_agent(self, tmp_path) -> None:
        loop, agent, tts = make_loop(tmp_path)

        turn = loop.step()

        assert turn is not None
        assert turn.transcript == "dime la hora"
        assert agent.calls == ["dime la hora"]
        assert tts.spoken == ["Son las diez."]
        assert turn.reply == "Son las diez."

    def test_no_wake_word_ignores_command(self, tmp_path) -> None:
        loop, agent, _ = make_loop(
            tmp_path, transcript="dime la hora"
        )
        assert loop.step() is None
        assert agent.calls == []

    def test_wake_word_only_reports(self, tmp_path) -> None:
        events = []
        loop, agent, _ = make_loop(tmp_path, transcript="hola liz")
        loop._on_event = events.append  # type: ignore[method-assign]

        assert loop.step() is None
        assert agent.calls == []
        assert any("wake word" in event for event in events)

    def test_empty_transcript_returns_none(self, tmp_path) -> None:
        loop, agent, _ = make_loop(tmp_path, transcript="   ")
        assert loop.step() is None
        assert agent.calls == []

    def test_tts_failure_still_returns_turn(self, tmp_path) -> None:
        events = []
        loop, agent, _ = make_loop(tmp_path, tts=FakeTTS(fail=True))
        loop._on_event = events.append  # type: ignore[method-assign]

        turn = loop.step()

        assert turn is not None
        assert turn.reply == "Son las diez."
        assert any("voice:" in event for event in events)

    def test_talk_skips_wake_word(self, tmp_path) -> None:
        loop, agent, _ = make_loop(tmp_path, transcript="dime la hora")

        turn = loop.step(require_wake_word=False)

        assert turn is not None
        assert agent.calls == ["dime la hora"]

    def test_recording_is_deleted(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)
        loop.step()
        assert not (tmp_path / "capture.wav").exists()

    def test_recorder_error_propagates(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)

        def boom(seconds: float) -> Path:
            raise VoiceError("mic busy")

        loop._recorder = boom  # type: ignore[method-assign]
        with pytest.raises(VoiceError, match="mic busy"):
            loop.step()

    def test_thread_start_and_stop(self, tmp_path) -> None:
        loop, agent, _ = make_loop(tmp_path, pcm=pcm_silence())

        assert loop.running is False
        assert loop.start() is True
        assert loop.running is True
        assert loop.start() is False  # already running

        loop.stop()
        assert loop.running is False

    def test_thread_answers_transcript(self, tmp_path) -> None:
        events = []
        loop, agent, _ = make_loop(tmp_path, transcript="hola liz dime la hora")
        loop._on_event = events.append  # type: ignore[method-assign]

        loop.start()
        deadline = time.time() + 5
        while time.time() < deadline and not agent.calls:
            time.sleep(0.05)
        loop.stop()

        assert agent.calls == ["dime la hora"]
        assert any(event.startswith("Liz:") for event in events)

    def test_stop_without_start_is_safe(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)
        loop.stop()  # must not raise

    def test_create_voice_loop_uses_factories(self, monkeypatch) -> None:
        agent = FakeAgent()
        loop = create_voice_loop(agent, _settings())  # type: ignore[arg-type]
        assert loop.agent is agent
        assert loop.wake_words  # parsed from settings


# --- commands -----------------------------------------------------------


class TestVoiceCommands:
    def test_help_lists_voice_commands(self) -> None:
        handler = _make_handler()
        result = handler.handle("/help")
        assert "/talk" in result.output
        assert "/voice" in result.output

    def test_voice_without_support(self) -> None:
        handler = _make_handler()
        result = handler.handle("/voice")
        assert "not enabled" in result.output

    def test_voice_status(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, stt_ok=False)
        handler = _make_handler(voice=loop)

        result = handler.handle("/voice")

        assert "STT" in result.output
        assert "not installed" in result.output
        assert "hola liz" in result.output
        assert "whisper.cpp" in result.output

    def test_voice_status_with_engines(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)
        handler = _make_handler(voice=loop)

        result = handler.handle("/voice")

        assert "fake-stt" in result.output
        assert "fake-tts" in result.output
        assert "stopped" in result.output

    def test_voice_on_without_stt_shows_install(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, stt_ok=False)
        handler = _make_handler(voice=loop)

        result = handler.handle("/voice on")

        assert "whisper" in result.output
        assert loop.running is False

    def test_voice_on_starts_listener(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, pcm=pcm_silence())
        handler = _make_handler(voice=loop)

        result = handler.handle("/voice on")
        try:
            assert "Voice on" in result.output
            assert loop.running is True
        finally:
            loop.stop()

    def test_voice_on_twice_warns(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, pcm=pcm_silence())
        handler = _make_handler(voice=loop)
        loop.start()
        try:
            result = handler.handle("/voice on")
            assert "already running" in result.output
        finally:
            loop.stop()

    def test_voice_off(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, pcm=pcm_silence())
        handler = _make_handler(voice=loop)
        loop.start()

        result = handler.handle("/voice off")

        assert "Voice off" in result.output
        assert loop.running is False

    def test_voice_off_when_stopped(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)
        handler = _make_handler(voice=loop)

        result = handler.handle("/voice off")

        assert "not running" in result.output

    def test_voice_unknown_action(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)
        handler = _make_handler(voice=loop)

        result = handler.handle("/voice banana")

        assert "Usage" in result.output

    def test_talk_without_support(self) -> None:
        handler = _make_handler()
        result = handler.handle("/talk")
        assert "not enabled" in result.output

    def test_talk_without_stt(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, stt_ok=False)
        handler = _make_handler(voice=loop)

        result = handler.handle("/talk")

        assert "whisper.cpp" in result.output

    def test_talk_success(self, tmp_path) -> None:
        loop, agent, _ = make_loop(tmp_path, transcript="dime la hora")
        handler = _make_handler(voice=loop)

        result = handler.handle("/talk")

        assert 'You said: "dime la hora"' in result.output
        assert "Son las diez." in result.output
        assert agent.calls == ["dime la hora"]

    def test_talk_silence(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path, pcm=pcm_silence())
        handler = _make_handler(voice=loop)

        result = handler.handle("/talk")

        assert "No voice detected" in result.output

    def test_talk_reports_voice_error(self, tmp_path) -> None:
        loop, _, _ = make_loop(tmp_path)
        handler = _make_handler(voice=loop)

        def boom(seconds: float) -> Path:
            raise VoiceError("mic busy")

        loop._recorder = boom  # type: ignore[method-assign]

        result = handler.handle("/talk")

        assert "Voice error: mic busy" in result.output

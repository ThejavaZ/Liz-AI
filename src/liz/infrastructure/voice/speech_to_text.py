"""Speech-to-text via whisper.cpp.

The engine is an external binary, keeping the project free of Python
ML dependencies: install whisper.cpp once, download a ggml model, and
point `WHISPER_BIN` / `WHISPER_MODEL` at them (the factory also probes
the usual locations).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from liz.core.voice import SpeechToText, VoiceError

_BINARIES = ("whisper-cli", "whisper-cpp", "whisper")
_MODEL_PREFERENCE = ("ggml-small", "ggml-base", "ggml-tiny")
_MODEL_DIRS = (
    Path("models"),
    Path("whisper.cpp/models"),
    Path.home() / ".cache" / "whisper",
    Path.home() / ".cache" / "whisper.cpp",
)
_BINARY_CANDIDATES = (
    Path("whisper.cpp/build/bin/whisper-cli"),
    Path("build/bin/whisper-cli"),
    Path.home() / ".local" / "bin" / "whisper-cli",
)


def find_binary(explicit: str | None = None) -> str | None:
    """Locate the whisper.cpp binary."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    for name in _BINARIES:
        found = shutil.which(name)
        if found:
            return found

    for candidate in _BINARY_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def find_model(explicit: str | None = None) -> Path | None:
    """Locate a ggml model, preferring better ones."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path

    for directory in _MODEL_DIRS:
        if not directory.is_dir():
            continue
        for preferred in _MODEL_PREFERENCE:
            match = next(directory.glob(f"{preferred}*.bin"), None)
            if match:
                return match
        any_model = next(directory.glob("ggml-*.bin"), None)
        if any_model:
            return any_model
    return None


class WhisperCppSTT(SpeechToText):
    """Runs the whisper.cpp CLI over a recorded WAV file."""

    name = "whisper.cpp"

    def __init__(
        self,
        binary: str,
        model: Path,
        language: str = "es",
        timeout: float = 120.0,
    ) -> None:
        self.binary = binary
        self.model = Path(model)
        self.language = language
        self.timeout = timeout

    def available(self) -> bool:
        return Path(self.binary).is_file() and self.model.is_file()

    def command(self, wav: Path, base: Path, no_prints: bool) -> list[str]:
        command = [
            self.binary,
            "-m", str(self.model),
            "-f", str(wav),
            "-l", self.language,
            "-of", str(base),
            "-otxt",
        ]
        if no_prints:
            command.append("--no-prints")
        return command

    def transcribe(self, wav: Path) -> str:
        wav = Path(wav)
        if not wav.is_file():
            raise VoiceError(f"no audio file: {wav}")

        workdir = Path(tempfile.mkdtemp(prefix="liz-stt-"))
        base = workdir / "transcript"
        text_file = base.with_suffix(".txt")

        def run(no_prints: bool) -> subprocess.CompletedProcess:
            return subprocess.run(
                self.command(wav, base, no_prints),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

        try:
            process = run(no_prints=True)
            if process.returncode != 0:
                # older builds do not know --no-prints
                process = run(no_prints=False)
            if process.returncode != 0:
                detail = (process.stderr or process.stdout or "").strip()
                raise VoiceError(
                    f"whisper.cpp failed: {detail.splitlines()[-1] if detail else 'unknown error'}"
                )

            if not text_file.is_file():
                raise VoiceError("whisper.cpp produced no transcript")
            return text_file.read_text(encoding="utf-8").strip()
        except subprocess.TimeoutExpired as error:
            raise VoiceError("transcription timed out") from error
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class NullSTT(SpeechToText):
    """No engine installed; reports how to get one."""

    name = "none"

    def available(self) -> bool:
        return False

    def transcribe(self, wav: Path) -> str:
        raise VoiceError(
            "no STT engine installed (install whisper.cpp and set "
            "WHISPER_BIN / WHISPER_MODEL, see /voice)"
        )


def create_stt(
    language: str = "es",
    binary: str | None = None,
    model: str | None = None,
) -> SpeechToText:
    """Pick whisper.cpp when both binary and model exist."""
    found_binary = find_binary(binary)
    found_model = find_model(model)

    if found_binary and found_model:
        return WhisperCppSTT(found_binary, found_model, language)
    return NullSTT()

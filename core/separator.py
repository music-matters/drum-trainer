"""
Stem separation using Meta's Demucs Python API.

Uses the demucs Python API directly rather than a subprocess so we control
all audio I/O with soundfile, sidestepping torchaudio's torchcodec backend
which requires specific FFmpeg shared DLLs that are difficult to satisfy on
Windows with a static FFmpeg build.

Model options (quality vs speed trade-off):
  htdemucs       — fast, good quality  (default)
  htdemucs_ft    — fine-tuned, slightly better drums isolation
  mdx_extra      — high quality, slower
"""
from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Block torchcodec BEFORE anything imports torchaudio.
#
# torchcodec links against FFmpeg 4/5 shared DLLs that are not present in a
# typical static FFmpeg install.  We stub the package so torchaudio sees it
# as importable (avoiding hard errors) but we never call torchaudio.save —
# all audio I/O goes through soundfile instead.
# ---------------------------------------------------------------------------
def _install_torchcodec_stub() -> None:
    stub = types.ModuleType("torchcodec")
    stub.__version__ = "0.0.0.stub"
    stub.__path__ = []  # marks it as a package

    for subname in (
        "torchcodec.encoders",
        "torchcodec.decoders",
        "torchcodec._internally_replaced_utils",
    ):
        sub = types.ModuleType(subname)
        # no-op the library loader so it never tries to dlopen anything
        sub.load_torchcodec_shared_libraries = lambda *a, **kw: None  # type: ignore
        sys.modules.setdefault(subname, sub)

    sys.modules.setdefault("torchcodec", stub)

_install_torchcodec_stub()

# ---------------------------------------------------------------------------
# Normal imports (safe now that torchcodec is stubbed)
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

STEM_NAMES = ("drums", "bass", "vocals", "other")


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------

def _load_for_demucs(
    audio_file: Path,
    target_sr: int,
    target_channels: int,
) -> torch.Tensor:
    """
    Load *audio_file* with soundfile, resample and channel-match as needed.

    Returns a float32 tensor of shape [channels, samples].
    """
    data, file_sr = sf.read(str(audio_file), dtype="float32", always_2d=True)
    # soundfile gives [samples, channels] → transpose to [channels, samples]
    data = data.T

    # Channel adjustment
    if data.shape[0] == 1 and target_channels == 2:
        data = np.repeat(data, 2, axis=0)
    elif data.shape[0] > target_channels:
        data = data[:target_channels]

    # Resample if needed
    if file_sr != target_sr:
        import librosa
        data = np.stack([
            librosa.resample(ch, orig_sr=file_sr, target_sr=target_sr)
            for ch in data
        ])

    return torch.from_numpy(data).float()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_stems(
    audio_file: Path,
    output_dir: Path,
    model: str = "htdemucs",
    device: str = "auto",
) -> dict[str, Path]:
    """
    Run demucs on *audio_file* and return paths to each stem.

    Args:
        audio_file: Path to the source WAV.
        output_dir: Root directory for output stems.
        model:      Demucs model name.
        device:     'cuda', 'cpu', or 'auto' (picks GPU if available).

    Returns:
        Dict mapping stem name → Path, plus a 'no_drums' mixed track.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else torch.device(device)

    # Load model weights (downloads on first use, cached thereafter)
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    demucs_model = get_model(model)
    demucs_model.eval()
    demucs_model.to(_device)

    target_sr = demucs_model.samplerate
    target_channels = demucs_model.audio_channels

    # Load and prepare audio
    wav = _load_for_demucs(audio_file, target_sr, target_channels)
    mix = wav.unsqueeze(0).to(_device)   # [1, channels, samples]

    # Separate — no torchaudio.save involved
    with torch.no_grad():
        sources = apply_model(demucs_model, mix)   # [1, n_sources, channels, samples]

    # Write stems with soundfile
    stems_dir = output_dir / model / audio_file.stem
    stems_dir.mkdir(parents=True, exist_ok=True)

    stem_paths: dict[str, Path] = {}
    for i, name in enumerate(demucs_model.sources):
        audio = sources[0, i].cpu().float().numpy().T   # [samples, channels]
        path = stems_dir / f"{name}.wav"
        sf.write(str(path), audio, target_sr, subtype="FLOAT")
        stem_paths[name] = path

    # Build no_drums backing track
    no_drums_path = stems_dir / "no_drums.wav"
    if not no_drums_path.exists():
        _mix_stems(
            [stem_paths[s] for s in demucs_model.sources if s != "drums"],
            no_drums_path,
        )
    stem_paths["no_drums"] = no_drums_path
    return stem_paths


def _mix_stems(stem_files: list[Path], output_path: Path) -> None:
    """Sum multiple WAV stems into a single file (equal weights)."""
    arrays = []
    sr = None
    for f in stem_files:
        data, file_sr = sf.read(str(f), dtype="float32")
        if sr is None:
            sr = file_sr
        arrays.append(data)

    mixed = np.sum(arrays, axis=0)
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    sf.write(str(output_path), mixed, sr)

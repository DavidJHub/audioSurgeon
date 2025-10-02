import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audioPrepUtils import preprocess_audio_for_vad
from audioProcessing import run_pipeline
from measureActivity import vad_energy_adaptive_array


def test_preprocess_pipeline_runs_without_optional_dependencies():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    result = preprocess_audio_for_vad(
        y,
        sr,
        enable_leveler=False,
        use_pedalboard=False,
        use_webrtcvad=False,
        use_pitch_gate=False,
        use_pcen=False,
    )

    assert result["y"].shape == y.shape
    assert result["log_rms"].ndim == 1
    assert np.all(np.isfinite(result["log_rms"]))


def test_activity_detector_matches_processed_frames():
    sr = 8000
    duration = 0.5
    silence = np.zeros(int(sr * duration), dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration), endpoint=False))
    y = np.concatenate([silence, tone.astype(np.float32), silence])

    mask = vad_energy_adaptive_array(y, sr, returns="mask")
    segments = vad_energy_adaptive_array(y, sr, returns="segments")

    assert mask.ndim == 1
    assert all(0 <= start < end <= len(y) / sr for start, end in segments)


def test_pipeline_processes_audio(tmp_path):
    sr = 8000
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    wav_path = input_dir / "sample.wav"
    sf.write(wav_path, tone, sr)

    summary_json = tmp_path / "summary.json"

    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        preproc_kwargs={
            "use_pedalboard": False,
            "use_webrtcvad": False,
            "use_pitch_gate": False,
        },
        vol_window_sec=0.5,
        vol_hop_sec=0.25,
        summary_json=summary_json,
    )

    assert summary_json.exists()
    data = summary_json.read_text(encoding="utf-8")
    assert "sample.wav" in data

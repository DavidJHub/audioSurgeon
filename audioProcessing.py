"""Simple entry point that runs the audio processing pipeline with preset values."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from audioPreprocessing import main_process_batch
from audioShazam import merge_peaks, ncc_fft


# Default locations for batch processing. Update these paths to match your project.
DEFAULT_INPUT_DIR = Path("input_audios")
DEFAULT_OUTPUT_DIR = Path("processed_audios")

# Template configuration (set to ``None`` when the template stage is not required).
DEFAULT_TEMPLATE_PATH: Optional[Path] = None
DEFAULT_TEMPLATE_RESAMPLE_TO: Optional[int] = None

# Overlap/template detection defaults.
DEFAULT_TEMPLATE_THRESH = 0.5
DEFAULT_TEMPLATE_MIN_DISTANCE = 0.4
DEFAULT_TEMPLATE_NMS = 0.25

# Preprocessing stage defaults.
DEFAULT_PREPROC_KWARGS: Dict[str, Any] = {
    "use_pedalboard": True,
    "use_webrtcvad": True,
    "use_pitch_gate": True,
}

# Batch processing behaviour toggles.
DEFAULT_PRESERVE_SUBDIRS = True
DEFAULT_OVERWRITE = True
DEFAULT_MONO = True
DEFAULT_FORCE_WAV_OUT = True

# Loudness trace configuration.
DEFAULT_VOL_WINDOW_SEC = 3.0
DEFAULT_VOL_HOP_SEC: Optional[float] = None

# Optional summary export locations.
DEFAULT_SUMMARY_JSON: Optional[Path] = None
DEFAULT_SUMMARY_EXCEL: Optional[Path] = None


def _build_detect_kwargs(template_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Return default detection kwargs when a template is provided."""

    if template_path is None:
        return None

    return {
        "thresh": DEFAULT_TEMPLATE_THRESH,
        "min_distance_sec": DEFAULT_TEMPLATE_MIN_DISTANCE,
        "nms_merge_sec": DEFAULT_TEMPLATE_NMS,
        "ncc_fft": ncc_fft,
        "merge_peaks": merge_peaks,
    }


def run_pipeline(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    template_path: Optional[Path] = DEFAULT_TEMPLATE_PATH,
    template_resample_to: Optional[int] = DEFAULT_TEMPLATE_RESAMPLE_TO,
    cut_kwargs: Optional[Dict[str, Any]] = None,
    detect_kwargs: Optional[Dict[str, Any]] = None,
    preproc_kwargs: Optional[Dict[str, Any]] = None,
    preserve_subdirs: bool = DEFAULT_PRESERVE_SUBDIRS,
    overwrite: bool = DEFAULT_OVERWRITE,
    mono: bool = DEFAULT_MONO,
    force_wav_out: bool = DEFAULT_FORCE_WAV_OUT,
    vol_window_sec: float = DEFAULT_VOL_WINDOW_SEC,
    vol_hop_sec: Optional[float] = DEFAULT_VOL_HOP_SEC,
    summary_json: Optional[Path] = DEFAULT_SUMMARY_JSON,
    summary_excel: Optional[Path] = DEFAULT_SUMMARY_EXCEL,
) -> None:
    """Execute ``main_process_batch`` using the configured defaults."""

    effective_detect_kwargs = detect_kwargs
    if effective_detect_kwargs is None:
        effective_detect_kwargs = _build_detect_kwargs(template_path)

    effective_preproc_kwargs = DEFAULT_PREPROC_KWARGS.copy()
    if preproc_kwargs:
        effective_preproc_kwargs.update(preproc_kwargs)

    df = main_process_batch(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        template_path=str(template_path) if template_path else None,
        template_resample_to=template_resample_to,
        cut_kwargs=cut_kwargs,
        detect_kwargs=effective_detect_kwargs,
        preproc_kwargs=effective_preproc_kwargs,
        preserve_subdirs=preserve_subdirs,
        overwrite=overwrite,
        mono=mono,
        force_wav_out=force_wav_out,
        verbose=True,
        vol_window_sec=vol_window_sec,
        vol_hop_sec=vol_hop_sec,
        summary_excel_path=str(summary_excel) if summary_excel else None,
    )

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            df.to_json(orient="records", indent=2, default_handler=str),
            encoding="utf-8",
        )


def main() -> None:
    """Run the pipeline with the module defaults."""

    run_pipeline()


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()

"""Convenience CLI for running the audio processing pipeline on a folder of files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from audioPreprocessing import main_process_batch
from audioShazam import merge_peaks, ncc_fft


DEFAULT_TEMPLATE_THRESH = 0.5
DEFAULT_TEMPLATE_MIN_DISTANCE = 0.4
DEFAULT_TEMPLATE_NMS = 0.25


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the audio processing pipeline over a directory of audio files.",
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing input audio files")
    parser.add_argument("output_dir", type=Path, help="Destination folder for processed audio")
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional hangup template to use for detection",
    )
    parser.add_argument(
        "--template-resample-to",
        type=int,
        default=None,
        help="If provided, resample the template waveform to this sample rate",
    )
    parser.add_argument(
        "--template-thresh",
        type=_positive_float,
        default=DEFAULT_TEMPLATE_THRESH,
        help="Detection threshold when using a template (default: %(default)s)",
    )
    parser.add_argument(
        "--template-min-distance",
        type=_positive_float,
        default=DEFAULT_TEMPLATE_MIN_DISTANCE,
        help="Minimum separation (seconds) between template detections",
    )
    parser.add_argument(
        "--template-nms-merge",
        type=_positive_float,
        default=DEFAULT_TEMPLATE_NMS,
        help="NMS merge window (seconds) for template detections",
    )
    parser.add_argument(
        "--keep-subdirs",
        action="store_true",
        help="Preserve the sub-directory structure under the output folder",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite processed audio that already exists",
    )
    parser.add_argument(
        "--stereo",
        dest="mono",
        action="store_false",
        help="Process audio in stereo instead of converting to mono",
    )
    parser.add_argument(
        "--skip-wav",
        dest="force_wav_out",
        action="store_false",
        help="Keep the original file extension when writing processed audio",
    )
    parser.add_argument(
        "--disable-pedalboard",
        action="store_true",
        help="Disable the optional pedalboard processing stage",
    )
    parser.add_argument(
        "--disable-webrtcvad",
        action="store_true",
        help="Disable the optional WebRTC VAD refinement stage",
    )
    parser.add_argument(
        "--disable-pitch-gate",
        action="store_true",
        help="Disable the optional pitch-based gating stage",
    )
    parser.add_argument(
        "--vol-window",
        type=_positive_float,
        default=3.0,
        help="Window size in seconds for the diagnostic loudness trace",
    )
    parser.add_argument(
        "--vol-hop",
        type=float,
        default=None,
        help="Hop size in seconds for the diagnostic loudness trace (defaults to the window)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to store the resulting dataframe summary in JSON format",
    )
    parser.add_argument(
        "--summary-excel",
        type=Path,
        default=None,
        help="Optional path to store the resulting dataframe summary in Excel format",
    )
    parser.set_defaults(overwrite=True, mono=True, force_wav_out=True)
    return parser


def _build_detect_kwargs(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.template is None:
        return None
    return {
        "thresh": args.template_thresh,
        "min_distance_sec": args.template_min_distance,
        "nms_merge_sec": args.template_nms_merge,
        "ncc_fft": ncc_fft,
        "merge_peaks": merge_peaks,
    }


def _build_preproc_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "use_pedalboard": not args.disable_pedalboard,
        "use_webrtcvad": not args.disable_webrtcvad,
        "use_pitch_gate": not args.disable_pitch_gate,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    detect_kwargs = _build_detect_kwargs(args)
    preproc_kwargs = _build_preproc_kwargs(args)

    df = main_process_batch(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        template_path=str(args.template) if args.template else None,
        template_resample_to=args.template_resample_to,
        cut_kwargs=None,
        detect_kwargs=detect_kwargs,
        preproc_kwargs=preproc_kwargs,
        preserve_subdirs=args.keep_subdirs,
        overwrite=args.overwrite,
        mono=args.mono,
        force_wav_out=args.force_wav_out,
        verbose=True,
        vol_window_sec=args.vol_window,
        vol_hop_sec=args.vol_hop,
        summary_excel_path=str(args.summary_excel) if args.summary_excel else None,
    )

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(df.to_json(orient="records", indent=2, default_handler=str), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

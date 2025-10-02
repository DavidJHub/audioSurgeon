import warnings

import librosa
import numpy as np
from typing import Tuple, List

try:  # pragma: no cover - optional project-specific dependency
    import database.dbConfig as dbcfg
except ImportError:  # pragma: no cover - exercised outside the main project
    class _DummyCfg:
        HF_TOKEN = None

    dbcfg = _DummyCfg()  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pyannote.audio import Pipeline
    HAVE_PYANONTE = True
except ImportError:  # pragma: no cover - exercised when pyannote is unavailable
    Pipeline = None  # type: ignore
    HAVE_PYANONTE = False

try:
    from .constants import (
        DEFAULT_FRAME_LENGTH,
        DEFAULT_HOP_LENGTH,
        DEFAULT_N_FFT,
        DEFAULT_WIN_LENGTH,
    )
except ImportError:  # pragma: no cover - compatibility for alternate package names
    try:
        from audio.constants import (  # type: ignore
            DEFAULT_FRAME_LENGTH,
            DEFAULT_HOP_LENGTH,
            DEFAULT_N_FFT,
            DEFAULT_WIN_LENGTH,
        )
    except ImportError:  # pragma: no cover - final fallback for script usage
        from constants import (  # type: ignore
            DEFAULT_FRAME_LENGTH,
            DEFAULT_HOP_LENGTH,
            DEFAULT_N_FFT,
            DEFAULT_WIN_LENGTH,
        )


PIPE = None
if HAVE_PYANONTE:
    try:  # pragma: no cover - network dependent initialisation
        PIPE = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=dbcfg.HF_TOKEN
        )
    except Exception:
        PIPE = None

def osd_segments_from_array(y: np.ndarray, sr: int):
    """Devuelve [(start, end), ...] con solapamiento (>=2 voces)."""
    if PIPE is None or torch is None:
        warnings.warn(
            "pyannote o torch no están instalados; se omite la detección de solapamiento",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    # pyannote acepta entrada en memoria como dict con "waveform" y "sample_rate"
    wav = torch.from_numpy(y.astype("float32")).unsqueeze(0)  # (1, T)
    out = PIPE({"waveform": wav, "sample_rate": sr})          # :contentReference[oaicite:3]{index=3}
    # out es una Annotation; el timeline .support() son los tramos con overlap
    segs = [(float(s.start), float(s.end)) for s in out.get_timeline().support()]
    return segs

def segment_centroids(
    y: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el tiempo central y el centroide espectral de cada segmento de audio.

    Parameters
    ----------
    y : np.ndarray
        Señal de audio mono.
    sr : int
        Frecuencia de muestreo.
    segments : list[(float, float)]
        Segmentos en segundos.

    Returns
    -------
    times : np.ndarray
        Vector con el tiempo central (segundos) de cada segmento.
    centroids : np.ndarray
        Vector con el centroide espectral promedio de cada segmento (Hz).
    """
    times = []
    centroids = []

    for (start, end) in segments:
        s = int(start * sr)
        e = int(end * sr)
        if e > len(y):
            e = len(y)
        if e <= s:
            continue
        y_seg = y[s:e]
        # tiempo central
        t_center = (start + end) / 2.0
        times.append(t_center)

        # centroide espectral (promedio del segmento)
        spec_cent = librosa.feature.spectral_centroid(y=y_seg, sr=sr)[0]
        centroids.append(float(np.mean(spec_cent)))
    return np.array(times), np.array(centroids)



# =================== Overlap (heurística) + Centroides ===================
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Optional, Dict, Union

def segment_centroids(
    y: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (times, centroids) para cada segmento."""
    times, cents = [], []
    n = len(y)
    for (start, end) in segments:
        s = max(0, int(round(start * sr)))
        e = min(n, int(round(end   * sr)))
        if e <= s: 
            continue
        y_seg = y[s:e]
        times.append((start + end) / 2.0)
        sc = librosa.feature.spectral_centroid(y=y_seg, sr=sr)[0]  # Hz
        cents.append(float(np.mean(sc)) if sc.size else 0.0)
    return np.asarray(times, dtype=float), np.asarray(cents, dtype=float)


def detect_overlap_segments(
    y: np.ndarray,
    sr: int,
    *,
    # Ventaneo
    win_ms: int = 25,
    hop_ms: int = 10,
    # Features
    use_pyin_voicing: bool = False,
    # Ponderaciones / umbral
    weights: Dict[str, float] = None,
    score_thresh: float = 0.9,
    smooth_ms: int = 250,
    min_run_ms: int = 120,
    # VAD opcional
    vad_mask: Optional[np.ndarray] = None,
    restrict_to_vad: bool = True,
    # Qué devolver
    returns: str = "segments",   # "segments" | "mask" | "score" | "centroids"
) -> Union[
    Tuple[np.ndarray, List[Tuple[float, float]]],  # returns="segments" -> (mask, segs)
    Tuple[np.ndarray, None],                       # returns="mask"     -> (mask, None)
    Tuple[np.ndarray, None],                       # returns="score"    -> (score, None)
    Tuple[np.ndarray, np.ndarray]                  # returns="centroids"-> (times, cents)
]:
    """Heurística rápida de Overlapped Speech con opción a devolver centroides."""
    assert y.ndim == 1, "Se espera señal mono."
    y = y.astype(np.float32, copy=False)

    hop = DEFAULT_HOP_LENGTH
    win = DEFAULT_WIN_LENGTH
    n_fft = DEFAULT_N_FFT

    # Espectrograma
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window="hann"))
    # Features
    rms  = librosa.feature.rms(S=S, frame_length=DEFAULT_FRAME_LENGTH)[0]
    dS   = np.diff(S, axis=1)
    flux = np.sqrt((np.clip(dS, 0, None)**2).sum(axis=0))
    flux = np.concatenate([[0.0], flux])
    flat = librosa.feature.spectral_flatness(S=S)[0]
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0]

    voicing = None
    if use_pyin_voicing:
        f0, vflag, vprob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=win,
            hop_length=hop,
            sr=sr,
        )
        voicing = np.nan_to_num(vprob, nan=0.0)

    # Normalización robusta
    def rz(x):
        med = np.median(x); iqr = np.percentile(x, 75) - np.percentile(x, 25)
        if iqr <= 1e-12: iqr = np.std(x) + 1e-12
        return (x - med) / iqr

    z_rms, z_flux, z_flat, z_zcr = rz(rms), rz(flux), rz(flat), rz(zcr)
    z_inv_voicing = -rz(voicing) if voicing is not None else None

    if weights is None:
        weights = dict(rms=0.8, flux=1.5, flat=1.5, zcr=0.7, inv_voicing=1.0)

    terms = [
        weights.get("flux", 0.0) * z_flux,
        weights.get("flat", 0.0) * z_flat,
        weights.get("rms", 0.0)  * z_rms,
        weights.get("zcr", 0.0)  * z_zcr,
    ]
    if z_inv_voicing is not None:
        terms.append(weights.get("inv_voicing", 1.0) * z_inv_voicing)

    score = np.sum(np.vstack(terms), axis=0).astype(np.float32)

    # Suavizado + restricción VAD
    smooth_frames = max(1, int(smooth_ms / hop_ms))
    score_s = uniform_filter1d(score, size=smooth_frames)
    if restrict_to_vad and vad_mask is not None:
        score_s = score_s * vad_mask.astype(np.float32)

    if returns == "score":
        return score_s, None

    # Umbral + runs mínimos
    mask = score_s >= score_thresh
    min_run_frames = max(1, int(min_run_ms / hop_ms))
    if min_run_frames > 1:
        from itertools import groupby
        fixed = mask.copy()
        i = 0
        for val, grp in groupby(mask):
            L = len(list(grp))
            if val and L < min_run_frames:
                fixed[i:i+L] = False
            i += L
        mask = fixed

    if returns == "mask":
        return mask, None

    # A segmentos
    t = np.arange(mask.size) * hop / sr
    segs: List[Tuple[float, float]] = []
    cur = None
    for i, f in enumerate(mask):
        if f and cur is None:
            cur = [t[i], None]
        elif (not f) and (cur is not None):
            cur[1] = t[i]
            segs.append((cur[0], cur[1]))
            cur = None
    if cur is not None:
        segs.append((cur[0], t[-1]))

    if returns == "segments":
        return mask, segs

    if returns == "centroids":
        times, cents = segment_centroids(y, sr, segs)
        return times, cents

    raise ValueError("returns debe ser 'segments', 'mask', 'score' o 'centroids'")

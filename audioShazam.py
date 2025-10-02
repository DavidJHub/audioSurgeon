import glob
import json
import os
from typing import Any, Callable, Iterable, List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import librosa, soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks

# ---------- Parámetros ----------
SR = 8000
HP = 1500.0   # Hz
LP = 3800.0   # Hz
PREEMPH = 0.97
NMS_MERGE_SEC = 0.02            # fusionar detecciones separadas < 20 ms
THRESH = 0.7                    # umbral NCC; calibra 0.65–0.8
MIN_DISTANCE_SEC = 0.01         # picos separados al menos 10 ms


def bandpass(y, sr, low, high, order=4):
    ny = 0.5*sr
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return filtfilt(b, a, y)

def preemphasis(y, a=0.97):
    y2 = np.empty_like(y)
    y2[0] = y[0]
    y2[1:] = y[1:] - a*y[:-1]
    return y2

def whiten(y):
    y = y - np.mean(y)
    std = np.std(y) + 1e-10
    return y / std

def preprocess(y, sr=SR):
    y = preemphasis(y, PREEMPH)
    y = bandpass(y, sr, HP, LP)
    return whiten(y)

def ncc_fft(x, h):
    """
    NCC entre x y plantilla h (h ya preprocesada).
    Devuelve arreglo de puntuaciones (len ~ len(x)-len(h)+1).
    Implementación eficiente con FFT y normalización por energía local.
    """
    n = len(x); m = len(h)
    if m > n:
        return np.array([])

    # correlación (x * h_rev)
    H = np.fft.rfft(h[::-1], n=n)
    X = np.fft.rfft(x, n=n)
    corr = np.fft.irfft(X * H, n=n)
    corr = corr[m-1:n]  # parte válida

    # energía local de x (ventana m) con suma acumulada
    x2 = x**2
    csum = np.cumsum(np.concatenate(([0.0], x2)))
    win_energy = csum[m:] - csum[:-m]  # longitud n-m+1

    # normalización
    denom = np.sqrt(win_energy) * (np.linalg.norm(h) + 1e-10)
    ncc = corr / (denom + 1e-10)
    return ncc

def merge_peaks(times, min_sep):
    times = np.array(sorted(times))
    if len(times) == 0: return []
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] >= min_sep:
            merged.append(t)
    return merged

# ---------- Pipeline ----------
def build_template(template_path, sr=SR):
    y, _ = librosa.load(template_path, sr=sr, mono=True)
    y = preprocess(y, sr)
    # recorta silencio por si la plantilla viene con márgenes grandes
    if len(y) > 0:
        thr = 0.1*np.max(np.abs(y))
        nz = np.where(np.abs(y) > thr)[0]
        if nz.size >= 1:
            y = y[max(0, nz[0]-int(0.002*sr)) : min(len(y), nz[-1]+int(0.002*sr))]
    # normaliza a norma 1 para estabilidad
    y = y / (np.linalg.norm(y) + 1e-10)
    return y

def detect_signature(audio_path, template_wave, sr=SR,
                     thresh=THRESH, min_distance_sec=MIN_DISTANCE_SEC):
    x, _ = librosa.load(audio_path, sr=sr, mono=True)
    x = preprocess(x, sr)

    ncc = ncc_fft(x, template_wave)
    if ncc.size == 0:
        return [], ncc
    # picos por umbral y distancia mínima
    distance = max(1, int(min_distance_sec * sr))
    peaks, _ = find_peaks(ncc, height=thresh, distance=distance)
    # tiempos (centro de la ventana)
    m = len(template_wave)
    times = (peaks + m//2) / sr
    # NMS temporal (merge cercano)
    times = merge_peaks(times, NMS_MERGE_SEC)
    return times, ncc


def detect_signature_from_array(
    y: np.ndarray,
    sr: int,
    template_wave: np.ndarray,
    *,
    thresh: float,
    min_distance_sec: float,
    nms_merge_sec: float,
    preprocess_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ncc_fft: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    merge_peaks: Callable[[np.ndarray, float], np.ndarray] = None,
) -> Tuple[List[float], np.ndarray]:
    """
    Detecta la 'firma' en un audio ya cargado.

    Parámetros
    ----------
    y : np.ndarray
        Señal mono.
    sr : int
        Sample rate de y (debe coincidir con template_wave).
    template_wave : np.ndarray
        Plantilla a correlacionar (misma tasa sr).
    thresh : float
        Umbral de altura para los picos de NCC.
    min_distance_sec : float
        Distancia mínima entre picos (en segundos).
    nms_merge_sec : float
        Ventana de 'merge' para NMS temporal (en segundos).
    preprocess_fn : callable | None
        Función opcional de preprocesado sobre y antes de correlacionar.
        Firma esperada: preprocess_fn(y, sr) -> y_proc
    ncc_fft : callable
        Función de correlación (NCC) en frecuencia: ncc_fft(x, template) -> np.ndarray
    merge_peaks : callable
        Función para fusionar picos cercanos: merge_peaks(times, nms_merge_sec) -> times_merged

    Retorna
    -------
    times : list[float]
        Tiempos (s) de los picos detectados (centro de la ventana).
    ncc : np.ndarray
        Curva de correlación normalizada.
    """
    assert ncc_fft is not None, "Debes pasar ncc_fft como función."
    assert merge_peaks is not None, "Debes pasar merge_peaks como función."
    x = y.astype(np.float32, copy=False)
    if preprocess_fn is not None:
        x = preprocess_fn(x, sr)

    ncc = ncc_fft(x, template_wave)
    if ncc is None or ncc.size == 0:
        return [], np.array([], dtype=np.float32)

    distance = max(1, int(min_distance_sec * sr))
    peaks, props = find_peaks(ncc, height=thresh, distance=distance)

    m = len(template_wave)
    times = (peaks + m // 2) / float(sr)
    if times.size:
        times = merge_peaks(times, nms_merge_sec)
    else:
        times = np.array([], dtype=float)

    return list(map(float, times)), ncc


def batch_detect_signature_to_df(
    input_dir: str,
    template_wave: np.ndarray,
    # parámetros de detect_signature (puedes omitir para usar sus defaults)
    sr: Optional[int] = None,
    thresh: Optional[float] = None,
    min_distance_sec: Optional[float] = None,
    # selección de archivos
    pattern: str = "**/*",
    exts: Iterable[str] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    # salidas
    save_df_path: Optional[str] = None,       # .csv o .parquet opcional
    store_ncc: bool = False,                  # si True, guarda el NCC por archivo en .npy
    ncc_out_dir: Optional[str] = None,        # dir donde guardar .npy (por defecto output_dir de DF)
    preserve_rel_paths: bool = True,          # guarda ruta relativa al input_dir además de la absoluta
    verbose: bool = True
) -> pd.DataFrame:
    """
    Recorre input_dir, corre detect_signature por archivo y devuelve un DataFrame con resultados.
    Columnas: file_path, rel_path, sr, times, n_peaks, ncc_max, ncc_mean, ncc_len, (ncc_path si store_ncc)

    save_df_path:
      - Si termina en .csv  -> guarda CSV
      - Si termina en .parquet -> guarda Parquet
    """
    # recolectar archivos
    all_paths = [
        p for p in glob.glob(os.path.join(input_dir, pattern), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    ]
    rows: List[Dict[str, Any]] = []

    if verbose:
        print(f"[batch_detect_signature] Encontrados {len(all_paths)} archivos de audio.")

    # carpeta para NCC si hace falta
    if store_ncc:
        base_dir = ncc_out_dir or (os.path.splitext(save_df_path)[0] + "_ncc" if save_df_path else os.path.join(input_dir, "_ncc"))
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    for idx, audio_path in enumerate(sorted(all_paths)):
        try:
            # llama detect_signature con los params provistos (o sus defaults)
            kwargs = {}
            if sr is not None: kwargs["sr"] = sr
            if thresh is not None: kwargs["thresh"] = thresh
            if min_distance_sec is not None: kwargs["min_distance_sec"] = min_distance_sec

            times, ncc = detect_signature(audio_path, template_wave, **kwargs)

            # métricas rápidas de NCC
            ncc_max  = float(np.max(ncc)) if ncc is not None and ncc.size else np.nan
            ncc_mean = float(np.mean(ncc)) if ncc is not None and ncc.size else np.nan
            ncc_len  = int(ncc.size) if ncc is not None else 0

            ncc_path = None
            if store_ncc and ncc is not None and ncc.size:
                # genera ruta paralela (preserva subdirs si preserve_rel_paths)
                if preserve_rel_paths:
                    rel = os.path.relpath(audio_path, input_dir)
                    stem, _ = os.path.splitext(rel)
                    out_ncc_path = os.path.join(base_dir, stem + "_ncc.npy")
                    os.makedirs(os.path.dirname(out_ncc_path), exist_ok=True)
                else:
                    stem, _ = os.path.splitext(os.path.basename(audio_path))
                    out_ncc_path = os.path.join(base_dir, stem + "_ncc.npy")
                np.save(out_ncc_path, ncc.astype(np.float32))
                ncc_path = out_ncc_path

            rows.append({
                "file_path": os.path.abspath(audio_path),
                "rel_path": os.path.relpath(audio_path, input_dir) if preserve_rel_paths else os.path.basename(audio_path),
                "sr": sr,  # el sr usado por detect_signature (si fue None, tu función habrá re-sampleado internamente)
                "times": list(map(float, times)) if times is not None else [],
                "n_peaks": int(len(times)) if times is not None else 0,
                "ncc_max": ncc_max,
                "ncc_mean": ncc_mean,
                "ncc_len": ncc_len,
                "ncc_path": ncc_path,
                "error": None
            })

            if verbose and (idx % 20 == 0):
                print(f"  [{idx+1}/{len(all_paths)}] {os.path.basename(audio_path)} — picos: {len(times)}")
        except Exception as e:
            rows.append({
                "file_path": os.path.abspath(audio_path),
                "rel_path": os.path.relpath(audio_path, input_dir) if preserve_rel_paths else os.path.basename(audio_path),
                "sr": sr,
                "times": [],
                "n_peaks": 0,
                "ncc_max": np.nan,
                "ncc_mean": np.nan,
                "ncc_len": 0,
                "ncc_path": None,
                "error": str(e)
            })
            if verbose:
                print(f"  [ERROR] {audio_path}: {e}")

    df = pd.DataFrame(rows)
    if save_df_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_df_path) or "."), exist_ok=True)
        if save_df_path.lower().endswith(".csv"):
            # Para CSV, serializamos 'times' como JSON para no perder estructura
            df_to_save = df.copy()
            df_to_save["times"] = df_to_save["times"].apply(lambda x: json.dumps(x))
            df_to_save.to_csv(save_df_path, index=False)
        elif save_df_path.lower().endswith(".parquet"):
            df.to_parquet(save_df_path, index=False)
        else:
            raise ValueError("save_df_path debe terminar en .csv o .parquet")
        if verbose:
            print(f"[batch_detect_signature] DF guardado en: {save_df_path}")
    return df
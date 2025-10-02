import os
import librosa
from matplotlib import pyplot as plt
import numpy as np
import soundfile as sf
import scipy.signal as sps
from typing import List, Tuple, Optional, Union





def cut_dial_start(
    audio: np.ndarray,
    samplerate: int,
    *,
    band=(420.0, 440.0),
    slice_seconds=40.0,
    intensity_ratio=0.75,
    min_duration_s=0.2,
    # robustez
    env_lp_hz=20.0,        # low-pass a la envolvente (≈50 ms)
    merge_gaps_s=0.05,     # fusiona huecos ≤ 50 ms entre runs
    # debug
    debug=False, debug_save=False, debug_out=None,
    stft_nfft=512, stft_hop=256, pad_hz=50
):
    """
    Detecta un tono (~band) al inicio (por envolvente) y corta el audio al final del tono.
    NO guarda nada en disco. Devuelve (y_out, dial_time_s).

    y_out: audio recortado desde el fin del tono (o audio original si no hay detección)
    dial_time_s: tiempo (s) del fin del tono, o None si no se detecta
    """
    # 1) Tramo a analizar
    n_slice = int(max(1, slice_seconds * samplerate))
    audio_slice = audio[:n_slice]

    # 2) Pasa-banda al tono (protegido por Nyquist)
    nyq = 0.5 * samplerate
    low = np.clip(band[0], 20.0, nyq * 0.95)
    high = np.clip(band[1], low + 1.0, nyq * 0.99)
    b_bp, a_bp = sps.butter(2, (low, high), btype='bandpass', fs=samplerate)
    y_bp = sps.filtfilt(b_bp, a_bp, audio_slice, method="pad")

    # 3) Envolvente (Hilbert) + suavizado (low-pass)
    env = np.abs(sps.hilbert(y_bp))
    if env_lp_hz is not None and env_lp_hz > 0:
        b_lp, a_lp = sps.butter(2, env_lp_hz, btype='low', fs=samplerate)
        env_s = sps.filtfilt(b_lp, a_lp, env, method="pad")
    else:
        env_s = env

    # 4) Umbral relativo sobre envolvente suavizada
    # (alternativa robusta: thr = np.percentile(env_s, 95) * intensity_ratio)
    env_max = float(np.max(env_s) + 1e-12)
    thr = intensity_ratio * env_max
    mask = env_s > thr

    # 5) Runs + fusión de huecos cortos (closing 1D)
    m = mask.astype(int)
    dif = np.diff(m, prepend=m[0])
    starts = list(np.where(dif == 1)[0])
    ends   = list(np.where(dif == -1)[0])
    if m[0] == 1:  starts = [0] + starts
    if m[-1] == 1: ends   = list(ends) + [len(m)-1]
    segs = [(int(s), int(e)) for s, e in zip(starts, ends)]

    merged = []
    gap_max = int(round(merge_gaps_s * samplerate))
    for seg in segs:
        if not merged:
            merged.append(seg)
            continue
        ps, pe = merged[-1]
        if seg[0] - pe <= gap_max:
            merged[-1] = (ps, seg[1])   # unir
        else:
            merged.append(seg)

    # 6) Filtrar por duración mínima
    min_len = int(round(min_duration_s * samplerate))
    candidates = [(s, e) for (s, e) in merged if (e - s + 1) >= min_len]

    # 7) Decidir corte
    dial_time_s = None
    if candidates:
        last_s, last_e = candidates[-1]
        dial_time_s = last_e / samplerate
        cut_idx = int(min(len(audio), max(0, round(dial_time_s * samplerate))))
        y_out = audio[cut_idx:]
    else:
        y_out = audio
        dial_time_s = None

    # 8) DEBUG: espectrograma + envolvente/umbral
    if debug:
        # Espectrograma del slice filtrado
        S = np.abs(librosa.stft(y_bp, n_fft=stft_nfft, hop_length=stft_hop))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=samplerate, hop_length=stft_hop)
        freqs = np.linspace(0, samplerate/2, S.shape[0])

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios':[3,1]})

        im = ax1.imshow(S_db, origin='lower', aspect='auto',
                        extent=[times[0] if times.size else 0, times[-1] if times.size else 0, freqs[0], freqs[-1]],
                        cmap='magma')
        ax1.set_ylim(max(0, low - pad_hz), min(samplerate/2, high + pad_hz))
        ax1.set_ylabel('Frecuencia (Hz)')
        ax1.set_title(f'Espectrograma (filtrado {low:.1f}–{high:.1f} Hz)')
        cbar = fig.colorbar(im, ax=ax1, pad=0.01)
        cbar.set_label('Amplitud (dB)')

        # Marcar segmentos candidatos
        for (s, e) in candidates:
            t0, t1 = s / samplerate, e / samplerate
            ax1.axvspan(t0, t1, color='white', alpha=0.18, lw=0)

        if dial_time_s is not None:
            ax1.axvline(dial_time_s, ls='--', lw=1.5, color='cyan', label=f'Cut @ {dial_time_s:.3f}s')
            ax1.legend(loc='upper right')

        # Envolvente y umbral
        t_env = np.arange(len(env_s)) / samplerate
        ax2.plot(t_env, env_s, label='Envelope (smoothed)')
        ax2.axhline(thr, color='r', ls='--', label='Threshold')
        ax2.fill_between(t_env, 0, env_s, where=mask, alpha=0.15, label='> thr')
        ax2.set_xlim(0, len(audio_slice)/samplerate)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Envolvente')
        ax2.legend(loc='upper right')

        if debug_save:
            if debug_out is None:
                debug_out = "dial_debug.png"
            fig.savefig(debug_out, dpi=150, bbox_inches='tight')
            print(f"[DEBUG] Figura guardada en: {debug_out}")
        plt.show()

    return y_out, dial_time_s



def split_activity_vs_background(
    y: np.ndarray,
    sr: int,
    activity_segments: List[Tuple[float, float]],
    *,
    save_files: bool = False,
    base_path: Optional[str] = None,   # p.ej. "/salida/carpeta/archivo" (sin sufijo)
) -> Union[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    A partir de una señal y su sr, separa audio ACTIVO (concatenación de segmentos)
    e INACTIVO (complemento). Opcionalmente guarda en disco.

    Parámetros
    ----------
    y : np.ndarray (mono)
        Señal de audio en flotantes (-1..1 idealmente).
    sr : int
        Frecuencia de muestreo.
    activity_segments : list[(float, float)]
        Lista de (inicio, fin) en segundos con actividad. Pueden venir sin ordenar.
        Si hay solapes, se fusionan.
    save_files : bool
        True → guarda y retorna rutas; False → retorna arrays.
    base_path : str | None
        Prefijo para construir nombres si se guarda (sin extensión).
        Ej.: "/ruta/archivo" → "/ruta/archivo_activ.wav" y "/ruta/archivo_noactiv.wav"

    Returns
    -------
    (str, str) si save_files=True → (ruta_activ, ruta_noactiv)
    (np.ndarray, np.ndarray) si save_files=False → (y_activ, y_noactiv)
    """

    if y.ndim != 1:
        raise ValueError("`y` debe ser mono (arreglo 1D).")

    n = len(y)
    if n == 0:
        y_active = np.array([], dtype=y.dtype)
        y_inactive = np.array([], dtype=y.dtype)
        if save_files:
            if not base_path:
                raise ValueError("Para guardar, provee base_path.")
            root, _ = os.path.splitext(base_path)
            active_path = root + "_activ.wav"
            inactive_path = root + "_noactiv.wav"
            sf.write(active_path, y_active, sr)
            sf.write(inactive_path, y_inactive, sr)
            return active_path, inactive_path
        else:
            return y_active, y_inactive

    # 1) Normaliza y valida segmentos: ordena, recorta a [0, n/sr], fusiona solapes
    #    (esto hace la función más robusta frente a VADs con bordes ruidosos)
    def _sanitize_and_merge(segments: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        # segundos → muestras (recortando)
        idx = []
        for s, e in segments:
            if e <= s:
                continue
            s_i = max(0, int(round(s * sr)))
            e_i = min(n, int(round(e * sr)))
            if e_i - s_i > 0:
                idx.append((s_i, e_i))
        if not idx:
            return []

        # ordenar
        idx.sort(key=lambda t: t[0])

        # fusionar solapes/adyacencias
        merged = [idx[0]]
        for a, b in idx[1:]:
            la, lb = merged[-1]
            if a <= lb:            # solapado o tocando
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    seg_idx = _sanitize_and_merge(activity_segments)

    # 2) Construir ACTIVO
    if seg_idx:
        active_chunks = [y[a:b] for a, b in seg_idx]
        y_active = np.concatenate(active_chunks).astype(y.dtype, copy=False)
    else:
        y_active = np.array([], dtype=y.dtype)

    # 3) Construir INACTIVO = complementos
    inactive_chunks = []
    last_end = 0
    for a, b in seg_idx:
        if a > last_end:
            inactive_chunks.append(y[last_end:a])
        last_end = b
    if last_end < n:
        inactive_chunks.append(y[last_end:])
    y_inactive = (np.concatenate(inactive_chunks).astype(y.dtype, copy=False)
                  if inactive_chunks else np.array([], dtype=y.dtype))

    # 4) Guardar o retornar arrays
    if save_files:
        if not base_path:
            raise ValueError("Para guardar, provee base_path (sin extensión).")
        root, _ = os.path.splitext(base_path)
        active_path = root + "_activ.wav"
        inactive_path = root + "_noactiv.wav"
        sf.write(active_path, y_active, sr)
        sf.write(inactive_path, y_inactive, sr)

        # chequeo suave de conservación de longitud
        if abs(n - (len(y_active) + len(y_inactive))) > 1:
            print("⚠️  Advertencia: ligeras discrepancias en la longitud total.")
        return active_path, inactive_path

    return y_active, y_inactive
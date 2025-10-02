import librosa, numpy as np
from scipy.ndimage import uniform_filter1d
import librosa, librosa.display, numpy as np
import matplotlib.pyplot as plt


def vad_energy_adaptive(path, win_ms=25, hop_ms=10, p_percentile=60, delta_db=3,
                        min_run_frames=3, smooth_ms=500,returns="segments"):
    y, sr = librosa.load(path, sr=8_000, mono=True)
    hop = 128
    win = 512
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    thr = np.percentile(rms_db, p_percentile) + delta_db
    act_mask = rms_db >= thr

    from itertools import groupby
    act_mask_enforced = np.zeros_like(act_mask, dtype=bool)
    idx = 0
    for val, grp in groupby(act_mask):
        length = len(list(grp))
        if val and length >= min_run_frames:
            act_mask_enforced[idx:idx+length] = True
        idx += length

    # Smooth with moving average window (half-second)
    smooth_frames = int(smooth_ms / hop_ms)
    prob = uniform_filter1d(act_mask_enforced.astype(float), size=smooth_frames)
    final_mask = prob >= 0.5          # 0.5 ↔ mayoria

    # Convert to segments
    segments = []
    t = np.arange(len(final_mask)) * hop / sr
    current = None
    for i, flag in enumerate(final_mask):
        if flag and current is None:
            current = [t[i], None]
        elif not flag and current is not None:
            current[1] = t[i]
            segments.append(tuple(current))
            current = None
    if current is not None:
        segments.append((current[0], t[-1]))
    if returns == "mask":
        return final_mask
    elif returns == "segments":
        return segments


def plot_vad_segments(audio_path,
                      segments,
                      sr=8_000,            
                      figsize=(14, 6),
                      color_active="#ff8113",
                      alpha_active=0.25,
                      show=True,
                      save_path=None):

    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    fig, (ax_wave, ax_bar) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw=dict(height_ratios=[3, 0.6]),
        sharex=True
    )

    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="steelblue")
    ax_wave.set(title="Forma de onda con segmentos de actividad", ylabel="Amplitud")

    for start, end in segments:
        ax_wave.axvspan(start, end, color=color_active, alpha=alpha_active)

    # ----- Panel 2: Barra binaria -----
    ax_bar.set(title="Línea de tiempo VAD", xlabel="Tiempo (s)")
    ax_bar.set_yticks([]); ax_bar.set_ylim(0, 1)
    for start, end in segments:
        ax_bar.axvspan(start, end, color=color_active, alpha=alpha_active)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Figura guardada en {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)

    return fig



from itertools import groupby
from typing import List, Tuple, Union

def vad_energy_adaptive_array(
    y: np.ndarray,
    sr: int,
    win_ms: int = 25,
    hop_ms: int = 10,
    p_percentile: float = 60,
    delta_db: float = 3,
    min_run_frames: int = 3,
    smooth_ms: int = 500,
    returns: str = "segments"
) -> Union[np.ndarray, List[Tuple[float, float]]]:
    """
    VAD (Voice Activity Detection) adaptativo basado en energía.
    Procesa directamente una señal `y` y su frecuencia de muestreo `sr`.

    Parameters
    ----------
    y : np.ndarray
        Señal de audio en mono.
    sr : int
        Frecuencia de muestreo.
    win_ms : int
        Ventana en milisegundos para RMS.
    hop_ms : int
        Hop en milisegundos para RMS.
    p_percentile : float
        Percentil sobre dB para calcular umbral dinámico.
    delta_db : float
        Margen por encima del percentil para decidir actividad.
    min_run_frames : int
        Número mínimo de frames consecutivos para aceptar un segmento activo.
    smooth_ms : int
        Ventana de suavizado (ms).
    returns : {"segments", "mask"}
        Qué retornar:
            - "mask": vector booleano por frames
            - "segments": lista de (inicio, fin) en segundos

    Returns
    -------
    np.ndarray (mask) o List[Tuple[float, float]] (segments)
    """

    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)

    # 1) Energía RMS
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # 2) Umbral dinámico
    thr = np.percentile(rms_db, p_percentile) + delta_db
    act_mask = rms_db >= thr

    # 3) Enforce: solo runs suficientemente largos
    act_mask_enforced = np.zeros_like(act_mask, dtype=bool)
    idx = 0
    for val, grp in groupby(act_mask):
        length = len(list(grp))
        if val and length >= min_run_frames:
            act_mask_enforced[idx:idx+length] = True
        idx += length

    # 4) Suavizado temporal (ventana ~ smooth_ms)
    smooth_frames = int(smooth_ms / hop_ms)
    prob = uniform_filter1d(act_mask_enforced.astype(float), size=smooth_frames)
    final_mask = prob >= 0.5

    if returns == "mask":
        return final_mask

    elif returns == "segments":
        # convertir a segmentos en segundos
        segments = []
        t = np.arange(len(final_mask)) * hop / sr
        current = None
        for i, flag in enumerate(final_mask):
            if flag and current is None:
                current = [t[i], None]
            elif not flag and current is not None:
                current[1] = t[i]
                segments.append(tuple(current))
                current = None
        if current is not None:
            segments.append((current[0], t[-1]))
        return segments

    else:
        raise ValueError("`returns` debe ser 'mask' o 'segments'")

if __name__ == "__main__":
    audio_test_path="data/Allianz/2025-07-24/ALLIZ_20250601-210200_1263943_CERRAJERIA_1007595176_129.mp3"
    segments = vad_energy_adaptive(audio_test_path, win_ms=25, hop_ms=50, p_percentile=10, delta_db=6,
                                   min_run_frames=3, smooth_ms=3_000)
        # 2. Graficar usando esos segmentos
    print(segments)  

    y, sr = librosa.load(audio_test_path, sr=None, mono=True)   

    plot_vad_segments(audio_test_path,
                    segments=segments,
                    sr=8_000,
                    save_path="vad_plot.png", 
                    show=True)       

    
    # ----- 1. Forma de onda -----
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr, color="steelblue")
    plt.title("Forma de onda")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.show()

    # ----- 2. Espectrograma -----
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                            x_axis="time", y_axis="hz")
    plt.title("Espectrograma (dB)")
    plt.colorbar(format="%+2.0f dB", label="Intensidad")
    plt.tight_layout()
    plt.show()



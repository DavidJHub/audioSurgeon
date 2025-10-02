import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from pydub import AudioSegment
from tqdm import tqdm
from audio.audioSegmentation import segment_speech

try:
    from .constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT
except ImportError:  # pragma: no cover - fallback when distributed under "audio"
    try:
        from audio.constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT  # type: ignore
    except ImportError:  # pragma: no cover - final fallback for script usage
        from constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT  # type: ignore




def _DbClassify(audio_file, start_time, duration=3):
    """
    Mide el nivel de volumen en decibelios (dBFS) de un archivo de audio,
    tomando un fragmento centrado en 'start_time' con longitud 'duration'.
    Retorna el valor dBFS y una clasificación (bajo/medio/alto).

    Parámetros:
    -----------
    audio_file : str
        Ruta del archivo de audio (e.g. 'audio.mp3').
    start_time : float
        Segundo en el que se centrará el fragmento de medición.
    duration   : float, opcional
        Duración (en segundos) del fragmento a medir. Por defecto 3 segundos.

    Returns:
    --------
    (float, str)
        Un tuple (db_value, classification), donde 'db_value' es el nivel en dBFS,
        y 'classification' es la etiqueta de volumen ('bajo', 'medio' o 'alto').
    """
    try:
        half_duration = duration / 2
        start_ms = int(max((start_time - half_duration), 0) * 1000)
        end_ms = int((start_time + half_duration) * 1000)

        audio_segment = AudioSegment.from_file(audio_file)[start_ms:end_ms]
        if len(audio_segment) == 0:
            raise ValueError("El fragmento de audio cargado está vacío.")

        db_value = audio_segment.dBFS

        if db_value < -45:
            classification = "bajo"
        elif -45 <= db_value < -20:
            classification = "medio"
        else:
            classification = "alto"

        return db_value, classification

    except FileNotFoundError:
        print(f"Archivo no encontrado: {audio_file}")
        return None, "Error: Archivo no encontrado"
    except ValueError as ve:
        return None, f"Error: {ve}"
    except Exception as e:
        return None, f"Error inesperado: {e}"
    


def measureDbAmplitude_df(directory, df,time_column,suffix, duration=5):
    """
    Mide el volumen (en decibelios) y la clasificación del volumen
    para cada fila de df, basándose en un archivo de audio almacenado
    en 'directory + row["file_name"]'.
    Se agregan columnas 'volume_db' y 'volume_classification'.

    Parámetros
    ----------
    directory : str
        Ruta al directorio donde se ubican los archivos de audio.
    df : pd.DataFrame
        DataFrame que contiene al menos la columna 'file_name'.
        Idealmente también otras columnas como 'start' y 'end'
        para calcular el instante de medición.
    duration : float, default=3
        Duración en segundos del fragmento que se tomará
        para medir el volumen.

    Returns
    -------
    pd.DataFrame
        El mismo DataFrame de entrada, con dos columnas nuevas:
        'volume_db' y 'volume_classification'.
    """
    mp='_'+suffix
    df['volume_db'+mp] = None
    df['volume_classification'+mp] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Measuring dB Volume"):
        file_path = os.path.join(directory, row['file_name'])

        db_value, classification = _DbClassify(
            audio_file=file_path,
            start_time=row[time_column],
            duration=duration
        )

        # Guardamos los resultados
        df.at[index, 'volume_db'+mp] = db_value
        df.at[index, 'volume_classification'+mp] = classification

    return df





def get_snr(audio_path, n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH):
    """
    Computes the SNR (in dB) over time using the segmentation mask.

    The average noise energy is estimated from the frames classified as non-speech.
    SNR per frame is calculated as:

        SNR = 10 * log10(frame_energy / noise_energy)

    Parameters:
        audio_path (str): Path to the audio file.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for frame analysis.

    Returns:
        mask (np.ndarray): Binary array with speech (1) and non-speech (0) frames.
        snr (np.ndarray): Array of SNR values (dB) for each frame.
        sr (int): Sampling rate of the audio.
    """
    mask, rms, sr = segment_speech(audio_path, n_fft, hop_length)

    # Estimate noise energy from the frames classified as non-speech.
    noise_frames = rms[mask == 0]
    # Avoid division by zero; if no noise frames are found, use a small epsilon.
    if len(noise_frames) == 0:
        noise_energy = np.finfo(float).eps
    else:
        noise_energy = np.mean(noise_frames)
        if noise_energy == 0:
            noise_energy = np.finfo(float).eps

    # Compute SNR in dB for each frame: 10 * log10(signal_energy / noise_energy)
    snr = 10 * np.log10(rms / noise_energy)

    return mask, snr, sr


def select_rich_window(
    audio_path: str,
    output_path: str | None = None,
    sr: int = 8000,
    win_sec: int = 30,
    hop_sec: int = 10,
    n_fft: int = DEFAULT_N_FFT,
    peak_thresh_db: float = -40.0,
):
    """
    Devuelve la ventana de `win_sec` segundos cuyo espectro presenta
    la mayor densidad de picos sobre el umbral `peak_thresh_db` (dB).

    Parámetros
    ----------
    audio_path : str
        Ruta al audio de entrada (WAV/MP3).
    output_path : str | None
        Si se indica, guarda la ventana seleccionada como WAV.
    sr : int
        Frecuencia de muestreo a la que se cargará el audio.
    win_sec : int
        Duración de la ventana en segundos (p.ej. 30 s).
    hop_sec : int
        Desplazamiento entre ventanas (p.ej. 10 s).
    n_fft : int
        Tamaño de FFT para el STFT.
    peak_thresh_db : float
        Umbral (en dB) para contar un pico.

    Retorna
    -------
    (start_time, end_time, y_window)
        `start_time` y `end_time` en segundos y la señal de la ventana.
    """
    # 1) Cargar audio mono
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    win_samp = int(win_sec * sr)
    hop_samp = int(hop_sec * sr)

    best_score = -np.inf
    best_start = 0

    # 2) Ventaneo deslizante
    for start in range(sr*40, len(y) - win_samp + 1, hop_samp):
        seg = y[start : start + win_samp]

        # 3) Espectrograma
        S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=n_fft // 4))
        S_db = librosa.amplitude_to_db(S, ref=np.max)  # dB

        # 4) Contar picos frame a frame
        peak_counts = [
            len(find_peaks(col, height=peak_thresh_db)[0]) for col in S_db.T
        ]
        score = np.sum(peak_counts)  # densidad total

        if score > best_score:
            best_score = score
            best_start = start

    best_end = best_start + win_samp
    y_best = y[best_start:best_end]

    if output_path:
        sf.write(output_path, y_best, sr)

    return best_start / sr, best_end / sr, y_best


def measure_spectral_flatness(
    audio_path,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
):
    """
    Computes spectral flatness for each frame.

    Returns:
        flatness: 1D array (n_frames,) of spectral flatness values.
        sr: Sampling rate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    return flatness, sr
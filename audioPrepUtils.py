import warnings

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, lfilter
import scipy.signal as sps
import soundfile as sf

try:  # pragma: no cover - optional dependency in some deployments
    from pedalboard import (
        Pedalboard,
        HighpassFilter,
        LowpassFilter,
        NoiseGate,
        Compressor,
        Limiter,
        Gain,
    )
    HAVE_PEDALBOARD = True
except ImportError:  # pragma: no cover - exercised when pedalboard is unavailable
    Pedalboard = HighpassFilter = LowpassFilter = NoiseGate = Compressor = Limiter = Gain = None
    HAVE_PEDALBOARD = False

try:  # pragma: no cover - optional dependency in some deployments
    import webrtcvad  # type: ignore
    HAVE_WEBRTCVAD = True
except ImportError:  # pragma: no cover - exercised when webrtcvad is unavailable
    webrtcvad = None  # type: ignore
    HAVE_WEBRTCVAD = False

try:  # pragma: no cover - optional dependency in some deployments
    import pyloudnorm as pyln
    HAVE_PYLOUDNORM = True
except ImportError:  # pragma: no cover - exercised when pyloudnorm is unavailable
    pyln = None  # type: ignore
    HAVE_PYLOUDNORM = False

try:
    from .constants import (
        DEFAULT_FRAME_LENGTH,
        DEFAULT_HOP_LENGTH,
        DEFAULT_N_FFT,
    )
except ImportError:  # pragma: no cover - fall back when packaged differently
    try:
        from audio.constants import (  # type: ignore
            DEFAULT_FRAME_LENGTH,
            DEFAULT_HOP_LENGTH,
            DEFAULT_N_FFT,
        )
    except ImportError:  # pragma: no cover - final fallback for script usage
        from constants import (  # type: ignore
            DEFAULT_FRAME_LENGTH,
            DEFAULT_HOP_LENGTH,
            DEFAULT_N_FFT,
        )


def _db_to_lin(db): return 10.0**(db/20.0)

def calculate_loudness(audio):
    return np.sqrt(np.mean(audio ** 2))


# Bandpass filter implementation
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# ---------- Parámetros ----------
SR = 8000
HP = 1500.0   # Hz
LP = 3800.0   # Hz
PREEMPH = 0.97
NMS_MERGE_SEC = 0.02            # fusionar detecciones separadas < 20 ms
THRESH = 0.7                    # umbral NCC; calibra 0.65–0.8
MIN_DISTANCE_SEC = 0.01         # picos separados al menos 10 ms

# ---------- Utilidades ----------



def _apply_pedalboard_chain(y, sr,
                            hp_hz=120.0, lp_hz=6000.0,
                            gate_thresh_db=None, gate_ratio=3.0, gate_attack_ms=8.0, gate_release_ms=120.0,
                            comp_thresh_db=-28.0, comp_ratio=5.0, comp_attack_ms=10.0, comp_release_ms=100.0,
                            makeup_gain_db=0.0, limit_thresh_db=-2.0, limit_release_ms=60.0):

    if not HAVE_PEDALBOARD:
        warnings.warn(
            "pedalboard no está instalado; se omite la cadena de procesamiento",
            RuntimeWarning,
            stacklevel=2,
        )
        return y.astype(np.float32, copy=False)

    # asegurar frecuencias válidas
    nyq = 0.5 * sr
    hp_hz = max(20.0, min(hp_hz, nyq * 0.95))
    lp_hz = None if lp_hz is None else max(200.0, min(lp_hz, nyq * 0.95))

    chain = []
    chain.append(HighpassFilter(cutoff_frequency_hz=hp_hz))
    if lp_hz is not None:
        chain.append(LowpassFilter(cutoff_frequency_hz=lp_hz))
    # NoiseGate (si gate_thresh_db es None, lo autocalibramos fuera)
    if gate_thresh_db is not None:
        chain.append(NoiseGate(threshold_db=gate_thresh_db, ratio=gate_ratio,
                               attack_ms=gate_attack_ms, release_ms=gate_release_ms))
    chain += [
        Compressor(threshold_db=comp_thresh_db, ratio=comp_ratio,
                   attack_ms=comp_attack_ms, release_ms=comp_release_ms),
        Gain(gain_db=makeup_gain_db),
        Limiter(threshold_db=limit_thresh_db, release_ms=limit_release_ms),
    ]
    pb = Pedalboard(chain)
    y32 = y.astype(np.float32, copy=False)
    return pb(y32, sr)

def _leveler(y, sr, mode="lufs", target_lufs=-18.0, limiter_ceiling_db=-1.0):
    """
    Normaliza sonoridad global y aplica un limitador simple de pico.
    mode: 'lufs' (requiere pyloudnorm) o 'peak' (normaliza pico).
    """
    y = y.astype(float)
    normalized = False
    if mode == "lufs":
        if pyln is None:
            warnings.warn(
                "pyloudnorm no está instalado; se aplica normalización por pico",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            meter = pyln.Meter(sr)
            loud = meter.integrated_loudness(y)
            y = pyln.normalize.loudness(y, loud, target_lufs)
            normalized = True
    if not normalized:
        # Fallback: normaliza a pico -1 dBFS (o al ceiling)
        # (si quieres RMS target, puedes añadirlo aquí)
        peak = np.max(np.abs(y)) + 1e-12
        tgt = _db_to_lin(limiter_ceiling_db)  # e.g., -1 dBFS
        if peak > 0:
            y = y * (tgt / peak)

    # Limitador suave de seguridad al ceiling (por si algún transitorio pasó)
    peak = np.max(np.abs(y)) + 1e-12
    ceiling = _db_to_lin(limiter_ceiling_db)
    if peak > ceiling:
        y = y * (ceiling / peak)
    return np.clip(y, -1.0, 1.0)


def preprocess_audio_for_vad(
    y: np.ndarray, sr: int,
    # LEVELER
    enable_leveler: bool = True, leveler_mode: str = "lufs",
    target_lufs: float = -18.0, limiter_ceiling_db: float = -1.0,
    # PEDALBOARD
    use_pedalboard: bool = True,
    pb_hp_hz: float = 120.0, pb_lp_hz: float | None = 6000.0,   # pon 3400 si sr=8k (ver abajo)
    pb_gate_thresh_db: float | None = None,  # None = auto
    pb_gate_ratio: float = 3.0, pb_gate_attack_ms: float = 8.0, pb_gate_release_ms: float = 120.0,
    pb_comp_thresh_db: float = -24.0, pb_comp_ratio: float = 3.0,
    pb_comp_attack_ms: float = 10.0, pb_comp_release_ms: float = 100.0,
    pb_makeup_gain_db: float = 0.0, pb_limit_thresh_db: float = -2.0, pb_limit_release_ms: float = 60.0,
    # FEATURES
    n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH,
    rms_band: tuple[int,int] | None = (300, 3400),  # para métrica de energía
    beta: float = 1e3, eps: float = 1e-8,
    use_pcen: bool = True,
    pcen_mels: int = 64, pcen_fmin: int = 80, pcen_fmax: int | None = None,
    pcen_time_constant: float = 0.06, pcen_gain: float = 0.98,
    pcen_bias: float = 2.0, pcen_power: float = 0.5,
    pcen_percentile: float = 88.0,
    use_pitch_gate: bool = True, f0_min: float = 70, f0_max: float = 400,
    use_webrtcvad: bool = True, vad_aggressiveness: int = 2, vad_frame_ms: int = 20,
    # DEBUG / EXPORT
    debug: bool = False, debug_audio_save: bool = False, debug_basepath: str = "preproc",
    stft_nfft: int = 1024, stft_hop: int | None = None, norm_preproc_peak: bool = True,
    # salvavidas
    safety_bypass: bool = True,        # si PB deja la señal “muerta”, volvemos a y_lev o sin gate
    min_peak_dbfs_after_pb: float = -30.0,   # si el pico queda por debajo de esto, algo va mal
    max_drop_db_after_pb: float = 20.0       # si cae > 20 dBFS respecto al leveller, revertir
):
    """
    Pipeline final:
      y -> leveler -> y_lev -> (auto gate?) -> pedalboard -> y_proc
      features (RMS/PCEN/pitch/VAD) SIEMPRE sobre y_proc
    """
    if y.ndim > 1: y = y.mean(axis=1)
    if pcen_fmax is None: pcen_fmax = sr // 2
    if stft_hop is None:  stft_hop = hop_length

    # 0) LEVELER
    y_lev = _leveler(y, sr, mode=leveler_mode, target_lufs=target_lufs, limiter_ceiling_db=limiter_ceiling_db) if enable_leveler else y.copy()

    # 0.1) Auto-umbral del gate (si no lo pasaste): estima piso de ruido de 0–2 s
    auto_gate_db = pb_gate_thresh_db
    if use_pedalboard and pb_gate_thresh_db is None:
        n_ref = min(len(y_lev), int(2.0*sr))
        ref = y_lev[:n_ref] if n_ref>0 else y_lev
        # dBFS de mediana (robusto) + margen
        ref_rms = np.sqrt(np.mean(ref**2) + 1e-12)
        ref_db  = 20*np.log10(ref_rms + 1e-12)
        auto_gate_db = max(-60.0, min(-30.0, ref_db + 6.0))  # entre -60 y -30 dBFS

    # 1) PEDALBOARD
    if use_pedalboard and not HAVE_PEDALBOARD:
        warnings.warn(
            "pedalboard no está instalado; se omite la cadena de procesamiento",
            RuntimeWarning,
            stacklevel=2,
        )
        use_pedalboard = False

    if use_pedalboard:
        # clamp LP a Nyquist por si sr=8k (nyq=4k): usa 3400–3800 típico PSTN
        nyq = 0.5*sr
        lp = None if pb_lp_hz is None else min(pb_lp_hz, nyq*0.95)
        y_pb = _apply_pedalboard_chain(
            y_lev, sr,
            hp_hz=pb_hp_hz, lp_hz=lp,
            gate_thresh_db=auto_gate_db, gate_ratio=pb_gate_ratio,
            gate_attack_ms=pb_gate_attack_ms, gate_release_ms=pb_gate_release_ms,
            comp_thresh_db=pb_comp_thresh_db, comp_ratio=pb_comp_ratio,
            comp_attack_ms=pb_comp_attack_ms, comp_release_ms=pb_comp_release_ms,
            makeup_gain_db=pb_makeup_gain_db,
            limit_thresh_db=pb_limit_thresh_db, limit_release_ms=pb_limit_release_ms
        )
        # métricas de seguridad
        def _peak_dbfs(sig):
            return 20*np.log10(np.max(np.abs(sig))+1e-12)
        pk_lev = _peak_dbfs(y_lev)
        pk_pb  = _peak_dbfs(y_pb)
        if safety_bypass and (pk_pb < min_peak_dbfs_after_pb or (pk_lev - pk_pb) > max_drop_db_after_pb):
            # reintenta SIN gate (por si el gate se comió todo)
            y_pb2 = _apply_pedalboard_chain(
                y_lev, sr,
                hp_hz=pb_hp_hz, lp_hz=lp,
                gate_thresh_db=None, gate_ratio=pb_gate_ratio,  # sin gate
                gate_attack_ms=pb_gate_attack_ms, gate_release_ms=pb_gate_release_ms,
                comp_thresh_db=pb_comp_thresh_db, comp_ratio=pb_comp_ratio,
                comp_attack_ms=pb_comp_attack_ms, comp_release_ms=pb_comp_release_ms,
                makeup_gain_db=pb_makeup_gain_db,
                limit_thresh_db=pb_limit_thresh_db, limit_release_ms=pb_limit_release_ms
            )
            pk_pb2 = 20*np.log10(np.max(np.abs(y_pb2))+1e-12)
            if pk_pb2 < min_peak_dbfs_after_pb:
                # último recurso: bypass total de PB
                y_proc = y_lev.copy()
            else:
                y_proc = y_pb2
        else:
            y_proc = y_pb
    else:
        y_proc = y_lev.copy()

    # 2) Métrica de energía en banda (sobre y_proc)
    y_for_rms = y_proc
    if rms_band is not None:
        b_bp, a_bp = sps.butter(2, rms_band, btype='bandpass', fs=sr)
        y_for_rms = sps.filtfilt(b_bp, a_bp, y_proc)

    rms = librosa.feature.rms(y=y_for_rms, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    log_rms = np.log1p(beta * np.maximum(rms, 0) + eps)

    # 3) PCEN (sobre y_proc)
    pcen_energy = pcen_thr = mel_db = mel_times = mel_freqs = pcen_arr = None
    if use_pcen:
        S = librosa.feature.melspectrogram(
            y=y_proc, sr=sr, n_fft=max(n_fft, DEFAULT_N_FFT), hop_length=hop_length,
            n_mels=pcen_mels, fmin=pcen_fmin, fmax=pcen_fmax or sr//2, power=1.0
        )
        mel_db = librosa.amplitude_to_db(S, ref=1.0)
        mel_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
        mel_freqs = librosa.mel_frequencies(n_mels=pcen_mels, fmin=pcen_fmin, fmax=(pcen_fmax or sr//2))
        P = librosa.pcen(S, sr=sr, hop_length=hop_length,
                         time_constant=pcen_time_constant, gain=pcen_gain,
                         bias=pcen_bias, power=pcen_power)
        pcen_arr = P
        pcen_energy = P.mean(axis=0)
        pcen_thr    = np.percentile(pcen_energy, pcen_percentile)

    # 4) Pitch (sobre y_proc)
    pitch_mask = np.zeros_like(log_rms, dtype=np.uint8)
    if use_pitch_gate:
        try:
            f0 = librosa.yin(
                y_proc,
                fmin=f0_min,
                fmax=f0_max,
                sr=sr,
                frame_length=max(DEFAULT_FRAME_LENGTH, n_fft),
                hop_length=hop_length,
            )
            pitch_mask = (~np.isnan(f0)).astype(np.uint8)
        except Exception:
            pass

    # 5) WebRTC VAD (sobre y_proc → resample a 16k si hace falta)
    vad_mask = np.zeros_like(log_rms, dtype=np.uint8)
    if use_webrtcvad:
        if not HAVE_WEBRTCVAD:
            warnings.warn(
                "webrtcvad no está instalado; se desactiva el VAD complementario",
                RuntimeWarning,
                stacklevel=2,
            )
            use_webrtcvad = False
        else:
            try:
                vad = webrtcvad.Vad(vad_aggressiveness)
                if sr != 16000:
                    y16k = librosa.resample(y_proc, orig_sr=sr, target_sr=16000)
                    sr_v = 16000
                else:
                    y16k = y_proc.copy(); sr_v = sr
                y16k = np.clip(y16k, -1, 1)
                pcm  = (y16k * 32767).astype(np.int16).tobytes()
                frame_len = int(sr_v * vad_frame_ms / 1000)  # 20ms -> 320 samples @16k
                step = frame_len
                frames = [pcm[i*2:(i+frame_len)*2] for i in range(0, len(y16k)-frame_len+1, step)]
                mask_v = np.array([vad.is_speech(fr, sr_v) for fr in frames], dtype=bool)

                hop_s     = hop_length / sr
                vad_times = np.arange(len(mask_v)) * (vad_frame_ms/1000.0)
                feat_len  = len(log_rms)
                feat_times = np.arange(feat_len) * hop_s
                vad_mask  = (np.interp(feat_times, vad_times, mask_v.astype(float), left=0, right=0) >= 0.5).astype(np.uint8)
            except Exception:
                pass

    # DEBUG: guardar etapas
    if debug_audio_save:
        sf.write(f"{debug_basepath}_leveler.wav", y_lev, sr)
        sf.write(f"{debug_basepath}_pb.wav",      y_proc, sr)  # post-PB (lo que sigue en el pipeline)
        y_dbg = y_for_rms.copy()
        if norm_preproc_peak:
            peak = np.max(np.abs(y_dbg)) + 1e-12
            y_dbg = (y_dbg/peak).astype(y_dbg.dtype)
        sf.write(f"{debug_basepath}_preproc.wav", y_dbg, sr)

    # DEBUG: STFT de la banda para inspección
    bp_spec_db = bp_times = bp_freqs = None
    if debug:
        Y = np.abs(librosa.stft(y_for_rms, n_fft=stft_nfft, hop_length=stft_hop))
        bp_spec_db = librosa.amplitude_to_db(Y, ref=1.0)
        bp_times = librosa.frames_to_time(np.arange(Y.shape[1]), sr=sr, hop_length=stft_hop)
        bp_freqs = np.linspace(0, sr/2, Y.shape[0])

    return {
        # Señales que seguirán en el pipeline:
        "y": y_proc,                 # <--- ESTA es la señal para segmentar
        "y_preproc": y_for_rms,      # banda para energía (opcional, debug)
        "sr": sr, "n_fft": n_fft, "hop_length": hop_length,
        # Features:
        "log_rms": log_rms,
        "pcen_energy": pcen_energy, "pcen_thr": pcen_thr,
        "pitch_mask": pitch_mask, "vad_mask": vad_mask,
        "t_frames": librosa.frames_to_time(np.arange(len(log_rms)), sr=sr, hop_length=hop_length),
        # Debug payloads:
        "bp_spec_db": bp_spec_db, "bp_times": bp_times, "bp_freqs": bp_freqs,
        "mel_db": mel_db, "mel_times": mel_times, "mel_freqs": mel_freqs, "pcen": pcen_arr,
        # Meta:
        "leveler_mode": leveler_mode, "target_lufs": target_lufs, "limiter_ceiling_db": limiter_ceiling_db
    }
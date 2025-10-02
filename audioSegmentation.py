import os, numpy as np, librosa, soundfile as sf, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter, binary_dilation
import scipy.signal as sps

try:
    from .constants import (
        DEFAULT_FRAME_LENGTH,
        DEFAULT_HOP_LENGTH,
        DEFAULT_N_FFT,
    )
except ImportError:  # pragma: no cover - fallback for alternate package names
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


def segment_speech(
    y: np.ndarray, sr: int,
    # K-means base (energía)
    n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH,
    silence_percentile: float = 5, speech_percentile: float = 99,
    smooth_size: int = 1, kmeans_n_init: int = 10, random_state: int = 0,
    beta: float = 1e3, eps: float = 1e-8,
    # Robustez extra
    rms_band: tuple[int,int] | None = (300, 3400),   # banda para RMS
    use_pcen: bool = True,                           # compresión/AGC tipo PCEN
    pcen_mels: int = 64, pcen_fmin: int = 80, pcen_fmax: int | None = None,
    pcen_time_constant: float = 0.06, pcen_gain: float = 0.98,
    pcen_bias: float = 2.0, pcen_power: float = 0.5,
    pcen_percentile: float = 85.0,                   # umbral PCEN (percentil)
    use_pitch_gate: bool = True, f0_min: float = 70, f0_max: float = 400,
    use_webrtcvad: bool = False, vad_aggressiveness: int = 2, vad_frame_ms: int = 20,
    # Estrictez para NO-actividad
    min_speech_s: float = 0.25,      # elimina tramos de habla demasiado cortos
    min_silence_s: float = 0.20,     # fusiona silencios cortos entre hablas
    speech_guard_s: float = 0.12,    # margen de seguridad ± alrededor de la voz
    # DEBUG
    debug: bool = False, debug_save: bool = True, debug_out: str | None = None,
    debug_audio_save: bool = False, debug_basepath: str = "seg_out",
    stft_nfft: int = DEFAULT_N_FFT, stft_hop: int = DEFAULT_HOP_LENGTH, fmax: float | None = 4000,
    db_range: int = 80
):
    """
    Devuelve: mask (1=habla por frame), log_rms, sr
    NO-actividad será estricta: ruido solo, sin voz (se usa compresión PCEN, pitch/VAD y margen).
    """
    if y.ndim > 1: y = y.mean(axis=1)
    if pcen_fmax is None: pcen_fmax = sr//2

    # -------- 1) Energía base (opcionalmente en banda de voz)
    y_rms = y
    if rms_band is not None:
        b_bp, a_bp = sps.butter(2, rms_band, btype='bandpass', fs=sr)
        y_rms = sps.filtfilt(b_bp, a_bp, y)

    rms = librosa.feature.rms(y=y_rms, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    log_rms = np.log1p(beta * np.maximum(rms, 0) + eps)

    # -------- 2) K-means (energía) → máscara base
    X = log_rms.reshape(-1, 1)
    silence_init = np.percentile(log_rms, silence_percentile)
    speech_init  = np.percentile(log_rms,  speech_percentile)
    init_centers = np.array([[silence_init], [speech_init]], dtype=float)
    kmeans = KMeans(n_clusters=2, init=init_centers, n_init=kmeans_n_init, random_state=random_state).fit(X)
    centers = kmeans.cluster_centers_.ravel()
    speech_label = int(np.argmax(centers))
    mask_energy = (kmeans.labels_ == speech_label).astype(np.uint8)

    # -------- 3) PCEN mel (AGC/compresión) → máscara extra
    mask_pcen = np.zeros_like(mask_energy)
    if use_pcen:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=stft_nfft, hop_length=hop_length,
            n_mels=pcen_mels, fmin=pcen_fmin, fmax=pcen_fmax, power=1.0
        )
        P = librosa.pcen(S, sr=sr, hop_length=hop_length,
                         time_constant=pcen_time_constant, gain=pcen_gain,
                         bias=pcen_bias, power=pcen_power)
        pcen_energy = np.mean(P, axis=0)
        thr_pcen = np.percentile(pcen_energy, pcen_percentile)
        mask_pcen = (pcen_energy >= thr_pcen).astype(np.uint8)

    # -------- 4) Periodicidad (pitch) → máscara de voz
    mask_pitch = np.zeros_like(mask_energy)
    if use_pitch_gate:
        try:
            f0 = librosa.yin(
                y,
                fmin=f0_min,
                fmax=f0_max,
                sr=sr,
                frame_length=max(DEFAULT_FRAME_LENGTH, n_fft),
                hop_length=hop_length,
            )
            mask_pitch = (~np.isnan(f0)).astype(np.uint8)
        except Exception:
            pass

    # -------- 5) WebRTC VAD (opcional; muy estricto)
    mask_vad = np.zeros_like(mask_energy)
    if use_webrtcvad:
        try:
            import webrtcvad
            vad = webrtcvad.Vad(vad_aggressiveness)
            # asegurar 16k mono int16
            if sr != 16000:
                y16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr_v = 16000
            else:
                y16k = y.copy(); sr_v = sr
            # int16
            y16k = np.clip(y16k, -1, 1)
            pcm = (y16k * 32767).astype(np.int16).tobytes()
            frame_len = int(sr_v * vad_frame_ms / 1000)   # 160, 320, 480 samples for 10/20/30ms @16k
            step = frame_len
            frames = [pcm[i*2:(i+frame_len)*2] for i in range(0, len(y16k)-frame_len+1, step)]
            # map VAD frames → feature frames (hop_length @ sr)
            hop_s = hop_length / sr
            vad_times = np.arange(len(frames)) * (vad_frame_ms/1000.0)
            mask_v = np.array([vad.is_speech(fr, sr_v) for fr in frames], dtype=bool)
            # upsample: asignar a frames de features
            feat_frames = len(mask_energy)
            feat_times  = np.arange(feat_frames) * hop_s
            mask_vad = np.interp(feat_times, vad_times, mask_v.astype(float), left=0, right=0) >= 0.5
            mask_vad = mask_vad.astype(np.uint8)
        except Exception:
            pass

    # -------- 6) Combinación estricta de “voz”
    mask = (mask_energy | mask_pcen | mask_pitch | mask_vad).astype(np.uint8)

    # suavizado + limpieza de habla corta y fusión de silencios cortos
    if smooth_size and smooth_size > 1:
        mask = median_filter(mask, size=smooth_size)

    def runs_from_mask(m):
        m = m.astype(int)
        dif = np.diff(m, prepend=m[0])
        starts = np.where(dif == 1)[0]
        ends   = np.where(dif == -1)[0]
        if m[0] == 1: starts = np.r_[0, starts]
        if m[-1] == 1: ends = np.r_[ends, len(m)-1]
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    hop_s = hop_length / sr
    # quitar hablas < min_speech_s
    cleaned = np.zeros_like(mask)
    for s,e in runs_from_mask(mask):
        if (e - s + 1) * hop_s >= min_speech_s:
            cleaned[s:e+1] = 1
    # fusionar silencios < min_silence_s
    merged = np.zeros_like(mask)
    prev = None
    for s,e in runs_from_mask(cleaned):
        if prev is None:
            prev = [s,e]; continue
        gap = (s - prev[1] - 1) * hop_s
        if gap < min_silence_s:
            prev[1] = e
        else:
            merged[prev[0]:prev[1]+1] = 1
            prev = [s,e]
    if prev is not None:
        merged[prev[0]:prev[1]+1] = 1
    mask = merged

    # margen de seguridad: dilatar voz para que NO-actividad sea súper estricta
    guard_frames = max(1, int(round(speech_guard_s / hop_s)))
    mask = binary_dilation(mask.astype(bool), structure=np.ones(2*guard_frames+1)).astype(np.uint8)

    # -------- 7) DEBUG (comparación justa + WAVs)
    if debug:
        # construir actividad / no-actividad concatenadas
        active_chunks, inactive_chunks = [], []
        last_end = 0
        for s,e in runs_from_mask(mask):
            a0 = int(s*hop_length); a1 = int((e+1)*hop_length)
            a0 = max(0, min(a0, len(y))); a1 = max(0, min(a1, len(y)))
            if a0 > last_end: inactive_chunks.append(y[last_end:a0])
            if a1 > a0:       active_chunks.append(y[a0:a1])
            last_end = a1
        if last_end < len(y): inactive_chunks.append(y[last_end:])
        y_act   = np.concatenate(active_chunks)   if active_chunks   else np.array([], dtype=y.dtype)
        y_noact = np.concatenate(inactive_chunks) if inactive_chunks else np.array([], dtype=y.dtype)

        if debug_audio_save:
            sf.write(f"{debug_basepath}_activity.wav",   y_act,   sr)
            sf.write(f"{debug_basepath}_noactivity.wav", y_noact, sr)
            print(f"[DEBUG] Guardados: {debug_basepath}_activity.wav | {debug_basepath}_noactivity.wav")

        # figura
        t_frames = librosa.frames_to_time(np.arange(len(log_rms)), sr=sr, hop_length=hop_length)
        fig = plt.figure(figsize=(12, 8))
        gs  = fig.add_gridspec(2, 2, height_ratios=[1.1, 2.0], hspace=0.35, wspace=0.15)
        ax_top = fig.add_subplot(gs[0,:]); ax_a = fig.add_subplot(gs[1,0]); ax_b = fig.add_subplot(gs[1,1])
        ax_top.plot(t_frames, log_rms, label='log_rms')
        for s,e in runs_from_mask(mask):
            ax_top.axvspan(t_frames[s], t_frames[min(e, len(t_frames)-1)], color='tab:green', alpha=0.15, lw=0)
        ax_top.set_title('log_rms y regiones detectadas (verde = habla, dilatada)')
        ax_top.set_xlabel('Tiempo (s)'); ax_top.set_ylabel('log_rms'); ax_top.legend(loc='upper right')

        def _spec(ax, sig, title):
            if sig.size == 0:
                ax.text(0.5,0.5,'Sin datos',ha='center',va='center'); ax.set_axis_off(); return None
            S = np.abs(librosa.stft(sig, n_fft=stft_nfft, hop_length=stft_hop))
            S_db = librosa.amplitude_to_db(S, ref=1.0)  # dBFS común
            times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=stft_hop)
            freqs = np.linspace(0, sr/2, S.shape[0])
            im = ax.imshow(S_db, origin='lower', aspect='auto',
                           extent=[times[0] if times.size else 0, times[-1] if times.size else 0, freqs[0], freqs[-1]],
                           cmap='magma', vmin=-db_range, vmax=0)
            if fmax is not None: ax.set_ylim(0, min(fmax, sr/2))
            ax.set_xlabel('Tiempo (s)'); ax.set_ylabel('Frecuencia (Hz)'); ax.set_title(title)
            return im
        im_a = _spec(ax_a, y_act,   'ACTIVIDAD (dBFS, escala común)')
        im_b = _spec(ax_b, y_noact, 'NO ACTIVIDAD (dBFS, escala común)')
        if im_a is not None:
            cbar = fig.colorbar(im_a, ax=[ax_a, ax_b], pad=0.01); cbar.set_label('Amplitud (dBFS)')
        if debug_save:
            if debug_out is None: debug_out = 'segment_speech_debug.png'
            fig.savefig(debug_out, dpi=150, bbox_inches='tight'); print(f"[DEBUG] Figura guardada en: {debug_out}")
        plt.show()

    return mask, log_rms, sr
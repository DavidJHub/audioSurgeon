import os, traceback
import pandas as pd
from pydub import AudioSegment
from pydub.utils import which

def concatenate_audios_main(download_path: str) -> None:

    # -- 0. Confirmar que FFmpeg existe --
    AudioSegment.converter = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    if not os.path.isfile(AudioSegment.converter):
        raise RuntimeError(
            f"FFmpeg no encontrado en {AudioSegment.converter}. "
            "Instálalo o ajusta la ruta."
        )

    # -- 1. Reúne .mp3 y .wav --
    audio_files = [
        f for f in os.listdir(download_path)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    if not audio_files:
        print("No hay audios mp3/wav en la carpeta.")
        return

    # -- 2. Extrae info del nombre --
    def extract_info(fname):
        parts = fname.split("_")
        if len(parts) == 4:
            return dict(
                archivo=fname,
                fecha=parts[1],
                lead_id=parts[2],
                idagent=parts[3],
            )
        if len(parts) == 5:
            return dict(
                archivo=fname,
                fecha=parts[1],
                lead_id=parts[2],
                idagent=parts[3],
                idclient=parts[4]
            )
        elif len(parts) == 6:
            return dict(
                archivo=fname,
                fecha=parts[1],
                lead_id=parts[2],
                idcall=parts[3],
                idclient=parts[4],
                phone=parts[5]
            )
        else:
            raise ValueError(f"Nombre inesperado: {fname}")

    df = pd.DataFrame([extract_info(f) for f in audio_files])

    # -- 3. Obtén sólo lead_id con ≥2 archivos --
    grupos = (
        df.groupby("lead_id")
          .filter(lambda g: len(g) > 1)
          .sort_values("fecha")           # orden cronológico
          .groupby("lead_id")["archivo"]
          .apply(list)
    )

    if grupos.empty:
        print("No hay grupos con más de un audio para concatenar.")
        return

    # -- 4. Concatenación + borrado seguro --
    def concat(files, out_path):
        concat_audio = AudioSegment.empty()
        for f in files:
            ruta = os.path.join(download_path, f)
            if not os.path.isfile(ruta):
                raise FileNotFoundError(ruta)
            ext = os.path.splitext(f)[1][1:].lower()
            concat_audio += AudioSegment.from_file(ruta, format=ext)
        concat_audio.export(out_path, format="mp3")

    for lead_id, files in grupos.items():
        result_name = f"{os.path.splitext(files[-1])[0]}-concat.mp3"
        out_path = os.path.join(download_path, result_name)

        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} ya existe.")
            continue

        try:
            print(f"[INFO] Concatenando {len(files)} audios → {result_name}")
            concat(files, out_path)
        except Exception:
            print(f"[ERROR] Falló lead_id {lead_id}")
            traceback.print_exc()
            continue      # no borres nada si falla
        else:
            # sólo borra cuando todo ha ido bien
            for f in files:
                try:
                    os.remove(os.path.join(download_path, f))
                except OSError as e:
                    print(f"[WARN] No se pudo borrar {f}: {e}")

    print("Proceso terminado.")

# Ejemplo:
# concatenate_audios_main(r"C:\grabaciones")

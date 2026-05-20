import io

import librosa
import numpy as np


def estimate_snr(audio_bytes: bytes) -> float:
    """
    Estima SNR aproximado desde bytes de audio.
    """

    try:
        # Leer desde memoria
        audio_buffer = io.BytesIO(audio_bytes)

        # Cargar audio
        y, sr = librosa.load(audio_buffer, sr=None, mono=True)

        if len(y) == 0:
            return 0.0

        # RMS por frame
        rms = librosa.feature.rms(y=y)[0]

        if len(rms) < 2:
            return 0.0

        # Percentiles robustos
        noise_floor = np.percentile(rms, 20)
        signal_level = np.percentile(rms, 95)

        noise_floor = max(noise_floor, 1e-10)

        # SNR en dB
        snr_db = 20 * np.log10(signal_level / noise_floor)

        return round(float(snr_db), 2)

    except Exception as e:
        print(f"Error calculando SNR: {e}")
        return 0.0

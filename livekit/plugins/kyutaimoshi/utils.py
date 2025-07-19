import numpy as np


def float32_to_int16_pcm(float_audio: np.ndarray) -> np.ndarray:
    return np.clip((float_audio * 32767), -32768, 32767).astype(np.int16)
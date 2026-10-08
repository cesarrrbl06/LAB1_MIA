import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

def compute_psnr(original: np.ndarray, filtered: np.ndarray, data_range: float = 1.0) -> float:
    """
    Calcula la relación señal-ruido de pico (PSNR) en dB entre la imagen original y la filtrada.
    """
    return float(psnr_fn(original, filtered, data_range=data_range))

def compute_ssim(original: np.ndarray, filtered: np.ndarray, data_range: float = 1.0) -> float:
    """
    Calcula el índice de similitud estructural (SSIM) entre la imagen original y la filtrada.
    """
    return float(ssim_fn(original, filtered, data_range=data_range))

def evaluate_quality(original: np.ndarray, filtered: np.ndarray) -> dict[str, float]:
    """
    Devuelve un diccionario con las métricas principales (PSNR en dB, SSIM y MSE).
    """
    mse = float(np.mean((original - filtered) ** 2))
    psnr_val = compute_psnr(original, filtered)
    ssim_val = compute_ssim(original, filtered)
    return {
        "PSNR (dB)": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "MSE": round(mse, 6)
    }


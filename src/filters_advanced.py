import numpy as np
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

def filter_bilateral(
    img: np.ndarray, 
    win_size: int = 5, 
    sigma_color: float = 0.05, 
    sigma_spatial: float = 1.5
) -> np.ndarray:
    """
    Filtro Bilateral: suaviza combinando proximidad espacial y similitud radiométrica (preserva bordes).
    """
    return denoise_bilateral(
        img,
        win_size=win_size,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial
    )

def filter_total_variation(img: np.ndarray, weight: float = 0.1) -> np.ndarray:
    """
    Filtrado por Variación Total (Total Variation Chambolle / ROF model).
    Excelente para preservar transiciones abruptas y bordes eliminando ruido gaussiano.
    """
    return denoise_tv_chambolle(img, weight=weight)


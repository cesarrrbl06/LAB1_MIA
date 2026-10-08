import numpy as np
from scipy import ndimage

def filter_mean(img: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Aplica un filtro de media espacial de tamaño size x size.
    """
    return ndimage.uniform_filter(img, size=size, mode="reflect")

def filter_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica un filtro gaussiano espacial 2D con desviación típica sigma.
    """
    return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")

def filter_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Aplica un filtro de mediana espacial con ventana size x size.
    Especialmente eficaz para la eliminación de ruido impulsivo (sal y pimienta).
    """
    return ndimage.median_filter(img, size=size, mode="reflect")


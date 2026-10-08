import numpy as np
from scipy import ndimage

def filter_mean(img: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Applies a spatial 2D uniform mean filter of size x size.
    """
    return ndimage.uniform_filter(img, size=size, mode="reflect")

def filter_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Applies a spatial 2D Gaussian filter with standard deviation sigma.
    """
    return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")

def filter_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Applies a spatial 2D median filter with a window of size x size.
    Particularly effective at removing impulse (salt-and-pepper) noise.
    """
    return ndimage.median_filter(img, size=size, mode="reflect")

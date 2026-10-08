import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def filter_nlm(
    img: np.ndarray, 
    patch_size: int = 5, 
    patch_distance: int = 6, 
    h: float = None, 
    fast_mode: bool = True
) -> np.ndarray:
    """
    Applies the Non-Local Means (NLM) filtering algorithm to a grayscale image in [0.0, 1.0].
    
    Parameters
    ----------
    img : np.ndarray
        Grayscale input image [0.0, 1.0].
    patch_size : int, optional
        2D size of patches used for comparison, defaults to 5.
    patch_distance : int, optional
        Maximal distance in pixels where to search for similar patches, defaults to 6.
    h : float, optional
        Filter parameter controlling smoothing strength. 
        If None, estimated automatically from image noise standard deviation.
    fast_mode : bool, optional
        If True, use accelerated patch-based algorithm.
        
    Returns
    -------
    np.ndarray
        NLM filtered image.
    """
    sigma_est = np.mean(estimate_sigma(img))
    if h is None:
        h = 0.8 * sigma_est if sigma_est > 1e-4 else 0.05

    return denoise_nl_means(
        img,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        sigma=sigma_est,
        fast_mode=fast_mode
    )

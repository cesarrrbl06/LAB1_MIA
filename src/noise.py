import numpy as np

# Standard noise intensity levels for the lab assignment
GAUSS_LEVELS = {
    "low": 0.01,
    "medium": 0.08,
    "high": 0.20,
    # Spanish aliases for compatibility
    "bajo": 0.01,
    "medio": 0.08,
    "alto": 0.20
}

IMPULSE_LEVELS = {
    "low": 0.02,
    "medium": 0.08,
    "high": 0.20,
    # Spanish aliases for compatibility
    "bajo": 0.02,
    "medio": 0.08,
    "alto": 0.20
}

def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator = None) -> np.ndarray:
    """
    Adds zero-mean additive Gaussian noise with standard deviation sigma.
    
    Parameters
    ----------
    img : np.ndarray
        Input image normalized to [0.0, 1.0].
    sigma : float
        Standard deviation of the Gaussian distribution.
    rng : np.random.Generator, optional
        Random number generator for reproducible experiments.
        
    Returns
    -------
    np.ndarray
        Noisy image clipped to [0.0, 1.0].
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img + noise, 0.0, 1.0)

def add_impulse_noise(
    img: np.ndarray, 
    amount: float, 
    salt_vs_pepper: float = 0.5, 
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Adds impulse noise (salt and pepper) to an image.
    
    Parameters
    ----------
    img : np.ndarray
        Input image normalized to [0.0, 1.0].
    amount : float
        Overall proportion of corrupted pixels (between 0.0 and 1.0).
    salt_vs_pepper : float, optional
        Proportion of 'salt' (white=1.0) vs 'pepper' (black=0.0). Defaults to 0.5.
    rng : np.random.Generator, optional
        Random number generator for reproducible experiments.
        
    Returns
    -------
    np.ndarray
        Noisy image with impulse artifacts in [0.0, 1.0].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    noisy = img.copy()
    num_noisy = int(np.ceil(amount * img.size))
    
    # Random flat indices
    coords = rng.choice(img.size, size=num_noisy, replace=False)
    
    # Split into salt and pepper
    num_salt = int(np.ceil(num_noisy * salt_vs_pepper))
    salt_idx = coords[:num_salt]
    pepper_idx = coords[num_salt:]
    
    flat = noisy.ravel()
    flat[salt_idx] = 1.0
    flat[pepper_idx] = 0.0
    
    return noisy

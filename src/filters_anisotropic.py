import numpy as np

def filter_perona_malik(
    img: np.ndarray,
    n_iter: int = 15,
    kappa: float = 0.05,
    gamma: float = 0.20,
    option: int = 1
) -> np.ndarray:
    """
    Applies Perona and Malik anisotropic diffusion (gradient-adaptive smoothing).
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale input image normalized to [0.0, 1.0].
    n_iter : int, optional
        Number of diffusion iterations (defaults to 15).
    kappa : float, optional
        Conduction coefficient threshold (edge sensitivity parameter, defaults to 0.05).
    gamma : float, optional
        Integration constant (time step). Must be <= 0.25 for numerical stability in 2D (defaults to 0.20).
    option : int, optional
        1: Exponential conduction c(g) = exp(-(g/kappa)^2) -> privileges high-contrast edges.
        2: Rational conduction c(g) = 1 / (1 + (g/kappa)^2) -> privileges wider regions.
        
    Returns
    -------
    np.ndarray
        Anisotropically filtered image.
    """
    diff = img.astype(np.float32).copy()
    k2 = np.float32(kappa * kappa)
    g = np.float32(gamma)
    
    # Pre-allocate gradient buffers once to avoid reallocations in the loop
    delta_n = np.empty_like(diff)
    delta_s = np.empty_like(diff)
    delta_e = np.empty_like(diff)
    delta_w = np.empty_like(diff)
    
    for _ in range(n_iter):
        delta_n[:-1, :] = diff[1:, :] - diff[:-1, :]
        delta_n[-1, :] = 0.0
        
        delta_s[1:, :]  = diff[:-1, :] - diff[1:, :]
        delta_s[0, :] = 0.0
        
        delta_e[:, :-1] = diff[:, 1:] - diff[:, :-1]
        delta_e[:, -1] = 0.0
        
        delta_w[:, 1:]  = diff[:, :-1] - diff[:, 1:]
        delta_w[:, 0] = 0.0
        
        if option == 1:
            c_n = np.exp(-(delta_n * delta_n) / k2)
            c_s = np.exp(-(delta_s * delta_s) / k2)
            c_e = np.exp(-(delta_e * delta_e) / k2)
            c_w = np.exp(-(delta_w * delta_w) / k2)
        elif option == 2:
            c_n = 1.0 / (1.0 + (delta_n * delta_n) / k2)
            c_s = 1.0 / (1.0 + (delta_s * delta_s) / k2)
            c_e = 1.0 / (1.0 + (delta_e * delta_e) / k2)
            c_w = 1.0 / (1.0 + (delta_w * delta_w) / k2)
        else:
            raise ValueError("option must be 1 or 2.")
            
        diff += g * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)
        
    return np.clip(diff, 0.0, 1.0)

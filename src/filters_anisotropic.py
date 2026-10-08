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
    diff = img.astype(np.float64).copy()
    
    for _ in range(n_iter):
        # Finite difference directional gradients (North, South, East, West)
        delta_n = np.zeros_like(diff)
        delta_s = np.zeros_like(diff)
        delta_e = np.zeros_like(diff)
        delta_w = np.zeros_like(diff)
        
        delta_n[:-1, :] = diff[1:, :] - diff[:-1, :]
        delta_s[1:, :]  = diff[:-1, :] - diff[1:, :]
        delta_e[:, :-1] = diff[:, 1:] - diff[:, :-1]
        delta_w[:, 1:]  = diff[:, :-1] - diff[:, 1:]
        
        # Conduction coefficients
        if option == 1:
            c_n = np.exp(-(delta_n / kappa) ** 2)
            c_s = np.exp(-(delta_s / kappa) ** 2)
            c_e = np.exp(-(delta_e / kappa) ** 2)
            c_w = np.exp(-(delta_w / kappa) ** 2)
        elif option == 2:
            c_n = 1.0 / (1.0 + (delta_n / kappa) ** 2)
            c_s = 1.0 / (1.0 + (delta_s / kappa) ** 2)
            c_e = 1.0 / (1.0 + (delta_e / kappa) ** 2)
            c_w = 1.0 / (1.0 + (delta_w / kappa) ** 2)
        else:
            raise ValueError("option must be 1 or 2.")
            
        # Update diffusion state
        diff += gamma * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)
        
    return np.clip(diff, 0.0, 1.0)

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
    Aplica el algoritmo Non-Local Means (NLM) a una imagen en escala de grises [0, 1].
    
    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises [0, 1].
    patch_size : int, optional
        Tamaño 2D del parche (ventana local de comparación), por defecto 5.
    patch_distance : int, optional
        Radio de búsqueda para parches similares, por defecto 6.
    h : float, optional
        Parámetro de filtrado que controla la fuerza del suavizado. 
        Si es None, se estima automáticamente en función de la desviación del ruido.
    fast_mode : bool, optional
        Si es True, utiliza aproximación acelerada por traslación de parches.
        
    Returns
    -------
    np.ndarray
        Imagen filtrada mediante NLM.
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


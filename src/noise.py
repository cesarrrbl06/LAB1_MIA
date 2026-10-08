import numpy as np

# Niveles estándar para la práctica
GAUSS_LEVELS = {
    "bajo": 0.01,
    "medio": 0.08,
    "alto": 0.20
}

IMPULSE_LEVELS = {
    "bajo": 0.02,
    "medio": 0.08,
    "alto": 0.20
}

def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator = None) -> np.ndarray:
    """
    Añade ruido gaussiano aditivo de media 0 y desviación típica sigma.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada normalizada en [0, 1].
    sigma : float
        Desviación estándar del ruido gaussiano.
    rng : np.random.Generator, optional
        Generador de números aleatorios para reproducibilidad.
        
    Returns
    -------
    np.ndarray
        Imagen ruidosa acotada a [0, 1].
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
    Añade ruido impulsivo (sal y pimienta) a la imagen.
    
    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada normalizada en [0, 1].
    amount : float
        Proporción total de píxeles afectados (entre 0.0 y 1.0).
    salt_vs_pepper : float, optional
        Proporción de píxeles 'sal' (blancos=1.0) frente a 'pimienta' (negros=0.0). Por defecto 0.5.
    rng : np.random.Generator, optional
        Generador de números aleatorios para reproducibilidad.
        
    Returns
    -------
    np.ndarray
        Imagen con ruido impulsivo en [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    noisy = img.copy()
    num_noisy = int(np.ceil(amount * img.size))
    
    # Índices planos aleatorios
    coords = rng.choice(img.size, size=num_noisy, replace=False)
    
    # Separar en sal y pimienta
    num_salt = int(np.ceil(num_noisy * salt_vs_pepper))
    salt_idx = coords[:num_salt]
    pepper_idx = coords[num_salt:]
    
    flat = noisy.ravel()
    flat[salt_idx] = 1.0
    flat[pepper_idx] = 0.0
    
    return noisy


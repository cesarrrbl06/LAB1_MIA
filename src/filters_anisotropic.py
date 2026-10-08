import numpy as np

def filter_perona_malik(
    img: np.ndarray,
    n_iter: int = 15,
    kappa: float = 0.05,
    gamma: float = 0.20,
    option: int = 1
) -> np.ndarray:
    """
    Aplica el filtrado anisotrópico de Perona y Malik (difusión adaptativa basada en gradiente).
    
    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada 2D en escala de grises [0, 1].
    n_iter : int, optional
        Número de iteraciones del proceso de difusión (por defecto 15).
    kappa : float, optional
        Parámetro de conductividad (umbral de gradiente que preserva bordes, por defecto 0.05).
    gamma : float, optional
        Paso de integración temporal. Debe ser <= 0.25 para estabilidad numérica 2D (por defecto 0.20).
    option : int, optional
        1: Función de conductividad exponencial c(g) = exp(-(g/kappa)^2) -> favorece bordes de alto contraste.
        2: Función de conductividad racional c(g) = 1 / (1 + (g/kappa)^2) -> favorece regiones más amplias.
        
    Returns
    -------
    np.ndarray
        Imagen filtrada mediante difusión anisotrópica.
    """
    diff = img.astype(np.float64).copy()
    
    for _ in range(n_iter):
        # Gradientes direccionales (N, S, E, W) usando diferencias finitas
        delta_n = np.zeros_like(diff)
        delta_s = np.zeros_like(diff)
        delta_e = np.zeros_like(diff)
        delta_w = np.zeros_like(diff)
        
        delta_n[:-1, :] = diff[1:, :] - diff[:-1, :]
        delta_s[1:, :]  = diff[:-1, :] - diff[1:, :]
        delta_e[:, :-1] = diff[:, 1:] - diff[:, :-1]
        delta_w[:, 1:]  = diff[:, :-1] - diff[:, 1:]
        
        # Coeficientes de conducción
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
            raise ValueError("option debe ser 1 o 2.")
            
        # Actualización de difusión
        diff += gamma * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)
        
    return np.clip(diff, 0.0, 1.0)


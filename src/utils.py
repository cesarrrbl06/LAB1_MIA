from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color

def get_project_root() -> Path:
    """Devuelve la ruta raíz del proyecto."""
    return Path(__file__).resolve().parent.parent

def load_gray(image_path: str | Path):
    """
    Carga una imagen y la convierte a escala de grises con valores float en el rango [0.0, 1.0].
    
    Parameters
    ----------
    image_path : str | Path
        Ruta del archivo de imagen.
        
    Returns
    -------
    numpy.ndarray
        Imagen en escala de grises float64 con rango [0, 1].
    """
    path = Path(image_path)
    if not path.is_absolute():
        # Resolver relativo a la raíz del proyecto si no existe en CWD
        if not path.exists():
            candidate = get_project_root() / path
            if candidate.exists():
                path = candidate
    
    img = io.imread(str(path))
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    return img_as_float(img)

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """
    Guarda una figura en results/figures/ para su inclusión en el informe PDF.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a guardar.
    filename : str
        Nombre del archivo (ej. 'ruido_gaussiano_mri.png').
    dpi : int, optional
        Resolución de salida (por defecto 300 dpi para calidad de memoria).
        
    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    output_dir = get_project_root() / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


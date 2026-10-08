from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent

def load_gray(image_path: str | Path):
    """
    Loads an image and converts it to grayscale with float values in [0.0, 1.0].
    
    Parameters
    ----------
    image_path : str | Path
        Path to the image file.
        
    Returns
    -------
    numpy.ndarray
        Grayscale float64 image scaled to [0.0, 1.0].
    """
    path = Path(image_path)
    if not path.is_absolute():
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
    Saves a figure to results/figures/ for inclusion in the final PDF report.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    filename : str
        Target file name (e.g. '01_gaussian_noise_mri.png').
    dpi : int, optional
        Resolution for publication quality (defaults to 300 dpi).
        
    Returns
    -------
    Path
        Path to the saved figure file.
    """
    output_dir = get_project_root() / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path

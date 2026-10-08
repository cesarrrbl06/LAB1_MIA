"""
Script principal del LAB 1: Métodos Avanzados de Preprocesado de Imagen Médica.
Comprueba la carga de datos, genera ruido y evalúa los filtros implementados.
"""

from pathlib import Path
from src.utils import load_gray, save_figure
from src.noise import (
    add_gaussian_noise, 
    add_impulse_noise, 
    GAUSS_LEVELS, 
    IMPULSE_LEVELS
)
from src.metrics import evaluate_quality
from src.filters_standard import filter_mean, filter_gaussian, filter_median
from src.filters_nlm import filter_nlm
from src.filters_anisotropic import filter_perona_malik
from src.filters_advanced import filter_bilateral, filter_total_variation

def run_pipeline_check():
    data_dir = Path("data/raw")
    img_files = list(data_dir.glob("*.png"))
    
    print(f"=== PRÁCTICA 1 MIA: Comprobación del Pipeline ===")
    print(f"Imágenes detectadas en '{data_dir}': {[f.name for f in img_files]}")
    
    if not img_files:
        print("ERROR: No se han encontrado imágenes en data/raw/")
        return

    test_img_path = img_files[0]
    print(f"\nProbando pipeline con: {test_img_path.name}")
    original = load_gray(test_img_path)
    print(f"- Imagen cargada con éxito. Tamaño: {original.shape}, Rango: [{original.min():.3f}, {original.max():.3f}]")

    # 1. Ruido
    noisy_gauss = add_gaussian_noise(original, sigma=GAUSS_LEVELS["medio"])
    noisy_impulse = add_impulse_noise(original, amount=IMPULSE_LEVELS["medio"])
    print(f"- Ruido gaussiano (medio): {evaluate_quality(original, noisy_gauss)}")
    print(f"- Ruido impulsivo (medio): {evaluate_quality(original, noisy_impulse)}")

    # 2. Filtros estándar
    f_mean = filter_mean(noisy_gauss, size=3)
    f_gauss = filter_gaussian(noisy_gauss, sigma=1.0)
    f_median = filter_median(noisy_impulse, size=3)
    print(f"- Filtro Media (sobre Gauss): {evaluate_quality(original, f_mean)}")
    print(f"- Filtro Gaussiano (sobre Gauss): {evaluate_quality(original, f_gauss)}")
    print(f"- Filtro Mediana (sobre Impulsivo): {evaluate_quality(original, f_median)}")

    # 3. NLM
    f_nlm = filter_nlm(noisy_gauss, patch_size=5, patch_distance=4, fast_mode=True)
    print(f"- Filtro NLM (sobre Gauss): {evaluate_quality(original, f_nlm)}")

    # 4. Anisotrópico (Perona & Malik)
    f_pm = filter_perona_malik(noisy_gauss, n_iter=10, kappa=0.05, option=1)
    print(f"- Filtro Perona-Malik (sobre Gauss): {evaluate_quality(original, f_pm)}")

    # 5. Método avanzado adicional (Bilateral & TV)
    f_bilateral = filter_bilateral(noisy_gauss, win_size=5, sigma_color=0.05, sigma_spatial=1.5)
    f_tv = filter_total_variation(noisy_gauss, weight=0.1)
    print(f"- Filtro Bilateral (sobre Gauss): {evaluate_quality(original, f_bilateral)}")
    print(f"- Filtro Total Variation (sobre Gauss): {evaluate_quality(original, f_tv)}")

    print("\n[OK] ¡Todos los módulos funcionan correctamente!")

if __name__ == "__main__":
    run_pipeline_check()

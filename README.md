# Práctica 1: Métodos Avanzados de Preprocesado de Imagen Médica
**Grado en Ingeniería Biomédica — Universidad Rey Juan Carlos (URJC)**  
*Asignatura: Análisis de Imagen Médica (MIA)*

---

## 📁 Estructura del Proyecto

```text
LAB1_MIA/
│
├── data/
│   ├── raw/                      # Imágenes clínicas originales (Brain MRI y Chest X-ray)
│   └── noisy/                    # (Opcional) Imágenes ruidosas generadas
│
├── notebooks/
│   └── LAB_1.ipynb               # Cuaderno de experimentación interactiva y visualización
│
├── src/                          # Código fuente modular en Python
│   ├── utils.py                  # Carga de imágenes (rutas relativas) y utilidades de guardado
│   ├── noise.py                  # Generación de ruido gaussiano e impulsivo (3 niveles)
│   ├── metrics.py                # Métricas objetivas de calidad (PSNR, SSIM, MSE)
│   ├── filters_standard.py       # Filtros estándar (media, gaussiano, mediana)
│   ├── filters_nlm.py            # Non-Local Means (skimage.restoration.denoise_nl_means)
│   ├── filters_anisotropic.py    # Difusión anisotrópica de Perona y Malik
│   └── filters_advanced.py       # Métodos avanzados del estado del arte (Bilateral / TV)
│
├── results/
│   ├── figures/                  # Gráficas e histogramas exportados para el informe
│   └── tables/                   # Tablas resumen de métricas numéricas
│
├── report/
│   ├── Lab_1_MIA.pdf             # Enunciado de la práctica
│   └── ...                       # Informe final en PDF (máximo 10 páginas)
│
├── main.py                       # Verificación rápida del pipeline completo
├── requirements.txt              # Dependencias del entorno Python
└── README.md                     # Documentación del repositorio
```

---

## 🚀 Requisitos e Instalación

Para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

---

## 🧪 Ejecución de Pruebas

Para comprobar que todo el pipeline y los módulos funcionan sin errores:

```bash
python main.py
```

Para continuar la experimentación interactiva y generar las visualizaciones para el informe:
1. Abre [notebooks/LAB_1.ipynb](file:///r:/Documents/LAB1MIA/LAB1_MIA/notebooks/LAB_1.ipynb).
2. Ejecuta las celdas de forma interactiva.

---

## 📦 Instrucciones para la Entrega Oficial
El archivo final debe ser un `.zip` nombrado según las instrucciones oficiales:
- Formato: `P1_MIA_BED_26_27_GroupX.zip` (ejemplo: `P1_MIA_BED_26_27_G1_2.zip`).
- Contenido requerido:
  1. Informe en PDF de máximo 10 páginas.
  2. Código fuente desarrollado (carpeta `src/`, `notebooks/` y `main.py`).
  3. Imágenes utilizadas (`data/raw/`).
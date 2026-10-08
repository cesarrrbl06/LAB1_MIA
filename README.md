# Lab 1: Advanced Medical Image Pre-Processing Methods
**Biomedical Engineering Degree — Universidad Rey Juan Carlos (URJC)**  
*Course: Medical Image Analysis (MIA)*

---

## 📁 Project Structure

```text
LAB1_MIA/
│
├── data/
│   └── raw/                           # Original grayscale clinical images
│       ├── Brain_MRI_131749_T1.png    # T1-weighted Brain MRI
│       └── Chest_Xray_PA_3-8-2010.png # Posteroanterior (PA) Chest X-ray
│
├── notebooks/
│   └── LAB_1.ipynb                    # Main experimentation notebook (English template)
│
├── src/                               # Modular Python package
│   ├── __init__.py
│   ├── utils.py                       # Image loading (relative paths) and high-res figure saving
│   ├── noise.py                       # Gaussian & Impulse noise generation (low, medium, high)
│   ├── metrics.py                     # Quantitative evaluation metrics (PSNR, SSIM, MSE)
│   ├── filters_standard.py            # Standard filters: Mean, Gaussian, and Median
│   ├── filters_nlm.py                 # Non-Local Means (skimage.restoration.denoise_nl_means)
│   ├── filters_anisotropic.py         # Perona and Malik anisotropic diffusion
│   └── filters_advanced.py            # Advanced literature filters (Bilateral & Total Variation)
│
├── results/
│   ├── figures/                       # Exported high-resolution plots for report (300 DPI)
│   └── tables/                        # Exported CSVs with numerical benchmarks
│
├── report/
│   ├── Lab_1_MIA.pdf                  # Official assignment prompt
│   └── (Final Report PDF)             # Final report document (max. 10 pages)
│
├── main.py                            # End-to-end pipeline verification script
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation and guide
```

---

## 🚀 Installation & Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`.

---

## 🧪 Pipeline Verification

To verify that all modules in `src/` load and process test images properly:

```bash
python main.py
```

---

## 📓 Running the Notebook

1. Open [notebooks/LAB_1.ipynb](file:///r:/Documents/LAB1MIA/LAB1_MIA/notebooks/LAB_1.ipynb).
2. Execute the setup cells and implement the `# TODO` sections for parameter exploration.
3. All plots are automatically saved into `results/figures/` via `save_figure(fig, filename)` for direct inclusion in the final report.

---

## 📦 Final Submission Format
The submission file must be a `.zip` named according to group guidelines:
- Name format: `P1_MIA_BED_26_27_GroupX.zip` (e.g. `P1_MIA_BED_26_27_G1_2.zip`).
- Deliverables required:
  1. Single PDF report (maximum 10 pages) including individual contributions and AI disclosure statement.
  2. Source code (`src/`, `notebooks/`, `main.py`).
  3. Original images used (`data/raw/`).
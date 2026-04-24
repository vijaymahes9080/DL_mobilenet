# 🔬 Research-Grade Deployment Guide

Follow these steps to deploy the full **Research-Grade Neural Synergy Pipeline** on Google Colab.

---

## 📂 Phase 1: Uploading to Google Drive

1.  **Open Google Drive**: Navigate to [drive.google.com](https://drive.google.com).
2.  **Create Root Folder**: Create a new folder named **`DL`** in your "My Drive".
3.  **Upload Pre-Structured Folders**: Drag and drop the following folders from `d:\current project\DL` into `DL/`:
    *   `notebooks/` (Contains the 11 research scripts)
    *   `dataset/` (Place your training images here)
    *   `models/`, `logs/`, `outputs/`, `processed_data/` (Pre-created for pipeline use)

---

## 🚀 Phase 2: Execution Workflow

Open the notebooks in Colab in the following order:

| Step | Notebook | Research Goal |
| :--- | :--- | :--- |
| 1 | `01_drive_setup.ipynb` | Environment init & folder verification. |
| 2 | `02_data_preprocessing.ipynb` | Data cleaning & class balancing. |
| 3 | `03_dataset_pipeline.ipynb` | High-speed `tf.data` configuration. |
| 4 | `04_model_architectures_comparison.ipynb` | EfficientNet vs MobileNet vs ResNet benchmarking. |
| 5 | `05_systematic_hyperparameter_tuning.ipynb` | Automated Grid Search (LR, BS, Dropout). |
| 6 | `06_advanced_metrics_evaluation.ipynb` | 16+ Statistical Metrics (MCC, Kappa, etc.). |
| 7 | `07_training_final_optimized_model.ipynb` | Final training of the "Champion Model". |
| 8 | `08_xai_explainability_visuals.ipynb` | Grad-CAM & Saliency Map visual evidence. |
| 9 | `09_ablation_study_performance_drop.ipynb` | Component contribution analysis. |
| 10 | `10_final_research_report.ipynb` | Automated PDF generation of findings. |
| 11 | `11_realtime_inference_optimization.ipynb` | TFLite optimization & Live Camera. |

---

## 💻 Phase 3: Local Colab Runtime (Optional)

If you want to run the training on your local GPU while using the Colab UI:

1.  **Install Jupyter HTTP Bridge**:
    ```bash
    pip install jupyter_http_over_ws
    jupyter serverextension enable --py jupyter_http_over_ws
    ```
2.  **Start Local Backend**:
    Run this command in your terminal inside the `DL` folder:
    ```bash
    jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retrying=False
    ```
3.  **Connect in Colab**:
    *   In the Colab UI, click the **Connect** button (top right).
    *   Select **Connect to a local runtime**.
    *   Enter the URL (usually `http://localhost:8888/`).

---

## 🏃 Phase 4: Standalone Local Training

If you don't want to use Jupyter/Colab at all, run the pre-configured training script:
```bash
python train_local.py
```
This script handles preprocessing, dataset building, and final training in one go.

## ⚠️ Critical Research Warnings
*   **GPU Usage**: Ensure **Runtime > Change runtime type > GPU** is selected for notebooks 04-09.
*   **Sequential Dependencies**: Each notebook generates logs or models required by the next. **Do not skip notebooks.**
*   **XAI Heatmaps**: If Grad-CAM heatmaps are blurry, try adjusting the `last_conv_layer_name` in Notebook 08 (e.g., `top_conv` for EfficientNet).


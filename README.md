# 🔬 Research-Grade Deep Learning Pipeline (EfficientNet-B0 & Beyond)

A production-ready, MLOps-driven pipeline for Google Colab. This system includes advanced statistical evaluation, multi-model benchmarking, and explainability (XAI).

---

## 📂 Project Structure (Google Drive)
```
/content/drive/MyDrive/DL/
├── dataset/           # Raw input images
├── processed_data/    # JSON configs and metadata
├── models/            # Benchmarked and final models (.h5, .tflite)
├── logs/              # Hyperparameter CSVs, TensorBoard logs
├── outputs/           # PDF Research Report, CMs, Loss Curves, XAI visuals
└── notebooks/         # Modular research scripts
```

---

## 📓 Research Workflow (Notebooks)

### 🏗️ Phase 1: Infrastructure
1. **`01_drive_setup.ipynb`**: Environment initialization.
2. **`02_data_preprocessing.ipynb`**: Data cleaning and balancing.
3. **`03_dataset_pipeline.ipynb`**: High-performance `tf.data` streams.

### 🔬 Phase 2: Benchmarking & Tuning
4. **`04_model_architectures_comparison.ipynb`**: B0 vs MobileNetV2 vs ResNet50.
5. **`05_systematic_hyperparameter_tuning.ipynb`**: Grid search over LR, Batch, Dropout.
6. **`06_advanced_metrics_evaluation.ipynb`**: 16+ research metrics (MCC, Kappa, etc.).

### 🚀 Phase 3: Final Training & Explainability
7. **`07_training_final_optimized_model.ipynb`**: Champion model training.
8. **`08_xai_explainability_visuals.ipynb`**: Grad-CAM and Saliency mapping.
9. **`09_ablation_study_performance_drop.ipynb`**: Component contribution analysis.

### 📑 Phase 4: Reporting & Deployment
10. **`10_final_research_report.ipynb`**: Automated PDF report generation.
11. **`11_realtime_inference_optimization.ipynb`**: TFLite conversion and live camera.

---

## ⚡ Technical Highlights
* **Metric Depth**: Computes Cohen's Kappa, MCC, Brier Score, and 13 other key indicators.
* **XAI**: Visualizes neural activations to confirm model focus on salient features.
* **Ablation**: Quantifies the "why" behind model performance by stripping components.
* **MLOps**: Fully automated logging and research report generation.


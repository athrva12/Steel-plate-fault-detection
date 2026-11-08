# SteelScope: Industrial Steel Plate Fault Detection

> **Coursework reference / learning starter:** personalize the UI, explain the method in your own words, and run the experiment yourself on BITS Virtual Lab. The course explicitly prohibits copying AI-generated solutions directly as submissions.

## a. Problem statement

Classify seven categories of steel plate surface defects from 27 numeric inspection features using five machine-learning algorithms. Compare their held-out classification performance and provide an interactive Streamlit frontend that *evaluates saved models without retraining*.

## b. Dataset descriptions

- **Name:** Steel Plates Faults, UCI Machine Learning Repository, dataset **198**.
- **Dataset link:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **Size:** 1,941 rows, 27 independent features, 7 fault types.
- **Classes:** Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults.
- **Features:** geometric coordinates, fault areas, luminosity, steel type, thickness, and other indices.
- **Original target:** seven one-hot label columns; converted to one `fault_type` column.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Citation:** Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI. https://doi.org/10.24432/C5J88N

**Note:** dataset rows describe already-observed faults. This is **fault-type classification**, not classification of healthy steel vs defective steel, and not image-based detection.

## c. GitHub repository link

`[https://github.com/athrva12/Steel-plate-fault-detection]`


## d. Models used and comparison

Five machine learning classification algorithms were
implemented and evaluated on the Steel Plates Faults
dataset using a stratified 80:20 train-test split.

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7198 | 0.9371 | 0.7464 | 0.7205 | 0.7309 | 0.6381 |
| Decision Tree | 0.7224 | 0.9106 | 0.7670 | 0.7075 | 0.7280 | 0.6433 |
| KNN | 0.7686 | 0.9220 | 0.7653 | 0.7866 | 0.7741 | 0.7044 |
| Naive Bayes | 0.6144 | 0.9238 | 0.6392 | 0.7190 | 0.6383 | 0.5542 |
| Support Vector Machine | 0.7532 | 0.9527 | 0.8013 | 0.7461 | 0.7673 | 0.6803 |

### Model Performance Observations

**Logistic Regression:**
Achieved 71.98% accuracy and 0.9371 AUC. It provided
reasonable classification results but performed below
KNN and SVM in overall accuracy.

**Decision Tree:**
Achieved 72.24% accuracy. Its performance was lower
than KNN and SVM, suggesting that the selected tree
configuration may not capture all fault patterns.

**K-Nearest Neighbors (KNN):**
Achieved the highest test accuracy of 76.86%,
with an F1 score of 0.7741 and MCC of 0.7044.
Distance-weighted voting improved its performance
compared with the earlier configuration.

**Naive Bayes:**
Achieved 61.44% accuracy, the lowest among
the five classifiers. Correlation between input
features may contribute to its weaker performance.

**Support Vector Machine (SVM):**
Achieved 75.32% test accuracy and the highest
AUC of 0.9527 and precision of 0.8013.
It was a strong competitor to KNN.

### Overall Winner

KNN achieved the highest accuracy, recall, F1,
and MCC on the held-out test dataset.

However, SVM achieved the best overall performance
during stratified 5-fold cross-validation.

### 5-Fold Cross-Validation Results

| Model | CV Accuracy | CV AUC | CV F1 | CV MCC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7145 | 0.8969 | 0.7111 | 0.6317 |
| Decision Tree | 0.7094 | 0.8803 | 0.7073 | 0.6281 |
| KNN | 0.7455 | 0.9085 | 0.7432 | 0.6753 |
| Naive Bayes | 0.5850 | 0.8697 | 0.5572 | 0.5217 |
| Support Vector Machine | 0.7713 | 0.9269 | 0.7702 | 0.7053 |

Based on cross-validation performance, SVM was
selected as the preferred overall classifier.

The cross-validation results demonstrate that model
selection should consider performance across multiple
training-validation splits rather than relying only
on a single test split.


## e. Live Streamlit App Link

**ADD YOUR REAL APP URL:** `[https://your-app-name.streamlit.app]`

### App features

- Upload labeled CSV containing 27 features and target `fault_type`.
- Choose any of five **saved, previously trained** models.
- Display Accuracy, AUC, Precision, Recall, F1, MCC.
- Show confusion matrix, classification report and class distribution.
- Compare five models on the same uploaded file; download metrics.

## f. BITS Virtual Lab proof

Insert **one genuine screenshot** showing your work running in BITS Virtual Lab in the submitted PDF. Do not submit a mock or fabricated screenshot.

## Project structure

```text
Steel_Plate_Faults_Assignment/
├── app.py
├── data_utils.py
├── train_models.py
├── verify_project.py
├── generate_report.py
├── requirements.txt
├── README.md
├── SETUP_GUIDE.md
├── model/                    # created during training
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metadata.json
├── train_data.csv            # created during training
├── test_data.csv             # created during training
├── model_metrics.csv         # created during training
└── screenshots/              # place genuine BITS lab screenshot here
```

## Getting started

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
python verify_project.py
python generate_report.py
python -m streamlit run app.py
```

In the app, upload `test_data.csv`, choose each model and inspect the evaluation output. If the first dataset download is blocked, obtain the **official UCI ZIP** and run `python train_models.py --dataset "/path/to/steel+plates+faults.zip"`.

## Deployment

Push the repository including all five `.joblib` files and test CSV, visit https://share.streamlit.io/ or https://streamlit.io/cloud, choose the repo and `main` branch, set entrypoint to `app.py`, and deploy. The cloud app **does not need to download the UCI dataset** because trained models and test CSV are already committed.

## Limitations and ethics

This is a research/teaching dataset of **known fault types**, not a safety-approved live manufacturing system. Classes are imbalanced; accuracy alone can be misleading. Model scores may vary with environment and library versions. Only load trusted `.joblib` artifacts.

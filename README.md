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

**ADD YOUR REAL REPOSITORY LINK AFTER PUSHING:** `[https://github.com/YOUR-USERNAME/YOUR-REPOSITORY]`

## d. Models used and comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | See `model_metrics.csv` after training | | | | | |
| Decision Tree | See `model_metrics.csv` after training | | | | | |
| KNN | See `model_metrics.csv` after training | | | | | |
| Naive Bayes | See `model_metrics.csv` after training | | | | | |
| Support Vector Machine | See `model_metrics.csv` after training | | | | | |

The scores are produced by `python train_models.py`. Values are **not pre-filled** until the actual dataset is downloaded and the models are trained.

### Experimental design

- Stratified 80:20 train/test split; fixed `random_state=42`.
- Preprocessing (StandardScaler) is fitted on training data only, within applicable scikit-learn pipelines.
- Accuracy, macro one-versus-rest AUC, macro Precision, macro Recall, macro F1 and multiclass MCC.
- Multiclass AUC uses `predict_proba`, with columns aligned to alphabetical class order.
- Test data is never used to fit the pipeline, and the app only loads `.joblib` models for evaluation.

### Observations about model performance (write from your actual run)

| Model | Observation to complete after training |
|---|---|
| Logistic Regression | Compare macro F1 and AUC with the other models; comment on linearity assumptions. |
| Decision Tree | Inspect class-wise mistakes and possible overfitting. |
| KNN | Describe the effect of feature scaling and neighbors. |
| Naive Bayes | Discuss the independence assumption and class-wise recall. |
| SVM | Compare macro F1 and MCC and runtime against alternatives. |
| Overall winner | Choose based on your measured metrics and justify using imbalance-aware metrics. |

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

# RUN AND SUBMIT — STEP-BY-STEP GUIDE

## A. Open the project in BITS Virtual Lab

1. Log into the BITS Virtual Lab provided for your course.
2. Open its terminal / Jupyter environment or VS Code, whichever is available.
3. Upload and extract `Steel_Plate_Faults_Assignment_Project.zip`, or upload the extracted project folder.
4. Navigate to the extracted `Steel_Plate_Faults_Assignment` directory.
5. Open a terminal there and check: `python --version` (Python 3.10+ recommended).

## B. Install dependencies

Run: `python -m pip install -r requirements.txt`

If `python` is not recognized, try `python3`. If virtual environments are permitted, create one with `python -m venv .venv` and activate it first. If the terminal says `Permission denied`, use a permitted virtual environment or `python -m pip install --user -r requirements.txt`.

## C. Download the UCI dataset and train five models

Run: `python train_models.py`

First run requires internet access to UCI. It downloads the official UCI Steel Plates Faults dataset ID 198, validates the one-hot fault labels, creates a stratified training/test split, trains five models, and saves model files.

**If internet is blocked on BITS Virtual Lab:** Download the official ZIP from https://archive.ics.uci.edu/dataset/198/steel+plates+faults in your browser, upload it into the project folder, then run:

`python train_models.py --dataset "steel+plates+faults.zip"`

If you extract the ZIP, use `python train_models.py --dataset "Faults.NNA"` instead.

Do **not** upload or replace data with randomly generated training records for the assignment.

## D. Validate outputs

Run: `python verify_project.py`

The project should have `train_data.csv`, `test_data.csv`, `model_metrics.csv`, and five `.joblib` models plus metadata in `model/`. The test CSV should contain 27 features plus `fault_type` and approximately 389 rows. Take **one genuine screenshot in BITS Virtual Lab** displaying the terminal results and lab desktop/context.

## E. Run Streamlit locally

Run: `python -m streamlit run app.py`

Open the URL printed by Streamlit (normally `http://localhost:8501`) in the environment that can access the server. If the Virtual Lab browser cannot access the app, use the lab's forwarded/preview port mechanism or test on your personal computer while keeping BITS Lab training proof.

In the left sidebar:
1. Select `Logistic Regression`.
2. Upload `test_data.csv`.
3. Verify Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Scroll to view confusion matrix and classification report.
5. Change dropdown to Decision Tree, KNN, Naive Bayes and SVM.
6. Expand **Compare all saved models** to view the comparison table.

## F. Prepare GitHub repository

1. Create a NEW PUBLIC GitHub repository. Example name: `steelscope-fault-classifier`.
2. Make your own meaningful improvements to code or UI and describe them in README.
3. From the project terminal run:

```
git init
git add .
git commit -m "Add steel fault dataset processing and training"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/steelscope-fault-classifier.git
git push -u origin main
```

Replace `YOUR-USERNAME`. Authenticate with GitHub when prompted. For existing repos, don't blindly re-run `git init` or `git remote add origin`. Ensure the repository includes saved models, requirements, README, test CSV, and source code.

## G. Deploy to Streamlit Cloud

1. Visit https://streamlit.io/cloud and sign in with GitHub.
2. Click **Create app** / **New app** (wording may change).
3. Select your GitHub repository and branch `main`.
4. Main file / app entrypoint: `app.py`.
5. Deploy. Open the generated `*.streamlit.app` link.
6. Upload `test_data.csv` and verify metrics + confusion matrix, and test every model.
7. If deployment fails, examine cloud logs and confirm saved `.joblib` models and `requirements.txt` are committed. Keep package versions compatible.

## H. Generate editable report

Run: `python generate_report.py` after successful training. This creates `SteelScope_Submission_DRAFT.docx` with real numerical results from `model_metrics.csv`.

Open in Word or LibreOffice, and **replace** name/ID/section, two real links, interpretations/reflection and actual BITS Virtual Lab screenshot. Export to a **single PDF**. The assignment requires the README content to appear inside the PDF (not merely a link), so verify that your final PDF includes the full current README text; you may append it after the report.

## I. Final checklist before submission

- [ ] UCI dataset has at least 500 records and 12 features (actual: 1,941 × 27)
- [ ] All five models are trained on the same dataset
- [ ] Six evaluation metrics for every model
- [ ] Test data in GitHub as CSV
- [ ] Five saved models available in GitHub
- [ ] Streamlit CSV upload, model selection, metrics and confusion matrix/report
- [ ] Live links opened and tested in an incognito tab
- [ ] Genuine BITS Virtual Lab screenshot added
- [ ] Entire README content included in final single PDF
- [ ] Personalization and original reflection added
- [ ] Final PDF submitted before course deadline

## Common errors

- `ModuleNotFoundError`: install `requirements.txt` into the same Python interpreter (`python -m pip`).
- UCI network error: download the ZIP with browser and use `--dataset`.
- `No module named streamlit`: `python -m pip install -r requirements.txt`.
- CSV missing target: upload saved `test_data.csv`, not `train_data.csv` or a blank template.
- AUC error: your uploaded test data must include all seven classes.
- Model missing on cloud: confirm `model/*.joblib` are tracked in GitHub.
- `git push` permission issue: ensure you're pushing to a repo you own and are logged in.
- Blank cloud page: review Streamlit application logs for stack trace.

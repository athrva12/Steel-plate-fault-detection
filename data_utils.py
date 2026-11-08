"""Dataset loading and evaluation helpers for UCI Steel Plates Faults"""
from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             classification_report, confusion_matrix)

DATASET_ID = 198
FEATURE_NAMES = [
    'X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas',
    'X_Perimeter','Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity',
    'Maximum_of_Luminosity','Length_of_Conveyer','TypeOfSteel_A300',
    'TypeOfSteel_A400','Steel_Plate_Thickness','Edges_Index','Empty_Index',
    'Square_Index','Outside_X_Index','Edges_X_Index','Edges_Y_Index',
    'Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
    'Orientation_Index','Luminosity_Index','SigmoidOfAreas'
]
CLASS_NAMES = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
TARGET = 'fault_type'

def _from_uci():
    from ucimlrepo import fetch_ucirepo
    result = fetch_ucirepo(id=DATASET_ID)
    x, flags = result.data.features.copy(), result.data.targets.copy()
    x = x.loc[:, FEATURE_NAMES]
    flags = flags.rename(columns={"K_Scratch": "K_Scatch"})
    flags = flags.loc[:, CLASS_NAMES]
    return x, flags

def _from_local(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        if TARGET in df.columns:
            return df.loc[:,FEATURE_NAMES].copy(), pd.get_dummies(df[TARGET]).reindex(columns=CLASS_NAMES,fill_value=0).astype(int)
        if all(c in df.columns for c in CLASS_NAMES):
            return df.loc[:,FEATURE_NAMES].copy(), df.loc[:,CLASS_NAMES].copy()
        raise ValueError('Local CSV must have fault_type or the seven one-hot target columns.')
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            matches = [n for n in z.namelist() if n.lower().endswith('.nna')]
            if not matches: raise ValueError('No .NNA data file found inside the UCI ZIP.')
            raw = z.read(matches[0])
        frame = pd.read_csv(io.BytesIO(raw), sep=r'\s+', header=None)
    elif path.suffix.lower() == '.nna':
        frame = pd.read_csv(path, sep=r'\s+', header=None)
    else:
        raise ValueError('Accepted local files: UCI .zip, Faults.NNA, or preprocessed .csv')
    if frame.shape[1] != 34:
        raise ValueError(f'Expected 34 numeric columns, got {frame.shape[1]}.')
    frame.columns = FEATURE_NAMES + CLASS_NAMES
    return frame[FEATURE_NAMES].copy(), frame[CLASS_NAMES].copy()

def load_dataset(local_file=None):
    if local_file:
        x, flags = _from_local(local_file)
    else:
        try:
            x, flags = _from_uci()
        except Exception as exc:
            raise RuntimeError(
                'Could not download dataset from UCI. Connect internet and rerun, '
                'or download the official UCI ZIP and execute: '
                'python train_models.py --dataset path/to/steel+plates+faults.zip'
            ) from exc
    if len(x) < 500 or len(FEATURE_NAMES) < 12:
        raise ValueError('Dataset does not meet assignment minimum size.')
    if x.isna().any().any() or flags.isna().any().any():
        raise ValueError('Missing values found; inspect the dataset before training.')
    x = x.apply(pd.to_numeric, errors='raise')
    flags = flags.apply(pd.to_numeric, errors='raise')
    if not flags.isin([0,1]).all().all() or not flags.sum(axis=1).eq(1).all():
        raise ValueError('Exactly one of the seven target flags must be 1 on every row.')
    labels = flags.idxmax(axis=1).rename(TARGET)
    return x.reset_index(drop=True), labels.reset_index(drop=True)

def validate_uploaded_test(df, required_features=FEATURE_NAMES):
    needed = set(required_features) | {TARGET}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    x = df.loc[:, required_features].copy()
    for col in required_features:
        x[col] = pd.to_numeric(x[col], errors='raise')
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Input features contain NaN or infinite values.')
    y = df[TARGET].astype(str)
    invalid = sorted(set(y) - set(CLASS_NAMES))
    if invalid:
        raise ValueError('Unknown fault labels: '+', '.join(invalid))
    return x, y

def evaluate(model, x, y):
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)
    # The model's classes_ can have a different order than CLASS_NAMES.
    classes = list(model.classes_)
    if set(classes) != set(CLASS_NAMES):
        raise ValueError('The fitted model is missing one or more classes.')
    auc_classes = sorted(CLASS_NAMES)
    ordered_proba = probabilities[:, [classes.index(label) for label in auc_classes]]
    return {
        'Accuracy': accuracy_score(y,predicted),
        'AUC': roc_auc_score(y, ordered_proba, multi_class='ovr', average='macro', labels=auc_classes),
        'Precision': precision_score(y,predicted,labels=CLASS_NAMES,average='macro',zero_division=0),
        'Recall': recall_score(y,predicted,labels=CLASS_NAMES,average='macro',zero_division=0),
        'F1': f1_score(y,predicted,labels=CLASS_NAMES,average='macro',zero_division=0),
        'MCC': matthews_corrcoef(y,predicted),
    }

def report_and_matrix(model, x, y):
    pred = model.predict(x)
    return (classification_report(y,pred,labels=CLASS_NAMES,zero_division=0),
            confusion_matrix(y,pred,labels=CLASS_NAMES))

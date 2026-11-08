"""Project verification after training."""
from pathlib import Path
import sys
import pandas as pd
import joblib
from data_utils import FEATURE_NAMES, CLASS_NAMES, validate_uploaded_test, evaluate
from train_models import MODELS
root=Path(__file__).resolve().parent
for path in ['test_data.csv','model_metrics.csv','model/metadata.json']:
    assert (root/path).exists(), f'Missing {path}; run python train_models.py'
test=pd.read_csv(root/'test_data.csv')
x,y=validate_uploaded_test(test)
assert len(x)>=100 and x.shape[1]==27 and set(y)==set(CLASS_NAMES)
for name,(slug,_) in MODELS.items():
    filename=root/'model'/f'{slug}.joblib'
    assert filename.exists(),f'Missing: {filename}'
    model=joblib.load(filename)
    score=evaluate(model,x,y)
    assert all(0<=v<=1 for v in score.values())
    print(f'{name:25} Accuracy={score["Accuracy"]:.4f} F1={score["F1"]:.4f}')
print('Five models, test CSV, schema and metrics verified')

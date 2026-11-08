"""Training the five mandatory classifiers """
import argparse
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from data_utils import load_dataset, evaluate, FEATURE_NAMES, CLASS_NAMES, TARGET

ROOT = Path(__file__).resolve().parent

MODELS = {
    'Logistic Regression': (
        'logistic_regression',
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.5,
                max_iter=2000,
                random_state=42
            ))
        ])
    ),

    'Decision Tree': (
        'decision_tree',
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
    ),

    'KNN': (
        'knn',
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ))
        ])
    ),

    'Naive Bayes': (
        'naive_bayes',
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB(
                var_smoothing=1e-8
            ))
        ])
    ),

    'Support Vector Machine': (
        'svm',
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=2.0,
                gamma='scale',
                probability=True,
                random_state=42
            ))
        ])
    )
}


def main(dataset=None):
    print('Loading UCI Steel Plates Faults dataset...')
    x,y = load_dataset(dataset)
    print(f'Dataset: {len(x)} rows, {x.shape[1]} features, {y.nunique()} classes')
    print('Class counts:\n'+y.value_counts().to_string())
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.20,random_state=42,stratify=y)
    
    train=pd.concat([x_train.reset_index(drop=True),y_train.reset_index(drop=True)],axis=1)
    test=pd.concat([x_test.reset_index(drop=True),y_test.reset_index(drop=True)],axis=1)
    train.to_csv(ROOT/'train_data.csv',index=False)
    test.to_csv(ROOT/'test_data.csv',index=False)
    (ROOT/'model').mkdir(exist_ok=True)
    summary=[]
    for name,(slug,estimator) in MODELS.items():
        print(f'Fitting {name}...')
        estimator.fit(x_train,y_train)
        scores=evaluate(estimator,x_test,y_test)
        summary.append({'ML Model Name':name,**scores})
        joblib.dump(estimator,ROOT/'model'/f'{slug}.joblib',compress=3)
    pd.DataFrame(summary).to_csv(ROOT/'model_metrics.csv',index=False)
    with open(ROOT/'model'/'metadata.json','w',encoding='utf-8') as f:
        json.dump({'features':FEATURE_NAMES,'classes':CLASS_NAMES,'target':TARGET,
                   'dataset':'UCI Steel Plates Faults (ID 198)',
                   'split':'80/20 stratified, random_state=42',
                   'evaluation':'macro OVR AUC; macro Precision, Recall, F1; multiclass MCC'},f,indent=2)
    print('\nSAVED MODEL METRICS (held-out test data):')
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print('\nCreated train_data.csv, test_data.csv, model_metrics.csv, model/*.joblib')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='Optional local UCI ZIP, Faults.NNA, or preprocessed CSV')
    args=parser.parse_args()
    main(args.dataset)

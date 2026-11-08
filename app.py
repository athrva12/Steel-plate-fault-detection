"""SteelScope | Interactive evaluation of saved multiclass classification models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from data_utils import (FEATURE_NAMES, CLASS_NAMES, TARGET, validate_uploaded_test,
                        evaluate, report_and_matrix)

ROOT=Path(__file__).resolve().parent
SAVED_MODELS={
    'Logistic Regression':'logistic_regression.joblib',
    'Decision Tree':'decision_tree.joblib',
    'KNN':'knn.joblib',
    'Naive Bayes':'naive_bayes.joblib',
    'Support Vector Machine':'svm.joblib',
}
st.set_page_config(page_title='SteelScope | Fault Classification',page_icon='🏭',layout='wide')
st.markdown('''<style>
    .block-container{padding-top:1.8rem;max-width:1200px}
    h1{font-weight:800;letter-spacing:-0.04em}
    div[data-testid="stMetric"]{border:1px solid #b7c2d099;border-radius:12px;padding:14px}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    filename=ROOT/'model'/SAVED_MODELS[model_name]
    if not filename.exists():
        raise FileNotFoundError(f'{filename.name} missing. Run python train_models.py first.')
    return joblib.load(filename)

st.title('🏭 SteelScope')
st.caption('Industrial steel plate fault classification · UCI ID 198 · 7 fault classes · 27 numerical features')
st.info('Educational demonstration: evaluates pre-trained models on a labeled test CSV. No model training occurs on uploaded data.')

with st.sidebar:
    st.header('Experiment controls')
    model_name=st.selectbox('Choose classifier',list(SAVED_MODELS))
    uploaded=st.file_uploader('Upload labeled test CSV',type=['csv'],help='Must include 27 feature columns plus fault_type.')
    st.markdown('**Expected target column:** `fault_type`')
    st.markdown('**Allowed labels:** '+', '.join(CLASS_NAMES))
    st.download_button('Download test CSV template',data=','.join(FEATURE_NAMES+[TARGET])+'\n',
                       file_name='steel_test_template.csv',mime='text/csv')
    if (ROOT/'test_data.csv').exists():
        st.download_button('Download packaged test data',
            data=(ROOT/'test_data.csv').read_bytes(),file_name='test_data.csv',mime='text/csv')

if uploaded is None:
    st.subheader('1. Start with test data')
    st.write('Upload the labeled `test_data.csv` created by `train_models.py` using the left sidebar.')
    st.write('This app loads saved models, computes six metrics and displays the multiclass confusion matrix and classification report.')
    if (ROOT/'model_metrics.csv').exists():
        st.subheader('Reference model comparison (saved training run)')
        st.dataframe(pd.read_csv(ROOT/'model_metrics.csv').round(4),hide_index=True,use_container_width=True)
    st.stop()

try:
    df=pd.read_csv(uploaded)
    if len(df)<7:
        raise ValueError('Test CSV needs at least seven rows and representation of all seven classes for AUC.')
    X,y=validate_uploaded_test(df)
    counts=y.value_counts()
    if set(counts.index)!=set(CLASS_NAMES):
        raise ValueError('AUC for all seven classes requires at least one test observation in every class.')
    model=load_model(model_name)
    scores=evaluate(model,X,y)
    report, matrix=report_and_matrix(model,X,y)
except Exception as exc:
    st.error(f'Unable to evaluate this CSV: {exc}')
    st.stop()

st.subheader(f'2. Results · {model_name}')
cols=st.columns(3)
for i,(key,val) in enumerate(scores.items()):
    cols[i%3].metric(key,f'{val:.4f}')
st.caption('AUC: macro one-vs-rest; Precision/Recall/F1: macro average; MCC: multiclass.')
st.divider()
lcol,rcol=st.columns([1.25,1])
with lcol:
    st.subheader('Confusion matrix')
    fig,ax=plt.subplots(figsize=(8.4,6.5))
    mesh=ax.imshow(matrix,cmap='Blues')
    for (i,j),value in np.ndenumerate(matrix):
        ax.text(j,i,str(value),ha='center',va='center',fontsize=9,
                color='white' if value > matrix.max()/2 else '#172437')
    ax.set_xticks(range(7),CLASS_NAMES,rotation=45,ha='right',fontsize=9)
    ax.set_yticks(range(7),CLASS_NAMES,fontsize=9)
    ax.set_xlabel('Predicted fault')
    ax.set_ylabel('Actual fault')
    fig.colorbar(mesh,ax=ax,fraction=.045,pad=.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with rcol:
    st.subheader('Class distribution')
    st.bar_chart(y.value_counts().reindex(CLASS_NAMES,fill_value=0))
    st.subheader('Classification report')
    st.code(report,language='text')

with st.expander('View first 10 uploaded examples'):
    st.dataframe(df.head(10),hide_index=True,use_container_width=True)
with st.expander('Compare all saved models on THIS uploaded CSV'):
    rows=[]
    for name in SAVED_MODELS:
        try:
            rows.append({'ML Model Name':name,**evaluate(load_model(name),X,y)})
        except Exception as exc:
            st.warning(f'{name}: {exc}')
    if rows:
        comparative=pd.DataFrame(rows)
        st.dataframe(comparative.round(4),hide_index=True,use_container_width=True)
        st.download_button('Download comparison results',comparative.to_csv(index=False),
                           'uploaded_data_metrics.csv','text/csv')
st.caption('Source: Buscema, Terzi & Tastle (2010), UCI Steel Plates Faults (CC BY 4.0).')

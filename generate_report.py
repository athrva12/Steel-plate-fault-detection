"""Generate a draft submission DOCX with REAL metrics from the saved training run."""
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT=Path(__file__).resolve().parent
METRICS=ROOT/'model_metrics.csv'

def make_report(path=None):
    if not METRICS.exists():
        raise FileNotFoundError('Run python train_models.py first to calculate REAL evaluation metrics.')
    metrics=pd.read_csv(METRICS)
    doc=Document()
    sec=doc.sections[0]
    sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7)
    sec.left_margin=Inches(.75);sec.right_margin=Inches(.75)
    styles=doc.styles
    styles['Normal'].font.name='Aptos';styles['Normal'].font.size=Pt(10)
    styles['Heading 1'].font.color.rgb=RGBColor(20,61,94)
    doc.add_heading('SteelScope | Steel Plate Fault Detection',0)
    doc.add_paragraph('Machine Learning - Assignment 2 | M.Tech (AIML)',style='Subtitle')
    doc.add_paragraph('Student name: [YOUR NAME]    BITS ID: [YOUR ID]    Section: [YOUR SECTION]')
    doc.add_heading('1. Mandatory submission links',1)
    doc.add_paragraph('GitHub repository: [INSERT WORKING GITHUB URL]')
    doc.add_paragraph('Live Streamlit app: [INSERT WORKING STREAMLIT URL]')
    doc.add_heading('2. Problem statement',1)
    doc.add_paragraph('Develop and compare five supervised classification models that assign a recorded steel plate surface fault to one of seven fault types using numerical descriptors, then expose saved models in an interactive Streamlit application.')
    doc.add_heading('3. Dataset description',1)
    doc.add_paragraph('Source: UCI Steel Plates Faults (Dataset ID 198), donated in 2010 by M. Buscema, S. Terzi and W. Tastle. Source URL: https://archive.ics.uci.edu/dataset/198/steel+plates+faults')
    doc.add_paragraph('1,941 observations; 27 numerical features; seven fault classes (Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps and Other_Faults). The original dataset has seven one-hot target columns, converted to a single fault_type label. Class frequencies are unequal.')
    doc.add_heading('4. Modeling procedure',1)
    doc.add_paragraph('A stratified 80%/20% train/test split (random_state=42) was used. StandardScaler is fitted only on the training split inside pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM; Decision Tree uses unscaled input. All five models train on the same training subset. The test subset is held out and used only for evaluation.')
    doc.add_paragraph('Accuracy, macro one-vs-rest AUC, macro Precision, macro Recall, macro F1 and multiclass MCC are reported. AUC is based on predicted probabilities, not hard labels.')
    doc.add_heading('5. Model comparison',1)
    columns=['ML Model Name','Accuracy','AUC','Precision','Recall','F1','MCC']
    table=doc.add_table(rows=1,cols=len(columns));table.style='Light Shading Accent 1'
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    for cell,col in zip(table.rows[0].cells,columns):cell.text=col
    for _,rec in metrics.iterrows():
        for cell,col in zip(table.add_row().cells,columns):
            cell.text=rec[col] if col=='ML Model Name' else f'{rec[col]:.4f}'
    doc.add_paragraph('All metrics above are generated from the saved held-out test split by train_models.py.')
    doc.add_heading('6. Observations and overall winner',1)
    for _,rec in metrics.iterrows():
        doc.add_paragraph(f'{rec["ML Model Name"]}: accuracy {rec["Accuracy"]:.3f}, macro F1 {rec["F1"]:.3f}, MCC {rec["MCC"]:.3f}. [ADD YOUR OWN INTERPRETATION OF CLASS ERRORS AND TRADEOFFS]')
    winner=metrics.sort_values(['F1','MCC'],ascending=False).iloc[0]['ML Model Name']
    doc.add_paragraph(f'Highest macro-F1 model on this split: {winner}. [EXPLAIN WHETHER THIS IS THE BEST CHOICE BASED ON YOUR PRIORITY METRIC AND CLASS-WISE ERRORS]')
    doc.add_heading('7. Streamlit features',1)
    doc.add_paragraph('CSV upload for held-out labeled test records; saved-model selection dropdown; six evaluation metrics; confusion matrix; classification report; class-distribution chart; comparison across all five pre-trained models. Uploaded records are evaluated without retraining.')
    doc.add_heading('8. BITS Virtual Lab execution screenshot',1)
    doc.add_paragraph('[INSERT A GENUINE SCREENSHOT OF YOUR OWN BITS VIRTUAL LAB RUN HERE. INCLUDE TERMINAL OR APP EXECUTION.]')
    doc.add_heading('9. Reflection and integrity',1)
    doc.add_paragraph('[DESCRIBE WHAT YOU MODIFIED, WHAT YOU LEARNED, YOUR TESTING EXPERIENCE, AND ANY LIMITATIONS. VERIFY ALL LINKS BEFORE SUBMISSION.]')
    doc.add_heading('10. Dataset citation',1)
    doc.add_paragraph('Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N (CC BY 4.0).')
    doc.add_page_break()
    doc.add_heading('Appendix A: README.md (complete content)',1)
    doc.add_paragraph('The assignment requires the complete README content in the final PDF. The text below is inserted from README.md at report-generation time. Check formatting before exporting.')
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    for line in readme.splitlines():
        line=line.strip()
        if line.startswith('# '): doc.add_heading(line[2:],level=1)
        elif line.startswith('## '): doc.add_heading(line[3:],level=2)
        elif line.startswith('### '): doc.add_heading(line[4:],level=3)
        elif line: doc.add_paragraph(line)
    path=Path(path or ROOT/'SteelScope_Submission_DRAFT.docx')
    doc.save(path)
    print(f'Draft report written: {path}')
    return path

if __name__=='__main__':
    make_report()

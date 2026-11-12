# Child malnutrition: analysis code

Code for the article "Weight, height and WHO Z-scores as predictors of MUAC-defined malnutrition in children aged 6 to 60 months in Noakhali and Lakshmipur, Bangladesh, after correcting for outcome-related data leakage: A cross-sectional study" (Maternal & Child Nutrition, under review).

## Contents

- `Malnutrition_Full_Analysis_Revised.ipynb`: the full analysis, already executed. It reproduces every computed table and figure in the article (Tables 1 to 7, Figures 1 to 5) and the Supporting Information (Tables S1 and S4 to S10, Figures S1 and S5), together with the sensitivity analyses reported in the text.
- `audit_flow.py`: the data-integrity audit (Section 3.1 and Table S11). It compares the analysis dataset with an earlier version that is not distributed, so it runs only where those files are available.
- `requirements.txt`: the exact package versions used.

Tables S2 and S3 are reporting checklists and are not computed.

## Where each analysis is in the notebook

| Notebook section | Output |
|---|---|
| 2 | Recalculation of the WHO Z-scores; birth-year corrections |
| 3 to 5 | Tables 1 to 5; events per predictor |
| 6 | Table 6, Table S1, Figure 1 (repeated cross-validation, thresholds chosen by inner cross-validation) |
| 7 | Figures 2 and 3, Table 7 (held-out evaluation, McNemar tests) |
| 8 | Figure 4, permutation importance, Figure 5, Figure S1 |
| 9 to 11 | Tables S4 to S7; optimism-corrected AUC |
| 12a, 12b | Table 6 with intervals for every metric; exact confidence intervals on the held-out set |
| 12c | Table S8 (correlations, variance inflation factors, coefficient sign stability) |
| 12d | Incremental value of each predictor set |
| 12e | Table S9 (sensitivity to penalty strength) |
| 12f | Calibration and Figure S5 |
| 12g | Table S10 (McNemar discordant pairs by outcome class) |
| 12h | Sensitivity analyses and recording checks |

## Data

The participant-level dataset is not publicly available. It may be requested from the corresponding author, subject to approval by the Noakhali Science and Technology University Ethics Committee. The notebook displays summary results only and contains no individual records.

## Running the notebook

1. Use Python 3.12.3.
2. Install the packages: `pip install -r requirements.txt`
3. Install pygrowup: `pip install --no-build-isolation pygrowup==0.8.2`. It needs `six` and `setuptools`, which the previous step installs; without them its installation fails on current Python versions.
4. Place the dataset at `data/Malnutrition_Analysis_Dataset_FINAL.csv`.
5. Run all cells. A full run takes about 45 minutes on a standard laptop. The random seed is fixed at 42.

## Running the audit script

`python audit_flow.py EARLIER.csv EARLIER_WITH_IDS.sav ANALYSIS_VERSION.xlsx`

It needs two further packages, `pyreadstat` and `openpyxl`, in addition to those above.

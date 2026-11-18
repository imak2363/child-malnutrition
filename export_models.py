"""Fit the seven Set C models of the article and save them for the prototype application.

The models, the training set and the classification thresholds are the same as in
Malnutrition_Full_Analysis_Revised.ipynb (Sections 6 and 7; settings in Table S4):

  * outcome: MUAC-defined malnutrition (MUAC below 12.5 cm); MUAC is not a predictor
  * predictors (Set C, in this order): age in months, sex (1 = male), district
    (1 = Noakhali), weight (kg), height (cm), WAZ, HAZ, WHZ
  * training set: the 372 children of the stratified 80/20 split (random seed 42)
  * threshold: Youden-optimal, chosen from out-of-fold predictions of an inner five-fold
    cross-validation within the training set

Usage:
    python export_models.py                       # reads data/Malnutrition_Analysis_Dataset_FINAL.csv
    python export_models.py DATA.csv OUTPUT_DIR   # explicit paths

Output: one .pkl file per model, and thresholds.json with each model's threshold.
The probabilities returned by models fitted with balanced class weights overestimate
absolute risk (Section 3.5 of the article); classify with the stored threshold, not 0.5.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
FEATURES = ['month', 'sex_male', 'area_noakhali', 'weight', 'height', 'waz', 'haz', 'whz']


def make_lr():
    return Pipeline([('s', StandardScaler()),
                     ('m', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))])


def make_dt():
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight='balanced', random_state=RANDOM_STATE)


def make_svm():
    return Pipeline([('s', StandardScaler()),
                     ('m', SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE))])


def make_mlp():
    return Pipeline([('s', StandardScaler()),
                     ('m', MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000, random_state=RANDOM_STATE))])


def make_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_depth=6, class_weight='balanced',
                                  random_state=RANDOM_STATE)


def make_gb():
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def make_stack():
    return StackingClassifier(
        estimators=[('lr', make_lr()), ('dt', make_dt()), ('rf', make_rf())],
        final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
        cv=5, n_jobs=-1)


# key used by the application -> (file name, constructor)
MODELS = {
    'logistic_regression': ('logistic_regression_set_c.pkl', make_lr),
    'decision_tree': ('decision_tree_set_c.pkl', make_dt),
    'svm_rbf': ('svm_rbf_set_c.pkl', make_svm),
    'mlp': ('mlp_set_c.pkl', make_mlp),
    'random_forest': ('random_forest_set_c.pkl', make_rf),
    'gradient_boosting': ('gradient_boosting_set_c.pkl', make_gb),
    'stacking_ensemble': ('stacking_ensemble_set_c.pkl', make_stack),
}


def inner_threshold(make_model, X_train, y_train):
    """Youden-optimal threshold from out-of-fold predictions of an inner five-fold
    cross-validation within the training data."""
    p = cross_val_predict(clone(make_model()), X_train, y_train,
                          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                          method='predict_proba')[:, 1]
    fpr, tpr, thr = roc_curve(y_train, p)
    return float(thr[np.argmax(tpr - fpr)])


def main(data_path='data/Malnutrition_Analysis_Dataset_FINAL.csv', out_dir='models'):
    df = pd.read_csv(data_path)
    X = np.column_stack([df.MONTH, (df.SEX == 'male').astype(int), (df.Area == 'Noakhali').astype(int),
                         df.WEIGHT, df.HEIGHT, df.WAZ, df.HAZ, df.WHZ]).astype(float)
    y = df.muac_malnourished.values
    train, _ = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # Two check cases with made-up round inputs, to confirm that a deployment loads the same models.
    checks = np.array([[44.06, 1, 1, 16, 98, 0.155, -0.759, 0.933],
                       [24.0, 1, 1, 7.5, 85.0, -3.5, -1.5, -3.8]])
    meta = {'features': FEATURES, 'outcome': 'MUAC below 12.5 cm', 'training_children': int(len(train)),
            'training_malnourished': int(y[train].sum()), 'random_state': RANDOM_STATE,
            'scikit_learn': sklearn.__version__, 'models': {}}
    print(f"Training set: {len(train)} children, {int(y[train].sum())} malnourished\n")
    print(f"{'model':<22}{'threshold':>10}{'check 1':>10}{'check 2':>10}")
    for key, (fname, make_model) in MODELS.items():
        model = make_model().fit(X[train], y[train])
        thr = inner_threshold(make_model, X[train], y[train])
        joblib.dump(model, out / fname)
        p = model.predict_proba(checks)[:, 1]
        meta['models'][key] = {'file': fname, 'threshold': round(thr, 6),
                               'check_probabilities': [round(float(v), 4) for v in p]}
        print(f"{key:<22}{thr:>10.3f}{p[0]:>10.3f}{p[1]:>10.3f}")
    (out / 'thresholds.json').write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(MODELS)} model files and thresholds.json to {out}/")


if __name__ == '__main__':
    main(*sys.argv[1:3])

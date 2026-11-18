# Serialized Machine Learning Models (Predictor Set C)

This directory hosts the serialized scikit-learn models (`.pkl` files) and their decision thresholds (`thresholds.json`) utilized by the FastAPI inference service.

---

## 🔗 Upstream Source & Training Repository

The full analysis pipeline, experimental validation, and model training code are maintained in the primary analysis repository:
* **Analysis Repository**: [https://github.com/imak2363/child-malnutrition](https://github.com/imak2363/child-malnutrition)
* **Interactive Analysis Notebook**: `Malnutrition_Full_Analysis_Revised.ipynb` (contains the complete 465-child cohort dataset, cross-validation pipeline, and reproductions of Tables 1–7 and Figures 1–5).

---

## 🛠 Model Training Procedure & Export Pipeline

The model files stored in this directory were generated directly using `export_models.py` adhering to the exact specifications in **Table S4 (Classifier settings)** and **Table S5 (Recommended model parameters)** of the manuscript.

### Step-by-Step Generation Workflow:

1. **Cohort Data Loading**:
   * Dataset: 465 children aged 6 to 60 months from Noakhali and Lakshmipur districts, Bangladesh.
   * Target outcome: Separately measured acute malnutrition defined as $\text{MUAC} < 12.5\text{ cm}$ (`muac_malnourished`).
   * Predictors (Set C): 8 parameters (`month`, `sex_male`, `area_noakhali`, `weight`, `height`, `waz`, `haz`, `whz`).

2. **Stratified Split**:
   * Stratified 80/20 train/test split with `random_state=42` ($n=372$ training, $n=93$ held-out test).

3. **Classifier Hyperparameters (Matching Table S4)**:
   * **Logistic Regression**: `StandardScaler()`, `LogisticRegression(max_iter=2000, class_weight='balanced')`
   * **Random Forest**: `RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, class_weight='balanced')`
   * **Decision Tree**: `DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight='balanced')`
   * **Multi-Layer Perceptron (MLP)**: `StandardScaler()`, `MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000, activation='relu')`
   * **Support Vector Machine (SVM RBF)**: `StandardScaler()`, `SVC(probability=True, class_weight='balanced')`
   * **Gradient Boosting**: `GradientBoostingClassifier(random_state=42)`
   * **Stacking Ensemble**: Meta-learner (`LogisticRegression`) combining Logistic Regression, Decision Tree, and Random Forest base models.

4. **Internal Cross-Validation Decision Thresholds**:
   * Each model's decision threshold was determined using Youden-optimal criteria on out-of-fold predictions from an inner 5-fold cross-validation on the training split, ensuring the threshold is selected on children the model was not fitted to.
   * **Logistic Regression classification threshold:** **`0.662`** (not 0.5).

5. **Execution Command**:
   To reproduce and re-export all models and thresholds into this directory, run:
   ```bash
   python export_models.py
   ```

---

## 📋 Deployed Models & Thresholds Summary

| Model Key | Serialized File | Architecture | Inner CV Threshold | Held-Out Test AUC | Held-Out Identified |
|:---|:---|:---|:---:|:---:|:---:|
| `logistic_regression` | `model_set_c_logistic_regression.pkl` | `Pipeline(StandardScaler, LogisticRegression)` | **0.662** | 0.723 | 3 / 7 (42.9%) |
| `decision_tree` | `model_set_c_decision_tree.pkl` | `DecisionTreeClassifier(depth=4)` | **0.783** | 0.526 | 1 / 7 (14.3%) |
| `svm_rbf` | `model_set_c_svm_rbf.pkl` | `Pipeline(StandardScaler, SVC)` | **0.099** | 0.703 | 3 / 7 (42.9%) |
| `mlp` | `model_set_c_mlp.pkl` | `Pipeline(StandardScaler, MLPClassifier)` | **0.131** | 0.689 | 2 / 7 (28.6%) |
| `random_forest` | `model_set_c_random_forest.pkl` | `RandomForestClassifier(300 trees, depth=6)` | **0.576** | 0.646 | 1 / 7 (14.3%) |
| `gradient_boosting` | `model_set_c_gradient_boosting.pkl` | `GradientBoostingClassifier()` | **0.106** | 0.631 | 1 / 7 (14.3%) |
| `stacking_ensemble` | `model_set_c_stacking_ensemble.pkl` | `StackingClassifier(LR + DT + RF)` | **0.599** | 0.729 | 2 / 7 (28.6%) |

> **Intended Use Note**: These models are deployed as a research prototype accompanying a published analysis. They have not been externally validated and must not be used to guide clinical decisions about individual children.

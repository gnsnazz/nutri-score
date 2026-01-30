import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocessor import build_preprocessor
from config import FEATURES, TARGET_COL, PLOTS_DIR

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    f1_score, roc_auc_score, roc_curve, auc, classification_report, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone


def make_cv(random_state = 42):
    return StratifiedKFold(n_splits = 10, shuffle = True, random_state = random_state)


def gridsearch_model(name, estimator, param_grid, x_train, y_train, scoring = "roc_auc", randomized = False, n_iter = 10, random_state = 42):

    pipe = ImbPipeline(steps=[("preprocessor", build_preprocessor()), ('oversampler', RandomOverSampler(random_state = 42)), ("model", estimator)])
    cv = make_cv(random_state)

    if randomized:
        search = RandomizedSearchCV(
            pipe,
            param_grid,
            n_iter = n_iter,
            scoring = scoring,
            cv = cv,
            n_jobs = 2,
            random_state = random_state,
            verbose = 1
        )
    else:
        search = GridSearchCV(
            pipe,
            param_grid,
            scoring = scoring,
            cv = cv,
            n_jobs = 2,
            verbose = 1
        )

    print(f"\n>>> Tuning {name}...")
    search.fit(x_train, y_train)

    print(f"   Best params: {search.best_params_}")
    print(f"   Best CV {scoring}: {search.best_score_:.4f}")

    return search.best_estimator_


def print_cv_report(model, x, y, name):
    print(f"\n  Report {name} (10-Fold CV)")
    # Calcola le metriche su tutti i fold
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = cross_validate(model, x, y, cv = 10, scoring = scoring, n_jobs = 2)

    for metric in scoring:
        key = f"test_{metric}"
        mean_val = results[key].mean()
        std_val = results[key].std()
        print(f"   {metric.capitalize():<10}: {mean_val:.4f} ± {std_val:.4f}")


def train_all_models(csv_path):
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        print(f"[Errore] Colonne mancanti nel CSV: {missing_cols}")
        return {}

    x = df[FEATURES]
    y = df[TARGET_COL]

    print(">> Splitting Dataset 80/20...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, stratify = y, random_state = 42
    )

    models = [
        ("Logistic Regression",
         LogisticRegression(max_iter=2000), {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}),
        (
            "Decision Tree",
            DecisionTreeClassifier(), {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 5]}),
        (
            "Random Forest",
            RandomForestClassifier(),
            {"model__n_estimators": [50, 100], "model__max_depth": [10, 20, None], "model__min_samples_leaf": [1, 4]}),
        (
            "KNN",
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]}),
        (
            "MLP",
            MLPClassifier(max_iter=500),
            {"model__hidden_layer_sizes": [(64,), (64, 32)], "model__alpha": [1e-4, 1e-3], "model__learning_rate_init": [1e-3]})
    ]

    trained_models = {}

    for name, est, grid in models:
        best_model = gridsearch_model(
            name, est, grid, x_train, y_train,
            randomized = (name == "MLP"), n_iter = 4
        )

        print_cv_report(best_model, x_train, y_train, name)

        # Grafico ROC
        safe_name = name.replace(" ", "_").lower()
        plot_path = f"{PLOTS_DIR}/roc_{safe_name}.png"
        plot_mean_roc_cv(best_model, x_train, y_train, f"{name} - ROC (CV)", plot_path)

        # Valutazione
        best_metrics = tune_threshold(best_model, x_train, y_train, label = name)
        best_thr = best_metrics["thr"]

        evaluate_model(best_model, x_test, y_test, name, threshold = best_thr)

        trained_models[name] = best_model

    return trained_models


def evaluate_model(model, x_test, y_test, name, threshold = 0.5):
    y_pred = predict_with_threshold(model, x_test, threshold)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = model.decision_function(x_test)

    print(f"\n- REPORT FINALE: {name} (Test Set | Thr={threshold:.2f}) -")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    try:
        print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    except:
        print("ROC AUC  : N/A")

    print("\n" + classification_report(y_test, y_pred))


def tune_threshold(model, x, y, label = "Model", thresholds = None, print_table = True):
    if not hasattr(model, "predict_proba"):
        return {"thr": 0.5}

    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    y_proba = model.predict_proba(x)[:, 1]

    if print_table:
        print(f"\n=== Tuning Soglia — {label} ===")
        print("thr   prec_1  rec_1   f1_1    FP    FN")
        print("----------------------------------------")

    best = {
        "thr": 0.5, "prec_1": 0.0, "rec_1": 0.0, "f1_1": 0.0, "fp": 0, "fn": 0
    }

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y, y_pred, average = None, labels = [0, 1], zero_division = 0
        )

        p1 = float(prec[1]) if len(prec) > 1 else 0.0
        r1 = float(rec[1]) if len(rec) > 1 else 0.0
        f1_1 = float(f1[1]) if len(f1) > 1 else 0.0

        cm = confusion_matrix(y, y_pred, labels = [0, 1])
        fp = int(cm[0, 1])  # Falsi Positivi
        fn = int(cm[1, 0])  # Falsi Negativi

        if print_table:
            print(f"{thr:0.2f}  {p1:0.4f}  {r1:0.4f}  {f1_1:0.4f}  {fp:4d}  {fn:3d}")

        if f1_1 > best["f1_1"]:
            best = {
                "thr": round(float(thr), 2),
                "prec_1": p1, "rec_1": r1, "f1_1": f1_1,
                "fp": fp, "fn": fn
            }

    print(
        f">>> Miglior soglia (max F1): {best['thr']:.2f} "
        f"| F1={best['f1_1']:.4f} | FP={best['fp']} FN={best['fn']}"
    )

    return best


def predict_with_threshold(model, x, threshold = 0.5):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[:, 1]
        return (probs >= threshold).astype(int)
    else:
        return model.predict(x)

def plot_mean_roc_cv(estimator, x, y, title, out_path, n_splits = 10, random_state = 42):
    """
    Genera il grafico ROC con Cross-Validation e lo salva su file.
    """
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    fig, ax = plt.subplots(figsize = (8, 6))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(x, y), start = 1):
        x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        est = clone(estimator)
        est.fit(x_tr, y_tr)

        if hasattr(est, "predict_proba"):
            y_score = est.predict_proba(x_va)[:, 1]
        else:
            y_score = est.decision_function(x_va)

        fpr, tpr, _ = roc_curve(y_va, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        ax.plot(fpr, tpr, alpha = 0.1, color = 'blue')

    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis = 0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.plot(mean_fpr, mean_tpr, color = 'b', linewidth = 2, label = f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color = 'grey', alpha = 0.2, label = "± 1 std. dev.")

    ax.plot([0, 1], [0, 1], linestyle = "--", linewidth = 1, color = 'r', label = 'Chance')

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc = "lower right")

    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"   >>> Salvataggio grafico in: {out_path}")
    fig.savefig(out_path, dpi = 200)
    plt.close(fig)

    return mean_auc, std_auc

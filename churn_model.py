"""Customer churn prediction: Logistic Regression vs Random Forest vs Gradient Boosting.

Usage: python churn_model.py path/to/Churn_Modelling.csv [output_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]  # identifiers, no predictive meaning
CATEGORICAL = ["Geography", "Gender"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary", "ZeroBalance", "BalanceToSalary"]


def load(path):
    df = pd.read_csv(path).drop(columns=ID_COLS)
    # Simple engineered features: ~36% of customers hold a zero balance, a distinct segment.
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["BalanceToSalary"] = df["Balance"] / df["EstimatedSalary"].clip(lower=1)
    return df.drop(columns=TARGET), df[TARGET]


def build_models():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL),
    ])
    return {
        "Logistic Regression": Pipeline([("pre", pre), ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED))]),
        "Random Forest": Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=SEED))]),
    }


def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def main(data_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = load(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, fitted, probs = {}, {}, {}
    for name, model in build_models().items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        # Pick the decision threshold on out-of-fold train predictions, never on the test set.
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_tr, oof)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        r = {"cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std(),
             "test_roc_auc": roc_auc_score(y_te, p), "test_pr_auc": average_precision_score(y_te, p)}
        for label, t in [("@0.5", 0.5), ("@tuned", thr)]:
            pred = (p >= t).astype(int)
            r[f"precision{label}"] = precision_score(y_te, pred)
            r[f"recall{label}"] = recall_score(y_te, pred)
            r[f"f1{label}"] = f1_score(y_te, pred)
        r["tuned_threshold"] = thr
        results[name], fitted[name], probs[name] = r, model, p
        print(f"\n=== {name} (tuned threshold {thr:.2f}) ===")
        print(classification_report(y_te, (p >= thr).astype(int), digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_te, (p >= thr).astype(int)))

    table = pd.DataFrame(results).T.round(3)
    print("\n", table.to_string())
    table.to_csv(out / "model_comparison.csv")

    # Feature importance: permutation importance on the test set (model-agnostic, comparable).
    feat_names = NUMERIC + CATEGORICAL
    imps = {}
    for name, model in fitted.items():
        pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                    random_state=SEED, n_jobs=-1)
        imps[name] = pd.Series(pi.importances_mean, index=X_te.columns)
    imp = pd.DataFrame(imps).loc[[c for c in X_te.columns if c in feat_names]]
    imp = imp.sort_values(max(results, key=lambda n: results[n]["test_roc_auc"]), ascending=False)
    imp.round(4).to_csv(out / "feature_importance.csv")
    print("\nPermutation importance (drop in test ROC-AUC):\n", imp.round(4).to_string())

    # Logistic regression coefficients give direction of effect.
    lr = fitted["Logistic Regression"]
    coef = pd.Series(lr.named_steps["clf"].coef_[0],
                     index=lr.named_steps["pre"].get_feature_names_out()).sort_values()
    coef.round(3).to_csv(out / "logreg_coefficients.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['test_roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    axes[0].legend(loc="lower right")
    best = max(results, key=lambda n: results[n]["test_roc_auc"])
    imp[best].sort_values().plot.barh(ax=axes[1], color="#3b6fb6")
    axes[1].set(title=f"Permutation importance – {best}", xlabel="Drop in ROC-AUC")
    fig.tight_layout()
    fig.savefig(out / "churn_results.png", dpi=130)

    with open(out / "metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best,
                   "test_churn_rate": float(y_te.mean()), "n_train": len(X_tr), "n_test": len(X_te)},
                  f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "output")

"""Export model parameters, ROC curves, feature distributions, and sample test data
to JSON for the interactive live demo.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL = ["Geography", "Gender"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary", "ZeroBalance", "BalanceToSalary"]

def main():
    df_raw = pd.read_csv("data/Churn_Modelling.csv")
    df = df_raw.drop(columns=ID_COLS).copy()
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["BalanceToSalary"] = df["Balance"] / df["EstimatedSalary"].clip(lower=1)
    
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    
    # Preprocessor
    num_scaler = StandardScaler()
    cat_encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    
    pre = ColumnTransformer([
        ("num", num_scaler, NUMERIC),
        ("cat", cat_encoder, CATEGORICAL),
    ])
    
    X_tr_trans = pre.fit_transform(X_tr)
    X_te_trans = pre.transform(X_te)
    
    # Feature names
    feature_names = list(pre.get_feature_names_out())
    
    # Train Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr.fit(X_tr_trans, y_tr)
    p_lr_te = lr.predict_proba(X_te_trans)[:, 1]
    
    # Train Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3,
                                    subsample=0.8, random_state=SEED)
    gb.fit(X_tr_trans, y_tr)
    p_gb_te = gb.predict_proba(X_te_trans)[:, 1]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf.fit(X_tr_trans, y_tr)
    p_rf_te = rf.predict_proba(X_te_trans)[:, 1]
    
    # Extract ROC curves
    roc_data = {}
    for name, p in [("Gradient Boosting", p_gb_te), ("Random Forest", p_rf_te), ("Logistic Regression", p_lr_te)]:
        fpr, tpr, thresholds = roc_curve(y_te, p)
        # Sample down to ~50 points for snappy rendering
        idx = np.linspace(0, len(fpr) - 1, 50).astype(int)
        roc_data[name] = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx]
        
    # Extract Threshold metrics across 0.05 to 0.95
    threshold_metrics = {}
    for name, p in [("Gradient Boosting", p_gb_te), ("Random Forest", p_rf_te), ("Logistic Regression", p_lr_te)]:
        t_list = np.linspace(0.05, 0.95, 37)
        t_data = []
        for t in t_list:
            pred = (p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            acc = float((tp + tn) / len(y_te))
            t_data.append({
                "threshold": round(float(t), 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            })
        threshold_metrics[name] = t_data

    # Extract Preprocessor scaler stats
    scaler_stats = {
        "numeric_features": NUMERIC,
        "mean": {col: float(m) for col, m in zip(NUMERIC, pre.named_transformers_["num"].mean_)},
        "scale": {col: float(s) for col, s in zip(NUMERIC, pre.named_transformers_["num"].scale_)},
        "categorical_features": CATEGORICAL,
        "cat_categories": {
            "Geography": list(pre.named_transformers_["cat"].categories_[0]),
            "Gender": list(pre.named_transformers_["cat"].categories_[1])
        }
    }
    
    # Logistic regression coefficients
    lr_weights = {
        "intercept": float(lr.intercept_[0]),
        "coefficients": {name: float(coef) for name, coef in zip(feature_names, lr.coef_[0])}
    }
    
    # Feature importances from Gradient Boosting & Random Forest
    rf_feature_importances = {name: round(float(v), 4) for name, v in zip(feature_names, rf.feature_importances_)}
    gb_feature_importances = {name: round(float(v), 4) for name, v in zip(feature_names, gb.feature_importances_)}
    
    # Key Exploratory Data Analysis insights
    eda_summary = {
        "total_customers": len(df_raw),
        "overall_churn_rate": round(float(df_raw["Exited"].mean()) * 100, 1),
        "products_churn": {
            "1 Product": round(float(df_raw[df_raw["NumOfProducts"] == 1]["Exited"].mean()) * 100, 1),
            "2 Products": round(float(df_raw[df_raw["NumOfProducts"] == 2]["Exited"].mean()) * 100, 1),
            "3 Products": round(float(df_raw[df_raw["NumOfProducts"] == 3]["Exited"].mean()) * 100, 1),
            "4 Products": round(float(df_raw[df_raw["NumOfProducts"] == 4]["Exited"].mean()) * 100, 1)
        },
        "geography_churn": {
            "France": round(float(df_raw[df_raw["Geography"] == "France"]["Exited"].mean()) * 100, 1),
            "Germany": round(float(df_raw[df_raw["Geography"] == "Germany"]["Exited"].mean()) * 100, 1),
            "Spain": round(float(df_raw[df_raw["Geography"] == "Spain"]["Exited"].mean()) * 100, 1)
        },
        "activity_churn": {
            "Active Member": round(float(df_raw[df_raw["IsActiveMember"] == 1]["Exited"].mean()) * 100, 1),
            "Inactive Member": round(float(df_raw[df_raw["IsActiveMember"] == 0]["Exited"].mean()) * 100, 1)
        },
        "age_brackets_churn": {
            "18-30": round(float(df_raw[df_raw["Age"].between(18, 30)]["Exited"].mean()) * 100, 1),
            "31-40": round(float(df_raw[df_raw["Age"].between(31, 40)]["Exited"].mean()) * 100, 1),
            "41-50": round(float(df_raw[df_raw["Age"].between(41, 50)]["Exited"].mean()) * 100, 1),
            "51-60": round(float(df_raw[df_raw["Age"].between(51, 60)]["Exited"].mean()) * 100, 1),
            "60+": round(float(df_raw[df_raw["Age"] > 60]["Exited"].mean()) * 100, 1)
        }
    }
    
    # Pick 25 diverse sample customers from test set for live batch testing / demo
    te_indices = X_te.index[:30]
    sample_customers = []
    for idx in te_indices:
        row_raw = df_raw.loc[idx]
        trans_idx = list(X_te.index).index(idx)
        sample_customers.append({
            "CustomerId": int(row_raw["CustomerId"]),
            "Surname": str(row_raw["Surname"]),
            "CreditScore": int(row_raw["CreditScore"]),
            "Geography": str(row_raw["Geography"]),
            "Gender": str(row_raw["Gender"]),
            "Age": int(row_raw["Age"]),
            "Tenure": int(row_raw["Tenure"]),
            "Balance": round(float(row_raw["Balance"]), 2),
            "NumOfProducts": int(row_raw["NumOfProducts"]),
            "HasCrCard": int(row_raw["HasCrCard"]),
            "IsActiveMember": int(row_raw["IsActiveMember"]),
            "EstimatedSalary": round(float(row_raw["EstimatedSalary"]), 2),
            "ActualExited": int(row_raw["Exited"]),
            "gb_churn_prob": round(float(p_gb_te[trans_idx]), 3),
            "rf_churn_prob": round(float(p_rf_te[trans_idx]), 3),
            "lr_churn_prob": round(float(p_lr_te[trans_idx]), 3)
        })

    # Extract Gradient Boosting model trees for direct client-side exact inference
    gb_trees = []
    for stage in gb.estimators_:
        t = stage[0].tree_
        gb_trees.append({
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": [round(float(v), 5) for v in t.threshold],
            "value": [round(float(v[0, 0]), 5) for v in t.value]
        })
    gb_serialized = {
        "learning_rate": gb.learning_rate,
        "init_raw": float(gb._raw_predict_init(X_te_trans[:1])[0, 0]),
        "trees": gb_trees
    }

    # Save to demo_data.json
    export_payload = {
        "scaler_stats": scaler_stats,
        "lr_weights": lr_weights,
        "gb_model": gb_serialized,
        "rf_feature_importances": rf_feature_importances,
        "gb_feature_importances": gb_feature_importances,
        "roc_data": roc_data,
        "threshold_metrics": threshold_metrics,
        "eda_summary": eda_summary,
        "sample_customers": sample_customers,
        "feature_names": feature_names
    }
    
    with open("demo_data.json", "w") as f:
        json.dump(export_payload, f)
    print("Exported demo_data.json successfully with", len(sample_customers), "sample customers and GB trees.")

if __name__ == "__main__":
    main()


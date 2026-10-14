# Customer churn prediction: results

**Data:** `Churn_Modelling.csv` from archive.zip. It has 10,000 bank customers and 14 columns, no missing values, and a 20.4% churn rate (`Exited`).
**Setup:** stratified 80/20 train/test split (seed 42) and 5-fold stratified CV on the training set. I dropped the identifier columns (RowNumber, CustomerId, Surname), standardized the numeric features and one-hot encoded Geography and Gender. I added two features, `ZeroBalance` and `BalanceToSalary`. LR and RF use class weighting to handle the imbalance. For each model, the decision threshold that maximizes F1 was chosen on out-of-fold training predictions, so the test set was never used to tune it.

## Model comparison (held-out test set, n = 2,000)

| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.766 ± 0.021 | 0.777 | 0.468 | 0.413 | 0.644 | 0.503 | 0.54 |
| Random Forest | 0.858 ± 0.012 | 0.860 | 0.690 | 0.656 | 0.614 | 0.635 | 0.52 |
| **Gradient Boosting** | **0.863 ± 0.009** | **0.872** | **0.725** | 0.625 | 0.651 | **0.638** | 0.30 |

Precision, recall and F1 are for the churn class at the tuned threshold. Results at the default 0.5 threshold are in `output/model_comparison.csv`. At 0.5, GB has precision 0.77 and recall 0.49.

**Recommendation:** use Gradient Boosting. It catches about 65% of churners, and about 62% of the customers it flags actually churn (265 true positives and 159 false alarms out of 2,000 customers). Random Forest is a close second. Logistic Regression trails by about 0.09 AUC because the main drivers are non-linear, as explained below.

## What drives churn

These are permutation importances on the test set, measured as the drop in ROC-AUC when a feature is shuffled. The direction of each effect comes from churn rates in the raw data.

1. **Age** is the strongest driver. Churn is 7.5% for ages 18–30, 34% for 40–50 and 56% for 50–60.
2. **NumOfProducts** is strongly non-linear. Churn is 28% with 1 product and 7.6% with 2, but 83% with 3 and 100% with 4. Logistic regression treats it as a straight line and misses this, which accounts for most of its gap to the tree models.
3. **IsActiveMember:** inactive members churn at 27%, compared with 14% for active members.
4. **Geography:** customers in Germany churn at 32%, compared with about 16% in France and Spain.
5. **Balance and Gender** have a smaller effect. Higher balances and female customers churn more.

CreditScore, Tenure, EstimatedSalary and HasCrCard contribute almost nothing.

## Files
- `churn_model.py`: the full pipeline. Run it with `python churn_model.py Churn_Modelling.csv [output_dir]` (needs pandas, scikit-learn and matplotlib).
- `output/model_comparison.csv`, `output/metrics.json`: all metrics.
- `output/feature_importance.csv`: permutation importance for all three models.
- `output/logreg_coefficients.csv`: signed LR coefficients.
- `output/churn_results.png`: ROC curves and the GB importance chart.

## Possible next steps
Add one-hot or interaction terms for NumOfProducts and Age to make logistic regression competitive and interpretable. Tune the GB hyperparameters with a CV search. Calibrate the probabilities if the scores will drive retention budgets.

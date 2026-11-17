# Loan approval prediction with ensemble models

> Course assignment from 2023, kept as it was written. The notes under "Reading the result" were added in November 2026.

Predicts whether a home-loan application is approved (614 applications, 422 approved) with
voting, bagging and boosting ensembles.

## Result

First notebook, one 80/20 split, 123 test applications:

| Model | Test accuracy |
|---|---|
| Logistic regression | 0.764 |
| Hard voting (logistic, tree, SVM) | 0.756 |
| XGBoost | 0.740 |
| Gradient boosting | 0.724 |
| SVM | 0.691 |
| Soft voting | 0.691 |
| Bagged SVM | 0.691 |
| Decision tree | 0.675 |

Approving everyone scores 0.691 on this test split.

The second notebook rebuilds the work as scikit-learn pipelines with grid search, saves the chosen
logistic regression to `best_model.pkl` and adds a small console form that scores one applicant.
Its logistic regression and XGBoost both reach 0.854 on their split; the final saved model scores
0.775 on a 169-row test split.

## Reading the result

- **Three of the eight models score exactly what "approve everyone" scores** (0.691), and the
  decision tree scores less.
- No ensemble beats the plain logistic regression. On 614 rows that is a common outcome.
- The two notebooks use different splits, so 0.764, 0.854 and 0.775 for logistic regression are
  not a progression. The gap shows how much one split of 123 rows can move the number.

## Newer work on the same topic

- [credit-risk-scorecard](https://github.com/faiqrasulov/credit-risk-scorecard): a scorecard against gradient boosting with repeated cross-validation, cost-based cutoffs, reason codes and a FastAPI scoring service.

## Files

| File | Content |
|---|---|
| `Loan Prediciton.ipynb` | First notebook: the ensembles |
| `Loan_Prediction_v2.ipynb` | Pipelines, grid search, saved model, console form |
| `Loan Prediciton.html` | Rendered copy of the first notebook |
| `best_model.pkl` | Saved pipeline (unpickle only with the scikit-learn version that wrote it) |
| `loan_prediction.csv` | The dataset |
| `About.txt` | Original brief |

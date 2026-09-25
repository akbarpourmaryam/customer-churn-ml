# Model evaluation and presentation guide

## Methodology

The project uses a stratified 60% training, 20% validation, and 20% test split. Stratification preserves the 26.5% churn rate in every partition.

1. Logistic Regression, Random Forest, and XGBoost are tuned with five-fold cross-validation on the training set. The dummy model supplies a minimum baseline.
2. Average precision (PR-AUC) is the tuning objective because churners are the minority class.
3. Each tuned model produces validation probabilities. Thresholds from 0.20 to 0.60 are compared, and the threshold with the highest validation F1 is retained.
4. Models are compared by validation PR-AUC. XGBoost wins narrowly and is refit on the combined training and validation data.
5. The winning model is evaluated once on the untouched test set.

This separation prevents test-set information from influencing model, hyperparameter, or threshold selection.

## Model comparison

| Model | Validation PR-AUC | Validation recall | Validation F1 |
|---|---:|---:|---:|
| XGBoost | 0.6476 | 0.7112 | 0.6303 |
| Logistic Regression | 0.6471 | 0.6551 | 0.6306 |
| Random Forest | 0.6410 | 0.6925 | 0.6403 |
| Dummy | 0.2654 | 0.0000 | 0.0000 |

XGBoost wins by only 0.0005 PR-AUC over Logistic Regression. That difference is not practically decisive. XGBoost is retained because it follows the predeclared selection rule, while Logistic Regression remains an attractive deployment alternative because it is simpler and easier to explain.

## Final test result

At its validation-selected threshold of 0.60, XGBoost achieves:

- Accuracy: 0.7764
- Balanced accuracy: 0.7556
- Precision: 0.5624
- Recall: 0.7112
- F1: 0.6281
- ROC-AUC: 0.8470
- PR-AUC: 0.6655

The confusion matrix contains 828 true negatives, 207 false positives, 108 false negatives, and 266 true positives.

## Why accuracy is insufficient

The dummy classifier predicts “no churn” for every customer. Because most customers do not churn, it still achieves about 73.5% accuracy. However, its recall and F1 for churn are zero. This is why the project also uses precision, recall, F1, balanced accuracy, ROC-AUC, PR-AUC, and the confusion matrix.

## Threshold selection

The model produces a probability. The threshold converts that probability into a retention action. Lower thresholds catch more churners but create more false-positive contacts; higher thresholds contact fewer customers but miss more churners.

F1 is used because no campaign economics were supplied. A production system should choose the threshold using retention value, contact and offer costs, and the cost of missing a churner.

## Feature importance

The strongest XGBoost features include contract type, fiber-optic internet, online security, payment method, and tenure. These are predictive associations—not evidence that changing one field will cause a customer to stay. Tree importance may also favor features that offer more split opportunities.

## Five-minute presentation outline

1. **Problem:** predict customer churn early enough to support retention activity.
2. **Data:** 7,043 telecom customers; 26.5% churn, so the classes are imbalanced.
3. **Pipeline:** validate, clean, engineer two features, split 60/20/20, preprocess inside each model pipeline.
4. **Models:** compare a dummy baseline, Logistic Regression, Random Forest, and XGBoost; tune with five-fold CV.
5. **Selection:** use PR-AUC rather than accuracy and tune the decision threshold separately.
6. **Result:** XGBoost reaches 0.6655 test PR-AUC and catches 71.1% of churners.
7. **Interpretation:** contracts, tenure, internet service, security, and payment method are important predictive signals.
8. **Engineering:** MLflow tracks experiments; the full preprocessing-plus-model bundle is versioned and exposed through FastAPI; 16 tests pass.
9. **Limitations:** static data, no business cost function, no causal claims, and no production drift monitoring yet.

## Short explanation to say aloud

> I treated this as a complete ML lifecycle rather than a notebook-only model. I kept the test set isolated, tuned three real classifiers with stratified cross-validation, compared them on a separate validation set using PR-AUC, and selected each decision threshold using validation F1. XGBoost narrowly won, although Logistic Regression was essentially tied and is easier to explain. On the untouched test set, XGBoost found 71% of churners with a PR-AUC of 0.666. I saved the preprocessing and model together, tracked experiments with MLflow, versioned the final artifact, added a prediction API, and covered the main behavior with automated tests.

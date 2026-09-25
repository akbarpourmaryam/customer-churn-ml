"""Presentation-ready Markdown reporting."""

from pathlib import Path


def write_evaluation_report(path, comparison, best_model_name, threshold, test_metrics, matrix, metadata):
    """Create a concise final report from reproducible training outputs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = comparison.round(4).to_markdown(index=False)
    tn, fp, fn, tp = matrix.ravel()
    content = f"""# Customer churn model evaluation

## Executive summary

Four classifiers were compared: a majority-class dummy baseline, logistic regression, random forest, and XGBoost. Models were tuned with stratified cross-validation on the training partition and compared on a separate validation partition using PR-AUC rather than accuracy. **{best_model_name}** was selected and evaluated once on the untouched test partition.

## Data and evaluation design

- Dataset rows: {metadata['dataset_rows']:,}
- Split: 60% train / 20% validation / 20% test, stratified by churn
- Hyperparameter objective: cross-validated average precision (PR-AUC)
- Model-selection metric: validation PR-AUC
- Threshold-selection metric: validation F1
- Selected probability threshold: {threshold:.2f}
- Random seed: {metadata['random_seed']}

## Validation model comparison

{table}

The top models are close. The winner is retained because it follows the predeclared validation PR-AUC rule; a simpler model may still be preferable when interpretability, latency, or maintenance dominates a very small metric difference.

Accuracy is not sufficient because only about 26.5% of customers churn. A model that predicts “no churn” for everyone achieves roughly 73.5% accuracy while finding zero churners. PR-AUC, recall, F1, ROC-AUC, and the confusion matrix expose that failure.

![Model comparison](model_comparison.png)

## Final test results for {best_model_name}

| Metric | Score |
|---|---:|
| Accuracy | {test_metrics['accuracy']:.4f} |
| Balanced accuracy | {test_metrics['balanced_accuracy']:.4f} |
| Precision | {test_metrics['precision']:.4f} |
| Recall | {test_metrics['recall']:.4f} |
| F1 | {test_metrics['f1']:.4f} |
| ROC-AUC | {test_metrics['roc_auc']:.4f} |
| PR-AUC | {test_metrics['pr_auc']:.4f} |

Confusion matrix counts: {tn} true negatives, {fp} false positives, {fn} false negatives, and {tp} true positives.

![Final evaluation curves](best_model_evaluation.png)

![Feature importance](best_model_feature_importance.png)

## Interpretation

Lowering the threshold usually catches more churners but contacts more customers who would not churn. The selected threshold maximizes validation F1 because business-specific retention costs were not supplied. In production, the threshold should instead maximize expected business value.

Feature importance describes predictive association, not causation. Tree importance can also favor variables with more possible split points; permutation or SHAP analysis would be appropriate follow-up work.

## Reproducibility

- Model version: `{metadata['model_version']}`
- Dataset SHA-256: `{metadata['dataset_sha256']}`
- Git commit at training time: `{metadata['git_commit']}`
- Configuration: `configs/training.yaml`
- Experiment tracking: local MLflow runs in `mlruns/`
"""
    path.write_text(content, encoding="utf-8")

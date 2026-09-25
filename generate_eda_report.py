"""Generate a compact EDA report and presentation plot."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import load_config


def main():
    config = load_config()
    data = pd.read_csv(config["paths"]["data_path"])
    reports_dir = config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.countplot(data=data, x="Churn", hue="Churn", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("Class distribution")
    contract_rates = data.groupby("Contract")["Churn"].apply(lambda values: values.eq("Yes").mean())
    contract_rates.sort_values().plot.barh(ax=axes[0, 1], color="#E45756")
    axes[0, 1].set(title="Churn rate by contract", xlabel="Churn rate", ylabel="")
    sns.histplot(data=data, x="tenure", hue="Churn", bins=24, element="step", ax=axes[1, 0])
    axes[1, 0].set_title("Tenure distribution by churn")
    sns.boxplot(data=data, x="Churn", y="MonthlyCharges", hue="Churn", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Monthly charges by churn")
    figure.tight_layout()
    figure.savefig(reports_dir / "eda_overview.png", dpi=150)
    plt.close(figure)

    report = f"""# Exploratory data analysis

## Dataset overview

- Rows: {len(data):,}
- Raw columns: {data.shape[1]}
- Churned customers: {data['Churn'].eq('Yes').sum():,}
- Overall churn rate: {data['Churn'].eq('Yes').mean():.1%}
- Blank/non-numeric `TotalCharges`: {data['TotalCharges'].isna().sum()} rows; these are new customers with zero tenure and are set to zero by cleaning.
- Median tenure: {data['tenure'].median():.0f} months
- Mean monthly charge: ${data['MonthlyCharges'].mean():.2f}

![EDA overview](eda_overview.png)

## Main observations

- Month-to-month customers churn at {data.loc[data['Contract'].eq('Month-to-month'), 'Churn'].eq('Yes').mean():.1%}, compared with {data.loc[data['Contract'].eq('Two year'), 'Churn'].eq('Yes').mean():.1%} for two-year customers.
- Fiber-optic customers churn at {data.loc[data['InternetService'].eq('Fiber optic'), 'Churn'].eq('Yes').mean():.1%}, higher than DSL customers.
- Electronic-check customers churn at {data.loc[data['PaymentMethod'].eq('Electronic check'), 'Churn'].eq('Yes').mean():.1%}, the highest rate among payment methods.
- Churn is more common among shorter-tenure customers and customers with higher monthly charges.

These observations are descriptive associations. They motivate model features but do not establish causal effects.
"""
    (reports_dir / "eda_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

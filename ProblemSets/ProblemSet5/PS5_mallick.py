#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ensure output folders exist
os.makedirs("images", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ---------------------------
# Part A: Data cleaning + Visualizations
# ---------------------------

def clean_data(file_path):
    """
    Clean and prepare the Tech_Use_Stress_Wellness dataset.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()
    df["total_tech_usage"] = (
        df["phone_usage_hours"] +
        df["laptop_usage_hours"] +
        df["tablet_usage_hours"] +
        df["tv_usage_hours"]
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[15, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "51+"]
    )
    return df


def create_visualization_1(df):
    """Scatter plot: tech usage vs. mental health (with trend line)."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df["total_tech_usage"], df["mental_health_score"],
                alpha=0.6, color="blue", s=50)
    z = np.polyfit(df["total_tech_usage"], df["mental_health_score"], 1)
    p = np.poly1d(z)
    plt.plot(df["total_tech_usage"], p(df["total_tech_usage"]), "r--", alpha=0.8)
    plt.xlabel("Total Daily Technology Usage (Hours)")
    plt.ylabel("Mental Health Score")
    plt.title("Technology Usage vs Mental Health Score")
    plt.tight_layout()
    plt.savefig("images/viz1_tech_vs_mental_health.png", dpi=300)
    plt.close()


def create_visualization_2(df):
    """Bar chart: average stress level by sleep quality."""
    plt.figure(figsize=(10, 6))
    sleep_stress = df.groupby("sleep_quality")["stress_level"].mean()
    bars = plt.bar(sleep_stress.index, sleep_stress.values,
                   color=["lightblue", "lightgreen", "gold", "orange", "salmon"])
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                 f"{h:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.xlabel("Sleep Quality Rating (1-5)")
    plt.ylabel("Average Stress Level")
    plt.title("Average Stress Level by Sleep Quality")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/viz2_sleep_vs_stress.png", dpi=300)
    plt.close()


def create_visualization_3(df):
    """Boxplot: mental health score by age group."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="age_group", y="mental_health_score", palette="Set2")
    sns.stripplot(data=df, x="age_group", y="mental_health_score",
                  color="black", alpha=0.5, size=3, jitter=True)
    plt.xlabel("Age Group")
    plt.ylabel("Mental Health Score")
    plt.title("Mental Health Scores by Age Group")
    plt.tight_layout()
    plt.savefig("images/viz3_mental_health_by_age.png", dpi=300)
    plt.close()

# ---------------------------
# Part B: Econometric models
# ---------------------------

def estimate_models(df):
    """Fit three regression models and return results."""
    model1 = ols("mental_health_score ~ total_tech_usage", data=df).fit()
    model2 = ols(
        "mental_health_score ~ total_tech_usage + age + sleep_quality + physical_activity_hours_per_week",
        data=df).fit()
    model3 = ols(
        "mental_health_score ~ total_tech_usage * age + sleep_quality",
        data=df).fit()
    return model1, model2, model3


def create_results_table(model1, model2, model3):
    """Save regression results to CSV and LaTeX tables."""
    results_data = [
        {
            "Model": "Basic",
            "Tech Usage Coef": f"{model1.params['total_tech_usage']:.3f}",
            "P-value": f"{model1.pvalues['total_tech_usage']:.3f}",
            "R-squared": f"{model1.rsquared:.3f}",
            "Observations": model1.nobs
        },
        {
            "Model": "With Controls",
            "Tech Usage Coef": f"{model2.params['total_tech_usage']:.3f}",
            "P-value": f"{model2.pvalues['total_tech_usage']:.3f}",
            "R-squared": f"{model2.rsquared:.3f}",
            "Observations": model2.nobs
        },
        {
            "Model": "With Interaction",
            "Tech Usage Coef": f"{model3.params['total_tech_usage']:.3f}",
            "P-value": f"{model3.pvalues['total_tech_usage']:.3f}",
            "R-squared": f"{model3.rsquared:.3f}",
            "Observations": model3.nobs
        }
    ]
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("tables/model_results.csv", index=False)
    results_df.to_latex("tables/model_results.tex", index=False)
    return results_df



def main_all():
    """Run Part A (visualizations) and Part B (models) together."""
    file_path = os.path.join("data", "Tech_Use_Stress_Wellness.csv")

    # Part A
    df_clean = clean_data(file_path)
    create_visualization_1(df_clean)
    create_visualization_2(df_clean)
    create_visualization_3(df_clean)

    # Part B
    m1, m2, m3 = estimate_models(df_clean)
    create_results_table(m1, m2, m3)

    print("\n All outputs generated successfully!")
    print("Check 'images/' for plots and 'tables/' for regression results.")

# ---------------------------
# Script entry
# ---------------------------
if __name__ == "__main__":
    main_all()



# In[ ]:





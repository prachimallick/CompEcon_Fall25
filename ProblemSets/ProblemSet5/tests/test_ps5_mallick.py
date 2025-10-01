# tests/test_ps5_mallick.py
import sys, os
# add project root (parent of tests/) to sys.path so pytest can import PS5_mallick
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from PS5_mallick import clean_data, create_visualization_1, estimate_models, create_results_table

def test_clean_data_returns_dataframe():
    df = clean_data(os.path.join("data", "Tech_Use_Stress_Wellness.csv"))
    assert isinstance(df, pd.DataFrame)
    assert "total_tech_usage" in df.columns
    assert "age_group" in df.columns

def test_visualization_creates_file():
    df = clean_data(os.path.join("data", "Tech_Use_Stress_Wellness.csv"))
    create_visualization_1(df)
    assert os.path.exists("images/viz1_tech_vs_mental_health.png")

def test_estimate_models_runs():
    df = clean_data(os.path.join("data", "Tech_Use_Stress_Wellness.csv"))
    m1, m2, m3 = estimate_models(df)
    assert hasattr(m1, "params")
    assert hasattr(m2, "params")
    assert hasattr(m3, "params")

def test_results_table_created():
    df = clean_data(os.path.join("data", "Tech_Use_Stress_Wellness.csv"))
    m1, m2, m3 = estimate_models(df)
    results_df = create_results_table(m1, m2, m3)
    assert isinstance(results_df, pd.DataFrame)
    assert os.path.exists("tables/model_results.csv")
    assert os.path.exists("tables/model_results.tex")

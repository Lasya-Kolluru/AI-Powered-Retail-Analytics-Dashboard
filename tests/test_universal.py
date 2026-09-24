"""
Comprehensive test suite verifying universal adaptability across ANY dataset structure.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
from app import sanitize_dataframe, profile_dataset, init_sqlite_db, run_sql_query


def test_user_retail_dataset():
    """Verify on user's 1000-row retail dataset."""
    df = pd.read_csv("data/sample_retail_data.csv")
    conn, clean_df, col_map = init_sqlite_db(df)
    prof = profile_dataset(clean_df)
    
    assert prof["primary_metric"] == "Total_Amount"
    assert prof["primary_dimension"] in ["Product_Category", "Gender"]
    
    res = run_sql_query(conn, "SELECT COUNT(*) as cnt, SUM(Total_Amount) as total FROM universal_data;")
    assert res["cnt"].iloc[0] == len(df)
    assert res["total"].iloc[0] > 0
    print("[PASS] User retail dataset test passed!")


def test_hr_dataset():
    """Verify on an arbitrary HR dataset with 0 retail columns."""
    hr_data = pd.DataFrame({
        "Employee ID": [101, 102, 103, 104, 105],
        "Full Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
        "Salary (USD)": [120000, 95000, 135000, 78000, 105000],
        "Performance Rating": [4.8, 3.9, 4.9, 4.2, 4.5],
    })
    conn, clean_df, col_map = init_sqlite_db(hr_data)
    prof = profile_dataset(clean_df)
    
    assert "Salary_USD" in clean_df.columns
    assert prof["primary_dimension"] == "Department"
    
    res = run_sql_query(conn, "SELECT Department, AVG(Salary_USD) as avg_sal FROM universal_data GROUP BY Department;")
    assert len(res) == 3
    print("[PASS] Arbitrary HR dataset test passed!")


def test_text_only_dataset():
    """Verify on a dataset with ZERO numeric columns."""
    text_data = pd.DataFrame({
        "Feedback_ID": ["FB-1", "FB-2", "FB-3", "FB-4"],
        "Category": ["Bug", "Feature Request", "Bug", "Billing"],
        "Sentiment": ["Negative", "Positive", "Negative", "Neutral"],
    })
    conn, clean_df, col_map = init_sqlite_db(text_data)
    prof = profile_dataset(clean_df)
    
    assert len(prof["numeric_cols"]) == 0
    assert prof["primary_dimension"] == "Category"
    
    res = run_sql_query(conn, "SELECT Category, COUNT(*) as cnt FROM universal_data GROUP BY Category;")
    assert len(res) == 3
    print("[PASS] Text-only dataset test passed!")


if __name__ == "__main__":
    test_user_retail_dataset()
    test_hr_dataset()
    test_text_only_dataset()
    print("[SUCCESS] All universal dataset tests passed successfully!")

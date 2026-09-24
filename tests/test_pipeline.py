"""
Test suite for Retail Analytics SQLite Engine and Pipeline Logic.
"""

import os
import sqlite3
import pandas as pd
import pytest


def test_sample_dataset_exists_and_valid():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_retail_data.csv")
    assert os.path.exists(csv_path), "Sample dataset does not exist."
    df = pd.read_csv(csv_path)
    assert len(df) >= 5, "Dataset should have at least 5 records."
    required_cols = {"Transaction_ID", "Product_Name", "Region", "Revenue", "Units_Sold"}
    assert required_cols.issubset(set(df.columns)), f"Missing required columns: {required_cols - set(df.columns)}"


def test_sqlite_in_memory_pipeline():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_retail_data.csv")
    df = pd.read_csv(csv_path)

    # In-memory SQLite DB
    conn = sqlite3.connect(":memory:")
    df.to_sql("retail_sales", conn, index=False, if_exists="replace")

    # 1. Top 5 products query
    q1 = """
        SELECT Product_Name, SUM(Revenue) AS Total_Revenue 
        FROM retail_sales 
        GROUP BY Product_Name 
        ORDER BY Total_Revenue DESC 
        LIMIT 5;
    """
    res1 = pd.read_sql_query(q1, conn)
    assert len(res1) <= 5
    assert "Total_Revenue" in res1.columns
    assert res1["Total_Revenue"].iloc[0] >= res1["Total_Revenue"].iloc[-1]

    # 2. Regional breakdown query
    q2 = """
        SELECT Region, SUM(Revenue) AS Total_Revenue, SUM(Units_Sold) AS Total_Units 
        FROM retail_sales 
        GROUP BY Region 
        ORDER BY Total_Revenue DESC;
    """
    res2 = pd.read_sql_query(q2, conn)
    assert len(res2) > 0
    assert "Region" in res2.columns
    assert res2["Total_Revenue"].sum() == df["Revenue"].sum()

    conn.close()


def test_app_import_and_compilation():
    import py_compile
    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    py_compile.compile(app_path, doraise=True)


if __name__ == "__main__":
    test_sample_dataset_exists_and_valid()
    test_sqlite_in_memory_pipeline()
    test_app_import_and_compilation()
    print("[SUCCESS] All pipeline unit tests passed successfully!")

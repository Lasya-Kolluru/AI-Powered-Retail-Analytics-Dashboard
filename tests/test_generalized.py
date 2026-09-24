"""
Test dynamic schema adaptability with user dataset.
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app import sanitize_dataframe, detect_semantic_roles, init_sqlite_db, run_sql_query

def test_user_dataset():
    df = pd.read_csv("data/sample_retail_data.csv")
    clean_df = sanitize_dataframe(df)
    roles = detect_semantic_roles(clean_df)
    
    assert roles["val_col"] == "Total_Amount", f"Expected Total_Amount, got {roles['val_col']}"
    assert roles["dim_col_1"] == "Product_Category", f"Expected Product_Category, got {roles['dim_col_1']}"
    assert roles["vol_col"] == "Quantity", f"Expected Quantity, got {roles['vol_col']}"
    assert roles["dim_col_2"] == "Gender", f"Expected Gender, got {roles['dim_col_2']}"
    
    conn, _ = init_sqlite_db(df)
    
    q1 = f"SELECT {roles['dim_col_1']}, SUM({roles['val_col']}) AS Total, COUNT(*) AS Count FROM retail_sales GROUP BY {roles['dim_col_1']} ORDER BY Total DESC;"
    res1 = run_sql_query(conn, q1)
    assert len(res1) == 3
    print("Test 1 Passed (Categories Breakdown):\n", res1)

    q2 = f"SELECT {roles['dim_col_2']}, SUM({roles['val_col']}) AS Total FROM retail_sales GROUP BY {roles['dim_col_2']};"
    res2 = run_sql_query(conn, q2)
    assert len(res2) == 2
    print("Test 2 Passed (Gender Breakdown):\n", res2)

    print("[SUCCESS] Generalized schema engine passed all tests!")

if __name__ == "__main__":
    test_user_dataset()

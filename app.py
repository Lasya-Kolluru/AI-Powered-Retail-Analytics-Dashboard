"""
AI-Powered Generalized Analytics Dashboard
Built with Streamlit, SQLite, Pandas, and the Google GenAI SDK.
Fully adaptive: works with ANY CSV, ANY schema, and ANY column configuration.
"""

import os
import re
import sqlite3
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal AI Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .ai-box {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.08) 0%, rgba(124, 77, 255, 0.08) 100%);
        border: 1px solid rgba(124, 77, 255, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
    }
    .badge-sql {
        display: inline-block;
        background-color: #0284c7;
        color: white;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .schema-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px;
        font-family: monospace;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Universal Data Ingestion & Semantic Schema Analyzer
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load default dataset bundled in repository or create robust fallback."""
    default_path = os.path.join(os.path.dirname(__file__), "data", "sample_retail_data.csv")
    if os.path.exists(default_path):
        try:
            return pd.read_csv(default_path)
        except Exception:
            pass
    # Fallback dataset
    return pd.DataFrame({
        "Transaction_ID": [1, 2, 3, 4, 5],
        "Product_Category": ["Beauty", "Clothing", "Electronics", "Clothing", "Beauty"],
        "Quantity": [3, 2, 1, 1, 2],
        "Price_per_Unit": [50, 500, 30, 500, 50],
        "Total_Amount": [150, 1000, 30, 500, 100],
    })


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize all column names into standard, safe SQL identifiers."""
    df_clean = df.copy()
    new_cols = []
    seen = {}
    for col in df_clean.columns:
        # Replace non-alphanumeric chars with underscore
        clean = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col).strip())
        clean = re.sub(r"^_+|_+$", "", clean)
        if not clean or clean[0].isdigit():
            clean = f"col_{clean}"
        # Ensure column uniqueness
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df_clean.columns = new_cols
    return df_clean


def detect_semantic_roles(df: pd.DataFrame):
    """
    Intelligently inspect any dataset to infer monetary values, quantities,
    categorical dimensions, and date columns for universal analytics.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    
    # Also check if any numeric columns act like low-cardinality categories
    for col in numeric_cols:
        if df[col].nunique() <= 10 and col not in categorical_cols:
            categorical_cols.append(col)

    # 1. Primary Value / Monetary Metric candidate
    val_priority = ["total_amount", "revenue", "amount", "total", "sales", "price", "gross", "cost", "value", "spend"]
    val_col = None
    for target in val_priority:
        for c in numeric_cols:
            if target in c.lower():
                val_col = c
                break
        if val_col:
            break
    if not val_col and numeric_cols:
        val_col = numeric_cols[-1]  # fallback to last numeric column

    # 2. Volume / Quantity Metric candidate
    vol_priority = ["quantity", "qty", "units_sold", "units", "volume", "count", "items", "num"]
    vol_col = None
    for target in vol_priority:
        for c in numeric_cols:
            if target in c.lower() and c != val_col:
                vol_col = c
                break
        if vol_col:
            break
    if not vol_col:
        for c in numeric_cols:
            if c != val_col:
                vol_col = c
                break

    # 3. Primary Dimension candidate (Category, Product, Department, etc.)
    cat1_priority = ["product_category", "category", "product_name", "product", "item", "department", "segment", "type", "name"]
    dim_col_1 = None
    for target in cat1_priority:
        for c in categorical_cols:
            if target in c.lower():
                dim_col_1 = c
                break
        if dim_col_1:
            break
    if not dim_col_1 and categorical_cols:
        # Choose categorical column with balanced cardinality
        cardinalities = [(c, df[c].nunique()) for c in categorical_cols if 1 < df[c].nunique() <= 100]
        if cardinalities:
            dim_col_1 = sorted(cardinalities, key=lambda x: x[1])[0][0]
        else:
            dim_col_1 = categorical_cols[0]

    # 4. Secondary Dimension candidate (Gender, Region, City, Payment, etc.)
    cat2_priority = ["gender", "region", "country", "city", "customer_segment", "payment", "status", "tier", "channel"]
    dim_col_2 = None
    for target in cat2_priority:
        for c in categorical_cols:
            if target in c.lower() and c != dim_col_1:
                dim_col_2 = c
                break
        if dim_col_2:
            break
    if not dim_col_2:
        for c in categorical_cols:
            if c != dim_col_1:
                dim_col_2 = c
                break

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "val_col": val_col,
        "vol_col": vol_col,
        "dim_col_1": dim_col_1,
        "dim_col_2": dim_col_2,
    }


def init_sqlite_db(df: pd.DataFrame) -> tuple[sqlite3.Connection, pd.DataFrame]:
    """Store sanitized DataFrame into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    sanitized_df = sanitize_dataframe(df)
    sanitized_df.to_sql("retail_sales", conn, index=False, if_exists="replace")
    return conn, sanitized_df


def run_sql_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Execute SQL query safely and return DataFrame."""
    return pd.read_sql_query(query, conn)


# -----------------------------------------------------------------------------
# 3. Gemini AI Analysis Engine (google-genai SDK)
# -----------------------------------------------------------------------------
def get_gemini_api_key(sidebar_key: str) -> str:
    """Resolve Gemini API key with priority: Streamlit secrets -> Environment -> Sidebar input."""
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        return st.secrets["GEMINI_API_KEY"].strip()
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY").strip()
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()
    return ""


def analyze_with_gemini(
    api_key: str,
    context_title: str,
    sql_query: str,
    result_df: pd.DataFrame,
    schema_summary: str,
    custom_user_prompt: str = "",
) -> str:
    """Invoke the Gemini model using the official google-genai SDK with model fallback."""
    if not api_key:
        return "⚠️ Gemini API Key not found. Please provide an API key in the sidebar or via `.streamlit/secrets.toml`."

    client = genai.Client(api_key=api_key)
    table_str = result_df.to_string(index=False)

    system_instruction = (
        "You are an elite Business Data Analyst and Strategic Consultant. "
        "Analyze the SQL query results extracted from the user's dataset. "
        "Your goal is to provide insightful, commercially actionable findings regardless of dataset domain.\n"
        "Format your analysis with:\n"
        "1. Executive Summary & Key Observations (highlight major proportions, top performers, or anomalies)\n"
        "2. 3 High-Impact Actionable Business Recommendations (practical, data-backed operational or revenue steps)\n"
        "3. Risk Factors & Strategic Considerations\n"
        "Use bolding for metrics and bullet points for high executive readability."
    )

    user_content = f"""
Dataset Schema Overview:
{schema_summary}

Metric / Query Focus: {context_title}
Executed SQL Query:
```sql
{sql_query}
```

SQL Query Results Data:
{table_str}

Additional User Focus / Instruction:
{custom_user_prompt if custom_user_prompt else "Provide the standard 3 actionable business recommendations based on these metrics."}
"""

    candidate_models = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-flash-latest"]
    last_err = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                ),
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            continue

    return f"❌ Error communicating with Gemini API: {str(last_err)}"


# -----------------------------------------------------------------------------
# 4. Streamlit Application Layout
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🛍️ Universal AI Retail & Business Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Adaptive In-Memory SQL Intelligence + Google Gemini AI • Works with any CSV schema</div>',
        unsafe_allow_html=True,
    )

    # --- Sidebar: Configuration & Data Source ---
    st.sidebar.header("⚙️ Configuration & Data")

    sidebar_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Enter your Google AI Studio Gemini API key. Alternatively, configure in .streamlit/secrets.toml",
    )

    gemini_key = get_gemini_api_key(sidebar_api_key)
    if gemini_key:
        st.sidebar.success("✅ Gemini API Key Connected")
    else:
        st.sidebar.warning("⚠️ No Gemini API Key configured. Add key to unlock AI Insights.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Any Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Any columns / size)", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(raw_df):,} rows × {len(raw_df.columns)} columns")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            raw_df = load_default_data()
    else:
        raw_df = load_default_data()
        st.sidebar.info(f"ℹ️ Using default dataset ({len(raw_df):,} rows)")

    # Initialize in-memory SQLite connection with generalized schema
    conn, clean_df = init_sqlite_db(raw_df)
    roles = detect_semantic_roles(clean_df)

    val_col = roles["val_col"]
    vol_col = roles["vol_col"]
    dim_1 = roles["dim_col_1"]
    dim_2 = roles["dim_col_2"]

    # --- Dynamic KPI Summary Bar ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Records", f"{len(clean_df):,}")

    with kpi2:
        if val_col:
            total_val = clean_df[val_col].sum()
            label = f"Total {val_col.replace('_', ' ').title()}"
            is_currency = any(w in val_col.lower() for w in ["amount", "revenue", "price", "sales", "cost", "total"])
            prefix = "$" if is_currency else ""
            st.metric(label, f"{prefix}{total_val:,.2f}")
        else:
            st.metric("Total Value", f"{len(clean_df):,} rows")

    with kpi3:
        if vol_col and vol_col != val_col:
            total_vol = clean_df[vol_col].sum()
            label = f"Total {vol_col.replace('_', ' ').title()}"
            st.metric(label, f"{total_vol:,.0f}")
        elif dim_1:
            st.metric(f"Unique {dim_1.replace('_', ' ').title()}", f"{clean_df[dim_1].nunique():,}")
        else:
            st.metric("Dimensions", f"{len(roles['categorical_cols'])}")

    with kpi4:
        if val_col and len(clean_df) > 0:
            avg_val = clean_df[val_col].mean()
            label = f"Avg {val_col.replace('_', ' ').title()}"
            is_currency = any(w in val_col.lower() for w in ["amount", "revenue", "price", "sales", "cost", "total"])
            prefix = "$" if is_currency else ""
            st.metric(label, f"{prefix}{avg_val:,.2f}")
        else:
            st.metric("Metrics", f"{len(roles['numeric_cols'])} numeric")

    st.markdown("---")

    # Schema summary for AI context
    schema_summary = ", ".join([f"{col} ({clean_df[col].dtype})" for col in clean_df.columns])

    # --- Dynamic Query Construction ---
    preset_queries = {}

    # Query 1: Primary Dimension Breakdown
    if dim_1 and val_col:
        preset_queries[f"📊 Breakdown by {dim_1.replace('_', ' ').title()}"] = (
            f"SELECT {dim_1}, "
            f"COUNT(*) AS Record_Count, "
            f"ROUND(SUM({val_col}), 2) AS Total_{val_col}, "
            f"ROUND(AVG({val_col}), 2) AS Avg_{val_col} "
            f"FROM retail_sales GROUP BY {dim_1} ORDER BY Total_{val_col} DESC LIMIT 10;"
        )

    # Query 2: Secondary Dimension Breakdown
    if dim_2 and val_col and dim_2 != dim_1:
        preset_queries[f"🌍 Distribution across {dim_2.replace('_', ' ').title()}"] = (
            f"SELECT {dim_2}, "
            f"COUNT(*) AS Total_Orders, "
            f"ROUND(SUM({val_col}), 2) AS Total_{val_col}, "
            f"ROUND(AVG({val_col}), 2) AS Avg_{val_col} "
            f"FROM retail_sales GROUP BY {dim_2} ORDER BY Total_{val_col} DESC;"
        )

    # Query 3: Cross-Dimensional Matrix
    if dim_1 and dim_2 and val_col:
        preset_queries[f"👥 Cross Analysis: {dim_1.replace('_', ' ').title()} × {dim_2.replace('_', ' ').title()}"] = (
            f"SELECT {dim_1}, {dim_2}, "
            f"COUNT(*) AS Volume, "
            f"ROUND(SUM({val_col}), 2) AS Total_{val_col} "
            f"FROM retail_sales GROUP BY {dim_1}, {dim_2} ORDER BY Total_{val_col} DESC LIMIT 15;"
        )

    # Query 4: Statistical & Distribution Overview
    if val_col:
        preset_queries[f"📈 Statistical Summary of {val_col.replace('_', ' ').title()}"] = (
            f"SELECT "
            f"ROUND(MIN({val_col}), 2) AS Min_Value, "
            f"ROUND(AVG({val_col}), 2) AS Average_Value, "
            f"ROUND(MAX({val_col}), 2) AS Max_Value, "
            f"ROUND(SUM({val_col}), 2) AS Aggregate_Sum, "
            f"COUNT(*) AS Sample_Size "
            f"FROM retail_sales;"
        )

    # Fallback if no numeric or categorical found
    if not preset_queries:
        first_col = clean_df.columns[0]
        preset_queries["📋 First 10 Records"] = f"SELECT * FROM retail_sales LIMIT 10;"

    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Adaptive SQL & AI Insights", "🛠️ Custom SQL Query Builder", "🔍 Schema & Raw Data"])

    # Tab 1: Built-in Adaptive Queries & AI Insights
    with tab1:
        st.subheader("Automated SQL Analytics Engine")
        st.caption("Queries are automatically synthesized based on your uploaded CSV column structure.")

        selected_query_label = st.selectbox(
            "Select an Analytical Query:",
            list(preset_queries.keys()),
            index=0,
        )

        query_sql = preset_queries[selected_query_label]

        with st.expander("🔎 View Executed SQL Statement", expanded=False):
            st.code(query_sql, language="sql")

        try:
            results_df = run_sql_query(conn, query_sql)
        except Exception as e:
            st.error(f"SQL execution error: {e}")
            results_df = pd.DataFrame()

        if not results_df.empty:
            col_data, col_chart = st.columns([1, 1])
            with col_data:
                st.markdown(f"**Query Results Table ({len(results_df)} rows)**")
                st.dataframe(results_df, use_container_width=True)

            with col_chart:
                st.markdown("**Visual Distribution**")
                numeric_cols = results_df.select_dtypes(include=["number"]).columns.tolist()
                text_cols = results_df.select_dtypes(include=["object"]).columns.tolist()

                if text_cols and numeric_cols:
                    chart_df = results_df.set_index(text_cols[0])[[numeric_cols[0]]]
                    st.bar_chart(chart_df, color="#1E88E5")
                elif numeric_cols:
                    st.bar_chart(results_df[numeric_cols[0]], color="#7C4DFF")
                else:
                    st.info("No numeric series available for auto-charting.")

            # AI Recommendation Section
            st.markdown("---")
            st.subheader("🤖 Gemini AI Business Recommendations")
            st.caption("Google GenAI analyzing in-memory SQL query results.")

            user_prompt_extra = st.text_input(
                "Add specific instructions or questions for the AI Analyst (Optional):",
                placeholder="e.g., Focus on profit margins or recommend target customer demographic...",
            )

            if st.button("✨ Generate AI Insights", type="primary", use_container_width=False):
                if not gemini_key:
                    st.error("Please configure your Gemini API Key in the sidebar or `.streamlit/secrets.toml` to generate insights.")
                else:
                    with st.spinner("Analyzing query metrics with Gemini..."):
                        ai_response = analyze_with_gemini(
                            api_key=gemini_key,
                            context_title=selected_query_label,
                            sql_query=query_sql,
                            result_df=results_df,
                            schema_summary=schema_summary,
                            custom_user_prompt=user_prompt_extra,
                        )
                        st.session_state["last_ai_insight"] = ai_response

            if "last_ai_insight" in st.session_state:
                st.markdown(
                    f"""
                    <div class="ai-box">
                        <span class="badge-sql">Gemini Strategic Analysis</span>
                        <br/>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["last_ai_insight"])

    # Tab 2: Custom SQL Query Builder & Playground
    with tab2:
        st.subheader("🛠️ Custom Query Builder")
        st.caption("Slice and aggregate ANY column in your dataset dynamically.")

        qb_col1, qb_col2, qb_col3, qb_col4 = st.columns(4)
        
        all_cols = clean_df.columns.tolist()
        num_cols = roles["numeric_cols"] if roles["numeric_cols"] else all_cols
        
        with qb_col1:
            grp_by = st.selectbox("Group By (Dimension):", all_cols, index=all_cols.index(dim_1) if dim_1 in all_cols else 0)
        with qb_col2:
            agg_col = st.selectbox("Aggregate Column (Metric):", num_cols, index=num_cols.index(val_col) if val_col in num_cols else 0)
        with qb_col3:
            agg_func = st.selectbox("Aggregation Function:", ["SUM", "AVG", "COUNT", "MAX", "MIN"], index=0)
        with qb_col4:
            limit_val = st.slider("Limit Rows:", min_value=5, max_value=50, value=10)

        built_query = f"SELECT {grp_by}, {agg_func}({agg_col}) AS {agg_func.lower()}_{agg_col}, COUNT(*) AS count FROM retail_sales GROUP BY {grp_by} ORDER BY {agg_func.lower()}_{agg_col} DESC LIMIT {limit_val};"
        
        st.code(built_query, language="sql")
        
        if st.button("Run Built Query"):
            try:
                b_res = run_sql_query(conn, built_query)
                st.dataframe(b_res, use_container_width=True)
            except Exception as e:
                st.error(f"Execution Error: {e}")

        st.markdown("---")
        st.subheader("💻 Free-Form SQL Editor")
        custom_sql = st.text_area("Write Any SQL Query:", value=f"SELECT * FROM retail_sales LIMIT 5;", height=100)
        
        if st.button("Execute Custom SQL"):
            try:
                c_res = run_sql_query(conn, custom_sql)
                st.dataframe(c_res, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")

    # Tab 3: Schema & Raw Data
    with tab3:
        st.subheader("Dataset Columns & Schema")
        st.markdown("**Available Table Columns:**")
        cols_html = "".join([f'<span class="schema-pill">{c} ({clean_df[c].dtype})</span>' for c in clean_df.columns])
        st.markdown(cols_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Raw Ingested Data Preview (First 100 rows)")
        st.dataframe(clean_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()

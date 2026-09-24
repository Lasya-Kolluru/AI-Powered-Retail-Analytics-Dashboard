"""
Universal AI-Powered Data Analytics Dashboard
Built with Streamlit, SQLite, Pandas, and the Google GenAI SDK.
Fully generalized for ANY domain, ANY column count, ANY row count, and ANY schema.
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Page Configuration & Modern Glassmorphic CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal AI Data Analytics Dashboard",
    page_icon="⚡",
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
        border: 1px solid rgba(124, 77, 255, 0.35);
        border-radius: 12px;
        padding: 22px;
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
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 6px;
        padding: 3px 9px;
        margin: 3px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Universal Data Ingestion & Schema Normalization
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load default dataset bundled in repository or return fallback."""
    default_path = os.path.join(os.path.dirname(__file__), "data", "sample_retail_data.csv")
    if os.path.exists(default_path):
        try:
            return pd.read_csv(default_path)
        except Exception:
            pass
    # Fallback tabular dataset
    return pd.DataFrame({
        "Transaction_ID": [1, 2, 3, 4, 5],
        "Product_Category": ["Beauty", "Clothing", "Electronics", "Clothing", "Beauty"],
        "Quantity": [3, 2, 1, 1, 2],
        "Price_per_Unit": [50, 500, 30, 500, 50],
        "Total_Amount": [150, 1000, 30, 500, 100],
    })


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Sanitize column names into standard, collision-free SQL identifiers
    while keeping a map of original-to-clean names for transparent UI display.
    """
    df_clean = df.copy()
    new_cols = []
    col_mapping = {}
    seen = {}
    
    for orig_col in df_clean.columns:
        # Replace non-alphanumeric chars with underscore
        clean = re.sub(r"[^0-9a-zA-Z_]+", "_", str(orig_col).strip())
        clean = re.sub(r"^_+|_+$", "", clean)
        if not clean or clean[0].isdigit():
            clean = f"col_{clean}"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
            
        new_cols.append(clean)
        col_mapping[clean] = orig_col
        
    df_clean.columns = new_cols
    return df_clean, col_mapping


def profile_dataset(df: pd.DataFrame):
    """
    Universally profile any tabular dataset regardless of domain, columns, or rows.
    Identifies numeric, categorical, date/time, and ID columns dynamically.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Categorical candidates
    categorical_cols = [c for c in all_cols if c not in numeric_cols]
    
    # Check if any numeric column has low cardinality (e.g., ratings, status codes, age groups)
    for c in numeric_cols:
        if df[c].nunique() <= 15 and c not in categorical_cols:
            categorical_cols.append(c)

    # Date/Time detection
    date_candidates = []
    for c in all_cols:
        if any(term in c.lower() for term in ["date", "time", "timestamp", "year", "month", "day", "created", "period"]):
            date_candidates.append(c)

    # Key Value Metric candidate (Revenue, Amount, Sales, Price, Score, Value, etc.)
    val_keywords = ["total_amount", "revenue", "amount", "sales", "price", "total", "cost", "score", "value", "profit", "spend"]
    primary_metric = None
    for kw in val_keywords:
        for c in numeric_cols:
            if kw in c.lower():
                primary_metric = c
                break
        if primary_metric:
            break
    if not primary_metric and numeric_cols:
        primary_metric = numeric_cols[-1]

    # Secondary Metric candidate (Quantity, Units, Count, Volume, etc.)
    vol_keywords = ["quantity", "qty", "units_sold", "units", "count", "volume", "items", "num"]
    secondary_metric = None
    for kw in vol_keywords:
        for c in numeric_cols:
            if kw in c.lower() and c != primary_metric:
                secondary_metric = c
                break
        if secondary_metric:
            break
    if not secondary_metric:
        for c in numeric_cols:
            if c != primary_metric:
                secondary_metric = c
                break

    # Primary Dimension candidate (Category, Product, Department, Region, Segment, Type, etc.)
    dim_keywords = ["category", "product", "item", "department", "segment", "region", "country", "type", "name", "gender", "status"]
    primary_dimension = None
    for kw in dim_keywords:
        for c in categorical_cols:
            if kw in c.lower():
                primary_dimension = c
                break
        if primary_dimension:
            break
    if not primary_dimension and categorical_cols:
        # Pick dimension with balanced cardinality (2 to 50 unique items)
        reasonable = [c for c in categorical_cols if 1 < df[c].nunique() <= 50]
        primary_dimension = reasonable[0] if reasonable else categorical_cols[0]

    # Secondary Dimension candidate
    secondary_dimension = None
    for c in categorical_cols:
        if c != primary_dimension:
            secondary_dimension = c
            break

    # Calculate Data Health Score (% non-null)
    total_cells = df.size
    non_null_cells = df.count().sum()
    data_health = (non_null_cells / total_cells * 100) if total_cells > 0 else 100.0

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_candidates,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "primary_dimension": primary_dimension,
        "secondary_dimension": secondary_dimension,
        "data_health": data_health,
    }


def init_sqlite_db(df: pd.DataFrame) -> tuple[sqlite3.Connection, pd.DataFrame, dict]:
    """Store sanitized DataFrame in an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    clean_df, col_mapping = sanitize_dataframe(df)
    clean_df.to_sql("universal_data", conn, index=False, if_exists="replace")
    # Also create an alias view for backward compatibility
    conn.execute("CREATE VIEW IF NOT EXISTS retail_sales AS SELECT * FROM universal_data;")
    return conn, clean_df, col_mapping


def run_sql_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Safely execute any SQL query against SQLite."""
    return pd.read_sql_query(query, conn)


# -----------------------------------------------------------------------------
# 3. Gemini AI Universal Analysis Engine (google-genai SDK)
# -----------------------------------------------------------------------------
def get_gemini_api_key(sidebar_key: str) -> str:
    """Hierarchical key resolution: Streamlit secrets -> Environment -> Sidebar input."""
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        return st.secrets["GEMINI_API_KEY"].strip()
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY").strip()
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()
    return ""


def call_gemini_with_fallback(api_key: str, system_instruction: str, user_prompt: str) -> str:
    """Call Google GenAI SDK with multi-model fallback to bypass transient high-demand limits."""
    if not api_key:
        return "⚠️ Gemini API Key not found. Please provide an API key in the sidebar or via `.streamlit/secrets.toml`."

    client = genai.Client(api_key=api_key)
    candidate_models = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-flash-latest"]
    last_err = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
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


def generate_sql_from_natural_language(api_key: str, natural_query: str, schema_info: str) -> str:
    """Use Gemini to translate any natural language question into valid SQLite query."""
    sys_prompt = (
        "You are an expert SQLite SQL developer. Given a table schema named `universal_data`, "
        "generate a single valid SQLite query that answers the user's question. "
        "Return ONLY the SQL statement enclosed in ```sql ... ``` code block. Do NOT include markdown text outside the code block."
    )
    user_prompt = f"Table: universal_data\nColumns and types:\n{schema_info}\n\nQuestion: {natural_query}"
    raw_response = call_gemini_with_fallback(api_key, sys_prompt, user_prompt)
    
    # Extract query from markdown if present
    match = re.search(r"```sql\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_response.strip().replace("`", "")


def analyze_with_gemini(
    api_key: str,
    context_title: str,
    sql_query: str,
    result_df: pd.DataFrame,
    schema_summary: str,
    custom_user_prompt: str = "",
) -> str:
    """Generate universal business intelligence and strategic recommendations."""
    table_str = result_df.head(50).to_string(index=False)

    system_instruction = (
        "You are an elite Principal Business Intelligence Consultant and Senior Data Strategist. "
        "Analyze the provided SQL query results derived from the user's dataset. "
        "Tailor your insights dynamically to the nature of the data (retail, sales, finance, operations, HR, etc.).\n"
        "Structure your output cleanly with:\n"
        "1. Executive Summary & Core Observations (patterns, top-performers, anomalies, concentration)\n"
        "2. 3 High-Impact Actionable Recommendations (prioritized, practical, high ROI)\n"
        "3. Operational Risks & Next Steps\n"
        "Use bolding for metrics and concise bullet points."
    )

    user_content = f"""
Dataset Schema:
{schema_summary}

Analysis Focus: {context_title}
Executed SQL Query:
```sql
{sql_query}
```

SQL Query Results ({len(result_df)} rows total):
{table_str}

Additional User Focus / Question:
{custom_user_prompt if custom_user_prompt else "Provide the standard 3 actionable strategic recommendations based on these results."}
"""

    return call_gemini_with_fallback(api_key, system_instruction, user_content)


# -----------------------------------------------------------------------------
# 4. Streamlit Application Layout
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">⚡ Universal AI Data Analytics & SQL Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Adaptive In-Memory SQL Processing + Gemini AI Analyst • Works with any CSV, any columns, and any rows</div>',
        unsafe_allow_html=True,
    )

    # --- Sidebar: Configuration & Universal File Ingestion ---
    st.sidebar.header("⚙️ Configuration & Data Source")

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
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Any domain, size, columns)", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(raw_df):,} rows × {len(raw_df.columns)} columns")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            raw_df = load_default_data()
    else:
        raw_df = load_default_data()
        st.sidebar.info(f"ℹ️ Using default dataset ({len(raw_df):,} rows × {len(raw_df.columns)} columns)")

    # Universal normalization and in-memory SQLite store
    conn, clean_df, col_mapping = init_sqlite_db(raw_df)
    profile = profile_dataset(clean_df)

    p_metric = profile["primary_metric"]
    s_metric = profile["secondary_metric"]
    p_dim = profile["primary_dimension"]
    s_dim = profile["secondary_dimension"]
    num_cols = profile["numeric_cols"]
    cat_cols = profile["categorical_cols"]
    date_cols = profile["date_cols"]

    # --- Dynamic Universal KPI Header ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Records", f"{len(clean_df):,}")

    with kpi2:
        if p_metric:
            total_val = clean_df[p_metric].sum()
            label = f"Total {col_mapping.get(p_metric, p_metric)}"
            is_currency = any(w in p_metric.lower() for w in ["amount", "revenue", "price", "sales", "cost", "total"])
            prefix = "$" if is_currency else ""
            st.metric(label, f"{prefix}{total_val:,.2f}")
        else:
            st.metric("Total Dimensions", f"{len(cat_cols)} columns")

    with kpi3:
        if s_metric and s_metric != p_metric:
            total_vol = clean_df[s_metric].sum()
            label = f"Total {col_mapping.get(s_metric, s_metric)}"
            st.metric(label, f"{total_vol:,.0f}")
        elif p_dim:
            st.metric(f"Unique {col_mapping.get(p_dim, p_dim)}", f"{clean_df[p_dim].nunique():,}")
        else:
            st.metric("Data Completeness", f"{profile['data_health']:.1f}%")

    with kpi4:
        if p_metric and len(clean_df) > 0:
            avg_val = clean_df[p_metric].mean()
            label = f"Avg {col_mapping.get(p_metric, p_metric)}"
            is_currency = any(w in p_metric.lower() for w in ["amount", "revenue", "price", "sales", "cost", "total"])
            prefix = "$" if is_currency else ""
            st.metric(label, f"{prefix}{avg_val:,.2f}")
        else:
            st.metric("Total Columns", f"{len(clean_df.columns)}")

    st.markdown("---")

    schema_summary = ", ".join([f"{c} ({clean_df[c].dtype})" for c in clean_df.columns])

    # --- Synthesize Dynamic Preset Queries based on actual columns ---
    preset_queries = {}

    if p_dim and p_metric:
        preset_queries[f"📊 Performance Breakdown by {col_mapping.get(p_dim, p_dim)}"] = (
            f"SELECT {p_dim}, "
            f"COUNT(*) AS Record_Count, "
            f"ROUND(SUM({p_metric}), 2) AS Total_{p_metric}, "
            f"ROUND(AVG({p_metric}), 2) AS Avg_{p_metric} "
            f"FROM universal_data GROUP BY {p_dim} ORDER BY Total_{p_metric} DESC LIMIT 15;"
        )

    if s_dim and p_metric and s_dim != p_dim:
        preset_queries[f"🌍 Distribution across {col_mapping.get(s_dim, s_dim)}"] = (
            f"SELECT {s_dim}, "
            f"COUNT(*) AS Total_Volume, "
            f"ROUND(SUM({p_metric}), 2) AS Total_{p_metric}, "
            f"ROUND(AVG({p_metric}), 2) AS Avg_{p_metric} "
            f"FROM universal_data GROUP BY {s_dim} ORDER BY Total_{p_metric} DESC LIMIT 15;"
        )

    if p_dim and s_dim and p_metric and p_dim != s_dim:
        preset_queries[f"👥 Cross-Tabulation: {col_mapping.get(p_dim, p_dim)} × {col_mapping.get(s_dim, s_dim)}"] = (
            f"SELECT {p_dim}, {s_dim}, "
            f"COUNT(*) AS Record_Count, "
            f"ROUND(SUM({p_metric}), 2) AS Total_{p_metric} "
            f"FROM universal_data GROUP BY {p_dim}, {s_dim} ORDER BY Total_{p_metric} DESC LIMIT 20;"
        )

    if p_metric:
        preset_queries[f"📈 Statistical Summary of {col_mapping.get(p_metric, p_metric)}"] = (
            f"SELECT "
            f"ROUND(MIN({p_metric}), 2) AS Minimum, "
            f"ROUND(AVG({p_metric}), 2) AS Average, "
            f"ROUND(MAX({p_metric}), 2) AS Maximum, "
            f"ROUND(SUM({p_metric}), 2) AS Total_Sum, "
            f"COUNT(*) AS Sample_Count "
            f"FROM universal_data;"
        )

    if p_dim and not p_metric:
        preset_queries[f"📋 Top Most Frequent {col_mapping.get(p_dim, p_dim)}"] = (
            f"SELECT {p_dim}, COUNT(*) AS Occurrences "
            f"FROM universal_data GROUP BY {p_dim} ORDER BY Occurrences DESC LIMIT 15;"
        )

    if not preset_queries:
        preset_queries["📋 Data Preview (First 15 Rows)"] = "SELECT * FROM universal_data LIMIT 15;"

    # --- Dashboard Navigation Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dynamic SQL Analytics & AI",
        "💬 Ask AI in Plain English (Text-to-SQL)",
        "🛠️ Visual Query Builder",
        "🔍 Schema & Data Profiling",
    ])

    # Tab 1: Dynamic SQL Queries & Gemini Insights
    with tab1:
        st.subheader("Automated SQL Analytics Engine")
        st.caption("Analytical queries automatically generated based on discovered columns and data types.")

        selected_query_label = st.selectbox("Choose an Analytical Query:", list(preset_queries.keys()), index=0)
        query_sql = preset_queries[selected_query_label]

        with st.expander("🔎 View Executed SQL Query", expanded=False):
            st.code(query_sql, language="sql")

        try:
            results_df = run_sql_query(conn, query_sql)
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
            results_df = pd.DataFrame()

        if not results_df.empty:
            col_data, col_chart = st.columns([1, 1])
            with col_data:
                st.markdown(f"**Results Table ({len(results_df)} records)**")
                st.dataframe(results_df, use_container_width=True)

            with col_chart:
                st.markdown("**Visual Distribution**")
                res_num = results_df.select_dtypes(include=[np.number]).columns.tolist()
                res_cat = results_df.select_dtypes(exclude=[np.number]).columns.tolist()

                if res_cat and res_num:
                    chart_df = results_df.set_index(res_cat[0])[[res_num[0]]]
                    st.bar_chart(chart_df, color="#1E88E5")
                elif res_num:
                    st.bar_chart(results_df[res_num[0]], color="#7C4DFF")
                else:
                    st.info("No numeric series available for visual charting.")

            st.markdown("---")
            st.subheader("🤖 Gemini Strategic Business Insights")
            user_prompt_extra = st.text_input(
                "Custom Strategic Question / Focus (Optional):",
                placeholder="e.g., Which segments yield the highest profitability or need promotional discounts?",
            )

            if st.button("✨ Generate AI Insights", type="primary"):
                if not gemini_key:
                    st.error("Please configure your Gemini API Key in the sidebar or `.streamlit/secrets.toml`.")
                else:
                    with st.spinner("Analyzing dataset with Gemini AI..."):
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
                    """
                    <div class="ai-box">
                        <span class="badge-sql">Gemini Strategic Recommendations</span>
                        <br/>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["last_ai_insight"])

    # Tab 2: Natural Language Querying (Text to SQL)
    with tab2:
        st.subheader("💬 Ask Your Data in Plain English")
        st.caption("Type any question about your dataset. Gemini will generate the SQL, execute it against your data, and show the answer.")

        user_natural_query = st.text_input(
            "What would you like to know about this dataset?",
            placeholder="e.g., Which category has the highest average amount? Or list top 5 customers with most orders.",
        )

        if st.button("🚀 Ask Question & Run SQL", type="primary", key="btn_nl_sql"):
            if not gemini_key:
                st.error("Please configure your Gemini API Key to use Text-to-SQL.")
            elif not user_natural_query.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Translating question to SQL and querying..."):
                    generated_sql = generate_sql_from_natural_language(
                        api_key=gemini_key,
                        natural_query=user_natural_query,
                        schema_info=schema_summary,
                    )
                    st.markdown("**Generated SQL Query:**")
                    st.code(generated_sql, language="sql")

                    try:
                        nl_result = run_sql_query(conn, generated_sql)
                        st.success(f"Returned {len(nl_result)} rows")
                        st.dataframe(nl_result, use_container_width=True)

                        # Auto-generate natural language summary
                        with st.spinner("Formulating AI answer..."):
                            nl_analysis = analyze_with_gemini(
                                api_key=gemini_key,
                                context_title=f"User Question: {user_natural_query}",
                                sql_query=generated_sql,
                                result_df=nl_result,
                                schema_summary=schema_summary,
                                custom_user_prompt="Provide a direct, concise executive answer to the user's specific question based on these results.",
                            )
                            st.markdown(nl_analysis)

                    except Exception as e:
                        st.error(f"SQL Execution Error on generated query: {e}")

    # Tab 3: Visual Query Builder & Free-Form Playground
    with tab3:
        st.subheader("🛠️ Interactive Visual Query Builder")
        st.caption("Slice, group, and aggregate ANY column in your dataset dynamically.")

        all_cols = clean_df.columns.tolist()
        numeric_selection = num_cols if num_cols else all_cols

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            grp_col = st.selectbox(
                "Group By (Dimension):",
                all_cols,
                index=all_cols.index(p_dim) if p_dim in all_cols else 0,
            )
        with q2:
            val_col_sel = st.selectbox(
                "Aggregate Target (Metric):",
                numeric_selection,
                index=numeric_selection.index(p_metric) if p_metric in numeric_selection else 0,
            )
        with q3:
            fn_sel = st.selectbox("Function:", ["SUM", "AVG", "COUNT", "MAX", "MIN"], index=0)
        with q4:
            limit_sel = st.slider("Row Limit:", min_value=5, max_value=100, value=15)

        built_sql = (
            f"SELECT {grp_col}, {fn_sel}({val_col_sel}) AS {fn_sel.lower()}_{val_col_sel}, COUNT(*) AS total_records "
            f"FROM universal_data GROUP BY {grp_col} ORDER BY {fn_sel.lower()}_{val_col_sel} DESC LIMIT {limit_sel};"
        )
        st.code(built_sql, language="sql")

        if st.button("Run Visual Query", key="btn_run_builder"):
            try:
                b_df = run_sql_query(conn, built_sql)
                st.dataframe(b_df, use_container_width=True)
                if len(b_df.columns) >= 2:
                    st.bar_chart(b_df.set_index(grp_col)[[f"{fn_sel.lower()}_{val_col_sel}"]], color="#0284c7")
            except Exception as e:
                st.error(f"Query Error: {e}")

        st.markdown("---")
        st.subheader("💻 Free-Form SQL Editor")
        st.caption("Write arbitrary SQL statements directly against `universal_data`.")
        custom_query = st.text_area("SQL Statement:", value="SELECT * FROM universal_data LIMIT 10;", height=100)

        if st.button("Execute SQL", key="btn_custom_sql"):
            try:
                c_df = run_sql_query(conn, custom_query)
                st.success(f"Executed successfully ({len(c_df)} rows)")
                st.dataframe(c_df, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")

    # Tab 4: Schema & Data Profiling
    with tab4:
        st.subheader("Dataset Profiling & Schema Architecture")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Total Columns", len(clean_df.columns))
        with p2:
            st.metric("Numeric Columns", len(num_cols))
        with p3:
            st.metric("Categorical Columns", len(cat_cols))

        st.markdown("**Discovered Columns & Data Types:**")
        pill_html = "".join([f'<span class="schema-pill">{c} ({clean_df[c].dtype})</span>' for c in clean_df.columns])
        st.markdown(pill_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Column Missing Values & Cardinality")
        summary_stats = []
        for c in clean_df.columns:
            summary_stats.append({
                "Column (Cleaned)": c,
                "Original Header": col_mapping.get(c, c),
                "Data Type": str(clean_df[c].dtype),
                "Unique Values": clean_df[c].nunique(),
                "Null Values": int(clean_df[c].isnull().sum()),
                "% Complete": f"{(1 - clean_df[c].isnull().mean())*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True)

        st.markdown("---")
        st.subheader("Data Preview (First 100 Rows)")
        st.dataframe(clean_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()

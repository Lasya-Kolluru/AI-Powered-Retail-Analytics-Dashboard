"""
AI-Powered Retail Analytics Dashboard
Built with Streamlit, SQLite, Pandas, and the Google GenAI SDK.
"""

import os
import sqlite3
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Retail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Sleek gradient header & metric card accents */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
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
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Helper Functions: Data Ingestion & SQLite Engine
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load bundled sample retail dataset."""
    default_path = os.path.join(os.path.dirname(__file__), "data", "sample_retail_data.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    # Inline fallback if CSV file is moved
    sample_records = [
        {"Transaction_ID": "TXN-101", "Date": "2024-01-05", "Product_Category": "Electronics", "Product_Name": "UltraHD 4K TV", "Region": "North", "Units_Sold": 15, "Revenue": 9750.0},
        {"Transaction_ID": "TXN-102", "Date": "2024-01-08", "Product_Category": "Apparel", "Product_Name": "Merino Sweater", "Region": "West", "Units_Sold": 45, "Revenue": 3825.0},
        {"Transaction_ID": "TXN-103", "Date": "2024-01-12", "Product_Category": "Electronics", "Product_Name": "Noise-Canceling Headset", "Region": "South", "Units_Sold": 60, "Revenue": 10800.0},
        {"Transaction_ID": "TXN-104", "Date": "2024-01-15", "Product_Category": "Home & Kitchen", "Product_Name": "Espresso Machine", "Region": "East", "Units_Sold": 18, "Revenue": 5760.0},
        {"Transaction_ID": "TXN-105", "Date": "2024-01-20", "Product_Category": "Footwear", "Product_Name": "Running Shoes", "Region": "North", "Units_Sold": 35, "Revenue": 4200.0},
    ]
    return pd.DataFrame(sample_records)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names to standard SQL identifiers."""
    df_clean = df.copy()
    df_clean.columns = [c.strip().replace(" ", "_").replace("-", "_").replace(".", "_") for c in df_clean.columns]
    return df_clean


def init_sqlite_db(df: pd.DataFrame) -> sqlite3.Connection:
    """Store DataFrame in an in-memory SQLite database connection."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    sanitized_df = sanitize_dataframe(df)
    sanitized_df.to_sql("retail_sales", conn, index=False, if_exists="replace")
    return conn


def run_sql_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Execute SQL query safely and return DataFrame."""
    return pd.read_sql_query(query, conn)


# -----------------------------------------------------------------------------
# 3. Gemini AI Analysis Engine (google-genai SDK)
# -----------------------------------------------------------------------------
def get_gemini_api_key(sidebar_key: str) -> str:
    """Resolve Gemini API key with priority: Streamlit secrets -> Environment -> Sidebar input."""
    # 1. Streamlit Secrets (used in deployment)
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        return st.secrets["GEMINI_API_KEY"].strip()
    # 2. Local Environment variable
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY").strip()
    # 3. Sidebar user override
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()
    return ""


def analyze_with_gemini(api_key: str, context_title: str, sql_query: str, result_df: pd.DataFrame, custom_user_prompt: str = "") -> str:
    """Invoke the Gemini model using the official google-genai SDK."""
    if not api_key:
        return "⚠️ Gemini API Key not found. Please provide an API key in the sidebar or via `.streamlit/secrets.toml`."

    client = genai.Client(api_key=api_key)

    # Format tabular result into clean formatted string without requiring tabulate
    table_str = result_df.to_string(index=False)

    system_instruction = (
        "You are an elite Retail Analytics Consultant and Lead Business Analyst. "
        "Analyze the SQL query results extracted from the retail transactions database. "
        "Structure your response professionally with:\n"
        "1. Executive Summary & Key Observations (high-level revenue patterns, outliers, or trends)\n"
        "2. 3 High-Impact Actionable Business Recommendations (clear, practical steps with expected commercial impact)\n"
        "3. Risk Factors & Operational Considerations\n"
        "Use bolding for metrics, bullet points, and concise, professional prose."
    )

    user_content = f"""
Metric Context: {context_title}
Executed SQL Query:
```sql
{sql_query}
```

SQL Query Results Data:
{table_str}

Additional User Question / Focus:
{custom_user_prompt if custom_user_prompt else "Provide the standard 3 actionable business recommendations based on these metrics."}
"""

    # Model cascade to gracefully handle transient 503 high-demand spikes
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
    # Header Banner
    st.markdown('<div class="main-title">🛍️ AI-Powered Retail Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Automated Retail Intelligence: In-Memory SQL Processing + Gemini Data Analyst</div>',
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
    st.sidebar.subheader("📂 Retail Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Retail CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(raw_df)} rows from upload")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            raw_df = load_default_data()
    else:
        raw_df = load_default_data()
        st.sidebar.info("ℹ️ Using default retail dataset (30 records)")

    # Initialize in-memory SQLite connection
    conn = init_sqlite_db(raw_df)

    # --- KPI Summary Bar ---
    cols = raw_df.columns.str.lower()
    has_revenue = "revenue" in cols
    has_units = "units_sold" in [c.lower().replace(" ", "_") for c in raw_df.columns]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Transactions", f"{len(raw_df):,}")
    with kpi2:
        if has_revenue:
            rev_col = raw_df.columns[cols == "revenue"][0]
            total_rev = raw_df[rev_col].sum()
            st.metric("Total Gross Revenue", f"${total_rev:,.2f}")
        else:
            st.metric("Total Gross Revenue", "N/A")
    with kpi3:
        if has_units:
            units_col = [c for c in raw_df.columns if "units" in c.lower()][0]
            total_units = raw_df[units_col].sum()
            st.metric("Total Units Sold", f"{int(total_units):,}")
        else:
            st.metric("Total Units Sold", "N/A")
    with kpi4:
        if has_revenue and len(raw_df) > 0:
            avg_txn = total_rev / len(raw_df)
            st.metric("Avg Order Value (AOV)", f"${avg_txn:,.2f}")
        else:
            st.metric("Avg Order Value", "N/A")

    st.markdown("---")

    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 SQL Analytics & AI Insights", "🔍 Raw Data & Schema", "💻 SQL Custom Playground"])

    # Tab 1: Built-in Analytical Queries & AI Insights
    with tab1:
        st.subheader("Automated SQL Analytics Engine")
        
        preset_queries = {
            "🏆 Top 5 Products by Revenue": (
                "SELECT Product_Name, Product_Category, SUM(Revenue) AS Total_Revenue, SUM(Units_Sold) AS Total_Units "
                "FROM retail_sales GROUP BY Product_Name ORDER BY Total_Revenue DESC LIMIT 5;"
            ),
            "🌍 Regional Sales & Revenue Breakdown": (
                "SELECT Region, SUM(Revenue) AS Total_Revenue, SUM(Units_Sold) AS Total_Units, "
                "ROUND(AVG(Revenue), 2) AS Avg_Transaction_Value "
                "FROM retail_sales GROUP BY Region ORDER BY Total_Revenue DESC;"
            ),
            "👥 Customer Segment & Payment Method Performance": (
                "SELECT Customer_Segment, Payment_Method, SUM(Revenue) AS Total_Revenue, COUNT(Transaction_ID) AS Transactions "
                "FROM retail_sales GROUP BY Customer_Segment, Payment_Method ORDER BY Total_Revenue DESC;"
            ),
            "📦 Category-level Volume & Yield": (
                "SELECT Product_Category, SUM(Revenue) AS Total_Revenue, SUM(Units_Sold) AS Total_Units, "
                "ROUND(SUM(Revenue) * 1.0 / SUM(Units_Sold), 2) AS Effective_Unit_Price "
                "FROM retail_sales GROUP BY Product_Category ORDER BY Total_Revenue DESC;"
            ),
        }

        selected_query_label = st.selectbox(
            "Select an Analytical Query:",
            list(preset_queries.keys()),
            index=0,
        )

        query_sql = preset_queries[selected_query_label]

        # Display Code & Execute Query
        with st.expander("🔎 View Executed SQL Query", expanded=False):
            st.code(query_sql, language="sql")

        try:
            results_df = run_sql_query(conn, query_sql)
        except Exception as e:
            st.error(f"SQL execution error: {e}")
            return

        col_data, col_chart = st.columns([1, 1])
        with col_data:
            st.markdown(f"**Query Results Table ({len(results_df)} records)**")
            st.dataframe(results_df, use_container_width=True)

        with col_chart:
            st.markdown("**Visual Distribution**")
            # Automatically chart first numeric column against first categorical column
            numeric_cols = results_df.select_dtypes(include=["number"]).columns.tolist()
            text_cols = results_df.select_dtypes(include=["object"]).columns.tolist()
            if numeric_cols and text_cols:
                chart_df = results_df.set_index(text_cols[0])[[numeric_cols[0]]]
                st.bar_chart(chart_df, color="#1E88E5")
            elif numeric_cols:
                st.bar_chart(results_df[numeric_cols[0]], color="#7C4DFF")
            else:
                st.info("No numeric series available for auto-charting.")

        # AI Recommendation Section
        st.markdown("---")
        st.subheader("🤖 AI Data Analyst Insights")
        st.caption("Powered by Google GenAI (Gemini) acting on in-memory SQL query results.")

        user_prompt_extra = st.text_input(
            "Add specific instructions or questions for the AI Analyst (Optional):",
            placeholder="e.g., Focus on which underperforming regions require discount promotions...",
        )

        generate_col, status_col = st.columns([1, 3])
        with generate_col:
            generate_button = st.button("✨ Generate AI Insights", type="primary", use_container_width=True)

        if generate_button:
            if not gemini_key:
                st.error("Please configure your Gemini API Key in the sidebar or `.streamlit/secrets.toml` to generate insights.")
            else:
                with st.spinner("Analyzing SQL metrics with Gemini..."):
                    ai_response = analyze_with_gemini(
                        api_key=gemini_key,
                        context_title=selected_query_label,
                        sql_query=query_sql,
                        result_df=results_df,
                        custom_user_prompt=user_prompt_extra,
                    )
                    st.session_state["last_ai_insight"] = ai_response

        if "last_ai_insight" in st.session_state:
            st.markdown(
                f"""
                <div class="ai-box">
                    <span class="badge-sql">Gemini 2.5 Flash Retail Analyst</span>
                    <br/>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state["last_ai_insight"])

    # Tab 2: Raw Data & Schema
    with tab2:
        st.subheader("Raw Ingested Dataset")
        st.dataframe(raw_df, use_container_width=True)
        
        st.subheader("SQLite Schema (`retail_sales`)")
        schema_df = run_sql_query(conn, "PRAGMA table_info(retail_sales);")
        st.dataframe(schema_df[["cid", "name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)

    # Tab 3: Custom SQL Query Playground
    with tab3:
        st.subheader("💻 Interactive SQL Playground")
        st.caption("Write arbitrary queries against the `retail_sales` in-memory table.")
        
        default_custom = "SELECT Region, COUNT(*) as Orders, SUM(Revenue) as Total_Revenue FROM retail_sales GROUP BY Region;"
        custom_sql = st.text_area("SQL Statement:", value=default_custom, height=120)
        
        if st.button("Run SQL Query", key="run_custom_sql"):
            try:
                custom_res = run_sql_query(conn, custom_sql)
                st.success(f"Returned {len(custom_res)} rows")
                st.dataframe(custom_res, use_container_width=True)
                
                # Option to trigger AI on custom query result
                if gemini_key and st.button("✨ Analyze This Custom Query with AI", key="ai_custom"):
                    with st.spinner("Analyzing custom SQL results with Gemini..."):
                        custom_ai = analyze_with_gemini(
                            api_key=gemini_key,
                            context_title="Custom SQL Query Execution",
                            sql_query=custom_sql,
                            result_df=custom_res,
                        )
                        st.markdown(custom_ai)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")


if __name__ == "__main__":
    main()

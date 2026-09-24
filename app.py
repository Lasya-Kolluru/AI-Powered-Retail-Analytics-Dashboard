"""
Universal AI-Powered 3D Data Analytics & SQL Intelligence Dashboard
Enhanced with Interactive 3D Movable Visualizations, Three.js 3D Pipeline, Glassmorphism, and Gemini AI.
Fully adaptive: works with ANY CSV, ANY schema, and ANY column configuration.
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Page Configuration & 3D Glassmorphic Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal 3D AI Data Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* 3D Ambient Background & Lighting */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(30, 136, 229, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(124, 77, 255, 0.14) 0%, transparent 45%),
                    #0a0d14;
    }

    /* 3D Title with Neon Aurora Gradient */
    .main-title-3d {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 45%, #9d4edd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 4px 14px rgba(0, 210, 255, 0.25));
    }

    .sub-title-3d {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-bottom: 1.4rem;
    }

    /* 3D Glassmorphic Metric Cards */
    .metric-card-3d {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 18px 20px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6),
                    0 0 20px -2px rgba(30, 136, 229, 0.2),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.2);
        transform: perspective(1000px) translateZ(0);
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .metric-card-3d:hover {
        transform: perspective(1000px) translateY(-6px) translateZ(15px);
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.7),
                    0 0 30px 2px rgba(124, 77, 255, 0.35),
                    inset 0 1px 2px 0 rgba(255, 255, 255, 0.3);
        border-color: rgba(124, 77, 255, 0.45);
    }

    /* 3D Glass AI Box */
    .ai-box-3d {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.08) 0%, rgba(124, 77, 255, 0.09) 100%);
        border: 1px solid rgba(124, 77, 255, 0.35);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5),
                    0 0 25px -5px rgba(124, 77, 255, 0.25);
        margin-top: 18px;
    }

    .badge-3d {
        display: inline-block;
        background: linear-gradient(135deg, #0284c7 0%, #7c4dff 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
        margin-bottom: 12px;
    }

    .schema-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 3px 10px;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Interactive 3D Three.js Pipeline Component (Drag & Rotate in Real-Time 3D)
# -----------------------------------------------------------------------------
def render_3d_pipeline_hero():
    """Renders a movable, rotatable 3D cybernetic node system via Three.js."""
    three_js_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { margin: 0; overflow: hidden; background: transparent; }
            #canvas3d { width: 100%; height: 210px; display: block; }
            #instructions {
                position: absolute; bottom: 8px; right: 16px;
                color: rgba(255,255,255,0.45); font-family: monospace; font-size: 11px;
                pointer-events: none; user-select: none;
            }
        </style>
    </head>
    <body>
        <div id="canvas3d"></div>
        <div id="instructions">🖱️ Click & Drag to Orbit in 3D • Scroll to Zoom</div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            const container = document.getElementById('canvas3d');
            const width = container.clientWidth || window.innerWidth;
            const height = 210;

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(0, 1.2, 7.5);

            const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
            scene.add(ambientLight);
            const pointLight = new THREE.PointLight(0x00d2ff, 2.5, 50);
            pointLight.position.set(5, 5, 5);
            scene.add(pointLight);
            const purpleLight = new THREE.PointLight(0x9d4edd, 2.5, 50);
            purpleLight.position.set(-5, -3, 3);
            scene.add(purpleLight);

            // 3D Nodes representing the Architecture Pipeline
            const group = new THREE.Group();
            scene.add(group);

            const nodesData = [
                { x: -4.2, y: 0, z: 0, color: 0x00d2ff, label: "Raw Data" },
                { x: -1.4, y: 0.6, z: 0.5, color: 0x3a7bd5, label: "Pandas Engine" },
                { x: 1.4, y: -0.5, z: -0.5, color: 0x7c4dff, label: "In-Memory SQLite" },
                { x: 4.2, y: 0, z: 0, color: 0x00f5d4, label: "Gemini AI" }
            ];

            const nodeMeshes = [];
            nodesData.forEach(data => {
                const geom = new THREE.DodecahedronGeometry(0.55, 1);
                const mat = new THREE.MeshStandardMaterial({
                    color: data.color,
                    metalness: 0.7,
                    roughness: 0.2,
                    wireframe: false
                });
                const mesh = new THREE.Mesh(geom, mat);
                mesh.position.set(data.x, data.y, data.z);
                group.add(mesh);
                nodeMeshes.push(mesh);

                // Add wireframe outer shell
                const wireGeom = new THREE.DodecahedronGeometry(0.72, 1);
                const wireMat = new THREE.MeshBasicMaterial({ color: data.color, wireframe: true, transparent: true, opacity: 0.35 });
                const wireMesh = new THREE.Mesh(wireGeom, wireMat);
                mesh.add(wireMesh);
            });

            // 3D Connecting Beams
            for (let i = 0; i < nodesData.length - 1; i++) {
                const p1 = new THREE.Vector3(nodesData[i].x, nodesData[i].y, nodesData[i].z);
                const p2 = new THREE.Vector3(nodesData[i+1].x, nodesData[i+1].y, nodesData[i+1].z);
                const curve = new THREE.LineCurve3(p1, p2);
                const tubeGeom = new THREE.TubeGeometry(curve, 20, 0.04, 8, false);
                const tubeMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.45 });
                group.add(new THREE.Mesh(tubeGeom, tubeMat));
            }

            // Floating Cyber Particles
            const particleCount = 120;
            const particleGeom = new THREE.BufferGeometry();
            const positions = new Float32Array(particleCount * 3);
            for(let i=0; i<particleCount*3; i+=3){
                positions[i] = (Math.random() - 0.5) * 12;
                positions[i+1] = (Math.random() - 0.5) * 4;
                positions[i+2] = (Math.random() - 0.5) * 6;
            }
            particleGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const particleMat = new THREE.PointsMaterial({ color: 0x38bdf8, size: 0.06, transparent: true, opacity: 0.6 });
            const particleSystem = new THREE.Points(particleGeom, particleMat);
            group.add(particleSystem);

            // Mouse Interactive Orbiting
            let isDragging = false;
            let prevMouseX = 0;
            let prevMouseY = 0;

            window.addEventListener('mousedown', (e) => {
                isDragging = true;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            });

            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                const deltaX = e.clientX - prevMouseX;
                const deltaY = e.clientY - prevMouseY;
                group.rotation.y += deltaX * 0.008;
                group.rotation.x += deltaY * 0.008;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            });

            window.addEventListener('mouseup', () => { isDragging = false; });

            window.addEventListener('wheel', (e) => {
                camera.position.z += e.deltaY * 0.003;
                camera.position.z = Math.max(3.5, Math.min(12, camera.position.z));
            }, { passive: true });

            // Animation Loop
            function animate() {
                requestAnimationFrame(animate);
                if (!isDragging) {
                    group.rotation.y += 0.0035;
                }
                nodeMeshes.forEach((mesh, idx) => {
                    mesh.rotation.x += 0.01 * (idx % 2 === 0 ? 1 : -1);
                    mesh.rotation.y += 0.015;
                });
                renderer.render(scene, camera);
            }
            animate();

            window.addEventListener('resize', () => {
                const newWidth = container.clientWidth || window.innerWidth;
                camera.aspect = newWidth / height;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, height);
            });
        </script>
    </body>
    </html>
    """
    components.html(three_js_code, height=215)


# -----------------------------------------------------------------------------
# 3. Universal Schema Normalizer & Data Profiler
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
    return pd.DataFrame({
        "Transaction_ID": [1, 2, 3, 4, 5],
        "Product_Category": ["Beauty", "Clothing", "Electronics", "Clothing", "Beauty"],
        "Quantity": [3, 2, 1, 1, 2],
        "Price_per_Unit": [50, 500, 30, 500, 50],
        "Total_Amount": [150, 1000, 30, 500, 100],
    })


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sanitize all column names into standard SQL identifiers and record mapping."""
    df_clean = df.copy()
    new_cols = []
    col_mapping = {}
    seen = {}
    for orig_col in df_clean.columns:
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
    """Dynamically profile any tabular dataset regardless of column count or domain."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    for c in numeric_cols:
        if df[c].nunique() <= 15 and c not in categorical_cols:
            categorical_cols.append(c)

    # Key Value Metric candidate
    val_keywords = ["total_amount", "revenue", "amount", "sales", "price", "total", "cost", "value", "profit", "spend"]
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

    # Secondary Metric candidate
    vol_keywords = ["quantity", "qty", "units_sold", "units", "count", "volume", "items", "age"]
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

    # Primary Dimension candidate
    dim_keywords = ["category", "product", "item", "department", "segment", "region", "gender", "type", "name"]
    primary_dimension = None
    for kw in dim_keywords:
        for c in categorical_cols:
            if kw in c.lower():
                primary_dimension = c
                break
        if primary_dimension:
            break
    if not primary_dimension and categorical_cols:
        primary_dimension = categorical_cols[0]

    # Secondary Dimension candidate
    secondary_dimension = None
    for c in categorical_cols:
        if c != primary_dimension:
            secondary_dimension = c
            break

    total_cells = df.size
    non_null_cells = df.count().sum()
    data_health = (non_null_cells / total_cells * 100) if total_cells > 0 else 100.0

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "primary_dimension": primary_dimension,
        "secondary_dimension": secondary_dimension,
        "data_health": data_health,
    }


def init_sqlite_db(df: pd.DataFrame) -> tuple[sqlite3.Connection, pd.DataFrame, dict]:
    """Store sanitized DataFrame into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    clean_df, col_mapping = sanitize_dataframe(df)
    clean_df.to_sql("universal_data", conn, index=False, if_exists="replace")
    conn.execute("CREATE VIEW IF NOT EXISTS retail_sales AS SELECT * FROM universal_data;")
    return conn, clean_df, col_mapping


def run_sql_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Execute SQL query safely and return DataFrame."""
    return pd.read_sql_query(query, conn)


# -----------------------------------------------------------------------------
# 4. Google GenAI SDK Engine
# -----------------------------------------------------------------------------
def get_gemini_api_key(sidebar_key: str) -> str:
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        return st.secrets["GEMINI_API_KEY"].strip()
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY").strip()
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()
    return ""


def call_gemini_with_fallback(api_key: str, system_instruction: str, user_prompt: str) -> str:
    """Multi-model fallback invocation across candidate Gemini models."""
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
    """Natural Language to SQLite SQL query translation."""
    sys_prompt = (
        "You are an expert SQLite SQL developer. Given table `universal_data`, "
        "generate a single valid SQLite query that answers the question. "
        "Return ONLY the SQL enclosed in ```sql ... ```. No additional conversation."
    )
    user_prompt = f"Table: universal_data\nSchema:\n{schema_info}\n\nQuestion: {natural_query}"
    raw = call_gemini_with_fallback(api_key, sys_prompt, user_prompt)
    match = re.search(r"```sql\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip().replace("`", "")


def analyze_with_gemini(
    api_key: str,
    context_title: str,
    sql_query: str,
    result_df: pd.DataFrame,
    schema_summary: str,
    custom_user_prompt: str = "",
) -> str:
    """Strategic business analysis with Gemini."""
    table_str = result_df.head(40).to_string(index=False)
    system_instruction = (
        "You are a Senior Strategic Data Intelligence Advisor. "
        "Analyze the provided SQL query results extracted from the dataset. "
        "Formulate your response with:\n"
        "1. Executive Summary & Core Observations (patterns, outliers, revenue/volume concentration)\n"
        "2. 3 High-Impact Actionable Business Recommendations (strategic, prioritized, practical)\n"
        "3. Strategic Opportunities & Operational Considerations\n"
        "Use bolding for metrics and concise bullet points."
    )

    user_content = f"""
Dataset Schema:
{schema_summary}

Focus: {context_title}
Executed SQL Query:
```sql
{sql_query}
```

Results Data ({len(result_df)} records):
{table_str}

Additional User Question:
{custom_user_prompt if custom_user_prompt else "Provide the standard 3 actionable strategic recommendations based on these metrics."}
"""
    return call_gemini_with_fallback(api_key, system_instruction, user_content)


# -----------------------------------------------------------------------------
# 5. Streamlit Application Layout
# -----------------------------------------------------------------------------
def main():
    # Header & 3D Interactive Hero
    st.markdown('<div class="main-title-3d">🌌 Universal 3D AI Data Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title-3d">Adaptive In-Memory SQL Processing + Interactive 3D Visualizations + Google Gemini AI</div>',
        unsafe_allow_html=True,
    )

    # Render Movable 3D Cyber Architecture Hero
    with st.expander("🌐 Interactive 3D Data Pipeline Architecture (Click to View / Drag to Orbit)", expanded=False):
        render_3d_pipeline_hero()

    # --- Sidebar: Configuration & Universal Ingestion ---
    st.sidebar.header("⚙️ Configuration & Data Source")

    sidebar_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Enter your Google AI Studio Gemini API key or configure in .streamlit/secrets.toml",
    )

    gemini_key = get_gemini_api_key(sidebar_api_key)
    if gemini_key:
        st.sidebar.success("✅ Gemini API Connected")
    else:
        st.sidebar.warning("⚠️ No Gemini API Key configured.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Any Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Any size, any columns)", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(raw_df):,} rows × {len(raw_df.columns)} cols")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            raw_df = load_default_data()
    else:
        raw_df = load_default_data()
        st.sidebar.info(f"ℹ️ Active dataset ({len(raw_df):,} rows × {len(raw_df.columns)} cols)")

    # Universal normalization
    conn, clean_df, col_mapping = init_sqlite_db(raw_df)
    profile = profile_dataset(clean_df)

    p_metric = profile["primary_metric"]
    s_metric = profile["secondary_metric"]
    p_dim = profile["primary_dimension"]
    s_dim = profile["secondary_dimension"]
    num_cols = profile["numeric_cols"]
    cat_cols = profile["categorical_cols"]

    # --- 3D Glassmorphic KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card-3d">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">TOTAL RECORDS</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-top: 4px;">{len(clean_df):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if p_metric:
            total_val = clean_df[p_metric].sum()
            label = col_mapping.get(p_metric, p_metric).replace("_", " ").title()
            is_currency = any(w in p_metric.lower() for w in ["amount", "revenue", "price", "sales", "cost", "total"])
            prefix = "$" if is_currency else ""
            val_str = f"{prefix}{total_val:,.2f}"
        else:
            label = "DIMENSIONS"
            val_str = f"{len(cat_cols)} categories"

        st.markdown(
            f"""
            <div class="metric-card-3d">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">TOTAL {label.upper()}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #38bdf8; margin-top: 4px;">{val_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        if s_metric and s_metric != p_metric:
            label = col_mapping.get(s_metric, s_metric).replace("_", " ").title()
            val_str = f"{clean_df[s_metric].sum():,.0f}"
        elif p_dim:
            label = f"UNIQUE {col_mapping.get(p_dim, p_dim).replace('_', ' ').title()}"
            val_str = f"{clean_df[p_dim].nunique():,}"
        else:
            label = "COMPLETENESS"
            val_str = f"{profile['data_health']:.1f}%"

        st.markdown(
            f"""
            <div class="metric-card-3d">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">{label.upper()}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #c084fc; margin-top: 4px;">{val_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        if p_metric and len(clean_df) > 0:
            label = f"AVG {col_mapping.get(p_metric, p_metric).replace('_', ' ').title()}"
            is_currency = any(w in p_metric.lower() for w in ["amount", "revenue", "price", "sales", "cost", "total"])
            prefix = "$" if is_currency else ""
            val_str = f"{prefix}{clean_df[p_metric].mean():,.2f}"
        else:
            label = "TOTAL COLUMNS"
            val_str = f"{len(clean_df.columns)}"

        st.markdown(
            f"""
            <div class="metric-card-3d">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">{label.upper()}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #4ade80; margin-top: 4px;">{val_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    schema_summary = ", ".join([f"{c} ({clean_df[c].dtype})" for c in clean_df.columns])

    # Dynamic Queries
    preset_queries = {}
    if p_dim and p_metric:
        preset_queries[f"📊 Performance Breakdown by {col_mapping.get(p_dim, p_dim)}"] = (
            f"SELECT {p_dim}, COUNT(*) AS Record_Count, "
            f"ROUND(SUM({p_metric}), 2) AS Total_{p_metric}, "
            f"ROUND(AVG({p_metric}), 2) AS Avg_{p_metric} "
            f"FROM universal_data GROUP BY {p_dim} ORDER BY Total_{p_metric} DESC LIMIT 15;"
        )

    if s_dim and p_metric and s_dim != p_dim:
        preset_queries[f"🌍 Distribution across {col_mapping.get(s_dim, s_dim)}"] = (
            f"SELECT {s_dim}, COUNT(*) AS Volume, "
            f"ROUND(SUM({p_metric}), 2) AS Total_{p_metric}, "
            f"ROUND(AVG({p_metric}), 2) AS Avg_{p_metric} "
            f"FROM universal_data GROUP BY {s_dim} ORDER BY Total_{p_metric} DESC LIMIT 15;"
        )

    if p_dim and s_dim and p_metric and p_dim != s_dim:
        preset_queries[f"👥 Cross-Tabulation: {col_mapping.get(p_dim, p_dim)} × {col_mapping.get(s_dim, s_dim)}"] = (
            f"SELECT {p_dim}, {s_dim}, COUNT(*) AS Count, "
            f"ROUND(SUM({p_metric}), 2) AS Total_{p_metric} "
            f"FROM universal_data GROUP BY {p_dim}, {s_dim} ORDER BY Total_{p_metric} DESC LIMIT 20;"
        )

    if p_metric:
        preset_queries[f"📈 Statistical Summary of {col_mapping.get(p_metric, p_metric)}"] = (
            f"SELECT ROUND(MIN({p_metric}), 2) AS Minimum, "
            f"ROUND(AVG({p_metric}), 2) AS Average, "
            f"ROUND(MAX({p_metric}), 2) AS Maximum, "
            f"ROUND(SUM({p_metric}), 2) AS Total_Sum, "
            f"COUNT(*) AS Sample_Count FROM universal_data;"
        )

    if not preset_queries:
        preset_queries["📋 Data Preview (First 15 Rows)"] = "SELECT * FROM universal_data LIMIT 15;"

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dynamic Analytics & AI Insights",
        "🌐 Movable 3D Visualizer Studio",
        "💬 Plain English Text-to-SQL",
        "🛠️ Visual Query Builder",
        "🔍 Schema & Data Profiling",
    ])

    # Tab 1: Analytics & Gemini
    with tab1:
        st.subheader("Automated SQL Analytics Engine")
        selected_query_label = st.selectbox("Select an Analytical Query:", list(preset_queries.keys()), index=0)
        query_sql = preset_queries[selected_query_label]

        with st.expander("🔎 View Executed SQL Query", expanded=False):
            st.code(query_sql, language="sql")

        try:
            results_df = run_sql_query(conn, query_sql)
        except Exception as e:
            st.error(f"SQL execution error: {e}")
            results_df = pd.DataFrame()

        if not results_df.empty:
            col_t, col_p = st.columns([1, 1])
            with col_t:
                st.markdown(f"**Results Table ({len(results_df)} records)**")
                st.dataframe(results_df, use_container_width=True)

            with col_p:
                st.markdown("**Dynamic Interactive Chart**")
                res_num = results_df.select_dtypes(include=[np.number]).columns.tolist()
                res_cat = results_df.select_dtypes(exclude=[np.number]).columns.tolist()

                if res_cat and res_num:
                    fig = px.bar(
                        results_df,
                        x=res_cat[0],
                        y=res_num[0],
                        color=res_num[0],
                        color_continuous_scale="Viridis",
                        template="plotly_dark",
                        title=f"{res_num[0]} by {res_cat[0]}",
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                elif res_num:
                    fig = px.histogram(results_df, x=res_num[0], template="plotly_dark", color_discrete_sequence=["#38bdf8"])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("🤖 Gemini Strategic AI Recommendations")
            user_prompt_extra = st.text_input(
                "Strategic Question / Target Area (Optional):",
                placeholder="e.g., How can we boost underperforming customer segments?",
            )

            if st.button("✨ Generate AI Insights", type="primary"):
                if not gemini_key:
                    st.error("Please configure your Gemini API Key in the sidebar or `.streamlit/secrets.toml`.")
                else:
                    with st.spinner("Gemini is analyzing the data..."):
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
                    <div class="ai-box-3d">
                        <span class="badge-3d">Gemini Strategic Intelligence</span>
                        <br/>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["last_ai_insight"])

    # Tab 2: Movable 3D Visualizer Studio
    with tab2:
        st.subheader("🌐 Movable 3D Visualizer Studio")
        st.caption("Interact with your data in 3-dimensional space. Click and drag to orbit, rotate 360°, scroll to zoom, and hover for coordinates.")

        all_cols = clean_df.columns.tolist()
        num_candidates = num_cols if len(num_cols) >= 2 else all_cols

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
        with ctrl1:
            x_axis = st.selectbox("X-Axis (3D):", all_cols, index=all_cols.index(p_dim) if p_dim in all_cols else 0)
        with ctrl2:
            y_axis = st.selectbox("Y-Axis (3D):", num_candidates, index=num_candidates.index(p_metric) if p_metric in num_candidates else 0)
        with ctrl3:
            z_axis = st.selectbox("Z-Axis (3D Depth):", num_candidates, index=1 if len(num_candidates) > 1 else 0)
        with ctrl4:
            color_dim = st.selectbox("Color Dimension:", all_cols, index=all_cols.index(p_dim) if p_dim in all_cols else 0)

        # 3D Plotly Interactive Scatter
        plot_sample = clean_df.head(2000) if len(clean_df) > 2000 else clean_df
        
        try:
            fig_3d = px.scatter_3d(
                plot_sample,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color_dim,
                opacity=0.85,
                color_continuous_scale="Plasma" if x_axis in num_cols else None,
                template="plotly_dark",
                title=f"Movable 3D Scatter Space: {x_axis} × {y_axis} × {z_axis} (Sample of {len(plot_sample):,} rows)",
            )
            fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color="rgba(255,255,255,0.4)")))
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(10, 15, 26, 0.7)", gridcolor="rgba(255, 255, 255, 0.1)"),
                    yaxis=dict(backgroundcolor="rgba(10, 15, 26, 0.7)", gridcolor="rgba(255, 255, 255, 0.1)"),
                    zaxis=dict(backgroundcolor="rgba(10, 15, 26, 0.7)", gridcolor="rgba(255, 255, 255, 0.1)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, b=0, t=40),
                height=650,
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.info(f"Could not generate 3D scatter with selected dimensions: {e}. Try picking numeric columns for Y and Z axes.")

        # Additional 3D Interactive Donut / Sunburst Drilldown
        st.markdown("---")
        st.subheader("🍩 Interactive Drill-Down Sunburst")
        if p_dim and s_dim and p_metric:
            try:
                sun_df = clean_df.groupby([p_dim, s_dim])[p_metric].sum().reset_index().head(60)
                fig_sun = px.sunburst(
                    sun_df,
                    path=[p_dim, s_dim],
                    values=p_metric,
                    color=p_metric,
                    color_continuous_scale="Turbo",
                    template="plotly_dark",
                    title=f"Hierarchical Sunburst: {col_mapping.get(p_dim, p_dim)} ➔ {col_mapping.get(s_dim, s_dim)} by {col_mapping.get(p_metric, p_metric)}",
                )
                fig_sun.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=520)
                st.plotly_chart(fig_sun, use_container_width=True)
            except Exception as e:
                st.info(f"Sunburst chart notice: {e}")

    # Tab 3: Text to SQL
    with tab3:
        st.subheader("💬 Ask Your Data in Plain English")
        st.caption("Ask anything about your dataset. Gemini will generate the exact SQL query, execute it in SQLite, and return the answer.")

        user_q = st.text_input(
            "Enter your question:",
            placeholder="e.g., Which product category generated the highest total revenue?",
        )

        if st.button("🚀 Ask Question & Run SQL", type="primary", key="btn_nl"):
            if not gemini_key:
                st.error("Please configure your Gemini API Key in the sidebar or `.streamlit/secrets.toml`.")
            elif not user_q.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Translating question to SQL..."):
                    gen_sql = generate_sql_from_natural_language(
                        api_key=gemini_key,
                        natural_query=user_q,
                        schema_info=schema_summary,
                    )
                    st.markdown("**Generated SQL:**")
                    st.code(gen_sql, language="sql")

                    try:
                        q_res = run_sql_query(conn, gen_sql)
                        st.success(f"Executed ({len(q_res)} rows returned)")
                        st.dataframe(q_res, use_container_width=True)

                        with st.spinner("Formulating AI executive answer..."):
                            nl_ans = analyze_with_gemini(
                                api_key=gemini_key,
                                context_title=f"User Question: {user_q}",
                                sql_query=gen_sql,
                                result_df=q_res,
                                schema_summary=schema_summary,
                                custom_user_prompt="Provide a direct, concise executive answer to the user's specific question based on these results.",
                            )
                            st.markdown(nl_ans)
                    except Exception as e:
                        st.error(f"SQL execution error on generated query: {e}")

    # Tab 4: Visual Query Builder
    with tab4:
        st.subheader("🛠️ Interactive Query Builder & SQL Studio")
        qb1, qb2, qb3, qb4 = st.columns(4)
        all_cols = clean_df.columns.tolist()
        num_candidates = num_cols if num_cols else all_cols

        with qb1:
            g_col = st.selectbox("Group By Dimension:", all_cols, index=all_cols.index(p_dim) if p_dim in all_cols else 0)
        with qb2:
            t_col = st.selectbox("Target Metric:", num_candidates, index=num_candidates.index(p_metric) if p_metric in num_candidates else 0)
        with qb3:
            func = st.selectbox("Aggregate Function:", ["SUM", "AVG", "COUNT", "MAX", "MIN"])
        with qb4:
            lim = st.slider("Row Limit:", min_value=5, max_value=100, value=15)

        built_q = f"SELECT {g_col}, {func}({t_col}) AS {func.lower()}_{t_col}, COUNT(*) AS total_count FROM universal_data GROUP BY {g_col} ORDER BY {func.lower()}_{t_col} DESC LIMIT {lim};"
        st.code(built_q, language="sql")

        if st.button("Run Visual Query", key="btn_run_vq"):
            try:
                res_b = run_sql_query(conn, built_q)
                st.dataframe(res_b, use_container_width=True)
                if len(res_b.columns) >= 2:
                    fig_b = px.bar(res_b, x=g_col, y=f"{func.lower()}_{t_col}", color=f"{func.lower()}_{t_col}", template="plotly_dark", color_continuous_scale="Viridis")
                    fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_b, use_container_width=True)
            except Exception as e:
                st.error(f"Query Error: {e}")

        st.markdown("---")
        st.subheader("💻 Free-Form SQL Editor")
        custom_q = st.text_area("SQL Statement:", value="SELECT * FROM universal_data LIMIT 10;", height=100)
        if st.button("Execute Arbitrary SQL", key="btn_run_arb"):
            try:
                res_c = run_sql_query(conn, custom_q)
                st.success(f"Executed ({len(res_c)} rows)")
                st.dataframe(res_c, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")

    # Tab 5: Schema & Data Profiling
    with tab5:
        st.subheader("Dataset Architecture & Profiling")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Total Columns", len(clean_df.columns))
        with p2:
            st.metric("Numeric Columns", len(num_cols))
        with p3:
            st.metric("Categorical Columns", len(cat_cols))

        st.markdown("**Discovered Columns & Data Types:**")
        pills = "".join([f'<span class="schema-pill">{c} ({clean_df[c].dtype})</span>' for c in clean_df.columns])
        st.markdown(pills, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Column Cardinality & Null Analysis")
        stats = []
        for c in clean_df.columns:
            stats.append({
                "Column (Cleaned)": c,
                "Original Header": col_mapping.get(c, c),
                "Data Type": str(clean_df[c].dtype),
                "Unique Values": clean_df[c].nunique(),
                "Null Values": int(clean_df[c].isnull().sum()),
                "% Complete": f"{(1 - clean_df[c].isnull().mean())*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(stats), use_container_width=True)

        st.markdown("---")
        st.subheader("Raw Data Preview (First 100 Rows)")
        st.dataframe(clean_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()

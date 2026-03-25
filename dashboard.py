"""
Demand Planning & Forecasting Dashboard
A Streamlit-based web UI for supply chain demand analysis.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import DemandDataPreprocessor
from src.models import ARIMAModel, ExponentialSmoothingModel, ProphetModel
from src.utils import get_config, get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.12);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-delta-up {
        color: #059669;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-delta-down {
        color: #dc2626;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Status badges */
    .badge-ok {
        background: #ecfdf5;
        color: #059669;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warn {
        background: #fef3c7;
        color: #d97706;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-alert {
        background: #fee2e2;
        color: #dc2626;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Anomaly table */
    .anomaly-row {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        font-weight: 500;
        color: #334155;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def detect_anomalies_zscore(values, threshold=3.0):
    """Detect anomalies using z-score method."""
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    z_scores = np.abs((values - mean) / std)
    return list(np.where(z_scores > threshold)[0])


def render_metric_card(label, value, delta=None, delta_type="up"):
    delta_html = ""
    if delta:
        cls = "metric-delta-up" if delta_type == "up" else "metric-delta-down"
        arrow = "↑" if delta_type == "up" else "↓"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def create_demand_chart(series, title="Demand Over Time"):
    """Create an interactive demand time series chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series["demand"],
        mode="lines",
        name="Demand",
        line=dict(color="#6366f1", width=2),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.08)",
    ))

    # Rolling average
    if len(series) > 7:
        rolling = series["demand"].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=rolling,
            mode="lines",
            name="7-Day Average",
            line=dict(color="#f97316", width=2, dash="dot"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="DM Sans")),
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="DM Sans"),
    )

    return fig


def create_forecast_chart(historical, forecast_df, horizon, sku_id):
    """Create forecast chart with confidence intervals."""
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical["demand"],
        mode="lines",
        name="Historical",
        line=dict(color="#6366f1", width=2),
    ))

    # Forecast dates
    last_date = historical.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(forecast_df["upper_bound"]) + list(forecast_df["lower_bound"][::-1]),
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df["forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=4),
    ))

    # Divider line
    fig.add_vline(
        x=last_date, line_dash="dash",
        line_color="#94a3b8", line_width=1,
        annotation_text="Forecast Start",
        annotation_position="top",
    )

    fig.update_layout(
        title=dict(text=f"Demand Forecast: {sku_id}", font=dict(size=16, family="DM Sans")),
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="DM Sans"),
    )

    return fig


def create_anomaly_chart(series, anomaly_indices):
    """Create chart highlighting anomalies."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series["demand"],
        mode="lines",
        name="Demand",
        line=dict(color="#6366f1", width=1.5),
    ))

    # Highlight anomalies
    if anomaly_indices:
        anomaly_dates = series.index[anomaly_indices]
        anomaly_values = series["demand"].iloc[anomaly_indices]

        fig.add_trace(go.Scatter(
            x=anomaly_dates,
            y=anomaly_values,
            mode="markers",
            name="Anomalies",
            marker=dict(color="#ef4444", size=10, symbol="diamond",
                        line=dict(color="#ffffff", width=2)),
        ))

    # Mean line
    mean_val = series["demand"].mean()
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Mean: {mean_val:.0f}")

    # Threshold lines
    std_val = series["demand"].std()
    fig.add_hline(y=mean_val + 3 * std_val, line_dash="dot", line_color="#f97316",
                  annotation_text="Upper Threshold")
    fig.add_hline(y=max(0, mean_val - 3 * std_val), line_dash="dot", line_color="#f97316",
                  annotation_text="Lower Threshold")

    fig.update_layout(
        title=dict(text="Anomaly Detection", font=dict(size=16, family="DM Sans")),
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="DM Sans"),
    )

    return fig


def create_comparison_chart(data, preprocessor, sku_ids):
    """Create multi-SKU comparison chart."""
    fig = go.Figure()
    colors = ["#6366f1", "#10b981", "#f97316", "#ec4899", "#8b5cf6"]

    for i, sku in enumerate(sku_ids):
        series = preprocessor.get_sku_series(data, sku)
        weekly = series["demand"].resample("W").mean()
        fig.add_trace(go.Scatter(
            x=weekly.index,
            y=weekly.values,
            mode="lines",
            name=sku,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title=dict(text="Weekly Average Demand Comparison", font=dict(size=16, family="DM Sans")),
        xaxis_title="Week",
        yaxis_title="Avg Units",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="DM Sans"),
    )

    return fig


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

@st.cache_data
def load_data(filepath):
    preprocessor = DemandDataPreprocessor()
    raw = preprocessor.load(filepath)
    return preprocessor.preprocess(raw)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📦 Demand Planner")
    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload demand data (CSV)",
        type=["csv"],
        help="CSV with columns: date, sku_id, demand, warehouse"
    )

    if uploaded_file:
        # Save uploaded file temporarily
        tmp_path = Path("/tmp/uploaded_demand.csv")
        tmp_path.write_bytes(uploaded_file.getvalue())
        data_path = str(tmp_path)
    else:
        data_path = "sample_data/demand_data.csv"
        if not Path(data_path).exists():
            st.warning("No data found. Run `python sample_data/generate.py` first.")
            st.stop()

    data = load_data(data_path)
    preprocessor = DemandDataPreprocessor()
    skus = preprocessor.list_skus(data)

    st.markdown("---")
    st.markdown("#### Settings")

    selected_sku = st.selectbox("Select SKU", skus, index=0)

    warehouses = ["All"] + sorted(data["warehouse"].unique().tolist()) if "warehouse" in data.columns else ["All"]
    selected_warehouse = st.selectbox("Warehouse", warehouses, index=0)

    horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7)

    model_choice = st.selectbox(
        "Forecasting Model",
        ["Ensemble", "ARIMA", "Exponential Smoothing", "Prophet"],
        index=0,
    )
    model_map = {
        "Ensemble": "ensemble",
        "ARIMA": "arima",
        "Exponential Smoothing": "exponential_smoothing",
        "Prophet": "prophet",
    }

    st.markdown("---")
    st.markdown("#### Data Summary")
    summary = preprocessor.summary(data)
    st.markdown(f"**Rows:** {summary['total_rows']:,}")
    st.markdown(f"**SKUs:** {summary['sku_count']}")
    if summary["date_range"]:
        st.markdown(f"**From:** {summary['date_range']['start'][:10]}")
        st.markdown(f"**To:** {summary['date_range']['end'][:10]}")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📦 Demand Planning Dashboard</h1>
    <p>Supply chain forecasting and anomaly detection powered by AI</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Get SKU Data
# ─────────────────────────────────────────────

warehouse_filter = None if selected_warehouse == "All" else selected_warehouse
series = preprocessor.get_sku_series(data, selected_sku, warehouse_filter)

if series.empty:
    st.error(f"No data found for {selected_sku}")
    st.stop()


# ─────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────

recent_7 = series["demand"].tail(7).mean()
previous_7 = series["demand"].iloc[-14:-7].mean() if len(series) > 14 else recent_7
change_pct = ((recent_7 - previous_7) / previous_7 * 100) if previous_7 > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        "Avg Daily Demand",
        f"{series['demand'].mean():,.0f}",
    )

with col2:
    render_metric_card(
        "Last 7-Day Avg",
        f"{recent_7:,.0f}",
        f"{abs(change_pct):.1f}% vs prior week",
        "up" if change_pct >= 0 else "down",
    )

with col3:
    render_metric_card(
        "Peak Demand",
        f"{series['demand'].max():,.0f}",
    )

with col4:
    render_metric_card(
        "Data Points",
        f"{len(series):,}",
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_history, tab_forecast, tab_anomaly, tab_compare, tab_report = st.tabs([
    "📈 History", "🔮 Forecast", "🚨 Anomalies", "📊 Compare", "📋 Report"
])


# ── History Tab ──
with tab_history:
    st.plotly_chart(
        create_demand_chart(series, f"Demand History: {selected_sku}"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Demand Distribution</div>', unsafe_allow_html=True)
        fig_dist = px.histogram(
            series, x="demand", nbins=40,
            color_discrete_sequence=["#6366f1"],
            template="plotly_white",
        )
        fig_dist.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Day-of-Week Pattern</div>', unsafe_allow_html=True)
        dow = series.copy()
        dow["day"] = dow.index.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_avg = dow.groupby("day")["demand"].mean().reindex(range(7))
        fig_dow = go.Figure(go.Bar(
            x=day_names,
            y=dow_avg.values,
            marker_color=["#6366f1" if i < 5 else "#f97316" for i in range(7)],
        ))
        fig_dow.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            template="plotly_white",
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_dow, use_container_width=True)


# ── Forecast Tab ──
with tab_forecast:
    run_forecast = st.button("🔮 Run Forecast", type="primary", use_container_width=True)

    if run_forecast:
        with st.spinner(f"Running {model_choice} forecast for {horizon} days..."):
            cfg = get_config()["forecasting"]
            models_to_run = []

            selected_model = model_map[model_choice]

            if selected_model == "ensemble":
                models_to_run = ["arima", "exponential_smoothing", "prophet"]
            else:
                models_to_run = [selected_model]

            results = {}
            progress = st.progress(0)

            for i, name in enumerate(models_to_run):
                model_cfg = cfg["models"].get(name, {})
                if not model_cfg.get("enabled", True):
                    continue

                try:
                    if name == "arima":
                        m = ARIMAModel()
                    elif name == "exponential_smoothing":
                        m = ExponentialSmoothingModel()
                    elif name == "prophet":
                        m = ProphetModel()
                    else:
                        continue

                    m.fit(series)
                    pred = m.predict(horizon)
                    results[name] = pred
                except Exception as e:
                    st.warning(f"{name} failed: {e}")

                progress.progress((i + 1) / len(models_to_run))

            progress.empty()

            if not results:
                st.error("All models failed. Check your data.")
            else:
                # Ensemble or single
                if selected_model == "ensemble" and len(results) > 1:
                    weights = cfg.get("ensemble", {}).get("weights", {})
                    total_weight = sum(weights.get(n, 1.0) for n in results)
                    forecast_vals = sum(
                        results[n]["forecast"] * (weights.get(n, 1.0) / total_weight)
                        for n in results
                    )
                    lower_vals = np.minimum.reduce([results[n]["lower_bound"] for n in results])
                    upper_vals = np.maximum.reduce([results[n]["upper_bound"] for n in results])

                    forecast_df = pd.DataFrame({
                        "forecast": forecast_vals,
                        "lower_bound": lower_vals,
                        "upper_bound": upper_vals,
                    })
                    model_used = f"Ensemble ({', '.join(results.keys())})"
                else:
                    name = list(results.keys())[0]
                    forecast_df = results[name]
                    model_used = name.replace("_", " ").title()

                # Store in session
                st.session_state["forecast_df"] = forecast_df
                st.session_state["forecast_model"] = model_used
                st.session_state["forecast_sku"] = selected_sku
                st.session_state["forecast_horizon"] = horizon

                # Chart
                st.plotly_chart(
                    create_forecast_chart(series, forecast_df, horizon, selected_sku),
                    use_container_width=True,
                )

                # Forecast KPIs
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    render_metric_card("Model", model_used)
                with fc2:
                    render_metric_card("Avg Daily Forecast", f"{forecast_df['forecast'].mean():,.0f}")
                with fc3:
                    render_metric_card("Total Projected", f"{forecast_df['forecast'].sum():,.0f}")
                with fc4:
                    safety = forecast_df["forecast"].sum() * 0.15
                    render_metric_card("Safety Stock (15%)", f"{safety:,.0f}")

                # Forecast table
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📋 View Daily Forecast Table"):
                    last_date = series.index[-1]
                    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
                    display_df = forecast_df.copy()
                    display_df.index = dates
                    display_df.index.name = "Date"
                    display_df = display_df.round(2)
                    st.dataframe(display_df, use_container_width=True)

    elif "forecast_df" in st.session_state and st.session_state.get("forecast_sku") == selected_sku:
        forecast_df = st.session_state["forecast_df"]
        st.plotly_chart(
            create_forecast_chart(series, forecast_df, st.session_state["forecast_horizon"], selected_sku),
            use_container_width=True,
        )


# ── Anomaly Tab ──
with tab_anomaly:
    cfg_anom = get_config()["anomaly_detection"]
    threshold = st.slider("Z-Score Threshold", 1.5, 5.0, cfg_anom["zscore_threshold"], 0.5)

    values = series["demand"].values.astype(float)
    anomaly_indices = detect_anomalies_zscore(values, threshold)

    st.plotly_chart(
        create_anomaly_chart(series, anomaly_indices),
        use_container_width=True,
    )

    mean_val = np.mean(values)

    col_an1, col_an2, col_an3 = st.columns(3)
    with col_an1:
        badge = "badge-ok" if len(anomaly_indices) == 0 else ("badge-warn" if len(anomaly_indices) < 5 else "badge-alert")
        render_metric_card("Anomalies Found", str(len(anomaly_indices)))
    with col_an2:
        spikes = sum(1 for i in anomaly_indices if values[i] > mean_val)
        render_metric_card("Spikes", str(spikes))
    with col_an3:
        drops = sum(1 for i in anomaly_indices if values[i] < mean_val)
        render_metric_card("Drops", str(drops))

    if anomaly_indices:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Anomaly Details</div>', unsafe_allow_html=True)

        anomaly_data = []
        for idx in anomaly_indices:
            val = float(values[idx])
            date = series.index[idx]
            deviation = (val - mean_val) / mean_val * 100
            anomaly_data.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Demand": round(val, 2),
                "Expected (Mean)": round(mean_val, 2),
                "Deviation": f"{deviation:+.1f}%",
                "Type": "🔺 Spike" if val > mean_val else "🔻 Drop",
                "Severity": "High" if abs(deviation) > 100 else "Medium",
            })

        st.dataframe(pd.DataFrame(anomaly_data), use_container_width=True, hide_index=True)

    # Store for report
    st.session_state["anomaly_count"] = len(anomaly_indices)
    st.session_state["anomaly_indices"] = anomaly_indices


# ── Compare Tab ──
with tab_compare:
    compare_skus = st.multiselect(
        "Select SKUs to compare",
        skus,
        default=skus[:3],
    )

    if compare_skus:
        st.plotly_chart(
            create_comparison_chart(data, preprocessor, compare_skus),
            use_container_width=True,
        )

        # Summary table
        st.markdown('<div class="section-header">SKU Comparison Summary</div>', unsafe_allow_html=True)
        comp_data = []
        for sku in compare_skus:
            s = preprocessor.get_sku_series(data, sku)
            trend = "📈 Up" if s["demand"].tail(7).mean() > s["demand"].head(7).mean() else "📉 Down"
            comp_data.append({
                "SKU": sku,
                "Avg Demand": f"{s['demand'].mean():,.0f}",
                "Std Dev": f"{s['demand'].std():,.0f}",
                "Min": f"{s['demand'].min():,.0f}",
                "Max": f"{s['demand'].max():,.0f}",
                "Trend": trend,
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ── Report Tab ──
with tab_report:
    st.markdown('<div class="section-header">Generate Demand Report</div>', unsafe_allow_html=True)

    if st.button("📋 Generate Report", type="primary", use_container_width=True):
        report_date = datetime.now().strftime("%Y-%m-%d")

        report_lines = [
            f"# Demand Planning Report: {selected_sku}",
            f"**Generated:** {report_date}",
            "",
            "## Executive Summary",
            "",
            f"Analysis of demand data for **{selected_sku}** covering {len(series)} data points.",
            "",
            "## Historical Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Average Daily Demand | {series['demand'].mean():,.2f} units |",
            f"| Std Deviation | {series['demand'].std():,.2f} units |",
            f"| Min Demand | {series['demand'].min():,.2f} units |",
            f"| Max Demand | {series['demand'].max():,.2f} units |",
            f"| Data Points | {len(series)} |",
            "",
        ]

        # Add forecast if available
        if "forecast_df" in st.session_state and st.session_state.get("forecast_sku") == selected_sku:
            fdf = st.session_state["forecast_df"]
            report_lines.extend([
                "## Forecast",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Model | {st.session_state.get('forecast_model', 'N/A')} |",
                f"| Horizon | {st.session_state.get('forecast_horizon', 'N/A')} days |",
                f"| Avg Daily Forecast | {fdf['forecast'].mean():,.2f} units |",
                f"| Total Projected | {fdf['forecast'].sum():,.2f} units |",
                f"| Safety Stock (15%) | {fdf['forecast'].sum() * 0.15:,.2f} units |",
                "",
            ])

        # Add anomalies
        anomaly_count = st.session_state.get("anomaly_count", 0)
        report_lines.extend([
            "## Anomaly Detection",
            "",
            f"**Anomalies Detected:** {anomaly_count}",
            "",
            "## Recommendations",
            "",
            f"1. Plan inventory for approximately {series['demand'].mean() * 30:,.0f} units per month based on historical averages.",
            f"2. Maintain safety stock of ~{series['demand'].mean() * 30 * 0.15:,.0f} units (15% buffer).",
        ])

        if anomaly_count > 0:
            report_lines.append("3. Investigate detected anomalies for root causes (promotions, supply disruptions, data quality).")
        else:
            report_lines.append("3. Demand patterns are stable. No anomalies require investigation.")

        report_lines.extend([
            "",
            "---",
            f"*Generated by Demand Planning Dashboard on {report_date}*",
        ])

        report_text = "\n".join(report_lines)

        st.markdown(report_text)

        st.download_button(
            "⬇️ Download Report (Markdown)",
            report_text,
            file_name=f"demand_report_{selected_sku}_{report_date}.md",
            mime="text/markdown",
            use_container_width=True,
        )

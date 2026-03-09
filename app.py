"""
app.py
======
Main Streamlit application for "The Superpowers Alignment Tax" project.

Architecture
------------
- 5 navigation pages via sidebar radio buttons
- @st.cache_data / @st.cache_resource for all expensive operations
- Session state for scenario results
- Full integration with data_generator, alignment_model, visualizations

All estimates are historical precedent-based and NOT deterministic predictions.
"""

from __future__ import annotations

import io
import sys
import os

# ── Ensure the app directory is on sys.path so local modules resolve ──────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import pandas as pd
import streamlit as st

# ── Page configuration (MUST be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="The Superpowers Alignment Tax",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**The Superpowers Alignment Tax**: A structured analytical framework estimating "
            "the economic costs and benefits of great-power alignment choices for "
            "African states. All data is calibrated to real-world "
            "patterns. NOT deterministic predictions."
        )
    },
)

# ── Local imports ─────────────────────────────────────────────────────────────
from data_generator import (
    load_all_data,
    generate_unga_voting_data,
    generate_diplomatic_signals,
    generate_economic_dependency,
    generate_historical_precedents,
    generate_ghana_deep_dive,
    FOCUS_COUNTRIES,
)
from alignment_model import (
    AlignmentVector,
    AlignmentTaxCalculator,
    BehavioralModifiers,
    ScenarioEngine,
    PanelEstimator,
)
from visualizations import (
    alignment_space_3d,
    economic_exposure_radar,
    alignment_tax_waterfall,
    historical_precedent_timeline,
    ghana_dashboard,
    alignment_heatmap,
    scenario_comparison_bar,
    credibility_signal_chart,
    loss_aversion_curve,
    BRAND,
)

# ═════════════════════════════════════════════════════════════════════════════
# Custom CSS — brand-consistent styling
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #13343B;
    }

    /* ── App background ── */
    .stApp {
        background-color: #FCFAF6;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1B474D;
        border-right: 1px solid #13343B;
    }
    [data-testid="stSidebar"] * {
        color: #FCFAF6 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem;
        font-weight: 500;
        padding: 2px 0;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.82rem;
        color: #BCE2E7 !important;
    }
    [data-testid="stSidebar"] a {
        color: #BCE2E7 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(188, 226, 231, 0.25) !important;
    }

    /* ── Header band ── */
    .alignment-header {
        background: linear-gradient(135deg, #1B474D 0%, #20808D 100%);
        padding: 1.4rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(27,71,77,0.18);
    }
    .alignment-header h1 {
        color: #FCFAF6 !important;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .alignment-header p {
        color: #BCE2E7 !important;
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
    }

    /* ── Section headings ── */
    h2 {
        color: #1B474D;
        font-size: 1.35rem;
        font-weight: 600;
        border-bottom: 2px solid #20808D;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    h3 {
        color: #20808D;
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #F3F3EE;
        border: 1px solid #E5E3D4;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        box-shadow: 0 1px 4px rgba(27,71,77,0.07);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #848456 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #1B474D !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #F3F3EE;
        border-radius: 8px 8px 0 0;
        padding: 0 0.5rem;
        gap: 0.2rem;
        border-bottom: 2px solid #E5E3D4;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #848456;
        padding: 0.6rem 1rem;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #1B474D !important;
        background: white !important;
        border-bottom: 2px solid #20808D !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #20808D;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.55rem 1.4rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #1B474D;
        color: white;
        border: none;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #F3F3EE;
        border: 1px solid #E5E3D4;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #1B474D;
    }

    /* ── Disclaimer banner ── */
    .disclaimer-banner {
        background: linear-gradient(90deg, #FFF8E7 0%, #FFF3D0 100%);
        border: 1px solid #FFC553;
        border-left: 4px solid #FFC553;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: #6E522B;
        margin: 0.75rem 0 1.25rem 0;
    }
    .disclaimer-banner strong {
        color: #A84B2F;
    }

    /* ── Info cards ── */
    .info-card {
        background: #F3F3EE;
        border: 1px solid #E5E3D4;
        border-left: 4px solid #20808D;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        font-size: 0.85rem;
        color: #13343B;
        margin: 0.5rem 0;
    }
    .info-card.warning {
        border-left-color: #A84B2F;
        background: #FFF8F6;
    }
    .info-card.success {
        border-left-color: #848456;
    }

    /* ── Dataframe styling ── */
    .stDataFrame {
        border: 1px solid #E5E3D4;
        border-radius: 6px;
    }

    /* ── Selectbox / slider ── */
    .stSelectbox > div > div {
        border-color: #D6D4C5 !important;
        border-radius: 6px !important;
    }
    .stSlider > div > div > div > div {
        background: #20808D !important;
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #E5E3D4;
        margin: 1.5rem 0;
    }

    /* ── Layer badge ── */
    .layer-badge {
        display: inline-block;
        background: #1B474D;
        color: #BCE2E7;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# Data loading — cached
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def _load_all():
    return load_all_data()


@st.cache_data(show_spinner=False)
def _unga():
    return generate_unga_voting_data()


@st.cache_data(show_spinner=False)
def _diplom():
    return generate_diplomatic_signals()


@st.cache_data(show_spinner=False)
def _econ():
    return generate_economic_dependency()


@st.cache_data(show_spinner=False)
def _precedents():
    return generate_historical_precedents()


@st.cache_data(show_spinner=False)
def _ghana():
    return generate_ghana_deep_dive()


@st.cache_resource(show_spinner=False)
def _engine():
    return ScenarioEngine()


@st.cache_resource(show_spinner=False)
def _panel():
    return PanelEstimator()


# ═════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═════════════════════════════════════════════════════════════════════════════

if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None
if "compare_df" not in st.session_state:
    st.session_state.compare_df = None
if "last_scenario_key" not in st.session_state:
    st.session_state.last_scenario_key = None


# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════


def _fmt_mn(val: float) -> str:
    """Format a USD million value for display."""
    if abs(val) >= 1000:
        return f"${val / 1000:+.2f}B"
    return f"${val:+.0f}M"


def _fmt_bn(val: float) -> str:
    return f"${val:.2f}B"


def _disclaimer():
    st.markdown(
        '<div class="disclaimer-banner">'
        "⚠ <strong>All estimates are historical precedent-based and NOT "
        "deterministic predictions.</strong> Data is calibrated to "
        "real-world patterns from public sources (UNGA voting records, World Bank, "
        "IMF, AGOA schedules). Confidence intervals reflect parameter uncertainty, "
        "not probability forecasts."
        "</div>",
        unsafe_allow_html=True,
    )


def _scenario_to_csv(result: dict) -> bytes:
    """Flatten a scenario result dict to CSV bytes."""
    rows = []
    rows.append(
        {
            "field": "country",
            "value": result["country"],
        }
    )
    rows.append({"field": "posture", "value": result["posture"]})
    rows.append({"field": "crisis_type", "value": result["crisis_type"]})
    rows.append(
        {
            "field": "total_alignment_tax_mn",
            "value": result["total_alignment_tax_mn"],
        }
    )
    rows.append({"field": "ci_lower_mn", "value": result["ci_lower_mn"]})
    rows.append({"field": "ci_upper_mn", "value": result["ci_upper_mn"]})
    rows.append({"field": "gross_gains_mn", "value": result["gross_gains_mn"]})
    rows.append({"field": "gross_costs_mn", "value": result["gross_costs_mn"]})
    rows.append({"field": "lock_in_cost_mn", "value": result["lock_in_cost_mn"]})
    rows.append(
        {
            "field": "credibility_score",
            "value": result["behavioral_modifiers"]["credibility"]["credibility_score"],
        }
    )
    rows.append(
        {
            "field": "audience_cost_multiplier",
            "value": result["behavioral_modifiers"]["audience_costs"][
                "audience_cost_multiplier"
            ],
        }
    )
    rows.append(
        {
            "field": "lock_in_probability",
            "value": result["behavioral_modifiers"]["escalation_lockin"][
                "lock_in_probability"
            ],
        }
    )
    # Channel breakdown
    for ch_key, ch_val in result["channel_breakdown"].items():
        for k, v in ch_val.items():
            if isinstance(v, (int, float)):
                rows.append({"field": f"channel.{ch_key}.{k}", "value": v})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em;">
                ⚖️ The Alignment Tax
            </div>
            <div style="font-size: 0.78rem; color: #BCE2E7; margin-top: 0.2rem; line-height: 1.4;">
                Quantifying the economic cost of<br>
                great-power alignment for African states
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🌍  Overview",
            "📡  Alignment Signal Coding",
            "💰  Economic Dependency",
            "⚖️  Alignment Tax Calculator",
            "🇬🇭  Ghana Deep Dive",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    with st.expander("📖 Methodology", expanded=False):
        st.markdown(
            """
            **Framework Overview**

            The Alignment Tax is a four-layer analytical framework:

            **Layer 1 — Signal Coding**
            UNGA voting agreement rates (2000–2025), diplomatic statements,
            sanctions compliance, and military exercise participation are coded
            into composite alignment vectors.

            **Layer 2 — Economic Dependency Mapping**
            Bilateral trade, FDI stocks, development assistance, debt exposure,
            and military cooperation are quantified across US, China, and Russia.

            **Layer 3 — Alignment Tax Calculation**
            Net economic cost/benefit of adopting each alignment posture, accounting
            for AGOA preferences, USAID/MCC flows, IMF/WB support, MSP access,
            Chinese FDI, secondary sanctions, and commodity routing risk.

            **Layer 4 — Behavioral Modifiers**
            Prospect theory loss aversion (λ=2.25), commitment credibility
            discounts, audience costs, and lock-in probability.

            **Calibration Sources**
            - Erik Voeten UNGA voting dataset
            - World Bank bilateral trade statistics
            - IMF Article IV reports, AGOA trade preference schedules
            - UNCTAD FDI database, AidData Chinese development finance
            - Kahneman-Tversky (1992) prospect theory

            **Confidence Intervals**
            ±25–30% parameter uncertainty bounds; bootstrapped from historical
            precedent episodes (N=4–18 per channel).
            """,
        )

    with st.expander("ℹ️ Data & Limitations", expanded=False):
        st.markdown(
            """
            **All data is synthetic**, generated with a fixed random seed (42)
            and calibrated to publicly available real-world patterns.

            **Key limitations:**
            - Alignment scores reflect UNGA voting only; bilateral signalling
              and covert preferences are not captured
            - Economic impact estimates assume linear channel responses; in
              practice, responses may be non-linear and context-dependent
            - Crisis-specific modifiers are analyst-calibrated; not derived
              from structural models
            - "Neutrality" is defined as equal diplomatic distance; in practice,
              active non-alignment requires significant diplomatic capital

            **This tool is for scenario exploration and research purposes only.
            It should not be used as the basis for policy decisions.**
            """,
        )

    with st.expander("📊 Focus Countries", expanded=False):
        for c in FOCUS_COUNTRIES:
            st.markdown(f"• {c}")

    st.divider()
    st.markdown(
        '<div style="font-size: 0.72rem; color: #BCE2E7; line-height: 1.6;">'
        "Synthetic data · Research instrument<br>"
        "Not a policy recommendation tool"
        "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ── PAGE 1: OVERVIEW ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def page_overview():
    st.markdown(
        '<div class="alignment-header">'
        "<h1>Africa's Alignment Landscape</h1>"
        "<p>54 African countries in US–China–Russia great-power alignment space, "
        "2000–2025 · UNGA voting-based scores</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _disclaimer()

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        unga_df = _unga()
        econ_df = _econ()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return

    latest = (
        unga_df.sort_values("year", ascending=False).drop_duplicates("country")
    )

    # ── Key metrics row ────────────────────────────────────────────────────────
    n_countries = unga_df["country"].nunique()
    total_trade = econ_df["total_trade_bn"].sum()
    avg_us_gap = (latest["china_alignment"] - latest["us_alignment"]).mean()
    avg_china = latest["china_alignment"].mean()
    avg_us = latest["us_alignment"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Countries Analyzed", f"{n_countries}", help="All 54 African states in UNGA dataset")
    with col2:
        st.metric("Focus Countries", f"{len(FOCUS_COUNTRIES)}", help="15 major African economies with full modelling")
    with col3:
        st.metric("Total Trade Exposure", f"${total_trade:.0f}B", help="Combined US+China+Russia bilateral trade (15 focus countries)")
    with col4:
        st.metric("Avg. China Alignment", f"{avg_china:.3f}", help="Mean UNGA agreement rate with China (2025)")
    with col5:
        st.metric(
            "Avg. Alignment Gap (CN-US)",
            f"{avg_us_gap:.3f}",
            delta=f"China leads by {avg_us_gap:.3f}",
            delta_color="off",
            help="Mean difference between China and US UNGA agreement rates"
        )

    st.divider()

    # ── 3D Scatter ─────────────────────────────────────────────────────────────
    st.subheader("3D Alignment Space — All 54 African Countries")
    st.markdown(
        '<div class="info-card">'
        "Each point represents a country's position in the three-dimensional space "
        "defined by UNGA voting agreement rates with the US, China, and Russia. "
        "The vast majority of African states cluster in the high China–Russia "
        "alignment quadrant (upper-right), with US agreement rates typically 20–35%. "
        "Rotate the chart to explore regional clustering patterns."
        "</div>",
        unsafe_allow_html=True,
    )
    try:
        fig_3d = alignment_space_3d(latest)
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"3D chart error: {e}")

    st.divider()

    # ── Alignment Heatmap with controls ───────────────────────────────────────
    st.subheader("Alignment Heatmap — Country Rankings")

    ctrl_col1, ctrl_col2 = st.columns([2, 3])
    with ctrl_col1:
        metric_choice = st.selectbox(
            "Alignment metric",
            options=["china_alignment", "us_alignment", "russia_alignment"],
            format_func=lambda x: {
                "china_alignment": "China Alignment (UNGA)",
                "us_alignment": "US Alignment (UNGA)",
                "russia_alignment": "Russia Alignment (UNGA)",
            }[x],
            index=0,
        )
    with ctrl_col2:
        year_val = st.slider(
            "Reference year",
            min_value=2000,
            max_value=2025,
            value=2025,
            step=1,
            help="Filter UNGA voting data to this year",
        )

    metric_desc = {
        "china_alignment": (
            "**China alignment** measures how often a country votes the same way as "
            "China in contested UNGA resolutions. The African average of ~0.72 reflects "
            "the structural convergence on sovereignty norms, development financing, and "
            "non-interference principles."
        ),
        "us_alignment": (
            "**US alignment** averages 20–35% for African states — reflecting systematic "
            "divergence on human rights resolutions, Israel-Palestine votes, and "
            "sovereignty-vs-intervention debates where US and African positions differ."
        ),
        "russia_alignment": (
            "**Russia alignment** has risen sharply since 2022. Countries like Mali, "
            "Burkina Faso, and Eritrea — with Wagner Group presence or military juntas — "
            "show the highest Russia UNGA agreement rates on this continent."
        ),
    }
    st.markdown(
        f'<div class="info-card">{metric_desc[metric_choice]}</div>',
        unsafe_allow_html=True,
    )

    try:
        fig_hm = alignment_heatmap(unga_df, metric=metric_choice, year=year_val)
        st.plotly_chart(fig_hm, use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ── PAGE 2: ALIGNMENT SIGNAL CODING ──────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def page_signal_coding():
    st.markdown(
        '<div class="alignment-header">'
        "<h1>"
        '<span class="layer-badge">Layer 1</span>'
        "Alignment Signal Coding"
        "</h1>"
        "<p>UNGA voting time-series · 2025 Iran crisis diplomatic signals · "
        "Credibility scoring methodology</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _disclaimer()

    try:
        unga_df = _unga()
        diplom_df = _diplom()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return

    tab_unga, tab_signals, tab_cred, tab_method = st.tabs(
        ["UNGA Voting Series", "Iran Crisis Signals", "Credibility Chart", "Methodology"]
    )

    # ── Tab 1: UNGA Voting ─────────────────────────────────────────────────────
    with tab_unga:
        st.subheader("UNGA Voting Alignment — Time Series")
        country_choice = st.selectbox(
            "Select country",
            options=FOCUS_COUNTRIES,
            index=0,
            key="unga_country",
        )

        c_df = unga_df[unga_df["country"] == country_choice].sort_values("year")

        if c_df.empty:
            st.warning("No voting data for this country.")
        else:
            import plotly.graph_objects as go

            fig = go.Figure()
            colors = {
                "us_alignment": BRAND["primary"],
                "china_alignment": BRAND["terra"],
                "russia_alignment": BRAND["mauve"],
            }
            labels = {
                "us_alignment": "US Alignment",
                "china_alignment": "China Alignment",
                "russia_alignment": "Russia Alignment",
            }
            for col, color in colors.items():
                fig.add_trace(
                    go.Scatter(
                        x=c_df["year"],
                        y=c_df[col],
                        mode="lines+markers",
                        name=labels[col],
                        line=dict(color=color, width=2.5),
                        marker=dict(size=5, color=color),
                        hovertemplate=f"<b>{labels[col]}</b><br>Year: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
                    )
                )

            # Event lines
            for yr, label in [(2003, "Iraq War"), (2014, "Crimea"), (2022, "Ukraine invasion")]:
                fig.add_vline(
                    x=yr,
                    line_dash="dot",
                    line_color=BRAND["grid"],
                    line_width=1.2,
                    opacity=0.7,
                )
                fig.add_annotation(
                    x=yr,
                    y=0.95,
                    text=label,
                    showarrow=False,
                    font=dict(size=9, color=BRAND["olive"]),
                    textangle=-60,
                    yref="y",
                )

            fig.update_layout(
                paper_bgcolor=BRAND["bg_offwhite"],
                plot_bgcolor=BRAND["bg_paper"],
                title=dict(
                    text=f"UNGA Alignment Scores — {country_choice} (2000–2025)",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=16, color=BRAND["text"]),
                ),
                xaxis=dict(
                    title="Year",
                    tickvals=list(range(2000, 2026, 2)),
                    gridcolor=BRAND["grid"],
                ),
                yaxis=dict(
                    title="UNGA Agreement Rate",
                    range=[0.0, 1.0],
                    gridcolor=BRAND["grid"],
                ),
                height=480,
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
                font=dict(family="Inter, Arial, sans-serif", color=BRAND["text"]),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor=BRAND["border"],
                    font_size=12,
                    font_family="Inter, Arial, sans-serif",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Latest stats
            latest_row = c_df.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Latest Year", int(latest_row["year"]))
            with m2:
                st.metric("US Agreement Rate", f"{latest_row['us_alignment']:.3f}")
            with m3:
                st.metric("China Agreement Rate", f"{latest_row['china_alignment']:.3f}")
            with m4:
                st.metric("Russia Agreement Rate", f"{latest_row['russia_alignment']:.3f}")

    # ── Tab 2: Iran Crisis Signals ─────────────────────────────────────────────
    with tab_signals:
        st.subheader("2025 Iran Crisis — Diplomatic Signal Coding (15 Focus Countries)")
        st.markdown(
            '<div class="info-card">'
            "Diplomatic signals are coded across five dimensions for the hypothetical "
            "2025 Iran–US/Israel escalation scenario: UN vote signal (−1 to +1 for US), "
            "diplomatic statement tone, sanctions compliance (0–1), military exercise "
            "participation (none/observer/participant), and commodity routing signal. "
            "A weighted composite score is derived."
            "</div>",
            unsafe_allow_html=True,
        )

        display_df = diplom_df.copy()
        display_df.columns = [
            "Country",
            "UN Vote Signal",
            "Statement Tone",
            "Sanctions Compliance",
            "Mil. Exercise",
            "Commodity Routing",
            "Composite US Align.",
            "Composite CN Align.",
            "Composite RU Align.",
        ]

        def _color_composite(val):
            if isinstance(val, float):
                if val > 0.15:
                    return "background-color: rgba(32,128,141,0.18)"
                elif val < -0.10:
                    return "background-color: rgba(168,75,47,0.18)"
            return ""

        styled = (
            display_df.style
            .format(
                {
                    "UN Vote Signal": "{:+.3f}",
                    "Statement Tone": "{:+.3f}",
                    "Sanctions Compliance": "{:.3f}",
                    "Commodity Routing": "{:+.3f}",
                    "Composite US Align.": "{:+.3f}",
                    "Composite CN Align.": "{:+.3f}",
                    "Composite RU Align.": "{:+.3f}",
                }
            )
            .applymap(_color_composite, subset=["Composite US Align."])
            .set_properties(**{"font-size": "12px"})
        )
        st.dataframe(styled, use_container_width=True, height=520)

        st.markdown(
            '<div class="info-card">'
            "<strong>Signal weights:</strong> UN vote signal 30% · Diplomatic statement "
            "tone 20% · Sanctions compliance 30% · Commodity routing 10% · "
            "Military exercise participation 10% (inverted for US alignment scoring). "
            "Composite scores range −1 (full anti-US) to +1 (full US alignment)."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Tab 3: Credibility Chart ───────────────────────────────────────────────
    with tab_cred:
        st.subheader("Alignment Credibility Signals")
        cred_country = st.selectbox(
            "Select country",
            options=FOCUS_COUNTRIES,
            index=0,
            key="cred_country",
        )
        try:
            fig_cred = credibility_signal_chart(cred_country, unga_df)
            st.plotly_chart(fig_cred, use_container_width=True)
        except Exception as e:
            st.error(f"Credibility chart error: {e}")

        # Compute and display scores
        bmod = BehavioralModifiers()
        for posture in ("US_ALIGNMENT", "CHINA_ALIGNMENT", "NEUTRALITY"):
            mult, info = bmod.commitment_credibility(cred_country, posture, unga_df)
            col_a, col_b, col_c = st.columns([2, 1, 4])
            with col_a:
                st.write(f"**{posture.replace('_', ' ')}**")
            with col_b:
                st.metric(
                    "Credibility",
                    f"{info['credibility_score']:.2f}",
                    help="0 = fully incredible, 1 = fully credible",
                )
            with col_c:
                st.caption(info["interpretation"])

    # ── Tab 4: Methodology ────────────────────────────────────────────────────
    with tab_method:
        st.subheader("Signal Coding Methodology")
        st.markdown(
            """
            #### UNGA Voting Alignment Score

            Based on the Erik Voeten UNGA voting dataset methodology. For each year,
            the **agreement rate** is calculated as:

            ```
            agreement_rate = (votes_matching_X) / (total_contested_resolutions)
            ```

            where contested resolutions exclude consensus votes. Calibration anchors:
            - US–Africa typical range: **20–35%** agreement
            - China–Africa typical range: **65–80%** agreement
            - Russia–Africa typical range: **50–65%** agreement (rising after 2022)

            #### Composite Alignment Score Formula (Iran Crisis)

            ```
            composite_US = 0.30 × UN_vote
                         + 0.20 × statement_tone
                         + 0.30 × sanctions_compliance
                         + 0.10 × commodity_routing
                         + 0.10 × (1 − military_exercise_level)
            ```

            #### Credibility Scoring

            Credibility measures how reliably a declared alignment posture predicts
            actual behaviour. It discounts the gross benefit of alignment accordingly:

            ```
            credibility = 0.5 × vote_consistency
                        + 0.3 × (1 − switching_penalty)
                        + 0.2 + trend_term
            benefit_multiplier = clip(0.5 + 0.5 × credibility, 0.5, 1.0)
            ```

            Countries with histories of rapid alignment switching (Mali, Burkina Faso,
            Guinea) receive a 0.3 switching penalty, reducing their credibility score
            and therefore the value major powers assign to their stated commitments.
            """
        )


# ═════════════════════════════════════════════════════════════════════════════
# ── PAGE 3: ECONOMIC DEPENDENCY MAPPING ──────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def page_economic_dependency():
    st.markdown(
        '<div class="alignment-header">'
        "<h1>"
        '<span class="layer-badge">Layer 2</span>'
        "Economic Dependency Mapping"
        "</h1>"
        "<p>Bilateral trade · FDI stocks · Development assistance · "
        "Debt exposure · Military cooperation indices</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _disclaimer()

    try:
        econ_df = _econ()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return

    # ── Country selector ───────────────────────────────────────────────────────
    sidebar_col, _ = st.columns([1, 3])

    sel_col1, sel_col2 = st.columns([2, 2])
    with sel_col1:
        primary_country = st.selectbox(
            "Primary country",
            options=FOCUS_COUNTRIES,
            index=0,
            key="econ_primary",
        )
    with sel_col2:
        compare_country = st.selectbox(
            "Compare with (optional)",
            options=["— None —"] + [c for c in FOCUS_COUNTRIES if c != primary_country],
            index=0,
            key="econ_compare",
        )

    st.divider()

    # ── Primary country metrics ────────────────────────────────────────────────
    row = econ_df[econ_df["country"] == primary_country].iloc[0]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("US Trade", _fmt_bn(row["us_trade_bn"]))
    with m2:
        st.metric("China Trade", _fmt_bn(row["china_trade_bn"]))
    with m3:
        st.metric("US ODA", f"${row['us_oda_mn']:.0f}M")
    with m4:
        st.metric("China Debt Share", f"{row['china_debt_pct_external']:.1f}%")
    with m5:
        st.metric("AGOA Value", f"${row['agoa_value_mn']:.0f}M")
    with m6:
        st.metric(
            "China Trade Dominance",
            f"{row['china_trade_dominance']:.1%}",
            help="China's share of this country's total great-power trade",
        )

    st.divider()

    # ── Radar + bilateral table ───────────────────────────────────────────────
    if compare_country == "— None —":
        # Single country
        try:
            fig_radar = economic_exposure_radar(primary_country, econ_df)
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.error(f"Radar error: {e}")
    else:
        # Side-by-side comparison
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            try:
                fig_r1 = economic_exposure_radar(primary_country, econ_df)
                fig_r1.update_layout(height=450)
                st.plotly_chart(fig_r1, use_container_width=True)
            except Exception as e:
                st.error(f"Radar (primary) error: {e}")
        with col_r2:
            try:
                fig_r2 = economic_exposure_radar(compare_country, econ_df)
                fig_r2.update_layout(height=450)
                st.plotly_chart(fig_r2, use_container_width=True)
            except Exception as e:
                st.error(f"Radar (comparison) error: {e}")

    # ── Detailed bilateral exposure table ─────────────────────────────────────
    st.subheader("Detailed Bilateral Exposure")

    def _build_exposure_table(r):
        rows = [
            ("US bilateral trade", _fmt_bn(r["us_trade_bn"]), _fmt_bn(r["china_trade_bn"]), _fmt_bn(r["russia_trade_bn"])),
            ("FDI stock", _fmt_bn(r["us_fdi_stock_bn"]), _fmt_bn(r["china_fdi_stock_bn"]), _fmt_bn(r["russia_fdi_stock_bn"])),
            ("Development assistance (ODA)", f"${r['us_oda_mn']:.0f}M", f"${r['china_oda_mn']:.0f}M", f"${r['russia_oda_mn']:.0f}M"),
            ("Military cooperation index (0–1)", f"{r['us_mil_coop_idx']:.3f}", f"{r['china_mil_coop_idx']:.3f}", f"{r['russia_mil_coop_idx']:.3f}"),
            ("Debt exposure (% external)", "Minimal", f"{r['china_debt_pct_external']:.1f}%", "Minimal"),
            ("AGOA trade value", f"${r['agoa_value_mn']:.0f}M", "—", "—"),
            ("BRI member", "—", "Yes" if r["bri_member"] else "No", "—"),
            ("IMF program active", "Influenced" if r["imf_program"] else "—", "—", "—"),
            ("MCC compact", "Yes" if r["mcc_compact"] else "—", "—", "—"),
            ("Primary commodities", r["primary_commodities"], r["primary_commodities"], r["primary_commodities"]),
        ]
        return pd.DataFrame(rows, columns=["Metric", "United States", "China", "Russia"])

    if compare_country == "— None —":
        tbl = _build_exposure_table(row)
        st.dataframe(tbl.set_index("Metric"), use_container_width=True)
    else:
        row2 = econ_df[econ_df["country"] == compare_country].iloc[0]
        tbl1 = _build_exposure_table(row)
        tbl2 = _build_exposure_table(row2)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(f"**{primary_country}**")
            st.dataframe(tbl1.set_index("Metric"), use_container_width=True)
        with t_col2:
            st.markdown(f"**{compare_country}**")
            st.dataframe(tbl2.set_index("Metric"), use_container_width=True)

    # ── Trade dominance metrics bar ────────────────────────────────────────────
    st.divider()
    st.subheader("China Trade Dominance — All Focus Countries")

    dom_df = econ_df[["country", "china_trade_dominance", "us_trade_share"]].copy()
    dom_df = dom_df.sort_values("china_trade_dominance", ascending=True)

    import plotly.graph_objects as go

    fig_dom = go.Figure()
    fig_dom.add_trace(
        go.Bar(
            y=dom_df["country"],
            x=dom_df["china_trade_dominance"],
            name="China share",
            orientation="h",
            marker_color=BRAND["terra"],
            text=[f"{v:.1%}" for v in dom_df["china_trade_dominance"]],
            textposition="outside",
        )
    )
    fig_dom.add_trace(
        go.Bar(
            y=dom_df["country"],
            x=dom_df["us_trade_share"],
            name="US share",
            orientation="h",
            marker_color=BRAND["primary"],
            text=[f"{v:.1%}" for v in dom_df["us_trade_share"]],
            textposition="outside",
        )
    )
    fig_dom.update_layout(
        paper_bgcolor=BRAND["bg_offwhite"],
        plot_bgcolor=BRAND["bg_paper"],
        title=dict(
            text="Great-Power Trade Share (% of bilateral total)",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="Share of great-power trade", tickformat=".0%", gridcolor=BRAND["grid"]),
        yaxis=dict(gridcolor=BRAND["grid"]),
        barmode="overlay",
        height=500,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
        font=dict(family="Inter, Arial, sans-serif", color=BRAND["text"]),
    )
    st.plotly_chart(fig_dom, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ── PAGE 4: ALIGNMENT TAX CALCULATOR (CORE) ──────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def page_calculator():
    st.markdown(
        '<div class="alignment-header">'
        "<h1>"
        '<span class="layer-badge">Layer 3 · Core</span>'
        "Alignment Tax Calculator"
        "</h1>"
        "<p>Quantify the net economic cost or benefit of each alignment posture "
        "under a specified crisis scenario · Channels · Behavioral modifiers · "
        "Historical precedents</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _disclaimer()

    engine = _engine()
    panel = _panel()

    # ── Parameter panel ────────────────────────────────────────────────────────
    with st.container():
        p_col1, p_col2, p_col3 = st.columns([2, 2, 2])

        with p_col1:
            country = st.selectbox("Country", options=FOCUS_COUNTRIES, index=0, key="calc_country")
            posture = st.selectbox(
                "Alignment posture",
                options=["US_ALIGNMENT", "CHINA_ALIGNMENT", "NEUTRALITY"],
                format_func=lambda x: {
                    "US_ALIGNMENT": "🇺🇸 US Alignment",
                    "CHINA_ALIGNMENT": "🇨🇳 China Alignment",
                    "NEUTRALITY": "⚖️ Neutrality",
                }[x],
                index=0,
                key="calc_posture",
            )
            crisis = st.selectbox(
                "Crisis type",
                options=["iran", "taiwan", "ukraine", "generic"],
                format_func=lambda x: {
                    "iran": "🇮🇷 Iran (2025 escalation)",
                    "taiwan": "🇹🇼 Taiwan (strait crisis)",
                    "ukraine": "🇺🇦 Ukraine (continued conflict)",
                    "generic": "⚡ Generic great-power crisis",
                }[x],
                index=0,
                key="calc_crisis",
            )

        with p_col2:
            severity = st.slider(
                "Crisis severity",
                min_value=1, max_value=5, value=3, step=1,
                help="1 = minor diplomatic dispute, 5 = existential military conflict",
            )
            power_response = st.slider(
                "Power response intensity",
                min_value=1, max_value=5, value=3, step=1,
                help="How aggressively great powers reward/punish alignment choices",
            )
            time_horizon = st.slider(
                "Time horizon (years)",
                min_value=1, max_value=5, value=3, step=1,
                help="Planning horizon over which costs/benefits are assessed",
            )

        with p_col3:
            opposition = st.slider(
                "Opposition strength",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Domestic political opposition capacity (0 = none, 1 = strong)",
            )
            prev_depth = st.slider(
                "Previous alignment depth",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Depth of prior alignment relationship with the chosen power",
            )
            entanglement = st.slider(
                "Institutional entanglement",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Depth of institutional ties (military, financial, trade agreements)",
            )

    st.divider()

    # ── Run scenario button ────────────────────────────────────────────────────
    scenario_key = f"{country}|{posture}|{crisis}|{severity}|{power_response}|{time_horizon}|{opposition:.2f}|{prev_depth:.2f}|{entanglement:.2f}"

    col_btn, col_exp = st.columns([2, 5])
    with col_btn:
        run_btn = st.button("▶ Run Scenario", type="primary", use_container_width=True)
    with col_exp:
        st.markdown(
            '<div style="padding: 0.5rem 0; font-size: 0.82rem; color: #848456;">'
            "Click Run Scenario to calculate the full alignment tax breakdown including "
            "channel-by-channel analysis, behavioral modifiers, and historical comparisons."
            "</div>",
            unsafe_allow_html=True,
        )

    if run_btn or (
        st.session_state.last_scenario_key == scenario_key
        and st.session_state.scenario_result is not None
    ):
        if run_btn or st.session_state.last_scenario_key != scenario_key:
            with st.spinner("Running scenario analysis..."):
                try:
                    result = engine.run_scenario(
                        country=country,
                        posture=posture,
                        crisis_type=crisis,
                        crisis_severity=severity,
                        power_response_intensity=power_response,
                        time_horizon=time_horizon,
                        opposition_strength=opposition,
                        previous_alignment_depth=prev_depth,
                        institutional_entanglement=entanglement,
                    )
                    compare_df = engine.compare_postures(
                        country=country,
                        crisis_type=crisis,
                        crisis_severity=severity,
                        power_response_intensity=power_response,
                        time_horizon=time_horizon,
                        opposition_strength=opposition,
                        previous_alignment_depth=prev_depth,
                        institutional_entanglement=entanglement,
                    )
                    st.session_state.scenario_result = result
                    st.session_state.compare_df = compare_df
                    st.session_state.last_scenario_key = scenario_key
                except Exception as e:
                    st.error(f"Scenario engine error: {e}")
                    return

        result = st.session_state.scenario_result
        compare_df = st.session_state.compare_df

        if result is None:
            return

        # ── Summary KPIs ───────────────────────────────────────────────────────
        net_tax = result["total_alignment_tax_mn"]
        ci_lo = result["ci_lower_mn"]
        ci_hi = result["ci_upper_mn"]

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric(
                "Net Alignment Tax",
                _fmt_mn(net_tax),
                delta="COST" if net_tax < 0 else "GAIN",
                delta_color="inverse" if net_tax < 0 else "normal",
            )
        with kpi2:
            st.metric("Gross Gains", f"${result['gross_gains_mn']:,.0f}M")
        with kpi3:
            st.metric("Gross Costs", f"${result['gross_costs_mn']:,.0f}M")
        with kpi4:
            st.metric("95% CI Lower", _fmt_mn(ci_lo))
        with kpi5:
            st.metric("95% CI Upper", _fmt_mn(ci_hi))

        # Interpretation text
        st.markdown(
            f'<div class="info-card">{result["interpretation"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Charts: waterfall + comparison side by side ────────────────────────
        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            st.subheader("Channel-by-Channel Breakdown")
            try:
                fig_wf = alignment_tax_waterfall(result)
                st.plotly_chart(fig_wf, use_container_width=True)
            except Exception as e:
                st.error(f"Waterfall chart error: {e}")

        with chart_col2:
            st.subheader("All Postures Compared")
            try:
                fig_comp = scenario_comparison_bar(compare_df)
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Comparison chart error: {e}")

        st.divider()

        # ── Behavioral modifiers panel ─────────────────────────────────────────
        st.subheader("Behavioral Modifiers")
        beh = result["behavioral_modifiers"]

        beh_col1, beh_col2, beh_col3 = st.columns(3)

        with beh_col1:
            cred = beh["credibility"]
            st.markdown("**Commitment Credibility**")
            st.metric("Score", f"{cred['credibility_score']:.3f}", help="0 = incredible, 1 = fully credible")
            st.metric("Benefit Multiplier", f"{cred['benefit_multiplier']:.3f}")
            st.caption(cred["interpretation"])

        with beh_col2:
            aud = beh["audience_costs"]
            st.markdown("**Audience Costs**")
            st.metric("Multiplier", f"{aud['audience_cost_multiplier']:.3f}", help="1 = no audience cost; lower = higher domestic cost")
            st.metric("Regime Constraint", f"{aud['regime_base_constraint']:.3f}")
            st.caption(aud["interpretation"])

        with beh_col3:
            lock = beh["escalation_lockin"]
            st.markdown("**Lock-in Risk**")
            st.metric("Lock-in Probability", f"{lock['lock_in_probability']:.1%}")
            st.metric("Expected Years Locked", f"{lock['expected_years_locked']:.1f} yrs")
            st.caption(lock["interpretation"])

        # Loss aversion
        st.divider()
        loss_col1, loss_col2 = st.columns([2, 3])

        with loss_col1:
            st.subheader("Loss Aversion (Prospect Theory)")
            la = beh["loss_aversion"]
            st.metric("Gains Considered", f"${la['gains_mn']:,.0f}M")
            st.metric("Losses Considered", f"${la['losses_mn']:,.0f}M")
            st.metric("Loss Aversion Weight (λ)", f"{la['lambda_weight']}×")
            st.metric("Prospect Value", _fmt_mn(la["prospect_value_mn"]))
            action_tag = "✅ Rational to act" if la["willingness_to_act"] else "🔴 Status quo preferred"
            st.markdown(f"**{action_tag}**")
            st.caption(la["interpretation"])

        with loss_col2:
            try:
                fig_la = loss_aversion_curve(
                    gains=float(la["gains_mn"]) if la["gains_mn"] > 0 else None,
                    losses=float(la["losses_mn"]) if la["losses_mn"] > 0 else None,
                    lambda_weight=la["lambda_weight"],
                )
                st.plotly_chart(fig_la, use_container_width=True)
            except Exception as e:
                st.error(f"Loss aversion curve error: {e}")

        # ── Historical precedents ──────────────────────────────────────────────
        st.divider()
        st.subheader("Historical Precedent Comparisons")

        hist = result["historical_comparisons"]
        if hist:
            for comp in hist:
                impact_sign = "📈" if (comp.get("impact_usd_mn") or 0) > 0 else "📉"
                conf_color = {
                    "high": "#20808D",
                    "medium": "#FFC553",
                    "low": "#A84B2F",
                }.get(str(comp.get("confidence", "medium")), "#848456")

                st.markdown(
                    f"""
                    <div class="info-card">
                        <strong>{impact_sign} {comp['country']} ({comp['year']})</strong>
                        &nbsp;
                        <span style="font-size:0.75rem; background:{conf_color}; color:white;
                                     padding:0.1rem 0.4rem; border-radius:3px;">
                            {str(comp.get('confidence','?')).upper()} CONFIDENCE
                        </span>
                        <br>
                        {comp['event']}
                        <br>
                        <span style="color:#848456; font-size:0.82rem;">
                            Impact: <strong>${comp.get('impact_usd_mn','?')}M</strong>
                            ({comp.get('impact_pct','?')}%) ·
                            Duration: {comp.get('duration_years','?')} years
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No directly comparable historical precedents found for this scenario.")

        # Historical timeline chart
        try:
            prec_df = _precedents()
            fig_tl = historical_precedent_timeline(prec_df)
            st.plotly_chart(fig_tl, use_container_width=True)
        except Exception as e:
            st.error(f"Timeline chart error: {e}")

        # ── Panel estimator ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Panel Estimator — DiD Estimates")
        st.markdown(
            '<div class="info-card">'
            "Simplified difference-in-differences estimates using historical alignment "
            "shift episodes. These are event-study analogues, NOT causal estimates. "
            "Standard errors are bootstrapped from N precedent observations."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            agoa_est = panel.estimate_agoa_revocation_effect(country)
            cn_est = panel.estimate_chinese_investment_response(country, posture)
            panel_summary = panel.full_panel_summary(country)

            est_col1, est_col2 = st.columns(2)
            with est_col1:
                st.markdown("**AGOA Revocation Effect (DiD)**")
                st.metric(
                    "Mean impact",
                    f"{agoa_est['estimate_pct']:.1f}%",
                    help=f"Bootstrapped SE: {agoa_est['se_pct']:.2f}pp",
                )
                st.metric("Absolute impact", f"${agoa_est['absolute_impact_mn']:.0f}M")
                st.caption(
                    f"N={agoa_est['n_episodes']} episodes · "
                    f"95% CI: [{agoa_est['ci_pct'][0]:.1f}%, {agoa_est['ci_pct'][1]:.1f}%] · "
                    f"Precedents: {', '.join(agoa_est['precedent_countries'])}"
                )
            with est_col2:
                st.markdown(f"**Chinese Investment Response ({posture.replace('_',' ')})**")
                st.metric(
                    "Estimated change",
                    f"{cn_est['estimate_pct']:+.1f}%",
                    help=f"SE: {cn_est['se_pct']:.2f}pp",
                )
                st.metric("Absolute impact", _fmt_mn(cn_est["absolute_impact_mn"]))
                st.caption(cn_est["interpretation"])

            st.dataframe(
                panel_summary.style.format(
                    {
                        "agoa_impact_mn": "${:,.0f}M",
                        "cn_invest_impact_mn": "${:,.0f}M",
                        "agoa_ci_lower": "{:.1f}%",
                        "agoa_ci_upper": "{:.1f}%",
                        "cn_ci_lower": "{:.1f}%",
                        "cn_ci_upper": "{:.1f}%",
                    }
                ),
                use_container_width=True,
            )
            st.caption(agoa_est["disclaimer"])
        except Exception as e:
            st.error(f"Panel estimator error: {e}")

        # ── Export ─────────────────────────────────────────────────────────────
        st.divider()
        csv_bytes = _scenario_to_csv(result)
        st.download_button(
            label="⬇ Download Scenario Results (CSV)",
            data=csv_bytes,
            file_name=f"alignment_tax_{country}_{posture}_{crisis}.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# ── PAGE 5: GHANA DEEP DIVE ───────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def page_ghana():
    st.markdown(
        '<div class="alignment-header">'
        "<h1>"
        '<span class="layer-badge">Layer 4</span>'
        "Ghana Deep Dive"
        "</h1>"
        "<p>Full granular exposure analysis · AGOA vulnerability · "
        "Chinese deal portfolio · IMF/MCC dependency · Commodity routing risk</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _disclaimer()

    try:
        ghana_data = _ghana()
        econ_df = _econ()
        engine = _engine()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return

    # ── Top KPIs ───────────────────────────────────────────────────────────────
    g_row = econ_df[econ_df["country"] == "Ghana"].iloc[0]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("US Trade", _fmt_bn(g_row["us_trade_bn"]))
    with k2:
        st.metric("China Trade", _fmt_bn(g_row["china_trade_bn"]))
    with k3:
        st.metric("AGOA Value", f"${g_row['agoa_value_mn']:.0f}M")
    with k4:
        st.metric("US ODA", f"${g_row['us_oda_mn']:.0f}M")
    with k5:
        st.metric("Chinese Debt %", f"{g_row['china_debt_pct_external']:.1f}%")
    with k6:
        st.metric("China Trade Dom.", f"{g_row['china_trade_dominance']:.1%}")

    st.divider()

    # ── 6-panel dashboard ──────────────────────────────────────────────────────
    st.subheader("Ghana Dashboard — Alignment Tax Exposure")
    try:
        fig_gh = ghana_dashboard(ghana_data)
        st.plotly_chart(fig_gh, use_container_width=True)
    except Exception as e:
        st.error(f"Ghana dashboard error: {e}")

    st.divider()

    # ── Scenario analysis: all 3 postures (Iran crisis) ───────────────────────
    st.subheader("Iran Crisis — All Posture Scenarios for Ghana")

    with st.spinner("Computing Ghana posture comparison..."):
        try:
            ghana_compare = engine.compare_postures("Ghana", crisis_type="iran")
            fig_ghana_comp = scenario_comparison_bar(ghana_compare)
            st.plotly_chart(fig_ghana_comp, use_container_width=True)

            # Tabular results
            display_compare = ghana_compare.copy()
            display_compare["total_tax_mn"] = display_compare["total_tax_mn"].apply(
                lambda v: f"${v:+,.0f}M"
            )
            display_compare["gross_gains_mn"] = display_compare["gross_gains_mn"].apply(
                lambda v: f"${v:,.0f}M"
            )
            display_compare["gross_costs_mn"] = display_compare["gross_costs_mn"].apply(
                lambda v: f"${v:,.0f}M"
            )
            display_compare["credibility"] = display_compare["credibility"].apply(
                lambda v: f"{v:.3f}"
            )
            display_compare["lock_in_prob"] = display_compare["lock_in_prob"].apply(
                lambda v: f"{v:.1%}"
            )
            display_compare["ci_lower"] = display_compare["ci_lower"].apply(
                lambda v: f"${v:+,.0f}M"
            )
            display_compare["ci_upper"] = display_compare["ci_upper"].apply(
                lambda v: f"${v:+,.0f}M"
            )
            display_compare.columns = [
                "Posture",
                "Net Tax",
                "Gross Gains",
                "Gross Costs",
                "Credibility",
                "Lock-in Prob.",
                "CI Lower",
                "CI Upper",
            ]
            st.dataframe(display_compare.set_index("Posture"), use_container_width=True)
        except Exception as e:
            st.error(f"Ghana scenario comparison error: {e}")

    st.divider()

    # ── AGOA vulnerability table ───────────────────────────────────────────────
    tab_agoa, tab_deals, tab_imf, tab_commod = st.tabs(
        ["AGOA Vulnerability", "Chinese Deals", "IMF/MCC Dependency", "Commodity Routing"]
    )

    with tab_agoa:
        st.subheader("AGOA Sector Vulnerability Assessment")
        st.markdown(
            '<div class="info-card">'
            "AGOA (African Growth and Opportunity Act) provides duty-free access to the "
            "US market. Ghana's ~$380M annual AGOA-eligible exports would be at risk if "
            "eligibility is suspended following an anti-US posture during the Iran crisis. "
            "Apparel and textiles face the highest tariff margins (12%)."
            "</div>",
            unsafe_allow_html=True,
        )

        agoa_df = ghana_data["agoa_sectors"].copy()

        def _vuln_color(val):
            colors = {
                "very high": "background-color: rgba(168,75,47,0.25)",
                "high": "background-color: rgba(168,75,47,0.12)",
                "medium": "background-color: rgba(255,197,83,0.20)",
                "low": "background-color: rgba(32,128,141,0.12)",
            }
            return colors.get(str(val).lower(), "")

        styled_agoa = (
            agoa_df.style
            .format(
                {
                    "exports_mn_usd": "${:,.1f}M",
                    "agoa_margin_pct": "{:.1f}%",
                    "jobs_supported": "{:,}",
                }
            )
            .applymap(_vuln_color, subset=["vulnerability"])
        )
        st.dataframe(styled_agoa, use_container_width=True)

        # Totals
        total_agoa = agoa_df["exports_mn_usd"].sum()
        high_risk = agoa_df[agoa_df["vulnerability"].isin(["high", "very high"])]["exports_mn_usd"].sum()
        total_jobs = agoa_df["jobs_supported"].sum()

        ta1, ta2, ta3 = st.columns(3)
        with ta1:
            st.metric("Total AGOA Exports", f"${total_agoa:.0f}M")
        with ta2:
            st.metric("High/Very High Risk", f"${high_risk:.0f}M", delta=f"{high_risk/total_agoa:.0%} of total", delta_color="inverse")
        with ta3:
            st.metric("Jobs at Risk (High+)", f"{agoa_df[agoa_df['vulnerability'].isin(['high','very high'])]['jobs_supported'].sum():,}")

    with tab_deals:
        st.subheader("Chinese Infrastructure & Economic Deal Portfolio")
        st.markdown(
            '<div class="info-card">'
            "Ghana's Chinese deal portfolio totals ~$4.03B, with the Sinohydro "
            "bauxite-for-infrastructure swap ($2B) representing the largest and most "
            "strategically significant commitment. These deals create lock-in through "
            "resource-backed repayment structures and cross-default clauses."
            "</div>",
            unsafe_allow_html=True,
        )

        deals_df = ghana_data["chinese_deals"].copy()

        status_colors = {
            "active": "background-color: rgba(32,128,141,0.15)",
            "completed": "background-color: rgba(132,132,86,0.15)",
            "negotiating": "background-color: rgba(255,197,83,0.25)",
        }

        def _status_color(val):
            return status_colors.get(str(val).lower(), "")

        styled_deals = (
            deals_df.style
            .format({"value_bn": "${:.2f}B"})
            .applymap(_status_color, subset=["status"])
            .set_properties(**{"font-size": "12px"})
        )
        st.dataframe(styled_deals, use_container_width=True)

        total_cn = deals_df["value_bn"].sum()
        active_cn = deals_df[deals_df["status"] == "active"]["value_bn"].sum()
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Portfolio", f"${total_cn:.2f}B")
        with d2:
            st.metric("Active Commitments", f"${active_cn:.2f}B")
        with d3:
            st.metric("Deals Count", len(deals_df))

        st.markdown(
            '<div class="info-card warning">'
            "<strong>Strategic concern:</strong> The Sinohydro bauxite-for-infrastructure "
            "arrangement requires annual bauxite off-take equivalents of ~$100M. Under a "
            "US alignment posture, China may invoke renegotiation clauses, threatening "
            "road infrastructure project continuity and future BRI financing access."
            "</div>",
            unsafe_allow_html=True,
        )

    with tab_imf:
        st.subheader("IMF & MCC Dependency Mapping")

        imf_df = ghana_data["imf_program"]
        mcc_df = ghana_data["mcc_compact"]

        i_col1, i_col2 = st.columns(2)

        with i_col1:
            st.markdown("**IMF Extended Credit Facility (2023)**")
            imf_row = imf_df.iloc[0]
            im1, im2 = st.columns(2)
            with im1:
                st.metric("Total Program", f"${imf_row['total_bn']:.1f}B")
                st.metric("Disbursed to Date", f"${imf_row['disbursed_bn']:.1f}B")
            with im2:
                st.metric("Status", imf_row["program_status"].title())
                st.metric("US Vote Share in IMF", f"{imf_row['us_vote_share_imf']}%")

            st.markdown(
                f"**Conditionality:** {imf_row['conditions']}"
            )
            st.markdown(
                f"**Alignment note:** {imf_row['alignment_conditionality']}"
            )
            st.markdown(
                '<div class="info-card">'
                "The US holds 16.5% of IMF voting shares — the only member with a "
                "de facto veto over major IMF decisions. Ghana's $3B ECF program "
                "critically depends on continued US support. A China alignment posture "
                "would increase IMF opposition risk by an estimated 20 percentage points."
                "</div>",
                unsafe_allow_html=True,
            )

        with i_col2:
            st.markdown("**MCC Compact History**")
            styled_mcc = (
                mcc_df.style
                .format({"value_mn": "${:,.0f}M", "leverage_ratio": "{:.1f}×"})
                .set_properties(**{"font-size": "12px"})
            )
            st.dataframe(styled_mcc, use_container_width=True)
            total_mcc = mcc_df["value_mn"].sum()
            st.metric("Total MCC Commitment", f"${total_mcc:,.0f}M")
            st.markdown(
                '<div class="info-card">'
                "MCC compacts require recipients to meet governance, rule-of-law, and "
                "democratic accountability thresholds. They cannot be maintained in "
                "parallel with significant anti-US diplomatic postures — as Tanzania's "
                "2016 suspension ($480M lost) demonstrated."
                "</div>",
                unsafe_allow_html=True,
            )

    with tab_commod:
        st.subheader("Commodity Routing & Export Destination Risk")
        st.markdown(
            '<div class="info-card">'
            "Ghana's commodity export structure creates asymmetric vulnerability: "
            "bauxite and manganese are China-dominated (68–72% of exports), while "
            "gold and cocoa flow primarily through European and UAE channels. "
            "An alignment posture that antagonises China would immediately affect "
            "bauxite off-take pricing under the Sinohydro arrangement."
            "</div>",
            unsafe_allow_html=True,
        )

        comm_df = ghana_data["commodity_exports"]
        styled_comm = (
            comm_df.style
            .format({"total_export_bn": "${:.2f}B"})
            .set_properties(**{"font-size": "12px"})
            .bar(
                subset=["china_pct"],
                color="rgba(168,75,47,0.25)",
            )
            .bar(
                subset=["us_pct"],
                color="rgba(32,128,141,0.25)",
            )
        )
        st.dataframe(styled_comm, use_container_width=True)

        total_exports = comm_df["total_export_bn"].sum()
        china_weighted = (comm_df["total_export_bn"] * comm_df["china_pct"] / 100).sum()
        us_weighted = (comm_df["total_export_bn"] * comm_df["us_pct"] / 100).sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Commodity Exports", f"${total_exports:.2f}B")
        with c2:
            st.metric("China-Routed (weighted)", f"${china_weighted:.2f}B",
                      delta=f"{china_weighted/total_exports:.1%} share")
        with c3:
            st.metric("US-Routed (weighted)", f"${us_weighted:.2f}B",
                      delta=f"{us_weighted/total_exports:.1%} share")

    # ── Export Ghana data ──────────────────────────────────────────────────────
    st.divider()
    export_frames = {
        "agoa_sectors": ghana_data["agoa_sectors"],
        "chinese_deals": ghana_data["chinese_deals"],
        "commodity_exports": ghana_data["commodity_exports"],
        "imf_program": ghana_data["imf_program"],
        "mcc_compact": ghana_data["mcc_compact"],
        "economic_timeline": ghana_data["economic_timeline"],
    }

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in export_frames.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    buf.seek(0)

    st.download_button(
        label="⬇ Download Ghana Report Data (Excel)",
        data=buf.getvalue(),
        file_name="ghana_alignment_tax_deep_dive.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════════════

_PAGE_MAP = {
    "🌍  Overview": page_overview,
    "📡  Alignment Signal Coding": page_signal_coding,
    "💰  Economic Dependency": page_economic_dependency,
    "⚖️  Alignment Tax Calculator": page_calculator,
    "🇬🇭  Ghana Deep Dive": page_ghana,
}

try:
    _PAGE_MAP[page]()
except KeyError:
    st.error(f"Unknown page: {page}")
except Exception as e:
    st.error(
        f"An unexpected error occurred loading this page: {e}\n\n"
        "Please check that all required modules (data_generator.py, "
        "alignment_model.py, visualizations.py) are present in the "
        "application directory."
    )
    import traceback
    with st.expander("Technical details"):
        st.code(traceback.format_exc())

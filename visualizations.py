"""
visualizations.py
=================
Plotly-based visualization functions for the Alignment Tax application.

All charts use Perplexity brand colors and are export-ready (SVG/PNG via Kaleido).
Charts use plotly.graph_objects for full control over every visual element.

Color palette
-------------
Primary:   #20808D (muted teal)
Secondary: #A84B2F (terra), #1B474D (dark teal), #BCE2E7 (light cyan),
           #944454 (mauve), #FFC553 (gold), #848456 (olive), #6E522B (brown)
Background:#FCFAF6 (off-white), #F3F3EE (paper white)
Text:      #13343B (dark teal/offblack)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any

# ── Brand palette ─────────────────────────────────────────────────────────────
BRAND = {
    "primary":     "#20808D",
    "terra":       "#A84B2F",
    "dark_teal":   "#1B474D",
    "light_cyan":  "#BCE2E7",
    "mauve":       "#944454",
    "gold":        "#FFC553",
    "olive":       "#848456",
    "brown":       "#6E522B",
    "bg_offwhite": "#FCFAF6",
    "bg_paper":    "#F3F3EE",
    "text":        "#13343B",
    "grid":        "#E5E3D4",
    "border":      "#D6D4C5",
}

# Ordered sequence for multi-series charts
PALETTE = [
    BRAND["primary"], BRAND["terra"],     BRAND["dark_teal"],
    BRAND["light_cyan"], BRAND["mauve"],  BRAND["gold"],
    BRAND["olive"],  BRAND["brown"],
]

POSTURE_COLORS: dict[str, str] = {
    "US_ALIGNMENT":    BRAND["primary"],
    "CHINA_ALIGNMENT": BRAND["terra"],
    "NEUTRALITY":      BRAND["gold"],
}

POWER_COLORS: dict[str, str] = {
    "US":     BRAND["primary"],
    "China":  BRAND["terra"],
    "Russia": BRAND["mauve"],
}

REGION_COLORS: dict[str, str] = {
    "West Africa":     BRAND["primary"],
    "East Africa":     BRAND["terra"],
    "Southern Africa": BRAND["dark_teal"],
    "North Africa":    BRAND["gold"],
    "Central Africa":  BRAND["mauve"],
}

# ── Common layout defaults ────────────────────────────────────────────────────
def _base_layout(**overrides) -> dict:
    """Return a base Plotly layout dict with brand styling."""
    layout = dict(
        paper_bgcolor = BRAND["bg_offwhite"],
        plot_bgcolor  = BRAND["bg_paper"],
        font          = dict(family="Inter, Arial, sans-serif",
                             color=BRAND["text"], size=12),
        title_font    = dict(family="Inter, Arial, sans-serif",
                             color=BRAND["text"], size=16, weight=600),
        margin        = dict(l=60, r=40, t=70, b=60),
        hoverlabel    = dict(bgcolor="white", bordercolor=BRAND["border"],
                             font_size=12, font_family="Inter, Arial, sans-serif"),
        legend        = dict(bgcolor="rgba(252,250,246,0.9)",
                             bordercolor=BRAND["border"], borderwidth=1,
                             font_size=11),
    )
    layout.update(overrides)
    return layout


# ═════════════════════════════════════════════════════════════════════════════
# 1. alignment_space_3d
# ═════════════════════════════════════════════════════════════════════════════

def alignment_space_3d(countries_df: pd.DataFrame) -> go.Figure:
    """
    3D scatter plot showing African countries in US–China–Russia alignment space.

    Parameters
    ----------
    countries_df : DataFrame with columns:
        country, region, us_alignment, china_alignment, russia_alignment
        (use latest year from generate_unga_voting_data())

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for region, color in REGION_COLORS.items():
        subset = countries_df[countries_df["region"] == region]
        if subset.empty:
            continue

        hover_text = [
            f"<b>{row['country']}</b><br>"
            f"US alignment: {row['us_alignment']:.3f}<br>"
            f"China alignment: {row['china_alignment']:.3f}<br>"
            f"Russia alignment: {row['russia_alignment']:.3f}<br>"
            f"Region: {row['region']}"
            for _, row in subset.iterrows()
        ]

        fig.add_trace(go.Scatter3d(
            x=subset["us_alignment"].values,
            y=subset["china_alignment"].values,
            z=subset["russia_alignment"].values,
            mode="markers+text",
            name=region,
            text=subset["country"].values,
            textposition="top center",
            textfont=dict(size=9, color=BRAND["text"]),
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                size=8, color=color, opacity=0.85,
                line=dict(color="white", width=0.8),
            ),
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="African Countries in Great-Power Alignment Space",
                x=0.5, xanchor="center",
            ),
            scene=dict(
                xaxis=dict(title=dict(text="US Alignment (UNGA vote agreement rate)", font=dict(size=11)),
                           range=[0.05, 0.55], gridcolor=BRAND["grid"],
                           backgroundcolor=BRAND["bg_paper"]),
                yaxis=dict(title=dict(text="China Alignment (UNGA vote agreement rate)", font=dict(size=11)),
                           range=[0.50, 0.95], gridcolor=BRAND["grid"],
                           backgroundcolor=BRAND["bg_paper"]),
                zaxis=dict(title=dict(text="Russia Alignment (UNGA vote agreement rate)", font=dict(size=11)),
                           range=[0.30, 0.90], gridcolor=BRAND["grid"],
                           backgroundcolor=BRAND["bg_paper"]),
                bgcolor=BRAND["bg_paper"],
            ),
            height=650,
        )
    )

    # Annotation: the neutrality plane
    fig.add_annotation(
        text=(
            "Note: Most African states cluster in high China–Russia alignment "
            "quadrant, reflecting UNGA voting patterns. "
            "US agreement rates typically 20–35%."
        ),
        xref="paper", yref="paper", x=0.0, y=-0.02,
        showarrow=False, font=dict(size=10, color=BRAND["olive"]),
        align="left",
    )

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2. economic_exposure_radar
# ═════════════════════════════════════════════════════════════════════════════

def economic_exposure_radar(
    country: str,
    dependency_data: pd.DataFrame,
) -> go.Figure:
    """
    Radar/spider chart showing economic exposure to each great power
    across multiple channels.

    Parameters
    ----------
    country        : country name
    dependency_data: from generate_economic_dependency()

    Returns
    -------
    go.Figure
    """
    row = dependency_data[dependency_data["country"] == country]
    if row.empty:
        raise ValueError(f"Country '{country}' not found in dependency_data.")
    d = row.iloc[0]

    # Normalise each channel to 0–1 for radar
    def norm(val: float, max_val: float) -> float:
        return float(np.clip(val / max_val, 0, 1))

    channels = ["Trade\nVolume", "FDI\nStock", "Development\nAssistance",
                "Debt\nExposure", "Military\nCooperation"]

    us_vals  = [
        norm(d["us_trade_bn"],       50),
        norm(d["us_fdi_stock_bn"],   30),
        norm(d["us_oda_mn"],       2000),
        norm(0,                    100),   # US holds minimal African debt
        norm(d["us_mil_coop_idx"],   1.0),
    ]
    cn_vals  = [
        norm(d["china_trade_bn"],        50),
        norm(d["china_fdi_stock_bn"],    30),
        norm(d["china_oda_mn"],        2000),
        norm(d["china_debt_pct_external"], 50),
        norm(d["china_mil_coop_idx"],     1.0),
    ]
    ru_vals  = [
        norm(d["russia_trade_bn"],       10),
        norm(d["russia_fdi_stock_bn"],    5),
        norm(d["russia_oda_mn"],        500),
        norm(0,                          50),
        norm(d["russia_mil_coop_idx"],   1.0),
    ]

    # Close the polygon
    categories = channels + [channels[0]]
    us_vals  = us_vals  + [us_vals[0]]
    cn_vals  = cn_vals  + [cn_vals[0]]
    ru_vals  = ru_vals  + [ru_vals[0]]

    rgba_fill = {
        "US":     "rgba(32,128,141,0.12)",
        "China":  "rgba(168,75,47,0.12)",
        "Russia": "rgba(148,68,84,0.12)",
    }

    fig = go.Figure()

    for name, vals, color, fill_key in [
        ("United States", us_vals, POWER_COLORS["US"],     "US"),
        ("China",         cn_vals, POWER_COLORS["China"],  "China"),
        ("Russia",        ru_vals, POWER_COLORS["Russia"], "Russia"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories,
            fill="toself", name=name,
            line=dict(color=color, width=2),
            fillcolor=rgba_fill[fill_key],
            opacity=0.85,
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Economic Exposure to Great Powers — {country}",
                x=0.5, xanchor="center",
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    tickvals=[0.25, 0.5, 0.75, 1.0],
                    ticktext=["Low", "Med", "High", "Max"],
                    gridcolor=BRAND["grid"],
                    linecolor=BRAND["border"],
                    tickfont=dict(size=10),
                ),
                angularaxis=dict(
                    linecolor=BRAND["border"],
                    gridcolor=BRAND["grid"],
                    tickfont=dict(size=11),
                ),
                bgcolor=BRAND["bg_paper"],
            ),
            height=520,
        )
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. alignment_tax_waterfall
# ═════════════════════════════════════════════════════════════════════════════

def alignment_tax_waterfall(scenario_result: dict) -> go.Figure:
    """
    Waterfall chart showing how gains and costs build up to the net alignment tax.

    Parameters
    ----------
    scenario_result : dict returned by ScenarioEngine.run_scenario()

    Returns
    -------
    go.Figure
    """
    bd   = scenario_result["channel_breakdown"]
    posture = scenario_result["posture"]
    country = scenario_result["country"]

    # Build waterfall components
    items: list[tuple[str, float, str]] = []  # (label, value, measure)

    # Gains first
    agoa_gain = bd["agoa"]["secured_mn"]
    if agoa_gain > 0:
        items.append(("AGOA Preferences", agoa_gain, "relative"))

    oda_gain = bd["usaid_mcc"]["secured_mn"]
    if oda_gain > 0:
        items.append(("USAID / MCC Assistance", oda_gain, "relative"))

    imf_gain = bd["imf_wb"]["expected_value_mn"]
    if imf_gain > 0:
        items.append(("IMF / World Bank Support", imf_gain, "relative"))

    msp_gain = bd["msp"]["expected_value_mn"]
    if msp_gain > 0:
        items.append(("Minerals Security Partnership", msp_gain, "relative"))

    cn_gain = bd["chinese_inv"]["secured_mn"]
    if cn_gain > 0:
        items.append(("Chinese FDI / BRI Investment", cn_gain, "relative"))

    bri_relief = bd["chinese_inv"].get("bri_debt_relief_potential_mn", 0)
    if bri_relief > 0:
        items.append(("BRI Debt Relief Potential", bri_relief, "relative"))

    # Costs (negative)
    agoa_cost = -bd["agoa"]["at_risk_mn"]
    if abs(agoa_cost) > 0:
        items.append(("AGOA Revocation Risk", agoa_cost, "relative"))

    oda_cost = -bd["usaid_mcc"]["at_risk_mn"]
    if abs(oda_cost) > 0:
        items.append(("USAID / MCC Reduction Risk", oda_cost, "relative"))

    cn_risk = -bd["chinese_inv"]["at_risk_mn"]
    if abs(cn_risk) > 0:
        items.append(("Chinese Investment Risk", cn_risk, "relative"))

    sanct_cost = -bd["sanctions"]["expected_cost_mn"]
    if abs(sanct_cost) > 0:
        items.append(("Secondary Sanctions Risk", sanct_cost, "relative"))

    commod_cost = -bd["commodity"]["expected_cost_mn"]
    if abs(commod_cost) > 0:
        items.append(("Commodity Routing Risk", commod_cost, "relative"))

    lock_cost = -scenario_result.get("lock_in_cost_mn", 0)
    if abs(lock_cost) > 0:
        items.append(("Lock-in Option Value Lost", lock_cost, "relative"))

    # Behavioral adjustments
    beh = scenario_result["behavioral_modifiers"]
    cred = beh["credibility"]["benefit_multiplier"]
    aud  = beh["audience_costs"]["audience_cost_multiplier"]
    cred_adj = (cred * aud - 1.0) * sum(v for _, v, _ in items if v > 0)
    if abs(cred_adj) > 5:
        items.append(("Credibility × Audience Cost Adj.", cred_adj, "relative"))

    # Total
    items.append(("NET ALIGNMENT TAX", scenario_result["total_alignment_tax_mn"], "total"))

    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    measures = [i[2] for i in items]

    # Colours: green for positive, red/terra for negative, dark teal for total
    marker_colors = []
    for i, (_, v, m) in enumerate(items):
        if m == "total":
            marker_colors.append(BRAND["dark_teal"])
        elif v >= 0:
            marker_colors.append(BRAND["primary"])
        else:
            marker_colors.append(BRAND["terra"])

    fig = go.Figure(go.Waterfall(
        name="Alignment Tax",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"${abs(v):.0f}M" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=BRAND["text"]),
        connector=dict(line=dict(color=BRAND["grid"], width=1.5, dash="dot")),
        increasing=dict(marker=dict(color=BRAND["primary"],
                                    line=dict(color=BRAND["dark_teal"], width=1))),
        decreasing=dict(marker=dict(color=BRAND["terra"],
                                    line=dict(color=BRAND["brown"], width=1))),
        totals=dict(marker=dict(color=BRAND["dark_teal"],
                                line=dict(color=BRAND["text"], width=1.5))),
    ))

    # CI band on total
    ci_lo = scenario_result["ci_lower_mn"]
    ci_hi = scenario_result["ci_upper_mn"]
    net   = scenario_result["total_alignment_tax_mn"]

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=(f"Alignment Tax Waterfall — {country} | {posture.replace('_', ' ')}<br>"
                      f"<sup>Net: ${net:+.0f}M  |  95% CI: [{ci_lo:+.0f}M, {ci_hi:+.0f}M]</sup>"),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(title="Estimated Economic Impact (USD million)",
                       gridcolor=BRAND["grid"], zeroline=True,
                       zerolinecolor=BRAND["text"], zerolinewidth=1.5),
            height=580,
            showlegend=False,
        )
    )

    fig.add_annotation(
        text="⚠ Estimates are historical precedent-based. NOT deterministic predictions.",
        xref="paper", yref="paper", x=0.5, y=-0.22,
        showarrow=False, font=dict(size=10, color=BRAND["olive"]),
        align="center",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4. historical_precedent_timeline
# ═════════════════════════════════════════════════════════════════════════════

def historical_precedent_timeline(episodes_df: pd.DataFrame) -> go.Figure:
    """
    Timeline of historical alignment-consequence episodes.

    Parameters
    ----------
    episodes_df : from generate_historical_precedents()

    Returns
    -------
    go.Figure
    """
    df = episodes_df.copy().sort_values("year")

    # Colour by impact direction
    direction_color = {
        "negative":           BRAND["terra"],
        "negative (US channel)": BRAND["terra"],
        "negative (West channel)": BRAND["terra"],
        "negative (risk)":    BRAND["mauve"],
        "positive":           BRAND["primary"],
        "positive (China)":   BRAND["dark_teal"],
        "positive (Russia)":  BRAND["olive"],
        "mixed":              BRAND["gold"],
    }

    # Assign y-position: stagger events in same year
    year_counts: dict[int, int] = {}
    y_positions = []
    for yr in df["year"]:
        count = year_counts.get(yr, 0)
        y_positions.append(count * 0.25)
        year_counts[yr] = count + 1

    df["y_pos"] = y_positions

    fig = go.Figure()

    # Baseline timeline
    fig.add_shape(
        type="line", x0=2000, x1=2025, y0=0, y1=0,
        line=dict(color=BRAND["border"], width=1.5),
    )

    # Plot episodes
    for _, row in df.iterrows():
        color = direction_color.get(str(row["impact_direction"]), BRAND["gold"])
        impact = row.get("estimated_impact_usd_mn", 0)
        size   = max(10, min(35, abs(float(impact or 0)) / 30 + 8))

        hover = (
            f"<b>{row['country']} ({int(row['year'])})</b><br>"
            f"{row['event']}<br>"
            f"Channel: {row['channel']}<br>"
            f"Impact: ${row.get('estimated_impact_usd_mn','?')}M "
            f"({row.get('estimated_impact_pct','?')}%)<br>"
            f"Confidence: {row.get('confidence','?')}<br>"
            f"Duration: {row.get('duration_years','?')} years"
        )

        fig.add_trace(go.Scatter(
            x=[row["year"]], y=[row["y_pos"]],
            mode="markers+text",
            name=str(row["impact_direction"]),
            showlegend=False,
            text=[row["country"]],
            textposition="top center",
            textfont=dict(size=9, color=BRAND["text"]),
            hovertext=[hover],
            hoverinfo="text",
            marker=dict(
                size=size, color=color,
                symbol="circle",
                line=dict(color="white", width=1.2),
                opacity=0.88,
            ),
        ))

        # Stem line
        fig.add_shape(
            type="line",
            x0=row["year"], x1=row["year"],
            y0=0, y1=row["y_pos"],
            line=dict(color=color, width=1, dash="dot"),
        )

    # Legend shapes
    for label, color in [
        ("Negative impact", BRAND["terra"]),
        ("Positive impact", BRAND["primary"]),
        ("Mixed / risk",    BRAND["gold"]),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color),
            name=label,
        ))

    # Decade markers
    for yr in [2005, 2010, 2015, 2020, 2025]:
        fig.add_vline(x=yr, line_dash="dot", line_color=BRAND["grid"],
                      line_width=1, opacity=0.5)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Historical Alignment–Consequence Episodes in Africa (2000–2025)",
                x=0.5, xanchor="center",
            ),
            xaxis=dict(title="Year", range=[1999, 2026],
                       tickvals=list(range(2000, 2026, 2)),
                       gridcolor=BRAND["grid"]),
            yaxis=dict(title="", showticklabels=False,
                       range=[-0.2, 1.2], gridcolor=BRAND["grid"]),
            height=480,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
        )
    )

    fig.add_annotation(
        text=(
            "Bubble size proportional to estimated USD impact. "
            "Color indicates impact direction. Confidence levels: high/medium/low."
        ),
        xref="paper", yref="paper", x=0.5, y=-0.22,
        showarrow=False, font=dict(size=10, color=BRAND["olive"]),
        align="center",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 5. ghana_dashboard
# ═════════════════════════════════════════════════════════════════════════════

def ghana_dashboard(ghana_data: dict) -> go.Figure:
    """
    Multi-panel dashboard for Ghana deep dive.

    Parameters
    ----------
    ghana_data : dict returned by generate_ghana_deep_dive()

    Returns
    -------
    go.Figure (3×2 subplot layout)
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "AGOA Exports by Sector (USD Million)",
            "Commodity Export Destinations (% of total)",
            "Chinese Infrastructure Deals Portfolio",
            "Economic Timeline: Key Alignment Shocks",
            "IMF & MCC Compact Overview",
            "Trade Dependency: US vs China",
        ],
        specs=[
            [{"type": "bar"},         {"type": "bar"}],
            [{"type": "table"},       {"type": "scatter"}],
            [{"type": "table"},       {"type": "bar"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: AGOA sectors ────────────────────────────────────────────
    agoa = ghana_data["agoa_sectors"]
    fig.add_trace(go.Bar(
        x=agoa["exports_mn_usd"], y=agoa["sector"],
        orientation="h",
        marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(agoa))],
        text=[f"${v:.0f}M" for v in agoa["exports_mn_usd"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Exports: $%{x:.0f}M<br>"
                      "AGOA margin: %{customdata[0]:.1f}%<extra></extra>",
        customdata=agoa[["agoa_margin_pct"]].values,
        showlegend=False,
    ), row=1, col=1)

    # ── Panel 2: Commodity destinations ──────────────────────────────────
    comm = ghana_data["commodity_exports"]
    commodities = comm["commodity"].values
    x_cats = ["US", "China", "EU", "UAE", "Other"]
    colors  = [BRAND["primary"], BRAND["terra"], BRAND["dark_teal"],
               BRAND["gold"], BRAND["olive"]]

    for i, (cat, col) in enumerate(zip(x_cats, colors)):
        col_name = cat.lower() + "_pct"
        fig.add_trace(go.Bar(
            name=cat, x=commodities,
            y=comm[col_name].values,
            marker_color=col,
            showlegend=(True if i < 5 else False),
            legendgroup="destinations",
        ), row=1, col=2)

    # ── Panel 3: Chinese deals table ─────────────────────────────────────
    deals = ghana_data["chinese_deals"]
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Deal</b>", "<b>Year</b>", "<b>Value ($B)</b>",
                    "<b>Sector</b>", "<b>Status</b>"],
            fill_color=BRAND["dark_teal"],
            font=dict(color="white", size=10),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                deals["deal"].values, deals["year"].values,
                [f"${v:.2f}B" for v in deals["value_bn"].values],
                deals["sector"].values, deals["status"].values,
            ],
            fill_color=[[BRAND["bg_paper"] if i % 2 == 0 else BRAND["bg_offwhite"]
                         for i in range(len(deals))]],
            font=dict(size=9, color=BRAND["text"]),
            align="left", height=22,
        ),
    ), row=2, col=1)

    # ── Panel 4: Economic timeline ────────────────────────────────────────
    tl = ghana_data["economic_timeline"]
    valence_color = {
        "US+": BRAND["primary"], "CN+": BRAND["terra"],
        "neutral": BRAND["gold"], "risk": BRAND["mauve"],
    }
    for _, row in tl.iterrows():
        col = valence_color.get(str(row["alignment_valence"]), BRAND["gold"])
        fig.add_trace(go.Scatter(
            x=[row["year"]],
            y=[row["economic_impact_mn"]],
            mode="markers+text",
            text=[row["alignment_valence"]],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=abs(row["economic_impact_mn"]) / 200 + 8,
                        color=col, opacity=0.9,
                        line=dict(color="white", width=1)),
            hovertext=[f"<b>{int(row['year'])}</b><br>{row['event']}<br>"
                       f"Impact: ${row['economic_impact_mn']:+.0f}M"],
            hoverinfo="text",
            showlegend=False,
        ), row=2, col=2)

    # Baseline on timeline (use add_shape with explicit xref/yref)
    fig.add_shape(
        type="line", x0=2010, x1=2026, y0=0, y1=0,
        line=dict(color=BRAND["border"], width=1, dash="dash"),
        xref="x4", yref="y4",
    )

    # ── Panel 5: IMF & MCC table ──────────────────────────────────────────
    imf = ghana_data["imf_program"]
    mcc = ghana_data["mcc_compact"]
    combined = pd.DataFrame([
        {"Program": f"IMF {imf.iloc[0]['facility']}", "Year": imf.iloc[0]["year_approved"],
         "Value": f"${imf.iloc[0]['total_bn']:.1f}B (total)",
         "Status": imf.iloc[0]["program_status"],
         "Notes": "US holds 16.5% IMF vote share; critical for approval"},
        {"Program": f"{mcc.iloc[0]['compact']}",
         "Year": mcc.iloc[0]["year_signed"],
         "Value": f"${mcc.iloc[0]['value_mn']:.0f}M",
         "Status": mcc.iloc[0]["status"],
         "Notes": mcc.iloc[0]["sector"]},
        {"Program": f"{mcc.iloc[1]['compact']}",
         "Year": mcc.iloc[1]["year_signed"],
         "Value": f"${mcc.iloc[1]['value_mn']:.0f}M",
         "Status": mcc.iloc[1]["status"],
         "Notes": mcc.iloc[1]["sector"]},
    ])

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Program</b>", "<b>Year</b>", "<b>Value</b>",
                    "<b>Status</b>", "<b>Notes</b>"],
            fill_color=BRAND["primary"],
            font=dict(color="white", size=10),
            align="left", height=28,
        ),
        cells=dict(
            values=[combined[c].values for c in combined.columns],
            fill_color=[[BRAND["bg_paper"] if i % 2 == 0 else BRAND["bg_offwhite"]
                         for i in range(len(combined))]],
            font=dict(size=9, color=BRAND["text"]),
            align="left", height=22,
        ),
    ), row=3, col=1)

    # ── Panel 6: US vs China trade comparison ────────────────────────────
    trade_channels = ["Trade Volume\n($B)", "FDI Stock\n($B)", "ODA\n($100M)",
                      "Mil. Coop\nIndex"]
    us_bars  = [3.2, 3.5, 6.2, 0.55]
    cn_bars  = [10.5, 5.8, 1.8, 0.25]

    fig.add_trace(go.Bar(
        name="United States", x=trade_channels, y=us_bars,
        marker_color=BRAND["primary"],
        text=[f"{v:.1f}" for v in us_bars], textposition="outside",
        legendgroup="trade_comp",
    ), row=3, col=2)
    fig.add_trace(go.Bar(
        name="China", x=trade_channels, y=cn_bars,
        marker_color=BRAND["terra"],
        text=[f"{v:.1f}" for v in cn_bars], textposition="outside",
        legendgroup="trade_comp",
    ), row=3, col=2)

    # ── Global layout ─────────────────────────────────────────────────────
    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Ghana Deep Dive — Alignment Tax Exposure Analysis",
                x=0.5, xanchor="center",
            ),
            height=1100,
            barmode="stack",   # stacked for destination chart
        )
    )

    # Fix barmode for panels that should be grouped
    fig.update_layout(barmode="group")  # group globally; stacked applied per trace if needed

    # Axis styling
    fig.update_xaxes(gridcolor=BRAND["grid"], tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=BRAND["grid"], tickfont=dict(size=9))

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6. alignment_heatmap
# ═════════════════════════════════════════════════════════════════════════════

def alignment_heatmap(
    countries_df: pd.DataFrame,
    metric: str = "china_alignment",
    year: Optional[int] = None,
) -> go.Figure:
    """
    Heatmap-style bar chart of African countries showing alignment metric.

    Parameters
    ----------
    countries_df : from generate_unga_voting_data() (or similar)
    metric       : column to display (e.g. 'china_alignment', 'us_alignment')
    year         : filter to specific year; None = latest available

    Returns
    -------
    go.Figure
    """
    df = countries_df.copy()
    if year is not None:
        df = df[df["year"] == year]
    else:
        df = df.sort_values("year", ascending=False).drop_duplicates("country")

    df = df.sort_values(metric, ascending=True)

    metric_labels = {
        "china_alignment":  "China Alignment (UNGA agreement rate)",
        "us_alignment":     "US Alignment (UNGA agreement rate)",
        "russia_alignment": "Russia Alignment (UNGA agreement rate)",
    }

    # Color-encode by region
    bar_colors = [REGION_COLORS.get(r, BRAND["gold"]) for r in df["region"]]

    # Custom colorscale for heatmap effect on bars
    def val_to_color(v: float, metric_name: str) -> str:
        if "china" in metric_name:
            r = int(32 + (168 - 32) * (1 - v))
            g = int(128 + (75 - 128) * (1 - v))
            b = int(141 + (47 - 141) * (1 - v))
        elif "us" in metric_name:
            r = int(27 + (32 - 27) * v)
            g = int(71 + (128 - 71) * v)
            b = int(77 + (141 - 77) * v)
        else:
            r = int(148 + (32 - 148) * v)
            g = int(68 + (128 - 68) * v)
            b = int(84 + (141 - 84) * v)
        return f"rgb({r},{g},{b})"

    bar_colors = [val_to_color(v, metric) for v in df[metric]]

    hover_text = [
        f"<b>{row['country']}</b><br>"
        f"{metric_labels.get(metric, metric)}: {row[metric]:.3f}<br>"
        f"Region: {row['region']}<br>"
        f"Year: {row.get('year', 'N/A')}"
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df[metric].values,
        y=df["country"].values,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="white", width=0.5),
        ),
        text=[f"{v:.3f}" for v in df[metric].values],
        textposition="outside",
        textfont=dict(size=9, color=BRAND["text"]),
        hovertext=hover_text,
        hoverinfo="text",
    ))

    year_label = str(year) if year else "Latest"
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"African Countries by {metric_labels.get(metric, metric)} ({year_label})",
                x=0.5, xanchor="center",
            ),
            xaxis=dict(
                title=metric_labels.get(metric, metric),
                range=[0, 1.05], gridcolor=BRAND["grid"],
            ),
            yaxis=dict(gridcolor=BRAND["grid"], tickfont=dict(size=9)),
            height=900,
            showlegend=False,
        )
    )

    # Threshold lines
    if "china" in metric:
        fig.add_vline(x=0.72, line_dash="dash", line_color=BRAND["terra"],
                      annotation_text="Avg China-Africa (0.72)",
                      annotation_position="top right",
                      annotation_font_size=10)
    elif "us" in metric:
        fig.add_vline(x=0.28, line_dash="dash", line_color=BRAND["primary"],
                      annotation_text="Avg US-Africa (0.28)",
                      annotation_position="top right",
                      annotation_font_size=10)
    elif "russia" in metric:
        fig.add_vline(x=0.55, line_dash="dash", line_color=BRAND["mauve"],
                      annotation_text="Avg Russia-Africa (0.55)",
                      annotation_position="top right",
                      annotation_font_size=10)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 7. scenario_comparison_bar
# ═════════════════════════════════════════════════════════════════════════════

def scenario_comparison_bar(scenarios: pd.DataFrame) -> go.Figure:
    """
    Side-by-side bar chart comparing alignment posture costs/benefits.

    Parameters
    ----------
    scenarios : DataFrame from ScenarioEngine.compare_postures()
                with columns: posture, total_tax_mn, gross_gains_mn,
                gross_costs_mn, ci_lower, ci_upper

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Gross gains bars
    fig.add_trace(go.Bar(
        name="Gross Gains",
        x=scenarios["posture"].values,
        y=scenarios["gross_gains_mn"].values,
        marker_color=BRAND["primary"],
        marker_line=dict(color=BRAND["dark_teal"], width=1.2),
        opacity=0.9,
        text=[f"+${v:.0f}M" for v in scenarios["gross_gains_mn"]],
        textposition="outside",
    ))

    # Gross costs bars (negative)
    fig.add_trace(go.Bar(
        name="Gross Costs",
        x=scenarios["posture"].values,
        y=[-v for v in scenarios["gross_costs_mn"].values],
        marker_color=BRAND["terra"],
        marker_line=dict(color=BRAND["brown"], width=1.2),
        opacity=0.9,
        text=[f"-${v:.0f}M" for v in scenarios["gross_costs_mn"]],
        textposition="outside",
    ))

    # Net tax markers
    fig.add_trace(go.Scatter(
        name="Net Alignment Tax",
        x=scenarios["posture"].values,
        y=scenarios["total_tax_mn"].values,
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color=BRAND["gold"],
                    line=dict(color=BRAND["dark_teal"], width=2)),
        text=[f"Net: ${v:+.0f}M" for v in scenarios["total_tax_mn"]],
        textposition="top center",
        textfont=dict(size=11, color=BRAND["text"]),
    ))

    # Error bars for CI
    if "ci_lower" in scenarios.columns and "ci_upper" in scenarios.columns:
        err_minus = scenarios["total_tax_mn"] - scenarios["ci_lower"]
        err_plus  = scenarios["ci_upper"] - scenarios["total_tax_mn"]
        fig.add_trace(go.Scatter(
            name="95% Confidence Interval",
            x=scenarios["posture"].values,
            y=scenarios["total_tax_mn"].values,
            mode="markers",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            error_y=dict(
                type="data", symmetric=False,
                array=err_plus.values,
                arrayminus=err_minus.values,
                color=BRAND["dark_teal"],
                thickness=2, width=8,
            ),
        ))

    fig.add_hline(y=0, line_dash="solid", line_color=BRAND["text"],
                  line_width=1.5, opacity=0.4)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Alignment Posture Comparison — Gains, Costs & Net Tax",
                x=0.5, xanchor="center",
            ),
            xaxis=dict(
                title="Alignment Posture",
                ticktext=["US Alignment", "China Alignment", "Neutrality"],
                tickvals=["US_ALIGNMENT", "CHINA_ALIGNMENT", "NEUTRALITY"],
            ),
            yaxis=dict(title="Estimated Economic Impact (USD million)",
                       gridcolor=BRAND["grid"], zeroline=True,
                       zerolinecolor=BRAND["text"], zerolinewidth=1.5),
            barmode="group",
            height=520,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
        )
    )

    fig.add_annotation(
        text="Diamond = net alignment tax (after behavioral adjustments). Error bars = 95% CI.",
        xref="paper", yref="paper", x=0.5, y=-0.24,
        showarrow=False, font=dict(size=10, color=BRAND["olive"]),
        align="center",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 8. credibility_signal_chart
# ═════════════════════════════════════════════════════════════════════════════

def credibility_signal_chart(
    country: str,
    voting_data: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Show how a country's credibility signals compound over time.

    Plots:
    - US/China/Russia alignment scores (2010–2025)
    - Credibility bands (consistency envelope)
    - Key alignment events annotated

    Parameters
    ----------
    country      : country name
    voting_data  : from generate_unga_voting_data(); loaded if None

    Returns
    -------
    go.Figure
    """
    if voting_data is None:
        from data_generator import generate_unga_voting_data
        voting_data = generate_unga_voting_data()

    df = voting_data[
        (voting_data["country"] == country) &
        (voting_data["year"] >= 2010)
    ].copy().sort_values("year")

    if df.empty:
        raise ValueError(f"No voting data for '{country}'")

    # Rolling 3-year consistency band
    df["us_ma"]  = df["us_alignment"].rolling(3, min_periods=1).mean()
    df["cn_ma"]  = df["china_alignment"].rolling(3, min_periods=1).mean()
    df["ru_ma"]  = df["russia_alignment"].rolling(3, min_periods=1).mean()
    df["us_std"] = df["us_alignment"].rolling(3, min_periods=1).std().fillna(0.01)
    df["cn_std"] = df["china_alignment"].rolling(3, min_periods=1).std().fillna(0.01)

    fig = go.Figure()

    # US alignment band
    fig.add_trace(go.Scatter(
        x=list(df["year"]) + list(df["year"])[::-1],
        y=list(df["us_ma"] + df["us_std"]) + list(df["us_ma"] - df["us_std"])[::-1],
        fill="toself", fillcolor="rgba(32,128,141,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["us_alignment"],
        mode="lines+markers", name="US Alignment",
        line=dict(color=BRAND["primary"], width=2.5),
        marker=dict(size=5, color=BRAND["primary"]),
    ))

    # China alignment band
    fig.add_trace(go.Scatter(
        x=list(df["year"]) + list(df["year"])[::-1],
        y=list(df["cn_ma"] + df["cn_std"]) + list(df["cn_ma"] - df["cn_std"])[::-1],
        fill="toself", fillcolor="rgba(168,75,47,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["china_alignment"],
        mode="lines+markers", name="China Alignment",
        line=dict(color=BRAND["terra"], width=2.5),
        marker=dict(size=5, color=BRAND["terra"]),
    ))

    # Russia alignment
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["russia_alignment"],
        mode="lines+markers", name="Russia Alignment",
        line=dict(color=BRAND["mauve"], width=2, dash="dash"),
        marker=dict(size=4, color=BRAND["mauve"]),
    ))

    # Credibility score as secondary axis
    us_consistency  = 1 - df["us_alignment"].std()
    cn_consistency  = 1 - df["china_alignment"].std()
    cred_series = (us_consistency + cn_consistency) / 2

    fig.add_trace(go.Scatter(
        x=df["year"],
        y=[cred_series] * len(df),
        mode="lines", name="Credibility Score",
        line=dict(color=BRAND["gold"], width=1.5, dash="dot"),
        yaxis="y2",
    ))

    # Key event annotations
    events = {
        2022: "Ukraine vote",
        2020: "COVID-19 realignment",
        2016: "UNGA shifts",
    }
    for yr, label in events.items():
        if yr in df["year"].values:
            fig.add_vline(x=yr, line_dash="dot", line_color=BRAND["grid"],
                          line_width=1, opacity=0.6)
            fig.add_annotation(x=yr, y=0.92, text=label,
                                showarrow=False, yref="y",
                                font=dict(size=9, color=BRAND["olive"]),
                                textangle=-45)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Alignment Credibility Signals — {country} (2010–2025)",
                x=0.5, xanchor="center",
            ),
            xaxis=dict(title="Year", gridcolor=BRAND["grid"],
                       tickvals=list(range(2010, 2026, 2))),
            yaxis=dict(title="UNGA Alignment Score", range=[0.05, 1.0],
                       gridcolor=BRAND["grid"]),
            yaxis2=dict(
                title=dict(text="Credibility Score", font=dict(color=BRAND["gold"])),
                range=[0, 1],
                overlaying="y", side="right",
                showgrid=False,
                tickfont=dict(color=BRAND["gold"], size=10),
            ),
            height=500,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
        )
    )

    fig.add_annotation(
        text=(
            "Shaded band = ±1 SD rolling consistency envelope. "
            "Narrow band = more credible alignment signal."
        ),
        xref="paper", yref="paper", x=0.5, y=-0.24,
        showarrow=False, font=dict(size=10, color=BRAND["olive"]),
        align="center",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 9. loss_aversion_curve
# ═════════════════════════════════════════════════════════════════════════════

def loss_aversion_curve(
    gains: Optional[float] = None,
    losses: Optional[float] = None,
    lambda_weight: float = 2.25,
) -> go.Figure:
    """
    Illustrate prospect theory asymmetry in the diplomatic context.

    Shows the Kahneman-Tversky value function alongside labelled examples
    of AGOA loss (existing benefit) vs MSP/BRI gain (prospective benefit).

    Parameters
    ----------
    gains         : optional specific gain value to annotate (USD million)
    losses        : optional specific loss value to annotate (USD million)
    lambda_weight : loss aversion coefficient (default 2.25)

    Returns
    -------
    go.Figure
    """
    # KT value function: v(x) = x^0.88 for gains; -lambda * (-x)^0.88 for losses
    alpha = 0.88

    x_gains  = np.linspace(0, 800, 300)
    x_losses = np.linspace(-800, 0, 300)

    v_gains  = x_gains ** alpha
    v_losses = -lambda_weight * ((-x_losses) ** alpha)

    fig = go.Figure()

    # Gains curve
    fig.add_trace(go.Scatter(
        x=x_gains, y=v_gains,
        mode="lines", name="Gains (concave)",
        line=dict(color=BRAND["primary"], width=3),
    ))

    # Losses curve
    fig.add_trace(go.Scatter(
        x=x_losses, y=v_losses,
        mode="lines", name=f"Losses (convex, λ={lambda_weight}×)",
        line=dict(color=BRAND["terra"], width=3),
    ))

    # Reference point
    fig.add_vline(x=0, line_dash="solid", line_color=BRAND["text"],
                  line_width=1.5, opacity=0.5)
    fig.add_hline(y=0, line_dash="solid", line_color=BRAND["text"],
                  line_width=1.5, opacity=0.5)

    # Specific annotations
    examples = []
    if gains is not None:
        v_g = float(gains ** alpha)
        examples.append((gains, v_g, f"MSP/BRI gain<br>${gains:.0f}M → PV ${v_g:.0f}",
                         BRAND["primary"], "top left"))
    if losses is not None:
        v_l = float(-lambda_weight * losses ** alpha)
        examples.append((-losses, v_l, f"AGOA loss<br>${losses:.0f}M → PV ${v_l:.0f}",
                         BRAND["terra"], "bottom right"))

    for x_ann, y_ann, text, color, pos in examples:
        fig.add_trace(go.Scatter(
            x=[x_ann], y=[y_ann],
            mode="markers+text",
            text=[text], textposition=pos,
            marker=dict(size=12, color=color,
                        symbol="star", line=dict(color="white", width=1)),
            showlegend=False,
            textfont=dict(size=10, color=color),
        ))
        fig.add_shape(
            type="line", x0=x_ann, y0=0, x1=x_ann, y1=y_ann,
            line=dict(color=color, width=1, dash="dot"),
        )

    # Asymmetry annotation: symmetric loss vs gain
    ref_val = 200.0
    gain_pv = ref_val ** alpha
    loss_pv = -lambda_weight * ref_val ** alpha

    fig.add_annotation(
        x=ref_val * 0.6, y=(gain_pv + loss_pv) / 2,
        text=(f"Same ${ref_val:.0f}M:<br>"
              f"Gain value = {gain_pv:.0f}<br>"
              f"Loss value = {loss_pv:.0f}<br>"
              f"Ratio: {abs(loss_pv/gain_pv):.1f}×"),
        showarrow=True, arrowhead=2, arrowcolor=BRAND["gold"],
        bordercolor=BRAND["gold"], borderwidth=1,
        bgcolor="white", font=dict(size=10, color=BRAND["text"]),
        ax=80, ay=-40,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=(f"Prospect Theory Value Function in Diplomatic Context<br>"
                      f"<sup>Loss aversion coefficient λ = {lambda_weight} "
                      f"(Kahneman-Tversky 1992)</sup>"),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(
                title="Economic Magnitude (USD million)",
                gridcolor=BRAND["grid"],
                zeroline=True, zerolinecolor=BRAND["text"],
            ),
            yaxis=dict(
                title="Subjective Value (Prospect Theory)",
                gridcolor=BRAND["grid"],
                zeroline=True, zerolinecolor=BRAND["text"],
            ),
            height=520,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.16),
        )
    )

    fig.add_annotation(
        text=(
            "African states weight loss of existing AGOA preferences more heavily "
            f"than equivalent gains from new BRI/MSP investment ({lambda_weight}× weighting). "
            "This creates a status-quo bias toward neutrality."
        ),
        xref="paper", yref="paper", x=0.5, y=-0.22,
        showarrow=False, font=dict(size=10, color=BRAND["olive"]),
        align="center", bgcolor="rgba(252,250,246,0.8)",
    )

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Module smoke test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_generator import (
        generate_unga_voting_data, generate_economic_dependency,
        generate_historical_precedents, generate_ghana_deep_dive,
    )
    from alignment_model import ScenarioEngine

    print("Testing alignment_space_3d...")
    unga = generate_unga_voting_data()
    latest = unga.sort_values("year", ascending=False).drop_duplicates("country")
    fig3d = alignment_space_3d(latest)
    print(f"  Traces: {len(fig3d.data)}")

    print("Testing economic_exposure_radar...")
    econ = generate_economic_dependency()
    fig_radar = economic_exposure_radar("Ghana", econ)
    print(f"  Traces: {len(fig_radar.data)}")

    print("Testing alignment_tax_waterfall...")
    engine = ScenarioEngine()
    scenario = engine.run_scenario("Ghana", "NEUTRALITY", "iran")
    fig_wf = alignment_tax_waterfall(scenario)
    print(f"  Traces: {len(fig_wf.data)}")

    print("Testing historical_precedent_timeline...")
    prec = generate_historical_precedents()
    fig_tl = historical_precedent_timeline(prec)
    print(f"  Traces: {len(fig_tl.data)}")

    print("Testing ghana_dashboard...")
    ghana = generate_ghana_deep_dive()
    fig_gh = ghana_dashboard(ghana)
    print(f"  Traces: {len(fig_gh.data)}")

    print("Testing alignment_heatmap...")
    fig_hm = alignment_heatmap(latest, metric="china_alignment")
    print(f"  Traces: {len(fig_hm.data)}")

    print("Testing scenario_comparison_bar...")
    comp = engine.compare_postures("Ghana", "iran")
    fig_comp = scenario_comparison_bar(comp)
    print(f"  Traces: {len(fig_comp.data)}")

    print("Testing credibility_signal_chart...")
    fig_cred = credibility_signal_chart("Ghana", unga)
    print(f"  Traces: {len(fig_cred.data)}")

    print("Testing loss_aversion_curve...")
    fig_la = loss_aversion_curve(gains=300, losses=380)
    print(f"  Traces: {len(fig_la.data)}")

    print("\nAll visualization smoke tests passed.")

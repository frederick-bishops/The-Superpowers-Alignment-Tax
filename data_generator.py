"""
data_generator.py
=================
Generates all synthetic data for the Alignment Tax application.

Synthetic data is calibrated to real-world values and patterns from public
sources (UN vote records, World Bank trade statistics, IMF debt data, AGOA
trade preference schedules). A fixed random seed ensures full reproducibility.

Modules:
    generate_unga_voting_data()         → 1A. UNGA voting alignment scores
    generate_diplomatic_signals()       → 1B. 2025 Iran-crisis diplomatic signals
    generate_economic_dependency()      → 1C. Bilateral economic exposure
    generate_historical_precedents()    → 1D. Historical alignment episodes
    generate_ghana_deep_dive()          → 1E. Ghana granular data
    load_all_data()                     → Convenience wrapper returning all DataFrames
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

# ── Reproducibility seed ──────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── Country master list ───────────────────────────────────────────────────────
AFRICAN_COUNTRIES: list[dict] = [
    # West Africa
    {"country": "Ghana",           "iso3": "GHA", "region": "West Africa",     "regime": "democracy"},
    {"country": "Nigeria",         "iso3": "NGA", "region": "West Africa",     "regime": "democracy"},
    {"country": "Senegal",         "iso3": "SEN", "region": "West Africa",     "regime": "democracy"},
    {"country": "Côte d'Ivoire",   "iso3": "CIV", "region": "West Africa",     "regime": "democracy"},
    {"country": "Guinea",          "iso3": "GIN", "region": "West Africa",     "regime": "military"},
    {"country": "Mali",            "iso3": "MLI", "region": "West Africa",     "regime": "military"},
    {"country": "Burkina Faso",    "iso3": "BFA", "region": "West Africa",     "regime": "military"},
    {"country": "Sierra Leone",    "iso3": "SLE", "region": "West Africa",     "regime": "democracy"},
    {"country": "Liberia",         "iso3": "LBR", "region": "West Africa",     "regime": "democracy"},
    {"country": "Togo",            "iso3": "TGO", "region": "West Africa",     "regime": "autocracy"},
    {"country": "Benin",           "iso3": "BEN", "region": "West Africa",     "regime": "democracy"},
    {"country": "Gambia",          "iso3": "GMB", "region": "West Africa",     "regime": "democracy"},
    {"country": "Guinea-Bissau",   "iso3": "GNB", "region": "West Africa",     "regime": "hybrid"},
    {"country": "Cabo Verde",      "iso3": "CPV", "region": "West Africa",     "regime": "democracy"},
    {"country": "Mauritania",      "iso3": "MRT", "region": "West Africa",     "regime": "hybrid"},
    # East Africa
    {"country": "Kenya",           "iso3": "KEN", "region": "East Africa",     "regime": "democracy"},
    {"country": "Ethiopia",        "iso3": "ETH", "region": "East Africa",     "regime": "hybrid"},
    {"country": "Tanzania",        "iso3": "TZA", "region": "East Africa",     "regime": "hybrid"},
    {"country": "Uganda",          "iso3": "UGA", "region": "East Africa",     "regime": "autocracy"},
    {"country": "Rwanda",          "iso3": "RWA", "region": "East Africa",     "regime": "hybrid"},
    {"country": "Mozambique",      "iso3": "MOZ", "region": "East Africa",     "regime": "hybrid"},
    {"country": "Zambia",          "iso3": "ZMB", "region": "East Africa",     "regime": "democracy"},
    {"country": "Malawi",          "iso3": "MWI", "region": "East Africa",     "regime": "democracy"},
    {"country": "Somalia",         "iso3": "SOM", "region": "East Africa",     "regime": "fragile"},
    {"country": "Burundi",         "iso3": "BDI", "region": "East Africa",     "regime": "autocracy"},
    {"country": "Djibouti",        "iso3": "DJI", "region": "East Africa",     "regime": "autocracy"},
    {"country": "Eritrea",         "iso3": "ERI", "region": "East Africa",     "regime": "autocracy"},
    {"country": "Comoros",         "iso3": "COM", "region": "East Africa",     "regime": "hybrid"},
    {"country": "Madagascar",      "iso3": "MDG", "region": "East Africa",     "regime": "hybrid"},
    {"country": "Seychelles",      "iso3": "SYC", "region": "East Africa",     "regime": "democracy"},
    # Southern Africa
    {"country": "South Africa",    "iso3": "ZAF", "region": "Southern Africa", "regime": "democracy"},
    {"country": "Angola",          "iso3": "AGO", "region": "Southern Africa", "regime": "hybrid"},
    {"country": "Zimbabwe",        "iso3": "ZWE", "region": "Southern Africa", "regime": "autocracy"},
    {"country": "Namibia",         "iso3": "NAM", "region": "Southern Africa", "regime": "democracy"},
    {"country": "Botswana",        "iso3": "BWA", "region": "Southern Africa", "regime": "democracy"},
    {"country": "Lesotho",         "iso3": "LSO", "region": "Southern Africa", "regime": "democracy"},
    {"country": "Eswatini",        "iso3": "SWZ", "region": "Southern Africa", "regime": "autocracy"},
    # North Africa
    {"country": "Egypt",           "iso3": "EGY", "region": "North Africa",    "regime": "autocracy"},
    {"country": "Morocco",         "iso3": "MAR", "region": "North Africa",    "regime": "hybrid"},
    {"country": "Algeria",         "iso3": "DZA", "region": "North Africa",    "regime": "autocracy"},
    {"country": "Tunisia",         "iso3": "TUN", "region": "North Africa",    "regime": "hybrid"},
    {"country": "Libya",           "iso3": "LBY", "region": "North Africa",    "regime": "fragile"},
    {"country": "Sudan",           "iso3": "SDN", "region": "North Africa",    "regime": "fragile"},
    # Central Africa
    {"country": "DRC",             "iso3": "COD", "region": "Central Africa",  "regime": "hybrid"},
    {"country": "Cameroon",        "iso3": "CMR", "region": "Central Africa",  "regime": "autocracy"},
    {"country": "Chad",            "iso3": "TCD", "region": "Central Africa",  "regime": "military"},
    {"country": "Central African Republic", "iso3": "CAF", "region": "Central Africa", "regime": "fragile"},
    {"country": "Republic of Congo", "iso3": "COG", "region": "Central Africa","regime": "autocracy"},
    {"country": "Gabon",           "iso3": "GAB", "region": "Central Africa",  "regime": "military"},
    {"country": "Equatorial Guinea","iso3": "GNQ", "region": "Central Africa", "regime": "autocracy"},
    {"country": "São Tomé & Príncipe","iso3": "STP","region": "Central Africa","regime": "democracy"},
    # Oceanic/Island
    {"country": "Mauritius",       "iso3": "MUS", "region": "East Africa",     "regime": "democracy"},
    {"country": "South Sudan",     "iso3": "SSD", "region": "East Africa",     "regime": "fragile"},
    {"country": "Niger",           "iso3": "NER", "region": "West Africa",     "regime": "military"},
]

FOCUS_COUNTRIES = [
    "Ghana", "Nigeria", "Kenya", "South Africa", "Ethiopia",
    "Egypt", "DRC", "Tanzania", "Senegal", "Morocco",
    "Côte d'Ivoire", "Angola", "Mozambique", "Zambia", "Rwanda",
]

# ── Regional alignment priors (empirically-informed) ─────────────────────────
REGION_PRIORS: dict[str, dict] = {
    "West Africa":     {"us": 0.28, "china": 0.68, "russia": 0.52},
    "East Africa":     {"us": 0.30, "china": 0.70, "russia": 0.50},
    "Southern Africa": {"us": 0.22, "china": 0.72, "russia": 0.60},
    "North Africa":    {"us": 0.35, "china": 0.65, "russia": 0.55},
    "Central Africa":  {"us": 0.24, "china": 0.73, "russia": 0.56},
}

# ── Country-specific calibration anchors ─────────────────────────────────────
COUNTRY_ANCHORS: dict[str, dict] = {
    "Ghana":       {"us": 0.33, "china": 0.65, "russia": 0.48},
    "Nigeria":     {"us": 0.28, "china": 0.68, "russia": 0.50},
    "South Africa":{"us": 0.20, "china": 0.74, "russia": 0.63},
    "Kenya":       {"us": 0.32, "china": 0.67, "russia": 0.49},
    "Ethiopia":    {"us": 0.25, "china": 0.73, "russia": 0.55},
    "Egypt":       {"us": 0.40, "china": 0.60, "russia": 0.52},
    "Morocco":     {"us": 0.38, "china": 0.62, "russia": 0.48},
    "DRC":         {"us": 0.24, "china": 0.73, "russia": 0.53},
    "Tanzania":    {"us": 0.22, "china": 0.75, "russia": 0.57},
    "Senegal":     {"us": 0.30, "china": 0.67, "russia": 0.50},
    "Côte d'Ivoire":{"us": 0.32,"china": 0.65, "russia": 0.49},
    "Angola":      {"us": 0.23, "china": 0.74, "russia": 0.59},
    "Mozambique":  {"us": 0.24, "china": 0.72, "russia": 0.58},
    "Zambia":      {"us": 0.27, "china": 0.70, "russia": 0.52},
    "Rwanda":      {"us": 0.30, "china": 0.68, "russia": 0.50},
    "Mali":        {"us": 0.15, "china": 0.68, "russia": 0.78},
    "Burkina Faso":{"us": 0.16, "china": 0.67, "russia": 0.77},
    "CAF":         {"us": 0.14, "china": 0.65, "russia": 0.80},
    "Zimbabwe":    {"us": 0.16, "china": 0.78, "russia": 0.67},
    "Algeria":     {"us": 0.22, "china": 0.70, "russia": 0.68},
    "Sudan":       {"us": 0.20, "china": 0.72, "russia": 0.66},
    "Eritrea":     {"us": 0.10, "china": 0.72, "russia": 0.75},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1A. UNGA Voting Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_unga_voting_data() -> pd.DataFrame:
    """
    Generate UNGA voting alignment scores for 54 African countries, 2000–2025.

    Calibration sources:
    - Erik Voeten UNGA voting dataset (agreement rates)
    - UN Bibliographic Information System (vote counts)
    - Typical US-Africa agreement: 20–35%
    - Typical China-Africa agreement: 65–80%
    - Typical Russia-Africa agreement: 50–65%

    Returns
    -------
    pd.DataFrame
        Columns: country, iso3, region, year, us_alignment, china_alignment,
                 russia_alignment, num_votes, agreement_rate_us,
                 agreement_rate_china, agreement_rate_russia
    """
    records: list[dict] = []
    years = list(range(2000, 2026))

    for c in AFRICAN_COUNTRIES:
        name    = c["country"]
        iso3    = c["iso3"]
        region  = c["region"]
        prior   = REGION_PRIORS[region]
        anchor  = COUNTRY_ANCHORS.get(name, prior)

        # Country-level random walk seed
        us_base  = anchor["us"]
        cn_base  = anchor["china"]
        ru_base  = anchor["russia"]

        # Year-on-year drift
        us_walk  = RNG.normal(0, 0.012, len(years)).cumsum()
        cn_walk  = RNG.normal(0, 0.010, len(years)).cumsum()
        ru_walk  = RNG.normal(0, 0.014, len(years)).cumsum()

        for i, yr in enumerate(years):
            us_align = float(np.clip(us_base  + us_walk[i], 0.10, 0.50))
            cn_align = float(np.clip(cn_base  + cn_walk[i], 0.55, 0.90))
            ru_align = float(np.clip(ru_base  + ru_walk[i], 0.35, 0.85))

            # Specific historical shocks
            if yr == 2022:                       # Ukraine vote shock
                if region == "Southern Africa":
                    ru_align = min(ru_align + 0.08, 0.85)
                    us_align = max(us_align - 0.05, 0.10)
                if name in ("Mali", "Burkina Faso", "Central African Republic"):
                    ru_align = min(ru_align + 0.12, 0.90)
                    us_align = max(us_align - 0.08, 0.08)
            if yr == 2003:                       # Iraq invasion vote
                us_align = max(us_align - 0.04, 0.10)
                cn_align = min(cn_align + 0.03, 0.88)

            # Agreement rates are votes cast with / total contested resolutions
            num_votes = int(RNG.integers(50, 80))
            agr_us = float(np.clip(RNG.normal(us_align,  0.025), 0.10, 0.50))
            agr_cn = float(np.clip(RNG.normal(cn_align,  0.020), 0.55, 0.90))
            agr_ru = float(np.clip(RNG.normal(ru_align,  0.025), 0.35, 0.85))

            records.append({
                "country":              name,
                "iso3":                 iso3,
                "region":               region,
                "year":                 yr,
                "us_alignment":         round(us_align, 4),
                "china_alignment":      round(cn_align, 4),
                "russia_alignment":     round(ru_align, 4),
                "num_votes":            num_votes,
                "agreement_rate_us":    round(agr_us, 4),
                "agreement_rate_china": round(agr_cn, 4),
                "agreement_rate_russia":round(agr_ru, 4),
            })

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1B. Diplomatic Signal Data (2025 Iran Crisis)
# ─────────────────────────────────────────────────────────────────────────────

def generate_diplomatic_signals() -> pd.DataFrame:
    """
    Generate diplomatic signal scores for 15 key African economies
    in the context of the hypothetical 2025 Iran–US/Israel crisis.

    Signal dimensions
    -----------------
    un_vote_signal         : -1 (aligned w/ Iran/China/Russia) to +1 (US)
    diplomatic_statement_tone: -1 (critical of US) to +1 (supportive of US)
    sanctions_compliance   : 0 (no compliance) to 1 (full US-sanctions compliance)
    military_exercise_participation: none / observer / participant
    commodity_routing_signal: -1 (routing away from US interests) to +1 (toward US)
    composite_us_alignment : weighted composite of above signals

    Returns
    -------
    pd.DataFrame
    """
    # Calibrated signal profiles per country
    signal_profiles: dict[str, dict] = {
        "Ghana": {
            "un_vote": 0.0, "statement": 0.10, "sanctions": 0.55,
            "mil_exercise": "observer", "commodity": 0.15,
        },
        "Nigeria": {
            "un_vote": -0.05, "statement": 0.05, "sanctions": 0.50,
            "mil_exercise": "observer", "commodity": 0.20,
        },
        "Kenya": {
            "un_vote": 0.05, "statement": 0.15, "sanctions": 0.60,
            "mil_exercise": "observer", "commodity": 0.10,
        },
        "South Africa": {
            "un_vote": -0.35, "statement": -0.30, "sanctions": 0.15,
            "mil_exercise": "participant",  # Mosi-3 naval exercise w/ China/Russia
            "commodity": -0.20,
        },
        "Ethiopia": {
            "un_vote": -0.20, "statement": -0.10, "sanctions": 0.30,
            "mil_exercise": "none", "commodity": -0.05,
        },
        "Egypt": {
            "un_vote": 0.10, "statement": 0.20, "sanctions": 0.70,
            "mil_exercise": "none", "commodity": 0.30,
        },
        "DRC": {
            "un_vote": -0.10, "statement": -0.05, "sanctions": 0.35,
            "mil_exercise": "none", "commodity": -0.15,
        },
        "Tanzania": {
            "un_vote": -0.15, "statement": -0.10, "sanctions": 0.25,
            "mil_exercise": "none", "commodity": -0.10,
        },
        "Senegal": {
            "un_vote": -0.05, "statement": 0.05, "sanctions": 0.50,
            "mil_exercise": "observer", "commodity": 0.10,
        },
        "Morocco": {
            "un_vote": 0.15, "statement": 0.25, "sanctions": 0.65,
            "mil_exercise": "none", "commodity": 0.25,
        },
        "Côte d'Ivoire": {
            "un_vote": 0.05, "statement": 0.10, "sanctions": 0.55,
            "mil_exercise": "none", "commodity": 0.10,
        },
        "Angola": {
            "un_vote": -0.20, "statement": -0.15, "sanctions": 0.25,
            "mil_exercise": "none", "commodity": -0.10,
        },
        "Mozambique": {
            "un_vote": -0.15, "statement": -0.10, "sanctions": 0.30,
            "mil_exercise": "none", "commodity": -0.05,
        },
        "Zambia": {
            "un_vote": -0.05, "statement": 0.00, "sanctions": 0.45,
            "mil_exercise": "none", "commodity": -0.10,
        },
        "Rwanda": {
            "un_vote": 0.05, "statement": 0.10, "sanctions": 0.55,
            "mil_exercise": "none", "commodity": 0.10,
        },
    }

    mil_map = {"none": 0, "observer": 0.5, "participant": 1.0}

    records: list[dict] = []
    for country, p in signal_profiles.items():
        noise_scale = 0.04
        uv   = float(np.clip(p["un_vote"]    + RNG.normal(0, noise_scale), -1, 1))
        st   = float(np.clip(p["statement"]  + RNG.normal(0, noise_scale), -1, 1))
        sc   = float(np.clip(p["sanctions"]  + RNG.normal(0, 0.03), 0, 1))
        cr   = float(np.clip(p["commodity"]  + RNG.normal(0, noise_scale), -1, 1))
        me   = p["mil_exercise"]

        # Weighted composite — sanctions compliance is high-signal
        composite = (0.30 * uv + 0.20 * st + 0.30 * sc +
                     0.10 * cr + 0.10 * (1 - mil_map[me]))
        composite = float(np.clip(composite, -1, 1))

        # Infer China/Russia alignment as rough inverse
        cn_signal = float(np.clip(-0.5 * uv - 0.3 * st + 0.2 * mil_map[me] - 0.2 * sc, -1, 1))
        ru_signal = float(np.clip(-0.4 * uv - 0.2 * st + 0.3 * mil_map[me] - 0.1 * sc, -1, 1))

        records.append({
            "country":                       country,
            "un_vote_signal":                round(uv, 3),
            "diplomatic_statement_tone":     round(st, 3),
            "sanctions_compliance":          round(sc, 3),
            "military_exercise_participation": me,
            "commodity_routing_signal":      round(cr, 3),
            "composite_us_alignment":        round(composite, 3),
            "composite_china_alignment":     round(cn_signal, 3),
            "composite_russia_alignment":    round(ru_signal, 3),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 1C. Economic Dependency Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_economic_dependency() -> pd.DataFrame:
    """
    Generate bilateral economic exposure for 15 African states.

    Real calibration anchors used:
    - Ghana: US trade ~$3.2B, China trade ~$10.5B, Chinese debt ~15% external
    - Nigeria: US trade ~$6B, China trade ~$19B
    - South Africa: US trade ~$18B, China trade ~$35B
    - DRC: China buys ~65% of cobalt output
    - Kenya: US trade ~$1.5B, China trade ~$7B
    - Egypt: US military aid ~$1.3B/yr, China trade ~$15B

    Returns
    -------
    pd.DataFrame
    """
    # Master calibration table: [us_trade, cn_trade, ru_trade,  # bUSD
    #                             us_fdi, cn_fdi, ru_fdi,        # bUSD stock
    #                             us_oda, cn_oda, ru_oda,         # mUSD
    #                             cn_debt_share,                  # % external debt
    #                             us_mil_idx, cn_mil_idx, ru_mil_idx,  # 0-1
    #                             primary_commodity]
    calibration: dict[str, dict] = {
        "Ghana": {
            "us_trade": 3.2, "cn_trade": 10.5, "ru_trade": 0.3,
            "us_fdi": 3.5, "cn_fdi": 5.8, "ru_fdi": 0.1,
            "us_oda": 620, "cn_oda": 180, "ru_oda": 5,
            "cn_debt_share": 15.2, "agoa_value": 380,
            "us_mil_idx": 0.55, "cn_mil_idx": 0.25, "ru_mil_idx": 0.08,
            "primary_commodity": "gold, cocoa, bauxite",
            "imf_program": True, "mcc_compact": True, "bri_member": True,
        },
        "Nigeria": {
            "us_trade": 6.0, "cn_trade": 19.0, "ru_trade": 0.8,
            "us_fdi": 8.5, "cn_fdi": 11.0, "ru_fdi": 0.5,
            "us_oda": 890, "cn_oda": 250, "ru_oda": 10,
            "cn_debt_share": 4.8, "agoa_value": 1200,
            "us_mil_idx": 0.50, "cn_mil_idx": 0.35, "ru_mil_idx": 0.10,
            "primary_commodity": "crude oil, LNG",
            "imf_program": False, "mcc_compact": False, "bri_member": True,
        },
        "Kenya": {
            "us_trade": 1.5, "cn_trade": 7.0, "ru_trade": 0.2,
            "us_fdi": 2.8, "cn_fdi": 4.2, "ru_fdi": 0.05,
            "us_oda": 780, "cn_oda": 120, "ru_oda": 3,
            "cn_debt_share": 21.5, "agoa_value": 430,
            "us_mil_idx": 0.60, "cn_mil_idx": 0.30, "ru_mil_idx": 0.05,
            "primary_commodity": "tea, coffee, horticulture",
            "imf_program": True, "mcc_compact": True, "bri_member": True,
        },
        "South Africa": {
            "us_trade": 18.0, "cn_trade": 35.0, "ru_trade": 1.2,
            "us_fdi": 22.0, "cn_fdi": 15.0, "ru_fdi": 0.8,
            "us_oda": 480, "cn_oda": 50, "ru_oda": 20,
            "cn_debt_share": 3.5, "agoa_value": 2800,
            "us_mil_idx": 0.35, "cn_mil_idx": 0.45, "ru_mil_idx": 0.40,
            "primary_commodity": "platinum, palladium, chrome, coal",
            "imf_program": False, "mcc_compact": False, "bri_member": True,
        },
        "Ethiopia": {
            "us_trade": 1.2, "cn_trade": 5.5, "ru_trade": 0.15,
            "us_fdi": 1.0, "cn_fdi": 8.0, "ru_fdi": 0.1,
            "us_oda": 1200, "cn_oda": 150, "ru_oda": 5,
            "cn_debt_share": 33.0, "agoa_value": 180,
            "us_mil_idx": 0.25, "cn_mil_idx": 0.55, "ru_mil_idx": 0.15,
            "primary_commodity": "coffee, sesame, oilseeds",
            "imf_program": True, "mcc_compact": False, "bri_member": True,
        },
        "Egypt": {
            "us_trade": 10.5, "cn_trade": 15.0, "ru_trade": 3.5,
            "us_fdi": 15.0, "cn_fdi": 7.5, "ru_fdi": 2.5,
            "us_oda": 1400, "cn_oda": 80, "ru_oda": 100,
            "cn_debt_share": 6.0, "agoa_value": 0,
            "us_mil_idx": 0.75, "cn_mil_idx": 0.30, "ru_mil_idx": 0.45,
            "primary_commodity": "oil, gas, phosphates",
            "imf_program": True, "mcc_compact": False, "bri_member": True,
        },
        "DRC": {
            "us_trade": 1.8, "cn_trade": 8.5, "ru_trade": 0.1,
            "us_fdi": 2.0, "cn_fdi": 12.0, "ru_fdi": 0.2,
            "us_oda": 650, "cn_oda": 90, "ru_oda": 3,
            "cn_debt_share": 19.0, "agoa_value": 65,
            "us_mil_idx": 0.30, "cn_mil_idx": 0.65, "ru_mil_idx": 0.12,
            "primary_commodity": "cobalt (65% to China), coltan, copper",
            "imf_program": True, "mcc_compact": False, "bri_member": True,
        },
        "Tanzania": {
            "us_trade": 0.9, "cn_trade": 4.8, "ru_trade": 0.1,
            "us_fdi": 1.5, "cn_fdi": 5.5, "ru_fdi": 0.08,
            "us_oda": 380, "cn_oda": 100, "ru_oda": 2,
            "cn_debt_share": 26.0, "agoa_value": 120,
            "us_mil_idx": 0.20, "cn_mil_idx": 0.50, "ru_mil_idx": 0.10,
            "primary_commodity": "gold, tanzanite, coffee",
            "imf_program": False, "mcc_compact": False, "bri_member": True,
        },
        "Senegal": {
            "us_trade": 0.7, "cn_trade": 3.0, "ru_trade": 0.2,
            "us_fdi": 1.2, "cn_fdi": 2.8, "ru_fdi": 0.1,
            "us_oda": 280, "cn_oda": 120, "ru_oda": 5,
            "cn_debt_share": 11.0, "agoa_value": 85,
            "us_mil_idx": 0.40, "cn_mil_idx": 0.35, "ru_mil_idx": 0.12,
            "primary_commodity": "oil (new), phosphates, fish",
            "imf_program": True, "mcc_compact": True, "bri_member": True,
        },
        "Morocco": {
            "us_trade": 4.8, "cn_trade": 6.0, "ru_trade": 0.5,
            "us_fdi": 5.5, "cn_fdi": 4.0, "ru_fdi": 0.3,
            "us_oda": 200, "cn_oda": 60, "ru_oda": 5,
            "cn_debt_share": 4.5, "agoa_value": 0,
            "us_mil_idx": 0.65, "cn_mil_idx": 0.35, "ru_mil_idx": 0.15,
            "primary_commodity": "phosphates, fertilizers, fish",
            "imf_program": False, "mcc_compact": True, "bri_member": False,
        },
        "Côte d'Ivoire": {
            "us_trade": 1.5, "cn_trade": 5.0, "ru_trade": 0.15,
            "us_fdi": 2.0, "cn_fdi": 3.5, "ru_fdi": 0.05,
            "us_oda": 130, "cn_oda": 130, "ru_oda": 2,
            "cn_debt_share": 13.0, "agoa_value": 250,
            "us_mil_idx": 0.35, "cn_mil_idx": 0.40, "ru_mil_idx": 0.08,
            "primary_commodity": "cocoa, cashew, oil",
            "imf_program": True, "mcc_compact": False, "bri_member": True,
        },
        "Angola": {
            "us_trade": 3.5, "cn_trade": 22.0, "ru_trade": 0.3,
            "us_fdi": 12.0, "cn_fdi": 18.0, "ru_fdi": 0.5,
            "us_oda": 60, "cn_oda": 200, "ru_oda": 5,
            "cn_debt_share": 41.0, "agoa_value": 50,
            "us_mil_idx": 0.20, "cn_mil_idx": 0.60, "ru_mil_idx": 0.15,
            "primary_commodity": "crude oil (major supplier to China)",
            "imf_program": True, "mcc_compact": False, "bri_member": True,
        },
        "Mozambique": {
            "us_trade": 0.8, "cn_trade": 3.5, "ru_trade": 0.1,
            "us_fdi": 10.0, "cn_fdi": 2.0, "ru_fdi": 0.3,
            "us_oda": 320, "cn_oda": 80, "ru_oda": 3,
            "cn_debt_share": 18.0, "agoa_value": 60,
            "us_mil_idx": 0.25, "cn_mil_idx": 0.35, "ru_mil_idx": 0.20,
            "primary_commodity": "LNG (TotalEnergies/ENI/ExxonMobil), coal",
            "imf_program": False, "mcc_compact": True, "bri_member": True,
        },
        "Zambia": {
            "us_trade": 0.6, "cn_trade": 4.5, "ru_trade": 0.1,
            "us_fdi": 1.0, "cn_fdi": 6.0, "ru_fdi": 0.2,
            "us_oda": 440, "cn_oda": 70, "ru_oda": 2,
            "cn_debt_share": 29.0, "agoa_value": 40,
            "us_mil_idx": 0.20, "cn_mil_idx": 0.55, "ru_mil_idx": 0.10,
            "primary_commodity": "copper, cobalt",
            "imf_program": True, "mcc_compact": False, "bri_member": True,
        },
        "Rwanda": {
            "us_trade": 0.4, "cn_trade": 1.8, "ru_trade": 0.05,
            "us_fdi": 0.8, "cn_fdi": 1.5, "ru_fdi": 0.05,
            "us_oda": 320, "cn_oda": 85, "ru_oda": 2,
            "cn_debt_share": 9.0, "agoa_value": 55,
            "us_mil_idx": 0.50, "cn_mil_idx": 0.40, "ru_mil_idx": 0.08,
            "primary_commodity": "coltan, cassiterite, tourism",
            "imf_program": False, "mcc_compact": False, "bri_member": False,
        },
    }

    records: list[dict] = []
    for country, d in calibration.items():
        # Add small calibrated noise (~3%) to base figures
        noise = lambda x: float(np.clip(x * (1 + RNG.normal(0, 0.03)), 0, 1e6))
        records.append({
            "country":             country,
            "us_trade_bn":         round(noise(d["us_trade"]), 2),
            "china_trade_bn":      round(noise(d["cn_trade"]), 2),
            "russia_trade_bn":     round(noise(d["ru_trade"]), 2),
            "us_fdi_stock_bn":     round(noise(d["us_fdi"]), 2),
            "china_fdi_stock_bn":  round(noise(d["cn_fdi"]), 2),
            "russia_fdi_stock_bn": round(noise(d["ru_fdi"]), 2),
            "us_oda_mn":           round(noise(d["us_oda"]), 1),
            "china_oda_mn":        round(noise(d["cn_oda"]), 1),
            "russia_oda_mn":       round(noise(d["ru_oda"]), 1),
            "china_debt_pct_external": round(noise(d["cn_debt_share"]), 1),
            "agoa_value_mn":       round(noise(d["agoa_value"]), 1),
            "us_mil_coop_idx":     round(float(np.clip(d["us_mil_idx"] + RNG.normal(0, 0.02), 0, 1)), 3),
            "china_mil_coop_idx":  round(float(np.clip(d["cn_mil_idx"] + RNG.normal(0, 0.02), 0, 1)), 3),
            "russia_mil_coop_idx": round(float(np.clip(d["ru_mil_idx"] + RNG.normal(0, 0.02), 0, 1)), 3),
            "primary_commodities": d["primary_commodity"],
            "imf_program":         d["imf_program"],
            "mcc_compact":         d["mcc_compact"],
            "bri_member":          d["bri_member"],
            # Derived: total trade exposure and dominance ratio
            "total_trade_bn":      round(noise(d["us_trade"] + d["cn_trade"] + d["ru_trade"]), 2),
            "china_trade_dominance": round(d["cn_trade"] / (d["us_trade"] + d["cn_trade"] + d["ru_trade"] + 0.01), 3),
            "us_trade_share":      round(d["us_trade"] / (d["us_trade"] + d["cn_trade"] + d["ru_trade"] + 0.01), 3),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 1D. Historical Precedent Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_historical_precedents() -> pd.DataFrame:
    """
    Generate historical episodes where alignment shifts had measurable
    economic or diplomatic consequences.

    Each record represents one country-episode pair with estimated impact.
    Confidence levels: high / medium / low (based on quality of available data).

    Returns
    -------
    pd.DataFrame
    """
    episodes: list[dict] = [
        # ── AGOA Revocations / Threats ─────────────────────────────────────
        {
            "country": "Mali", "iso3": "MLI", "year": 2022,
            "event": "AGOA eligibility suspended — military coup & Wagner deployment",
            "alignment_shift": "away_from_US", "channel": "AGOA trade preferences",
            "impact_direction": "negative",
            "estimated_impact_pct": -18.0,  # ~18% textile/apparel export loss
            "estimated_impact_usd_mn": -42,
            "confidence": "high",
            "mechanism": "Direct trade preference withdrawal",
            "duration_years": 3.0,
        },
        {
            "country": "Guinea", "iso3": "GIN", "year": 2022,
            "event": "AGOA suspended — military coup (Sept 2021)",
            "alignment_shift": "away_from_US", "channel": "AGOA trade preferences",
            "impact_direction": "negative",
            "estimated_impact_pct": -12.0,
            "estimated_impact_usd_mn": -18,
            "confidence": "high",
            "mechanism": "Direct trade preference withdrawal",
            "duration_years": 2.0,
        },
        {
            "country": "Ethiopia", "iso3": "ETH", "year": 2022,
            "event": "AGOA revoked — Tigray conflict human rights concerns",
            "alignment_shift": "neutral_costs", "channel": "AGOA trade preferences",
            "impact_direction": "negative",
            "estimated_impact_pct": -22.0,
            "estimated_impact_usd_mn": -48,
            "confidence": "high",
            "mechanism": "Conditional preference withdrawal",
            "duration_years": 2.5,
        },
        {
            "country": "Burkina Faso", "iso3": "BFA", "year": 2023,
            "event": "AGOA suspended — military junta & Wagner presence",
            "alignment_shift": "away_from_US", "channel": "AGOA trade preferences",
            "impact_direction": "negative",
            "estimated_impact_pct": -8.0,
            "estimated_impact_usd_mn": -12,
            "confidence": "high",
            "mechanism": "Direct trade preference withdrawal",
            "duration_years": 2.0,
        },
        # ── Taiwan Recognition Switches ─────────────────────────────────────
        {
            "country": "Burkina Faso", "iso3": "BFA", "year": 2018,
            "event": "Switched Taiwan recognition → PRC; BRI investment surge",
            "alignment_shift": "toward_China", "channel": "FDI/infrastructure",
            "impact_direction": "positive (China)",
            "estimated_impact_pct": 35.0,
            "estimated_impact_usd_mn": 600,
            "confidence": "medium",
            "mechanism": "Chinese investment package following diplomatic switch",
            "duration_years": 5.0,
        },
        # ── Iraq War 2003 ───────────────────────────────────────────────────
        {
            "country": "Guinea", "iso3": "GIN", "year": 2003,
            "event": "UNSC non-permanent member; US lobbying for Iraq resolution support",
            "alignment_shift": "neutral", "channel": "bilateral aid/debt relief",
            "impact_direction": "positive (US pressure)",
            "estimated_impact_pct": 8.0,
            "estimated_impact_usd_mn": 45,
            "confidence": "medium",
            "mechanism": "US debt relief tied to UNSC vote support (documented offers)",
            "duration_years": 1.0,
        },
        {
            "country": "Cameroon", "iso3": "CMR", "year": 2003,
            "event": "UNSC member; resisted US Iraq vote pressure",
            "alignment_shift": "neutral", "channel": "US bilateral relations",
            "impact_direction": "negative",
            "estimated_impact_pct": -5.0,
            "estimated_impact_usd_mn": -30,
            "confidence": "low",
            "mechanism": "Reduced US diplomatic attention and delayed aid",
            "duration_years": 2.0,
        },
        # ── Ukraine Votes 2022 ──────────────────────────────────────────────
        {
            "country": "South Africa", "iso3": "ZAF", "year": 2022,
            "event": "Abstained on UNGA ES-11/1 (Russia-Ukraine); subsequent US scrutiny",
            "alignment_shift": "neutral_toward_Russia", "channel": "AGOA/bilateral",
            "impact_direction": "negative (risk)",
            "estimated_impact_pct": -4.0,
            "estimated_impact_usd_mn": -180,
            "confidence": "medium",
            "mechanism": "AGOA eligibility review triggered; diplomatic downgrade",
            "duration_years": 3.0,
        },
        {
            "country": "Eritrea", "iso3": "ERI", "year": 2022,
            "event": "Voted AGAINST UNGA ES-11/1 (one of 5 states); Russian diplomatic gains",
            "alignment_shift": "toward_Russia", "channel": "Russian diplomatic/economic",
            "impact_direction": "positive (Russia)",
            "estimated_impact_pct": 15.0,
            "estimated_impact_usd_mn": 40,
            "confidence": "low",
            "mechanism": "Russian grain supply and diplomatic support",
            "duration_years": 2.0,
        },
        # ── Chinese Debt / BRI Consequences ────────────────────────────────
        {
            "country": "Zambia", "iso3": "ZMB", "year": 2020,
            "event": "Sovereign default; China holds 29% of external debt",
            "alignment_shift": "debt_dependency", "channel": "debt service/restructuring",
            "impact_direction": "negative",
            "estimated_impact_pct": -6.5,
            "estimated_impact_usd_mn": -580,
            "confidence": "high",
            "mechanism": "G20 Common Framework delayed by Chinese creditor negotiations",
            "duration_years": 4.0,
        },
        {
            "country": "Ethiopia", "iso3": "ETH", "year": 2021,
            "event": "G20 Common Framework debt relief delayed by China",
            "alignment_shift": "debt_dependency", "channel": "debt restructuring",
            "impact_direction": "negative",
            "estimated_impact_pct": -3.5,
            "estimated_impact_usd_mn": -320,
            "confidence": "high",
            "mechanism": "Chinese holdout in multilateral restructuring",
            "duration_years": 3.0,
        },
        {
            "country": "Angola", "iso3": "AGO", "year": 2019,
            "event": "Chinese debt renegotiation; oil-for-infrastructure swap revised",
            "alignment_shift": "debt_dependency", "channel": "commodity-linked debt",
            "impact_direction": "mixed",
            "estimated_impact_pct": -2.0,
            "estimated_impact_usd_mn": -150,
            "confidence": "medium",
            "mechanism": "Oil price decline triggered collateral renegotiation",
            "duration_years": 2.0,
        },
        # ── MCC / USAID Conditionality ──────────────────────────────────────
        {
            "country": "Tanzania", "iso3": "TZA", "year": 2016,
            "event": "MCC compact threshold program suspended — governance concerns",
            "alignment_shift": "governance_democratic_backsliding", "channel": "MCC assistance",
            "impact_direction": "negative",
            "estimated_impact_pct": -100.0,  # full suspension
            "estimated_impact_usd_mn": -480,
            "confidence": "high",
            "mechanism": "MCC eligibility criteria — rule of law/democratic governance",
            "duration_years": 6.0,
        },
        {
            "country": "Malawi", "iso3": "MWI", "year": 2023,
            "event": "MCC Compact II signed ($350M) — improved governance scores",
            "alignment_shift": "toward_US", "channel": "MCC development assistance",
            "impact_direction": "positive",
            "estimated_impact_pct": 25.0,
            "estimated_impact_usd_mn": 350,
            "confidence": "high",
            "mechanism": "MCC eligibility re-established; compact signed",
            "duration_years": 5.0,
        },
        # ── Wagner/Russia Military Alignment ───────────────────────────────
        {
            "country": "Mali", "iso3": "MLI", "year": 2021,
            "event": "Wagner Group deployment replaces French Barkhane; US suspends security aid",
            "alignment_shift": "toward_Russia", "channel": "US security/military aid",
            "impact_direction": "negative (US channel)",
            "estimated_impact_pct": -45.0,
            "estimated_impact_usd_mn": -110,
            "confidence": "high",
            "mechanism": "Direct US security assistance suspension",
            "duration_years": 4.0,
        },
        {
            "country": "Central African Republic", "iso3": "CAF", "year": 2018,
            "event": "Wagner deployment; Russian arms supply; EU mission downgraded",
            "alignment_shift": "toward_Russia", "channel": "EU/US aid and security",
            "impact_direction": "negative (West channel)",
            "estimated_impact_pct": -30.0,
            "estimated_impact_usd_mn": -85,
            "confidence": "medium",
            "mechanism": "EU suspended aid; US downgraded security cooperation",
            "duration_years": 5.0,
        },
        # ── Minerals Security Partnership ───────────────────────────────────
        {
            "country": "DRC", "iso3": "COD", "year": 2023,
            "event": "Minerals Security Partnership (MSP) inclusion — strategic minerals",
            "alignment_shift": "toward_US", "channel": "critical minerals investment",
            "impact_direction": "positive",
            "estimated_impact_pct": 12.0,
            "estimated_impact_usd_mn": 550,
            "confidence": "medium",
            "mechanism": "US/G7 commitment to finance alternative cobalt supply chains",
            "duration_years": 5.0,
        },
        {
            "country": "Zambia", "iso3": "ZMB", "year": 2023,
            "event": "Lobito Corridor / MSP investment commitment",
            "alignment_shift": "toward_US", "channel": "infrastructure/critical minerals",
            "impact_direction": "positive",
            "estimated_impact_pct": 18.0,
            "estimated_impact_usd_mn": 700,
            "confidence": "medium",
            "mechanism": "G7 Partnership for Global Infrastructure and Investment",
            "duration_years": 8.0,
        },
    ]

    return pd.DataFrame(episodes)


# ─────────────────────────────────────────────────────────────────────────────
# 1E. Ghana Deep Dive Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_ghana_deep_dive() -> Dict[str, pd.DataFrame]:
    """
    Detailed data for Ghana across six sub-dimensions.

    Returns
    -------
    dict of DataFrames:
        agoa_sectors, mcc_compact, imf_program, chinese_deals,
        commodity_exports, economic_timeline
    """
    # ── AGOA trade preferences by sector ────────────────────────────────────
    agoa_sectors = pd.DataFrame([
        {"sector": "Petroleum & Products",     "exports_mn_usd": 180.0, "agoa_margin_pct": 2.5,  "jobs_supported": 1200,  "vulnerability": "high"},
        {"sector": "Agricultural Products",    "exports_mn_usd": 85.0,  "agoa_margin_pct": 8.2,  "jobs_supported": 18000, "vulnerability": "high"},
        {"sector": "Cocoa & Preparations",     "exports_mn_usd": 42.0,  "agoa_margin_pct": 0.0,  "jobs_supported": 5500,  "vulnerability": "low"},
        {"sector": "Apparel & Textiles",       "exports_mn_usd": 38.0,  "agoa_margin_pct": 12.0, "jobs_supported": 8200,  "vulnerability": "very high"},
        {"sector": "Fresh Cut Flowers",        "exports_mn_usd": 12.0,  "agoa_margin_pct": 6.5,  "jobs_supported": 2100,  "vulnerability": "high"},
        {"sector": "Processed Foods",          "exports_mn_usd": 10.5,  "agoa_margin_pct": 10.5, "jobs_supported": 3400,  "vulnerability": "high"},
        {"sector": "Aluminum/Bauxite Products","exports_mn_usd": 8.2,   "agoa_margin_pct": 4.8,  "jobs_supported": 900,   "vulnerability": "medium"},
        {"sector": "Handicrafts & Art",        "exports_mn_usd": 4.3,   "agoa_margin_pct": 7.0,  "jobs_supported": 12000, "vulnerability": "medium"},
    ])

    # ── MCC Compact ──────────────────────────────────────────────────────────
    mcc_compact = pd.DataFrame([
        {"compact": "MCC I",  "year_signed": 2012, "value_mn": 498.0, "sector": "Power sector (NEDCO/ECG transformation)", "status": "completed", "leverage_ratio": 4.2},
        {"compact": "MCC II (DIGI)", "year_signed": 2020, "value_mn": 190.0, "sector": "Digital connectivity / GIFEC", "status": "active", "leverage_ratio": 2.8},
    ])

    # ── IMF Program ──────────────────────────────────────────────────────────
    imf_program = pd.DataFrame([
        {"facility": "Extended Credit Facility", "year_approved": 2023, "total_bn": 3.0,
         "disbursed_bn": 1.2, "conditions": "fiscal consolidation, SOE reform, FX liberalization",
         "us_vote_share_imf": 16.5, "program_status": "on-track",
         "alignment_conditionality": "indirect (US veto power over approval)"},
    ])

    # ── Chinese Infrastructure/Economic Deals ────────────────────────────────
    chinese_deals = pd.DataFrame([
        {"deal": "Sinohydro Bauxite-for-Infrastructure",   "year": 2018, "value_bn": 2.0,
         "sector": "Roads/infrastructure", "repayment": "bauxite off-take (annual $100M equiv)",
         "chinese_entity": "Sinohydro Corp (State)", "status": "active",
         "strategic_concern": "resource-backed lending, limited local content"},
        {"deal": "China CITIC Bank syndicated loan",       "year": 2019, "value_bn": 0.65,
         "sector": "Budget support", "repayment": "sovereign guarantee",
         "chinese_entity": "China CITIC Bank", "status": "active",
         "strategic_concern": "debt service burden, cross-default clauses"},
        {"deal": "Sinohydro Bui Dam refurbishment",        "year": 2015, "value_bn": 0.27,
         "sector": "Hydropower", "repayment": "cocoa revenue linked",
         "chinese_entity": "Sinohydro Corp", "status": "completed",
         "strategic_concern": "cocoa revenue diversion (COCOBOD)"},
        {"deal": "Goldfields China Mining JV (Tarkwa)",    "year": 2021, "value_bn": 0.38,
         "sector": "Gold mining", "repayment": "equity stake",
         "chinese_entity": "Chifeng Jilong (partially)",  "status": "active",
         "strategic_concern": "artisanal miner displacement, galamsey"},
        {"deal": "China Road and Bridge Corp — roads",      "year": 2020, "value_bn": 0.55,
         "sector": "Roads", "repayment": "government budget",
         "chinese_entity": "CRBC (State)", "status": "active",
         "strategic_concern": "procurement tied to Chinese contractors"},
        {"deal": "Bright Food / cocoa processing",         "year": 2022, "value_bn": 0.18,
         "sector": "Agro-processing", "repayment": "equity/profit-share",
         "chinese_entity": "Bright Food (State)", "status": "negotiating",
         "strategic_concern": "value-chain integration, export dependence"},
    ])

    # ── Commodity Exports by Destination ─────────────────────────────────────
    commodity_exports = pd.DataFrame([
        {"commodity": "Gold",    "total_export_bn": 6.8, "us_pct": 5,  "china_pct": 25, "eu_pct": 55, "uae_pct": 10, "other_pct": 5,  "notes": "Dubai re-export hub important"},
        {"commodity": "Cocoa",   "total_export_bn": 2.5, "us_pct": 12, "china_pct": 20, "eu_pct": 58, "uae_pct": 2,  "other_pct": 8,  "notes": "EU remains dominant buyer"},
        {"commodity": "Crude Oil","total_export_bn": 2.1, "us_pct": 18, "china_pct": 30, "eu_pct": 25, "uae_pct": 5,  "other_pct": 22, "notes": "Jubilee/TEN fields"},
        {"commodity": "Bauxite", "total_export_bn": 0.55,"us_pct": 2,  "china_pct": 72, "eu_pct": 15, "uae_pct": 2,  "other_pct": 9,  "notes": "Sinohydro deal shapes routing"},
        {"commodity": "Manganese","total_export_bn": 0.42,"us_pct": 3,  "china_pct": 68, "eu_pct": 20, "uae_pct": 1,  "other_pct": 8,  "notes": "China dominant buyer"},
        {"commodity": "Tuna/Fish","total_export_bn": 0.18,"us_pct": 22, "china_pct": 8,  "eu_pct": 45, "uae_pct": 5,  "other_pct": 20, "notes": "AGOA-eligible; EU market key"},
        {"commodity": "Rubber",  "total_export_bn": 0.12,"us_pct": 8,  "china_pct": 45, "eu_pct": 30, "uae_pct": 3,  "other_pct": 14, "notes": "China growing market share"},
    ])

    # ── Economic Timeline (key shocks for charting) ──────────────────────────
    economic_timeline = pd.DataFrame([
        {"year": 2012, "event": "MCC I Compact signed ($498M)", "alignment_valence": "US+", "economic_impact_mn": 498},
        {"year": 2015, "event": "Bui Dam / Sinohydro deal", "alignment_valence": "CN+", "economic_impact_mn": 270},
        {"year": 2017, "event": "Oil revenue disappointment; IMF ECF program", "alignment_valence": "neutral", "economic_impact_mn": -200},
        {"year": 2018, "event": "Sinohydro $2B bauxite deal signed", "alignment_valence": "CN+", "economic_impact_mn": 2000},
        {"year": 2019, "event": "CITIC syndicated loan", "alignment_valence": "CN+", "economic_impact_mn": 650},
        {"year": 2020, "event": "MCC II DIGI Compact signed ($190M)", "alignment_valence": "US+", "economic_impact_mn": 190},
        {"year": 2021, "event": "Debt distress signals; cedi depreciation", "alignment_valence": "neutral", "economic_impact_mn": -350},
        {"year": 2022, "event": "Sovereign debt crisis; Eurobond market closed", "alignment_valence": "risk", "economic_impact_mn": -1200},
        {"year": 2023, "event": "IMF ECF $3B approved; Domestic Debt Exchange", "alignment_valence": "US+", "economic_impact_mn": 3000},
        {"year": 2024, "event": "External debt restructuring (partial)", "alignment_valence": "neutral", "economic_impact_mn": 600},
        {"year": 2025, "event": "Iran crisis: AGOA eligibility review triggered", "alignment_valence": "risk", "economic_impact_mn": -380},
    ])

    return {
        "agoa_sectors":      agoa_sectors,
        "mcc_compact":       mcc_compact,
        "imf_program":       imf_program,
        "chinese_deals":     chinese_deals,
        "commodity_exports": commodity_exports,
        "economic_timeline": economic_timeline,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data() -> Dict[str, object]:
    """
    Load all synthetic datasets.

    Returns
    -------
    dict with keys:
        unga_voting, diplomatic_signals, economic_dependency,
        historical_precedents, ghana (sub-dict of DataFrames)
    """
    return {
        "unga_voting":           generate_unga_voting_data(),
        "diplomatic_signals":    generate_diplomatic_signals(),
        "economic_dependency":   generate_economic_dependency(),
        "historical_precedents": generate_historical_precedents(),
        "ghana":                 generate_ghana_deep_dive(),
    }


if __name__ == "__main__":
    # Smoke test
    data = load_all_data()
    print("UNGA voting data shape:      ", data["unga_voting"].shape)
    print("Diplomatic signals shape:    ", data["diplomatic_signals"].shape)
    print("Economic dependency shape:   ", data["economic_dependency"].shape)
    print("Historical precedents shape: ", data["historical_precedents"].shape)
    print("Ghana sub-tables:            ", list(data["ghana"].keys()))
    print("\nGhana AGOA sectors:\n",       data["ghana"]["agoa_sectors"])
    print("\nEconomic dependency sample:\n",data["economic_dependency"][["country","us_trade_bn","china_trade_bn","china_debt_pct_external"]].head())

# The Alignment Tax — Deployment Guide

## Quantifying the Economic Cost of Strategic Neutrality for African States During Great-Power Crises

### Project Structure

```
alignment-tax-app/
├── app.py                 # Main Streamlit application (1,855 lines)
├── data_generator.py      # Synthetic data generation module (901 lines)
├── alignment_model.py     # Core analytical engine (1,243 lines)
├── visualizations.py      # Plotly visualization functions (1,331 lines)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Local Development

```bash
# Clone or download the project
cd alignment-tax-app

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Deploy to Streamlit Community Cloud

1. **Push to GitHub**: Create a new repository and push all 5 files (app.py, data_generator.py, alignment_model.py, visualizations.py, requirements.txt).

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **No API keys or secrets are needed** — the app uses synthetic data calibrated to real-world values, so it works immediately without any external data sources.

### Architecture Overview

#### Layer 1 — Alignment Signal Coding
Codes each African state's revealed alignment over time with the US, China, and Russia using UNGA General Assembly voting patterns (calibrated to Erik Voeten's dataset). Each state receives a time-varying alignment vector in 3D space. The model extends this to the 2025 Iran crisis through diplomatic signal coding (UN votes, diplomatic statements, sanctions compliance, military exercise participation, commodity routing).

#### Layer 2 — Economic Dependency Mapping
Maps bilateral economic exposure for 15 focus African economies across 7 channels: trade volumes, FDI stock, official development assistance, debt holdings, military cooperation, commodity dependency, and conditional program access (AGOA, MCC, IMF, BRI).

#### Layer 3 — Alignment Tax Calculator
The core innovation. For any configurable crisis scenario, calculates the economic exposure on each bilateral channel under three postures (US alignment, China alignment, neutrality). Integrates four behavioral economics modifiers:
- **Commitment credibility** — voting consistency and rhetoric-action gap
- **Audience costs** — regime type, opposition strength, media freedom, colonial legacy
- **Escalation lock-in** — probability of being locked into an alignment posture
- **Loss aversion** — prospect theory (λ=2.25) asymmetry between losing existing benefits and gaining new ones

#### Layer 4 — Ghana Deep Dive
Full strategic alignment audit for Ghana: AGOA sector vulnerability, MCC compact exposure, IMF program dependency, Chinese infrastructure deal portfolio (Sinohydro $2B bauxite deal), commodity export routing, and complete scenario analysis.

### Data Sources & Calibration

All data is synthetic but calibrated to real-world values from:
- **UNGA Voting**: Erik Voeten dataset (Harvard Dataverse) — agreement rates calibrated to 20-35% US-Africa, 65-80% China-Africa
- **Trade**: UN Comtrade / World Bank WITS — e.g., Ghana-US trade ~$3.2B, Ghana-China ~$10.5B
- **FDI**: UNCTAD / IMF CDIS / World Bank Harmonized Bilateral FDI
- **ODA**: OECD DAC statistics
- **Debt**: World Bank International Debt Statistics — e.g., Chinese debt share of Angola's external debt ~41%
- **Military**: SIPRI Arms Transfers Database
- **AGOA**: USTR/ITC trade preference data
- **MCC/IMF**: Official program documentation

### Key Assumptions & Limitations

1. **Synthetic data**: All data is modeled, not live. Values are calibrated to publicly available figures but include noise and should not be cited as primary data.

2. **Causal attribution**: The model estimates exposure and precedent-based risk, NOT deterministic causal effects. The DiD estimator uses historical episodes (AGOA revocations, Taiwan recognition switches) for calibration.

3. **Behavioral parameters**: Loss aversion coefficient (λ=2.25), credibility discounts, and audience cost multipliers are based on published behavioral economics research but applied to diplomatic contexts with acknowledged uncertainty.

4. **Crisis-specific calibration**: The Iran crisis parameters are scenario-based. Different crises (Taiwan, Ukraine) have different severity and pressure intensity defaults.

5. **Static snapshots**: Economic dependency data represents a point-in-time snapshot. Real bilateral relationships evolve continuously.

### Module API Reference

```python
# Data generation
from data_generator import load_all_data, generate_unga_voting_data, 
    generate_diplomatic_signals, generate_economic_dependency,
    generate_historical_precedents, generate_ghana_deep_dive

# Model
from alignment_model import (
    AlignmentVector,           # 3D alignment position
    AlignmentTaxCalculator,    # Channel-by-channel tax calculation
    BehavioralModifiers,       # Credibility, audience costs, lock-in, loss aversion
    ScenarioEngine,            # Full scenario orchestrator
    PanelEstimator,            # DiD historical estimation
)

# Visualizations
from visualizations import (
    alignment_space_3d,        # 3D scatter of alignment positions
    economic_exposure_radar,   # Spider chart of bilateral exposure
    alignment_tax_waterfall,   # Waterfall of gains/costs
    historical_precedent_timeline,  # Timeline of alignment episodes
    ghana_dashboard,           # Multi-panel Ghana deep dive
    alignment_heatmap,         # Country ranking by alignment metric
    scenario_comparison_bar,   # Posture comparison
    credibility_signal_chart,  # Time-series credibility signals
    loss_aversion_curve,       # Prospect theory value function
)
```

### License

This project is provided for research and educational purposes. All data is synthetic.

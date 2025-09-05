# Bitcoin Cycle Forecast — Diminishing ROI Model

A comprehensive Bitcoin price and timing forecasting system using **diminishing returns analysis** with real-time market data integration and advanced visualization capabilities.

> **Not financial advice.** This repository is a learning tool for cycle analytics and quantitative modeling.

---

## Overview

This repository provides a **sophisticated Bitcoin cycle forecasting system** that combines:

1. **Diminishing ROI Price Model** (ρ ≈ 0.29): Peak multiples decay exponentially across cycles
2. **Saturating Lag Timing Model**: Halving-to-peak delays approach an asymptote  
3. **Rolling Backtest Calibration**: Out-of-sample bias correction
4. **Real-time Data Integration**: Live Bitcoin prices via Yahoo Finance
5. **Comprehensive Visualization**: Multiple plot types with cycle timelines and progress tracking

**Key Features:**
- **Professional visualizations** with real market data
- **Cycle progress tracking** with visual timelines
- **Confidence intervals** and uncertainty quantification
- **Historical backtesting** with bias calibration
- **Command-line interface** for easy usage
- **Modular architecture** for easy extension

**Current Model Status (as of September 2025):**
- **Price Model**: Diminishing ROI with ρ = 0.289 (29% decay per cycle)
- **Timing Model**: Saturating lag approaching L∞ = 552 days
- **Calibration**: Rolling backtests show 27.4% average price error
- **Data**: 4,000+ days of Bitcoin price history
- **Current Forecast**: Peak ~$120K (calibrated), expected March 2026

The system models two separate—but related—aspects of Bitcoin halving cycles using **natural logs** for multiplicative behavior and provides **readable uncertainty estimates**.

---

## Contents
- [Overview](#overview)
- [Data Model](#data-model)
- [Price Model: Halving-day Multiple (Diminishing ROI)](#price-model-halving-day-multiple-diminishing-roi)
- [Date Model: Halving→Peak Lag (Saturating)](#date-model-halvingpeak-lag-saturating)
- [Usage](#usage)
- [Design Choices & Assumptions](#design-choices--assumptions)
- [Limitations](#limitations)
- [Appendix: Symbols & Variables](#appendix-symbols--variables)

---

## Data Model

Each cycle is represented by:

```python
@dataclass
class CycleData:
    i: int                # cycle index (1, 2, 3, ...)
    halving: date         # halving day (UTC)
    H: float              # BTC price on halving day (USD)
    peak_date: date|None  # cycle peak date (intraday or ref close)
    P: float|None         # cycle peak price (USD)

    @property
    def M(self) -> float|None:  # peak multiple
        return P/H if P and H else None

    @property
    def L(self) -> int|None:    # lag (days) from halving to peak
        return (peak_date - halving).days if peak_date else None
```

---

## Price Model: Halving-day Multiple (Diminishing ROI)

We define the **peak multiple** for cycle \(i\) as:
\[
M_i \;=\; \frac{P_i}{H_i}
\]

Empirically, \(M_i\) decreases across cycles. We fit an **exponential decay**:
\[
\ln M_i \;=\; a + b\,(i-1)
\quad\Longrightarrow\quad
M_i \;=\; \alpha\,\rho^{\,i-1},\;\; \alpha=e^{a},\;\rho=e^{b}.
\]

**Key Insight**: When \(0 < \rho < 1\), this represents **diminishing returns on investment** - each cycle produces smaller percentage gains than the previous one.

### Fitting (OLS in log-space)
We collect \((x_i, y_i)\) with \(x_i=i-1\) and \(y_i=\ln M_i\), then perform **ordinary least squares** (OLS) to estimate \(a,b\).

### Forecast
For a future cycle \(j\) with known halving-day price \(H_j\):
\[
\boxed{\,P_{\text{raw}}(j) \;=\; H_j \cdot \alpha \cdot \rho^{\,j-1}\,}
\]

### Bias Calibration via Rolling Backtests
We use **out-of-sample errors** from rolling backtests to calibrate for systematic bias:
\[
\boxed{\,P_{\text{cal}} \;=\; P_{\text{raw}} \cdot e^{\mu}\,}
\]
where \(\mu\) is the mean log-error from historical predictions.

---

## Date Model: Halving→Peak Lag (Saturating)

Let \(L_i\) be the number of days from halving to peak in cycle \(i\). We model a **saturating** lag:
\[
\boxed{\,L_i \;=\; L_{\infty} - c\;\rho_{\text{lag}}^{\,i-1}},\quad 0<\rho_{\text{lag}}<1.
\]

### Forecast & Uncertainty
Given cycle \(j\) halving date \(T_j\):
\[
\boxed{\,\widehat{D}_j = T_j + \widehat{L}_j \text{ days}\,}
\]

---

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/kinqsradio/BitcoinPricePrediction.git
cd BitcoinPricePrediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For visualization features (optional)
pip install matplotlib numpy yfinance
```

**Dependencies:**
- Python 3.7+
- matplotlib (for visualization)
- numpy (for numerical operations)
- yfinance (for real-time Bitcoin data)
- pandas (data manipulation)

### Quick Start

```python
from datetime import date
from btc_forecast import CycleData, forecast_current_cycle

# Historical cycles
cycles_hist = [
    CycleData(i=1, halving=date(2012,11,28), H=12.35,  peak_date=date(2013,12,4),  P=1151.0),
    CycleData(i=2, halving=date(2016,7,9),   H=650.0,  peak_date=date(2017,12,17), P=19783.0),
    CycleData(i=3, halving=date(2020,5,11),  H=8846.0, peak_date=date(2021,11,10), P=69000.0),
]

# Current cycle
current_cycle = CycleData(i=4, halving=date(2024,4,19), H=64968.87)

res = forecast_current_cycle(cycles_hist, current_cycle, calibrate=True)

print("Raw peak price:", res["P_hat_raw"])
print("Calibrated peak price:", res["P_hat_cal"])
print("Raw peak date:", res["D_hat_raw"])
print("Calibrated peak date:", res["D_hat_cal"])
```

### Command Line Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Basic forecast (text output only)
python main.py

# Generate all visualizations and save to files
python main.py all

# Individual visualization modes:
python main.py full      # Full cycle forecast with timelines
python main.py price     # Price range visualization
python main.py model     # Model parameters comparison
```

**Output Files:**
- `btc_forecast_full.png` - Complete forecast with cycle timelines
- `btc_price_ranges.png` - Price confidence intervals
- `btc_model_comparison.png` - Model diagnostics

### Visualization Features

#### Full Forecast Plot (`btc_forecast_full.png`)
- **Historical Bitcoin price data** from Yahoo Finance (4,000+ days)
- **Cycle peak markers** with historical data points
- **Halving date indicators** for all cycles
- **Forecast ranges** with 68% and 95% confidence intervals
- **Cycle timelines** showing halving-to-peak periods for all cycles
- **Current cycle progress** with days from halving and days to expected peak
- **Real-time countdown** showing position in current cycle

#### Key Features:
- **Diminishing ROI Model** with ρ=0.29 (showing exponential decay)
- **Real-time Bitcoin data** integration via yfinance
- **Multiple cycle comparison** with historical context
- **Professional styling** with clear legends and annotations
- **High-resolution output** (300 DPI) for publications

### Interpreting Outputs

- **Raw vs Calibrated:** calibrated adjusts for the model's historical bias
- **Confidence Intervals:** log-normal bands showing forecast uncertainty
- **Cycle Timelines:** visual progress bars for each Bitcoin cycle
- **Rho Factor:** diminishing returns coefficient (0.29 = 29% decay per cycle)

---

## Design Choices & Assumptions

- Halving-day price is a stable anchor for cycle-level comparisons
- Peak multiples **decay exponentially** across cycles (diminishing ROI)
- Halving→peak lags **saturate** toward an asymptote
- Log-space errors are roughly **Gaussian** (hence log-normal price bands)
- Calibration uses **out-of-sample** errors (rolling), not in-sample residuals

---

## Limitations

- **Small-N problem:** just a handful of cycles limits statistical power
- **Regime changes** can break historical relationships
- **Choice of reference peak** affects results
- The lag CI is descriptive, not fully parametric

Treat forecasts and bands as **decision aids**, not guarantees.

---

## Appendix: Symbols & Variables

| Symbol | Meaning |
|---|---|
| \(i,j\) | cycle index (1,2,3,...) |
| \(H_i\) | halving-day price (USD) |
| \(P_i\) | peak price (USD) |
| \(M_i=P_i/H_i\) | peak multiple |
| \(\alpha=e^a\) | level parameter in multiple decay |
| \(\rho=e^b\) | per-cycle decay factor (0<\(\rho\)<1) |
| \(\sigma_{\ln}\) | log-residual std for price model |
| \(P_{\text{raw}}\) | raw peak-price forecast |
| \(P_{\text{cal}} = P_{\text{raw}} e^{\mu}\) | calibrated price (μ = mean log-error) |
| \(L_{\infty}\) | asymptotic lag |
| \(\widehat{D}_j\) | forecast peak date |

---

**License & Disclaimer:** Free to use and modify. This is educational material, provided **as-is** without warranties. Not financial advice.
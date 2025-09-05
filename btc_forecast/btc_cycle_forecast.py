# btc_cycle_forecast.py
# ------------------------------------------------------------
# A lightweight, self-contained module to forecast Bitcoin's
# cycle-top **price** and **date** using simple historical models.
#
# MODELS
# 1) Halving-day Multiple (diminishing ROI) price model
#    ln M_i = a + b*(i-1)  =>  M_i = alpha * rho**(i-1)
#    where M_i = P_i/H_i (peak multiple vs halving-day price)
#
# 2) Halving→Peak Lag (saturating) date model
#    L_i = L_inf - c * rho_lag**(i-1)
#    where L_i = (peak_date - halving_date).days
#
# NEW IN THIS VERSION
# - Robust lag fitting for **N>=3** cycles using OLS in the transformed ("Z-space") with
#   a grid search for L_inf (ensures all Z_i = L_inf - L_i > 0 and uses all data).
# - Rolling backtests (train on cycles <= i-1, predict i) to measure out-of-sample accuracy.
# - Bias calibration:
#     * Price: apply average **log-error** from rolling backtests to correct forecast bias.
#     * Date:  apply average **day error** from rolling backtests to shift peak-date.
# - Helper functions to add or load data in plain Python lists (no external deps).
#
# Both models remain intentionally simple and transparent so you can keep them updated.
# No internet access is required.
#
# DISCLAIMER: Not financial advice.
# ------------------------------------------------------------

from datetime import date, timedelta
from math import exp
from typing import List, Dict
from .cycle_data import CycleData
from .price_model import fit_peak_multiple_decay, forecast_peak_price, log_error_sigma, price_confidence_interval
from .date_model import fit_saturating_lag, forecast_peak_date, lag_uncertainty
from .backtests import rolling_backtest_price, rolling_backtest_lag, price_bias_from_backtest, date_bias_from_backtest

# ---------- Convenience runner ----------

def forecast_current_cycle(cycles_hist: List[CycleData], current_cycle: CycleData, calibrate: bool = True):
    """
    Fit both models on historical data and forecast price + date for the current cycle.
    If calibrate=True, apply bias learned from rolling backtests (out-of-sample errors).
    
    Returns a dict with:
      - raw model params + forecasts
      - bias-calibrated forecasts (price & date)
      - backtest summaries (for transparency)
    """
    # PRICE (raw)
    alpha, rho, r2_log = fit_peak_multiple_decay(cycles_hist)
    sigma_ln = log_error_sigma(cycles_hist, alpha, rho)
    P_hat = forecast_peak_price(current_cycle.H, current_cycle.i, alpha, rho)
    ci68 = price_confidence_interval(P_hat, sigma_ln, z=1.0)
    ci95 = price_confidence_interval(P_hat, sigma_ln, z=1.96)

    # DATE (raw)
    L_inf, c_param, rho_lag = fit_saturating_lag(cycles_hist)
    lags_hist = [c.L for c in cycles_hist if c.L is not None]
    lag_mu, lag_std = lag_uncertainty(lags_hist)
    D_hat = forecast_peak_date(current_cycle.halving, current_cycle.i, L_inf, c_param, rho_lag)

    # Rolling backtests for calibration
    bt_price = rolling_backtest_price(cycles_hist)
    bt_date  = rolling_backtest_lag(cycles_hist)

    # Bias calibration (multiplicative in log-space for price, additive in days for date)
    price_bias_mu, price_bias_sd = price_bias_from_backtest(bt_price)
    date_bias_mu,  date_bias_sd  = date_bias_from_backtest(bt_date)

    if calibrate:
        P_hat_cal = P_hat * exp(price_bias_mu)  # upward if model had tended to undershoot
        # Keep CI symmetric in log-space around the *calibrated* center
        ci68 = (P_hat_cal * exp(-1.0 * sigma_ln), P_hat_cal * exp(1.0 * sigma_ln))
        ci95 = (P_hat_cal * exp(-1.96 * sigma_ln), P_hat_cal * exp(1.96 * sigma_ln))
        D_hat_cal = D_hat + timedelta(days=int(round(date_bias_mu)))
    else:
        P_hat_cal = P_hat
        D_hat_cal = D_hat

    return {
        # Price model internals
        "alpha": alpha, "rho": rho, "r2_log": r2_log, "sigma_ln": sigma_ln,
        # Raw forecasts
        "P_hat_raw": P_hat, "P_ci68": ci68, "P_ci95": ci95,
        "L_inf": L_inf, "c_param": c_param, "rho_lag": rho_lag,
        "lag_mu": lag_mu, "lag_std": lag_std,
        "D_hat_raw": D_hat,
        # Calibration pieces
        "backtest_price": bt_price,
        "backtest_date": bt_date,
        "price_bias_mu": price_bias_mu, "price_bias_sd": price_bias_sd,
        "date_bias_mu": date_bias_mu, "date_bias_sd": date_bias_sd,
        # Calibrated outputs
        "P_hat_cal": P_hat_cal,
        "D_hat_cal": D_hat_cal,
    }


# ---------- Example runner ----------

if __name__ == "__main__":
    # Historical inputs (edit as you add more cycles)
    cycles_hist = [
        CycleData(i=1, halving=date(2012,11,28), H=12.35,  peak_date=date(2013,12,4),  P=1151.0),
        CycleData(i=2, halving=date(2016,7,9),   H=650.0,  peak_date=date(2017,12,17), P=19783.0),
        CycleData(i=3, halving=date(2020,5,11),  H=8846.0, peak_date=date(2021,11,10), P=69000.0),
        # When the current cycle finishes, append it here as CycleData(i=4, ..., P=..., peak_date=...)
    ]
    current_cycle = CycleData(i=4, halving=date(2024,4,19), H=64968.87)

    result = forecast_current_cycle(cycles_hist, current_cycle, calibrate=True)

    # --- PRINT SUMMARY ---
    print("PRICE MODEL: alpha={alpha:.6f}, rho={rho:.6f}, R2_log={r2_log:.3f}, sigma_ln={sigma_ln:.3f}".format(**result))
    print("Raw peak price (USD) ≈ {0:,.0f}".format(result["P_hat_raw"]))
    print("Calibrated peak price (USD) ≈ {0:,.0f}".format(result["P_hat_cal"]))
    print("  68% CI: [{0:,.0f}, {1:,.0f}]".format(*result["P_ci68"]))
    print("  95% CI: [{0:,.0f}, {1:,.0f}]".format(*result["P_ci95"]))

    print("\nDATE MODEL: L_inf={L_inf:.1f}, c={c_param:.1f}, rho_lag={rho_lag:.4f}".format(**result))
    print("Raw peak date ≈ {:%Y-%m-%d}".format(result["D_hat_raw"]))
    print("Calibrated peak date ≈ {:%Y-%m-%d}".format(result["D_hat_cal"]))
    print("  Lag mean/std (days): {lag_mu:.1f} / {lag_std:.1f}".format(**result))

    # Rolling backtest summaries
    bt_p = result["backtest_price"]
    bt_d = result["backtest_date"]
    if bt_p:
        avg_pct_err = sum(abs(r["pct_err"]) for r in bt_p) / len(bt_p)
        print("\nRolling backtest (price):")
        for r in bt_p:
            print(f"  Cycle {r['i']}: pred={r['P_hat']:.0f}, actual={r['P_actual']:.0f}, pct_err={r['pct_err']:+.2f}%")
        print(f"  Avg |pct_err| = {avg_pct_err:.2f}%  |  price bias mu (log)={result['price_bias_mu']:+.3f}")
    if bt_d:
        avg_abs_days = sum(abs(r["day_err"]) for r in bt_d) / len(bt_d)
        print("\nRolling backtest (date):")
        for r in bt_d:
            sign = "+" if r["day_err"]>=0 else "-"
            print(f"  Cycle {r['i']}: pred={r['D_hat']:%Y-%m-%d}, actual={r['D_actual']:%Y-%m-%d}, day_err={sign}{abs(r['day_err'])}d")
        print(f"  Avg |day_err| = {avg_abs_days:.1f} days  |  date bias (days)={result['date_bias_mu']:+.1f}")

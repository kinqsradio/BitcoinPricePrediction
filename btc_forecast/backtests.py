# backtests.py
# ------------------------------------------------------------
# Backtest functions for Bitcoin cycle forecasting.
# ------------------------------------------------------------

from datetime import date
from math import log, exp
from typing import List, Dict, Tuple
from .cycle_data import CycleData
from .price_model import fit_peak_multiple_decay, forecast_peak_price
from .date_model import fit_saturating_lag, forecast_peak_date

def rolling_backtest_price(cycles: List[CycleData]) -> List[Dict[str, float]]:
    """
    For i from 3..N (where P_i known), fit on cycles <= i-1 and predict cycle i.
    Returns list of dicts with predictions and errors (including log-error).
    """
    rows = []
    # Ensure cycles sorted by i
    data = sorted([c for c in cycles if c.P is not None], key=lambda c: c.i)
    for k in range(2, len(data)):  # index into data list (k=2 => third known cycle)
        train = data[:k]           # up to i-1
        test = data[k]             # cycle i
        alpha, rho, _ = fit_peak_multiple_decay(train)
        P_hat = forecast_peak_price(test.H, test.i, alpha, rho)
        log_err = log(test.P) - log(P_hat)     # >0 means model undershot; <0 overshot
        rows.append({
            "i": test.i,
            "H_i": test.H,
            "P_hat": P_hat,
            "P_actual": test.P,
            "abs_err": P_hat - test.P,
            "pct_err": (P_hat - test.P) / test.P * 100.0,
            "log_err": log_err,
        })
    return rows


def rolling_backtest_lag(cycles: List[CycleData]) -> List[Dict[str, float]]:
    """
    For i from 3..N (where L_i known), fit lag model on cycles <= i-1 and predict cycle i.
    Returns list of dicts with predictions and day errors.
    """
    rows = []
    data = sorted([c for c in cycles if c.L is not None], key=lambda c: c.i)
    for k in range(2, len(data)):
        train = data[:k]
        test = data[k]
        L_inf, c_param, rho_lag = fit_saturating_lag(train)
        D_hat = forecast_peak_date(test.halving, test.i, L_inf, c_param, rho_lag)
        day_err = (D_hat - test.peak_date).days  # positive => predicted late
        rows.append({
            "i": test.i,
            "D_hat": D_hat,
            "D_actual": test.peak_date,
            "day_err": day_err,
        })
    return rows


def price_bias_from_backtest(backtest_rows: List[Dict[str, float]]) -> Tuple[float, float]:
    """
    Compute mean and std of log-error from rolling price backtests.
    Bias factor to apply = exp(mean_log_error).
    """
    if not backtest_rows:
        return 0.0, 0.0
    logs = [r["log_err"] for r in backtest_rows]
    mu = sum(logs)/len(logs)
    if len(logs) < 2:
        return mu, 0.0
    var = sum((e - mu)**2 for e in logs) / (len(logs) - 1)
    return mu, var**0.5


def date_bias_from_backtest(backtest_rows: List[Dict[str, float]]) -> Tuple[float, float]:
    """
    Compute mean and std of day errors from rolling lag backtests.
    Positive mean => model tends to be late; negative => tends to be early.
    """
    if not backtest_rows:
        return 0.0, 0.0
    errs = [r["day_err"] for r in backtest_rows]
    mu = sum(errs)/len(errs)
    if len(errs) < 2:
        return mu, 0.0
    var = sum((e - mu)**2 for e in errs) / (len(errs) - 1)
    return mu, var**0.5
# price_model.py
# ------------------------------------------------------------
# Price model functions for Bitcoin cycle forecasting.
# ------------------------------------------------------------

from math import log, exp, sqrt
from typing import List, Tuple
from .cycle_data import CycleData

def fit_peak_multiple_decay(cycles: List[CycleData]) -> Tuple[float, float, float]:
    """
    Fit ln M_i = a + b*(i-1) via OLS on historical cycles with known peaks.
    Returns (alpha, rho, r2_log) where:
      alpha = exp(a)
      rho   = exp(b)  # diminishing ROI if 0 < rho < 1
      r2_log: R^2 in log space
    """
    xs, ys = [], []
    for c in cycles:
        if c.M is not None and c.M > 0:
            xs.append(c.i - 1)
            ys.append(log(c.M))
    if len(xs) < 2:
        raise ValueError("Need >=2 historical cycles with peaks to fit price model.")
    
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    Sxx = sum((x - x_bar)**2 for x in xs)
    Sxy = sum((x - x_bar)*(y - y_bar) for x, y in zip(xs, ys))
    b = Sxy / Sxx
    a = y_bar - b * x_bar

    ss_tot = sum((y - y_bar)**2 for y in ys)
    ss_res = sum((y - (a + b*x))**2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    alpha = exp(a)
    rho = exp(b)
    return alpha, rho, r2


def forecast_peak_price(H_j: float, j: int, alpha: float, rho: float) -> float:
    """P_hat = H_j * alpha * rho**(j-1)"""
    return H_j * alpha * (rho ** (j - 1))


def log_error_sigma(cycles: List[CycleData], alpha: float, rho: float) -> float:
    """
    Std dev of residuals in log space for CI construction.
    If not enough data, returns a conservative default (0.35).
    """
    resid = []
    for c in cycles:
        if c.M is not None and c.M > 0:
            lnM = log(c.M)
            pred = log(alpha) + (c.i - 1)*log(rho)
            resid.append(lnM - pred)
    if len(resid) < 2:
        return 0.35
    mu = sum(resid)/len(resid)
    var = sum((e - mu)**2 for e in resid) / (len(resid) - 1)
    return var**0.5


def price_confidence_interval(P_hat: float, sigma_ln: float, z: float = 1.96) -> Tuple[float, float]:
    """
    Log-normal CI: Peak ~ LogNormal( ln(P_hat), sigma_ln^2 )
    CI = [ P_hat * e^{-z*sigma_ln}, P_hat * e^{+z*sigma_ln} ]
    z ~ 1.0 for ~68% CI, 1.96 for ~95% CI.
    """
    return (P_hat * exp(-z * sigma_ln), P_hat * exp(z * sigma_ln))
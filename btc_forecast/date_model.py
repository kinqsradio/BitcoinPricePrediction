# date_model.py
# ------------------------------------------------------------
# Date model functions for Bitcoin cycle forecasting.
# ------------------------------------------------------------

from datetime import date, timedelta
from math import log, exp
from typing import List, Tuple
from .cycle_data import CycleData

def _ols_ab(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Small OLS helper for y = a + b x."""
    n = len(x)
    x_bar = sum(x)/n
    y_bar = sum(y)/n
    Sxx = sum((xi - x_bar)**2 for xi in x)
    Sxy = sum((xi - x_bar)*(yi - y_bar) for xi, yi in zip(x, y))
    b = Sxy / Sxx if Sxx else 0.0
    a = y_bar - b * x_bar
    return a, b


def fit_saturating_lag(cycles: List[CycleData]) -> Tuple[float, float, float]:
    """
    Robust fit for L_i = L_inf - c * rho_lag**(i-1).
    Uses all available cycles (N>=3) via a grid-search over L_inf and
    OLS on ln Z_i where Z_i = L_inf - L_i.
    
    Steps:
      1) Collect (i, L_i). Require at least 2 known lags.
      2) If N==2: fall back to a linear proxy like previous version.
      3) If N>=3:
         - Choose a grid for L_inf in (max(L_i)+1 ... max(L_i)+400).
         - For each candidate L_inf, compute Z_i = L_inf - L_i (>0 required).
         - Fit ln Z_i = a + b*(i-1) -> c = exp(a), rho = exp(b).
         - Compute SSE in L-space and pick the (L_inf, c, rho) minimizing SSE,
           while enforcing 0 < rho < 1 for a saturating shape.
    Returns: (L_inf, c, rho_lag)
    """
    lags = [(c.i, c.L) for c in cycles if c.L is not None]
    if len(lags) < 2:
        raise ValueError("Need at least two cycles with known lags to fit the lag model.")
    lags.sort(key=lambda t: t[0])
    idx = [i for i, _ in lags]
    Ls  = [L for _, L in lags]

    # With exactly 2 points, fall back to linear trend proxy (as before).
    if len(Ls) == 2:
        xs = [i - 1 for i in idx]
        a, b = _ols_ab(xs, Ls)
        # Project one cycle ahead to mimic "saturation"
        L_inf = a + b * (max(xs) + 1)
        c = max(1.0, L_inf - Ls[0])
        rho = 0.0
        return L_inf, c, rho

    # N>=3: robust grid over L_inf
    Lmax = max(Ls)
    best = (None, None, None, float("inf"))  # (L_inf, c, rho, SSE)
    # Grid step can be coarse; data is small. Use step=1 day.
    for L_inf in [Lmax + k for k in range(1, 401)]:  # up to +400 days beyond max observed lag
        Z = [L_inf - L for L in Ls]
        if any(z <= 0 for z in Z):
            continue
        # Fit ln Z = a + b*(i-1)
        try:
            lnZ = [log(z) for z in Z]
        except ValueError:
            continue
        xs = [i - 1 for i in idx]
        a, b = _ols_ab(xs, lnZ)
        c = exp(a)
        rho = exp(b)
        # Enforce saturating behavior
        if not (0 < rho < 1.0):
            continue
        # Compute SSE in L-space
        SSE = 0.0
        for ii, Li in zip(idx, Ls):
            Li_hat = L_inf - c * (rho ** (ii - 1))
            SSE += (Li - Li_hat) ** 2
        if SSE < best[3]:
            best = (L_inf, c, rho, SSE)

    # If grid search failed (pathological), fall back to linear proxy
    if best[0] is None:
        xs = [i - 1 for i in idx]
        a, b = _ols_ab(xs, Ls)
        L_inf = a + b * (max(xs) + 1)
        c = max(1.0, L_inf - Ls[0])
        rho = 0.0
        return L_inf, c, rho

    return best[0], best[1], best[2]  # L_inf, c, rho_lag


def forecast_peak_date(halving_date: date, j: int, L_inf: float, c: float, rho_lag: float) -> date:
    """D_hat = halving_date + (L_inf - c * rho_lag**(j-1)) days"""
    L_hat = L_inf - c * (rho_lag ** (j - 1))
    return halving_date + timedelta(days=int(round(L_hat)))


def lag_uncertainty(lags: List[int]) -> Tuple[float, float]:
    """Return (mean, sample std) of historical lags (days)."""
    if not lags:
        return 0.0, 0.0
    mu = sum(lags)/len(lags)
    if len(lags) < 2:
        return mu, 0.0
    var = sum((L - mu)**2 for L in lags) / (len(lags) - 1)
    return mu, var**0.5
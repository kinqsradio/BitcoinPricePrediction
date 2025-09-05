#!/usr/bin/env python3
# main.py
# ------------------------------------------------------------
# Main entry point for Bitcoin Cycle Forecast
# ------------------------------------------------------------

import sys
from datetime import date
from btc_forecast import (
    CycleData, forecast_current_cycle,
    plot_cycle_forecast, plot_price_ranges, plot_model_comparison
)

def main():
    # Historical inputs (edit as you add more cycles)
    cycles_hist = [
        CycleData(i=1, halving=date(2012,11,28), H=12.35,  peak_date=date(2013,12,4),  P=1151.0),
        CycleData(i=2, halving=date(2016,7,9),   H=650.0,  peak_date=date(2017,12,17), P=19783.0),
        CycleData(i=3, halving=date(2020,5,11),  H=8846.0, peak_date=date(2021,11,10), P=69000.0),
        # When the current cycle finishes, append it here as CycleData(i=4, ..., P=..., peak_date=...)
    ]
    current_cycle = CycleData(i=4, halving=date(2024,4,19), H=64968.87)

    result = forecast_current_cycle(cycles_hist, current_cycle, calibrate=True)

    # Print results
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

    # Check for visualization arguments
    if len(sys.argv) > 1:
        plot_type = sys.argv[1].lower()

        if plot_type == 'full':
            print("\nGenerating full cycle forecast visualization...")
            plot_cycle_forecast(cycles_hist, current_cycle, result)
        elif plot_type == 'price':
            print("\nGenerating price range visualization...")
            plot_price_ranges(result)
        elif plot_type == 'model':
            print("\nGenerating model comparison visualization...")
            plot_model_comparison(cycles_hist, result)
        elif plot_type == 'all':
            print("\nGenerating all visualizations...")
            plot_cycle_forecast(cycles_hist, current_cycle, result, 'btc_forecast_full.png')
            plot_price_ranges(result, 'btc_price_ranges.png')
            plot_model_comparison(cycles_hist, result, 'btc_model_comparison.png')
        else:
            print("\nUsage: python main.py [full|price|model|all]")
            print("  full  - Complete forecast visualization")
            print("  price - Price range visualization")
            print("  model - Model parameters comparison")
            print("  all   - Generate all plots and save to files")

if __name__ == "__main__":
    main()
# visualization.py
# ------------------------------------------------------------
# Visualization functions for Bitcoin cycle forecasting.
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import date, timedelta
from typing import List, Dict, Optional
import numpy as np
import yfinance as yf

from .cycle_data import CycleData

def plot_cycle_forecast(cycles_hist: List[CycleData],
                       current_cycle: CycleData,
                       forecast_result: Dict,
                       save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive visualization of the Bitcoin cycle forecast.

    Args:
        cycles_hist: Historical cycle data
        current_cycle: Current cycle data
        forecast_result: Result from forecast_current_cycle()
        save_path: Optional path to save the plot
    """

    # Fetch real Bitcoin price data from yfinance
    print("Fetching Bitcoin price data from Yahoo Finance...")
    end_date = date.today() + timedelta(days=730)  # 2 years ahead
    btc_data = yf.download('BTC-USD', start='2011-01-01', end=end_date.strftime('%Y-%m-%d'), progress=False)
    btc_prices = btc_data['Close']
    print(f"Fetched {len(btc_prices)} days of Bitcoin price data")

    # Set up the plot style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

    # Prepare data for plotting
    dates = []
    prices = []
    halving_dates = []
    peak_dates = []
    peak_prices = []

    for cycle in cycles_hist:
        if cycle.peak_date and cycle.P:
            dates.append(cycle.peak_date)
            prices.append(cycle.P)
            halving_dates.append(cycle.halving)
            peak_dates.append(cycle.peak_date)
            peak_prices.append(cycle.P)

    # Add current cycle halving date
    halving_dates.append(current_cycle.halving)

    # Convert dates for plotting
    dates_num = [mdates.date2num(d) for d in dates]
    halving_dates_num = [mdates.date2num(d) for d in halving_dates]
    peak_dates_num = [mdates.date2num(d) for d in peak_dates]

    # Plot 1: Price chart with forecasts
    # Plot real Bitcoin price data first (background)
    ax1.semilogy(btc_prices.index, btc_prices.values, 'gray', linewidth=1, alpha=0.6,
                label='Bitcoin Price (Yahoo Finance)')

    ax1.semilogy(dates, prices, 'bo-', linewidth=2, markersize=8, label='Historical Peaks')

    # Plot halving dates
    for i, (hd, hp) in enumerate(zip(halving_dates[:-1], halving_dates[1:])):
        ax1.axvline(x=hd, color='red', linestyle='--', alpha=0.7, label='Halving' if i == 0 else "")
        # Connect halving to peak with a line
        if i < len(peak_dates):
            ax1.plot([hd, peak_dates[i]], [cycles_hist[i].H, peak_prices[i]],
                    'g--', alpha=0.5, label='Cycle Range' if i == 0 else "")

    # Plot current halving
    ax1.axvline(x=current_cycle.halving, color='orange', linestyle='--',
               linewidth=2, label='Current Halving')

    # Add timelines for all cycles
    y_offset = 0.08  # Vertical spacing between timelines
    
    for i, cycle in enumerate(cycles_hist + [current_cycle]):
        if cycle.halving and (cycle.P or cycle == current_cycle):  # Has halving and either has peak or is current
            # Calculate peak date (actual for historical, expected for current)
            if cycle == current_cycle:
                peak_date = cycle.halving + timedelta(days=forecast_result['lag_mu'])
                is_current = True
            else:
                peak_date = cycle.peak_date
                is_current = False
            
            if peak_date:
                # Position each timeline below the chart with vertical offset
                y_pos = 10**(np.log10(ax1.get_ylim()[0]) + y_offset * (i + 1))
                
                # Draw timeline from halving to peak
                ax1.plot([cycle.halving, peak_date], [y_pos, y_pos],
                        color='purple' if is_current else 'gray', 
                        linewidth=3 if is_current else 2, 
                        alpha=0.8 if is_current else 0.6, 
                        solid_capstyle='butt')
                
                # Add markers
                ax1.plot([cycle.halving], [y_pos], 'ko', markersize=4, alpha=0.8)
                if is_current:
                    ax1.plot([date.today()], [y_pos], 'ro', markersize=6, alpha=0.9)
                ax1.plot([peak_date], [y_pos], 'k^' if not is_current else 'k^', 
                        markersize=6 if is_current else 4, alpha=0.8)
                
                # Add concise labels
                cycle_num = cycle.i
                if is_current:
                    days_from_halving = (date.today() - cycle.halving).days
                    days_to_peak = (peak_date - date.today()).days
                    status = f"C{cycle_num}: {days_from_halving}d"
                    if days_to_peak > 0:
                        status += f" | -{days_to_peak}d"
                    else:
                        status += f" | +{-days_to_peak}d"
                else:
                    cycle_length = (peak_date - cycle.halving).days
                    status = f"C{cycle_num}: {cycle_length}d"
                
                ax1.text(peak_date, y_pos * 1.05, status,
                        ha='center', va='bottom', fontsize=6, 
                        color='purple' if is_current else 'gray',
                        fontweight='bold')

    # Plot forecast ranges for both raw and calibrated dates
    forecast_date_raw = forecast_result['D_hat_raw']
    forecast_date_cal = forecast_result['D_hat_cal']
    forecast_date_raw_num = mdates.date2num(forecast_date_raw)
    forecast_date_cal_num = mdates.date2num(forecast_date_cal)

    # Price forecast ranges
    p_raw = forecast_result['P_hat_raw']
    p_cal = forecast_result['P_hat_cal']
    ci68 = forecast_result['P_ci68']
    ci95 = forecast_result['P_ci95']

    # Plot confidence intervals as filled areas around calibrated date
    ax1.fill_between([forecast_date_cal_num - 30, forecast_date_cal_num + 30],
                    [ci95[0], ci95[0]], [ci95[1], ci95[1]],
                    alpha=0.2, color='blue', label='95% CI')
    ax1.fill_between([forecast_date_cal_num - 30, forecast_date_cal_num + 30],
                    [ci68[0], ci68[0]], [ci68[1], ci68[1]],
                    alpha=0.3, color='green', label='68% CI')

    # Plot forecast points for both raw and calibrated with clear distinction
    ax1.semilogy([forecast_date_raw], [p_raw], 'r^', markersize=10, markeredgecolor='darkred',
                markerfacecolor='red', linewidth=2,
                label=f'Raw Forecast: ${p_raw:,.0f} ({forecast_date_raw.strftime("%Y-%m")})')
    ax1.semilogy([forecast_date_cal], [p_cal], 'gs', markersize=10, markeredgecolor='darkgreen',
                markerfacecolor='lime', linewidth=2,
                label=f'Calibrated: ${p_cal:,.0f} ({forecast_date_cal.strftime("%Y-%m")})')

    # Add connecting lines from last historical peak to forecasts
    if cycles_hist:
        last_peak_date = cycles_hist[-1].peak_date
        last_peak_price = cycles_hist[-1].P
        if last_peak_date and last_peak_price:
            # Line to raw forecast
            ax1.plot([last_peak_date, forecast_date_raw], [last_peak_price, p_raw],
                    'r--', linewidth=1, alpha=0.7, label='Raw Trend' if cycles_hist else "")
            # Line to calibrated forecast
            ax1.plot([last_peak_date, forecast_date_cal], [last_peak_price, p_cal],
                    'g--', linewidth=1, alpha=0.7, label='Calibrated Trend' if cycles_hist else "")

    # Add vertical lines for both dates with different styles
    ax1.axvline(x=forecast_date_raw, color='red', linestyle='--', linewidth=1, alpha=0.6,
               label=f'Raw Peak Date: {forecast_date_raw.strftime("%Y-%m-%d")}')
    ax1.axvline(x=forecast_date_cal, color='green', linestyle='-.', linewidth=1, alpha=0.6,
               label=f'Calibrated Peak Date: {forecast_date_cal.strftime("%Y-%m-%d")}')

    # Add smaller text annotations near the markers
    ax1.annotate(f'${p_raw:,.0f}', xy=(forecast_date_raw, p_raw),
                xytext=(5, 5), textcoords='offset points',
                fontsize=6, ha='left', color='red', fontweight='bold')

    ax1.annotate(f'${p_cal:,.0f}', xy=(forecast_date_cal, p_cal),
                xytext=(-5, -10), textcoords='offset points',
                fontsize=6, ha='right', color='green', fontweight='bold')

    ax1.set_title('Bitcoin Cycle Price Forecast\nDiminishing ROI Model (ρ=0.29) + Saturating Lag Model + Rolling Backtests', 
                fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=8)

    # Create a more detailed legend
    legend_entries = [
        'Bitcoin Price (Yahoo Finance)',
        'Historical Peaks',
        'Halving',
        'Cycle Range',
        'Current Halving',
        f'Raw Forecast: ${p_raw:,.0f} ({forecast_date_raw.strftime("%Y-%m")})',
        f'Calibrated: ${p_cal:,.0f} ({forecast_date_cal.strftime("%Y-%m")})',
        'Raw Trend',
        'Calibrated Trend',
        '95% CI',
        '68% CI',
        'Cycle Timelines'
    ]

    # Create custom legend handles
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=1, alpha=0.6),
        plt.Line2D([0], [0], color='blue', marker='o', linestyle='-', linewidth=2, markersize=6),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, alpha=0.5),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2),
        plt.Line2D([0], [0], color='red', marker='^', linestyle='', markersize=10, markerfacecolor='red', markeredgecolor='darkred'),
        plt.Line2D([0], [0], color='green', marker='s', linestyle='', markersize=10, markerfacecolor='lime', markeredgecolor='darkgreen'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, alpha=0.7),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, alpha=0.7),
        plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.2),
        plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.3),
        plt.Line2D([0], [0], color='purple', linewidth=3, alpha=0.8)
    ]

    ax1.legend(legend_elements, legend_entries, loc='upper left', fontsize=5, framealpha=0.9)

    ax1.grid(True, alpha=0.3)

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Cycle timing analysis
    ax2.plot(range(1, len(cycles_hist) + 1),
            [c.L for c in cycles_hist if c.L is not None],
            'bo-', linewidth=2, markersize=8, label='Historical Lag (days)')

    # Plot forecast lag
    cycle_numbers = list(range(1, len(cycles_hist) + 2))
    ax2.plot([len(cycles_hist) + 1], [forecast_result['lag_mu']],
            'r^', markersize=12, label=f'Forecast Lag: {forecast_result["lag_mu"]:.0f} days')

    # Add lag uncertainty
    lag_std = forecast_result['lag_std']
    ax2.fill_between([len(cycles_hist) + 1 - 0.2, len(cycles_hist) + 1 + 0.2],
                    [forecast_result['lag_mu'] - lag_std, forecast_result['lag_mu'] - lag_std],
                    [forecast_result['lag_mu'] + lag_std, forecast_result['lag_mu'] + lag_std],
                    alpha=0.3, color='red', label=f'±1σ ({lag_std:.0f} days)')

    ax2.set_title('Cycle Timing Analysis - Saturating Lag Model', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Cycle Number', fontsize=8)
    ax2.set_ylabel('Days from Halving to Peak', fontsize=8)
    ax2.legend(loc='upper left', fontsize=6)
    ax2.grid(True, alpha=0.3)

    # Add forecast information as text
    date_diff_days = (forecast_date_cal - forecast_date_raw).days
    info_text = f"""
    Forecast Summary:
    • Raw Price: ${p_raw:,.0f}
    • Calibrated: ${p_cal:,.0f}
    • 68% CI: ${ci68[0]:,.0f} - ${ci68[1]:,.0f}
    • 95% CI: ${ci95[0]:,.0f} - ${ci95[1]:,.0f}
    • Raw Peak Date: {forecast_date_raw.strftime('%Y-%m-%d')}
    • Calibrated Peak Date: {forecast_date_cal.strftime('%Y-%m-%d')}
    • Date Difference: {date_diff_days} days
    • Lag: {forecast_result['lag_mu']:.0f} ± {lag_std:.0f} days
    """

    # Add text box with forecast info (moved up to avoid hiding cycle timelines)
    ax1.text(0.98, 0.15, info_text, transform=ax1.transAxes,
            fontsize=5, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def plot_price_ranges(forecast_result: Dict, save_path: Optional[str] = None) -> None:
    """
    Create a focused plot showing only the price forecast ranges.

    Args:
        forecast_result: Result from forecast_current_cycle()
        save_path: Optional path to save the plot
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Extract forecast data
    p_cal = forecast_result['P_hat_cal']
    ci68 = forecast_result['P_ci68']
    ci95 = forecast_result['P_ci95']
    forecast_date = forecast_result['D_hat_cal']

    # Create price range visualization
    prices = [ci95[0], ci68[0], p_cal, ci68[1], ci95[1]]
    labels = ['95% Low', '68% Low', 'Most Likely', '68% High', '95% High']
    colors = ['#ff6b6b', '#ffa500', '#32cd32', '#ffa500', '#ff6b6b']

    bars = ax.bar(range(len(prices)), prices, color=colors, alpha=0.7)

    # Add value labels on bars
    for i, (bar, price) in enumerate(zip(bars, prices)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${price:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f'Bitcoin Price Forecast - Peak: {forecast_date.strftime("%B %Y")}\n'
                f'Most Likely: ${p_cal:,.0f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add percentage ranges
    pct_68 = (ci68[1] - ci68[0]) / p_cal * 100
    pct_95 = (ci95[1] - ci95[0]) / p_cal * 100

    ax.text(0.02, 0.98, f'68% Range: ±{pct_68:.1f}%\n95% Range: ±{pct_95:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Price range plot saved to: {save_path}")
    else:
        plt.show()

def plot_model_comparison(cycles_hist: List[CycleData],
                         forecast_result: Dict,
                         save_path: Optional[str] = None) -> None:
    """
    Plot model parameters and their evolution across cycles.

    Args:
        cycles_hist: Historical cycle data
        forecast_result: Result from forecast_current_cycle()
        save_path: Optional path to save the plot
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Peak multiples over time
    multiples = [c.M for c in cycles_hist if c.M is not None]
    cycles = [c.i for c in cycles_hist if c.M is not None]

    ax1.semilogy(cycles, multiples, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Peak Multiples (P/H) - Diminishing ROI Model', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Multiple')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lag times over time
    lags = [c.L for c in cycles_hist if c.L is not None]
    lag_cycles = [c.i for c in cycles_hist if c.L is not None]

    ax2.plot(lag_cycles, lags, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Days from Halving to Peak - Historical Data', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Days')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Model parameters
    params = ['alpha', 'rho', 'r2_log', 'sigma_ln']
    values = [forecast_result[p] for p in params]

    bars = ax3.bar(params, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax3.set_title('Price Model Parameters - Diminishing ROI', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)

    # Plot 4: Date model parameters
    date_params = ['L_inf', 'c_param', 'rho_lag']
    date_values = [forecast_result[p] for p in date_params]

    bars = ax4.bar(date_params, date_values, color=['purple', 'brown', 'pink'], alpha=0.7)
    ax4.set_title('Date Model Parameters - Saturating Lag', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, value in zip(bars, date_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    else:
        plt.show()
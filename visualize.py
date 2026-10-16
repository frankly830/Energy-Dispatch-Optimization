import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_baseline_results(results_df, save_path='figures/baseline_results.png'):
    """Create visualization of baseline dispatch results."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    week_hours = slice(0, 7 * 24)
    battery_discharge = results_df.get(
        'battery_discharge_mw', pd.Series(0.0, index=results_df.index)
    )
    battery_discharge_week = battery_discharge.iloc[week_hours]

    axes[0].plot(
        results_df['datetime'].iloc[week_hours],
        results_df['demand_mw'].iloc[week_hours],
        label='Demand',
        color='black',
        linewidth=2,
    )
    axes[0].fill_between(
        results_df['datetime'].iloc[week_hours],
        0,
        results_df['renewable_mw'].iloc[week_hours],
        label='Renewable',
        color='green',
        alpha=0.6,
    )
    axes[0].fill_between(
        results_df['datetime'].iloc[week_hours],
        results_df['renewable_mw'].iloc[week_hours],
        results_df['renewable_mw'].iloc[week_hours] + battery_discharge_week,
        label='Battery discharge',
        color='purple',
        alpha=0.6,
    )
    axes[0].fill_between(
        results_df['datetime'].iloc[week_hours],
        results_df['renewable_mw'].iloc[week_hours] + battery_discharge_week,
        results_df['renewable_mw'].iloc[week_hours]
        + battery_discharge_week
        + results_df['traditional_mw'].iloc[week_hours],
        label='Traditional',
        color='orange',
        alpha=0.6,
    )
    axes[0].set_ylabel('Generation (MW)')
    axes[0].set_title('Baseline Dispatch - Week View')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results_df['datetime'].iloc[::24],
        results_df['renewable_mw'].iloc[::24].rolling(7).mean(),
        label='Renewable (7-day avg)',
        color='green',
        linewidth=2,
    )
    axes[1].plot(
        results_df['datetime'].iloc[::24],
        results_df['traditional_mw'].iloc[::24].rolling(7).mean(),
        label='Traditional (7-day avg)',
        color='orange',
        linewidth=2,
    )
    axes[1].plot(
        results_df['datetime'].iloc[::24],
        battery_discharge.iloc[::24].rolling(7).mean(),
        label='Battery discharge (7-day avg)',
        color='purple',
        linewidth=2,
    )
    axes[1].set_ylabel('Generation (MW)')
    axes[1].set_title('Daily Average Generation (7-day rolling average)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(
        range(365),
        results_df['demand_mw'].iloc[::24][:365],
        label='Demand',
        alpha=0.6,
        color='blue',
    )
    axes[2].bar(
        range(365),
        results_df['renewable_mw'].iloc[::24][:365],
        label='Renewable',
        alpha=0.6,
        color='green',
    )
    axes[2].set_xlabel('Day of year')
    axes[2].set_ylabel('Daily average (MW)')
    axes[2].set_title('Daily Average Demand vs. Renewable Generation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")
    plt.show()


def plot_comparison(baseline_df, optimized_df, save_path='figures/comparison.png'):
    """Compare baseline vs. optimized dispatch."""

    baseline_df = baseline_df.copy()
    optimized_df = optimized_df.copy()
    baseline_df['datetime'] = pd.to_datetime(baseline_df['datetime'])
    optimized_df['datetime'] = pd.to_datetime(optimized_df['datetime'])
    baseline_battery_discharge = baseline_df.get(
        'battery_discharge_mw', pd.Series(0.0, index=baseline_df.index)
    )

    week = slice(0, 7 * 24)
    baseline_week = baseline_df.iloc[week]
    optimized_week = optimized_df.iloc[week]

    generator_mw_cols = [
        col for col in optimized_df.columns
        if col.endswith('_mw') and col not in {'unmet_mw', 'curtailed_renewable_mw'}
    ]
    optimized_thermal = (
        optimized_df[generator_mw_cols].sum(axis=1)
        if generator_mw_cols
        else pd.Series(0.0, index=optimized_df.index)
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(
        baseline_week['datetime'],
        baseline_week['demand_mw'],
        label='Demand',
        color='black',
        linewidth=2,
    )
    axes[0, 0].fill_between(
        baseline_week['datetime'],
        0,
        baseline_week['renewable_mw'],
        label='Renewable',
        color='green',
        alpha=0.5,
    )
    axes[0, 0].fill_between(
        baseline_week['datetime'],
        baseline_week['renewable_mw'],
        baseline_week['renewable_mw'] + baseline_battery_discharge.iloc[week],
        label='Battery discharge',
        color='purple',
        alpha=0.5,
    )
    axes[0, 0].fill_between(
        baseline_week['datetime'],
        baseline_week['renewable_mw'] + baseline_battery_discharge.iloc[week],
        baseline_week['renewable_mw']
        + baseline_battery_discharge.iloc[week]
        + baseline_week['traditional_mw'],
        label='Traditional',
        color='orange',
        alpha=0.5,
    )
    axes[0, 0].set_title('Baseline Dispatch')
    axes[0, 0].set_ylabel('Generation (MW)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        baseline_week['datetime'],
        baseline_week['demand_mw'],
        label='Demand',
        color='black',
        linewidth=2,
    )
    if generator_mw_cols:
        axes[0, 1].stackplot(
            baseline_week['datetime'],
            *[optimized_week[col].to_numpy() for col in generator_mw_cols],
            labels=generator_mw_cols,
            alpha=0.75,
        )
    axes[0, 1].plot(
        baseline_week['datetime'],
        optimized_thermal.iloc[week],
        label='Total thermal',
        color='tab:blue',
        linewidth=1.5,
    )
    if 'unmet_mw' in optimized_week.columns:
        axes[0, 1].plot(
            baseline_week['datetime'],
            optimized_week['unmet_mw'],
            label='Unmet demand',
            color='red',
            linewidth=1.5,
        )
    axes[0, 1].set_title('Optimized Dispatch')
    axes[0, 1].set_ylabel('Generation (MW)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    scenarios = ['Baseline', 'Optimized']
    baseline_cost = baseline_df['traditional_mw'].sum()
    optimized_cost = optimized_thermal.sum()
    costs = [baseline_cost, optimized_cost]
    colors = ['orange', 'green']

    bars = axes[1, 0].bar(scenarios, costs, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Thermal Generation (MWh)')
    axes[1, 0].set_title('Total Thermal Generation Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{cost:,.0f}',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    baseline_renewable_pct = baseline_df['renewable_mw'].sum() / (
        baseline_df['renewable_mw'].sum() + baseline_df['traditional_mw'].sum()
    ) * 100
    if 'curtailed_renewable_mw' in optimized_df.columns:
        optimized_renewable_pct = (
            1 - optimized_df['curtailed_renewable_mw'].sum() / max(1.0, baseline_df['renewable_mw'].sum())
        ) * 100
    else:
        optimized_renewable_pct = 0.0

    renewable_pct = [baseline_renewable_pct, optimized_renewable_pct]
    bars = axes[1, 1].bar(scenarios, renewable_pct, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Renewable %')
    axes[1, 1].set_title('Renewable Utilization %')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for bar, pct in zip(bars, renewable_pct):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plot saved to {save_path}")


if __name__ == '__main__':
    results_df = pd.read_csv('results/baseline_dispatch.csv')
    optimized_df = pd.read_csv('results/optimized_dispatch.csv')
    plot_comparison(results_df, optimized_df)

import pandas as pd
import numpy as np
from model import create_example_grid

from simulator import simulate_baseline_dispatch
from optimizer import BatteryOptimizedDispatch, OptimizedDispatch

# Load data
df_demand = pd.read_csv('data/demand.csv')
df_wind = pd.read_csv('data/wind.csv')
df_solar = pd.read_csv('data/solar.csv')

timestamps = pd.to_datetime(df_demand['datetime'])
demand = df_demand['demand_mw'].values
wind = df_wind['wind_mw'].values
solar = df_solar['solar_mw'].values

# Create grid
grid = create_example_grid(demand, wind, solar)

# Print summary of the grid
print("=" * 60)
print("ENERGY GRID SETUP")
print("=" * 60)
print(f"\nGenerators:")
for gen in grid.generators:
    print(f"  - {gen}")

print(f"\nRenewables:")
for ren in grid.renewables:
    print(f"  - {ren.name}: {ren.capacity}MW")

print(f"\nStorage: {grid.storages[0].name} ({grid.storages[0].capacity}MWh)")

print(f"\nDemand: min={demand.min():.0f}, max={demand.max():.0f}, avg={demand.mean():.0f} MW")
print(f"Wind: min={wind.min():.0f}, max={wind.max():.0f}, avg={wind.mean():.0f} MW")
print(f"Solar: min={solar.min():.0f}, max={solar.max():.0f}, avg={solar.mean():.0f} MW")

print("\nSetup complete! Ready for simulation.")

# Baseline
print("\n" + "="*60)
print("BASELINE DISPATCH")
print("="*60)
baseline_results = simulate_baseline_dispatch(grid, hours=len(demand))
baseline_results.print_summary()
baseline_results.to_dataframe().to_csv('results/baseline_dispatch.csv', index=False)
print("Baseline dispatch schedule saved to results/baseline_dispatch.csv.")

# Compare optimized dispatch with and without battery storage.
scenarios = [
    ("without_battery", OptimizedDispatch, "results/optimized_dispatch.csv"),
    (
        "with_battery",
        BatteryOptimizedDispatch,
        "results/optimized_dispatch_with_battery.csv",
    ),
]
comparison_rows = []

for scenario, optimizer_class, output_path in scenarios:
    print("\n" + "=" * 60)
    print(f"OPTIMIZED DISPATCH {scenario.replace('_', ' ').upper()}")
    print("=" * 60)

    # The baseline simulation mutates generator and battery state.
    grid.reset()
    optimizer = optimizer_class(
        grid, demand, wind + solar, hours=len(demand), timestamps=timestamps
    )
    if not optimizer.optimize():
        raise RuntimeError(f"Optimization failed for scenario: {scenario}")

    schedule = optimizer.get_dispatch_schedule()
    schedule.to_csv(output_path, index=False)
    print(f"Dispatch schedule saved to {output_path}.")

    results = optimizer.results
    comparison_rows.append(
        {
            "scenario": scenario,
            "objective_value_eur": results["total_cost"],
            "fuel_cost_eur": results["fuel_cost"],
            "startup_cost_eur": results["startup_cost"],
            "unmet_energy_mwh": sum(results["unmet"].values()),
            "curtailed_renewable_mwh": sum(results["curtailment"].values()),
            "battery_charge_mwh": sum(results["charge"].values()),
            "battery_discharge_mwh": sum(results["discharge"].values()),
        }
    )

comparison = pd.DataFrame(comparison_rows)
without_battery_cost = comparison.loc[
    comparison["scenario"] == "without_battery", "objective_value_eur"
].iloc[0]
with_battery_cost = comparison.loc[
    comparison["scenario"] == "with_battery", "objective_value_eur"
].iloc[0]
comparison["savings_vs_without_battery_pct"] = (
    (without_battery_cost - comparison["objective_value_eur"])
    / without_battery_cost
    * 100
    if without_battery_cost
    else 0.0
)
comparison.to_csv("results/optimization_comparison.csv", index=False)

print("\n" + "=" * 60)
print("OPTIMIZER COMPARISON")
print("=" * 60)
for row in comparison.to_dict("records"):
    print(
        f"{row['scenario']}: objective €{row['objective_value_eur']:,.0f}, "
        f"unmet {row['unmet_energy_mwh']:,.1f} MWh, "
        f"savings {row['savings_vs_without_battery_pct']:.2f}%"
    )
print("Comparison saved to results/optimization_comparison.csv.")

import pandas as pd
import numpy as np
from model import create_example_grid

from simulator import simulate_baseline_dispatch
from optimizer import OptimizedDispatch

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

# Optimized
print("\n" + "="*60)
print("OPTIMIZED DISPATCH")
print("="*60)
optimizer = OptimizedDispatch(grid, demand, wind + solar, hours=len(demand), timestamps=timestamps)

if optimizer.optimize():
    optimized_schedule = optimizer.get_dispatch_schedule()
    optimized_schedule.to_csv('results/optimized_dispatch.csv', index=False)
    
    print(f"\nOptimized dispatch schedule saved.")
    print(f"Optimized total cost: €{optimizer.results['total_cost']:,.0f}")
    
    # Calculate savings
    baseline_cost = baseline_results.total_cost
    optimized_cost = optimizer.results['total_cost']
    savings = (baseline_cost - optimized_cost) / baseline_cost * 100
    print(f"\nCost comparison:")
    print(f"  Baseline: €{baseline_cost:,.0f}")
    print(f"  Optimized: €{optimized_cost:,.0f}")
    print(f"  Savings: {savings:.2f}%")

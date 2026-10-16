import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create hourly data for one year
hours = 8760  # 365 days * 24 hours
dates = pd.date_range(start='2023-01-01', periods=hours, freq='h')

# Base demand (in MW)
base_demand = 1000

# Add daily pattern (higher in evening, lower at night)
hours_of_day = np.arange(hours) % 24
daily_factor = 1 + 0.3 * np.sin((hours_of_day - 6) * np.pi / 12)

# Add seasonal pattern (higher in winter)
days_of_year = np.arange(hours) // 24
seasonal_factor = 1 + 0.2 * np.cos(2 * np.pi * days_of_year / 365)

# Add random noise
noise = 1 + 0.05 * np.random.randn(hours)

# Combine
demand = base_demand * daily_factor * seasonal_factor * noise
demand = np.clip(demand, 500, 1500)  # Realistic bounds

df_demand = pd.DataFrame({
    'datetime': dates,
    'demand_mw': demand
})

df_demand.to_csv('data/demand.csv', index=False)
print(f"Demand: min={demand.min():.1f}, max={demand.max():.1f}, mean={demand.mean():.1f} MW")




# Create wind generation profile (0-600 MW capacity)
wind_capacity = 600

# Wind has seasonal and hourly variationb
wind_generation = np.zeros(hours)
for i in range(hours):
    day_of_year = i // 24
    hour_of_day = i % 24
    
    # Seasonal: more wind in winter
    seasonal = 0.5 + 0.3 * np.cos(2 * np.pi * day_of_year / 365)
    
    # Hourly variation (realistic but random)
    hourly = 0.5 + 0.3 * np.sin((hour_of_day - 6) * np.pi / 12)
    
    # Random noise
    noise = np.random.uniform(0.3, 1.0)
    
    wind_generation[i] = wind_capacity * seasonal * hourly * noise

wind_generation = np.clip(wind_generation, 0, wind_capacity)

df_wind = pd.DataFrame({
    'datetime': dates,
    'wind_mw': wind_generation
})

df_wind.to_csv('data/wind.csv', index=False)
print(f"Wind: min={wind_generation.min():.1f}, max={wind_generation.max():.1f}, mean={wind_generation.mean():.1f} MW")

## Generate solar profile (0-400 MW capacity)
solar_capacity = 400

solar_generation = np.zeros(hours)
for i in range(hours):
    day_of_year = i // 24
    hour_of_day = i % 24
    
    # Zero at night (0-4, 20-24), peak at midday (10-14)
    if hour_of_day < 5 or hour_of_day > 20:
        solar_generation[i] = 0
    else:
        # Bell curve peaking at hour 12
        solar_curve = np.cos((hour_of_day - 12) * np.pi / 8)
        solar_generation[i] = solar_capacity * max(0, solar_curve)
    
    # Add seasonal factor (stronger in summer)
    seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
    solar_generation[i] *= seasonal
    
    # Add noise
    solar_generation[i] *= (0.8 + 0.2 * np.random.rand())

solar_generation = np.clip(solar_generation, 0, solar_capacity)

df_solar = pd.DataFrame({
    'datetime': dates,
    'solar_mw': solar_generation
})

df_solar.to_csv('data/solar.csv', index=False)
print(f"Solar: min={solar_generation.min():.1f}, max={solar_generation.max():.1f}, mean={solar_generation.mean():.1f} MW")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot Week 1 (easier to see pattern)
week_hours = slice(0, 7*24)

axes[0].plot(dates[week_hours], demand[week_hours], label='Demand', color='blue', linewidth=2)
axes[0].set_ylabel('Demand (MW)')
axes[0].set_title('Electricity Demand - Week 1')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(dates[week_hours], wind_generation[week_hours], label='Wind', color='green', linewidth=2)
axes[1].set_ylabel('Wind Generation (MW)')
axes[1].set_title('Wind Generation - Week 1')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(dates[week_hours], solar_generation[week_hours], label='Solar', color='orange', linewidth=2)
axes[2].set_ylabel('Solar Generation (MW)')
axes[2].set_title('Solar Generation - Week 1')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/data_visualization.png', dpi=150)
plt.show()

print("Data visualization saved!")
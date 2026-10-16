import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
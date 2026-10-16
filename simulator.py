import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SimulationResult:
    """Store simulation results."""
    
    def __init__(self, hours):
        self.hours = hours
        self.timestamps = []
        self.demand = np.zeros(hours)
        self.renewable_gen = np.zeros(hours)
        self.traditional_gen = np.zeros(hours)
        self.unmet_demand = np.zeros(hours)
        self.total_cost = 0
        self.startup_costs = 0
        self.fuel_costs = 0
        self.generator_status = {}  # Track each generator's output
        
    def add_hour(self, hour, timestamp, demand, renewable, traditional, unmet, cost_data):
        self.timestamps.append(timestamp)
        self.demand[hour] = demand
        self.renewable_gen[hour] = renewable
        self.traditional_gen[hour] = traditional
        self.unmet_demand[hour] = unmet
        self.total_cost += cost_data['total']
        self.startup_costs += cost_data['startup']
        self.fuel_costs += cost_data['fuel']
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        return pd.DataFrame({
            'datetime': self.timestamps,
            'demand_mw': self.demand,
            'renewable_mw': self.renewable_gen,
            'traditional_mw': self.traditional_gen,
            'unmet_demand_mw': self.unmet_demand
        })
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total demand met: {(1 - self.unmet_demand.sum() / self.demand.sum()) * 100:.2f}%")
        print(f"Unmet demand (blackouts): {self.unmet_demand.sum():.0f} MWh")
        print(f"\nGeneration mix (avg):")
        print(f"  Renewable: {self.renewable_gen.mean():.0f} MW ({100*self.renewable_gen.mean()/self.demand.mean():.1f}%)")
        print(f"  Traditional: {self.traditional_gen.mean():.0f} MW ({100*self.traditional_gen.mean()/self.demand.mean():.1f}%)")
        print(f"\nCosts:")
        print(f"  Total: €{self.total_cost:,.0f}")
        print(f"  Fuel: €{self.fuel_costs:,.0f}")
        print(f"  Startup: €{self.startup_costs:,.0f}")
        print(f"  Average: €{self.total_cost / len(self.timestamps):,.0f}/hour")
        print("=" * 60)


def simulate_baseline_dispatch(grid, hours=8760):
    """
    Simulate grid with baseline dispatch rule:
    1. Use all available renewable generation
    2. Turn on cheapest generators (merit order)
    3. Only turn off if demand drops
    """
    
    results = SimulationResult(hours)
    start_date = datetime(2023, 1, 1)
    
    for hour in range(hours):
        timestamp = start_date + timedelta(hours=hour)
        
        # Step 1: Get demand and renewable generation
        demand = grid.demand_profile[hour]
        renewable_gen = grid.get_renewable_generation(hour)

        needed = max(0, demand - renewable_gen)

        # Update generator hour counters at the start of the hour.
        for gen in grid.generators:
            if gen.status == 1:
                gen.hours_on += 1
            else:
                gen.hours_off += 1
        
        # Step 2: Dispatch generators in merit order (cheapest first)
        generators_sorted = sorted(grid.generators, key=lambda g: g.fuel_cost)
        
        traditional_gen = 0
        hour_startup_cost = 0
        hour_fuel_cost = 0
        
        for gen in generators_sorted:
            if needed <= 0:
                # Turn off if we don't need more
                if gen.status == 1 and gen.can_turn_off():
                    gen.status = 0
                    gen.output = 0
                    gen.hours_on = 0
                    gen.hours_off = 0
            else:
                # Check if we should turn on
                if gen.status == 0 and gen.can_turn_on():
                    # Turn on if we need power and it's economical
                    gen.status = 1
                    gen.hours_off = 0
                    gen.hours_on = 1
                    hour_startup_cost += gen.startup_cost
                
                # Generate as much as needed (up to capacity)
                if gen.status == 1:
                    output = min(needed, gen.capacity - gen.output)
                    output = max(output, gen.min_output) if gen.output == 0 else output
                    gen.output = output
                    traditional_gen += output
                    hour_fuel_cost += gen.get_cost(output)
                    needed -= output
        
        # Step 3: Handle unmet demand
        unmet = max(0, needed)
        
        # Step 4: Store results
        cost_data = {
            'total': hour_fuel_cost + hour_startup_cost,
            'fuel': hour_fuel_cost,
            'startup': hour_startup_cost
        }
        
        results.add_hour(hour, timestamp, demand, renewable_gen, traditional_gen, unmet, cost_data)
        
        # Print progress
        if (hour + 1) % 2000 == 0:
            print(f"Simulated {hour + 1} hours...")
    
    return results


# Run simulation
if __name__ == '__main__':
    import pandas as pd
    from model import create_example_grid
    
    # Load data
    df_demand = pd.read_csv('data/demand.csv')
    df_wind = pd.read_csv('data/wind.csv')
    df_solar = pd.read_csv('data/solar.csv')
    
    demand = df_demand['demand_mw'].values
    wind = df_wind['wind_mw'].values
    solar = df_solar['solar_mw'].values
    
    # Create and run simulation
    grid = create_example_grid(demand, wind, solar)
    results = simulate_baseline_dispatch(grid)
    
    # Print summary
    results.print_summary()
    
    # Save results
    df_results = results.to_dataframe()
    df_results.to_csv('results/baseline_dispatch.csv', index=False)
    print("\nResults saved to results/baseline_dispatch.csv")

# Generator model
from matplotlib.pyplot import grid
import pandas as pd
import numpy as np

class Generator:
    def __init__(self, name, fuel_type, capacity, min_output, fuel_cost, startup_cost, ramp_rate, min_uptime=1, min_downtime=1):
        """
        Args:
            name (str): Name of the generator (e.g., 'Gas Plant 1')
            fuel_type (str): Type of fuel (e.g., 'natural_gas', 'coal', 'hydro', 'nuclear')
            capacity (float): Maximun output in MW
            min_output (float): Minimum stable output when on in MW -- usually 40% of capacity
            fuel_cost (float): Opreating cost in $/MWh
            startup_cost (float): Cost to start the generator in $
        """
        
        self.name = name
        self.fuel_type = fuel_type
        self.capacity = capacity
        self.min_output = min_output
        self.fuel_cost = fuel_cost
        self.startup_cost = startup_cost
        self.ramp_rate = ramp_rate
        self.min_uptime = min_uptime
        self.min_downtime = min_downtime
    
        self.status = 0 # 0=off, 1=on
        self.output = 0 # Current output in MW
        self.hours_on =0 # Consecutive hours on
        self.hours_off = min_downtime # Consecutive hours off
    
    def can_turn_on(self):
        """Check if the generator can be turned on based on minimum downtime."""
        return self.status ==0 and self.hours_off >= self.min_downtime
    
    def can_turn_off(self):
        """Check if the generator can be turned off based on minimum uptime."""
        return self.status == 1 and self.hours_on >= self.min_uptime
    
    def get_cost(self, output):
        """Calculate cost for generating 'output' MW"""
        if output == 0:
            return 0
        return output * self.fuel_cost # + self.startup_cost * (1 if self.status == 0 else 0)
    
    def __repr__(self):
        return f"{self.name} ({self.fuel_type}, {self.capacity} MW, €{self.fuel_cost}/MWh)"
    

class RenewableSource:
    """
    Represents a renewable energy source (wind or solar)
    """
    def __init__(self, name, capacity, generation_profile):
        """
        Args:
            name (str): Name of the renewable source (e.g., 'Wind', 'Solar', etc.)
            capacity (float): Maximum capacity in MW
            generation_profile (list of float): Expected generation as a percentage of capacity for each time period (e.g., hourly)
        """
        self.name = name
        self.capacity = capacity
        self.generation_profile = generation_profile

    def get_generation(self, hour):
        """
        Get available generation at given hour.
        """
        return self.generation_profile[hour]
    
class EnergyStorage:
    """
    Represents battery or pumped hyfro storage
    """

    def __init__(self, name, capacity, max_charge_rate, max_discharge_rate, efficiency=0.9):
        """
        Args:
        name (str): Name of the storage (e.g., 'Battery 1')
        capacity (float): Total energy capacity in MWh
        max_charge_rate (float): Maximum charge rate in MW
        max_discharge_rate (float): Maximum discharge rate in MW
        efficiency (float): Round-trip efficiency (0.9 = 90%)
        """
        self.name = name
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        if not 0 < efficiency <= 1:
            raise ValueError("efficiency must be greater than 0 and at most 1")
        self.efficiency = efficiency
        self.one_way_efficiency = np.sqrt(efficiency)

        self.stored_energy = capacity * 0.5 # start at 50% charge

    def can_charge(self, amount):
        """Check if ``amount`` MW can be charged for one hour."""
        return (
            0 <= amount <= self.max_charge_rate
            and amount * self.one_way_efficiency <= self.capacity - self.stored_energy
        )

    def can_discharge(self, amount):
        """Check if ``amount`` MW can be discharged for one hour."""
        return (
            0 <= amount <= self.max_discharge_rate
            and amount / self.one_way_efficiency <= self.stored_energy
        )

    def charge(self, amount):
        """Charge for one hour, returning whether the requested rate was accepted."""
        if self.can_charge(amount):
            self.stored_energy += amount * self.one_way_efficiency
            return True
        return False

    def discharge(self, amount):
        """Discharge for one hour, returning whether the requested rate was accepted."""
        if self.can_discharge(amount):
            self.stored_energy -= amount / self.one_way_efficiency
            return True
        return False

class EngergyGrid:
    """
    Represents the entire energy grid, including generators, renewable sources, and storage
    """
    def __init__(self):
        self.generators = []
        self.renewables = []
        self.storages = []

    def add_generator(self, generator):
        self.generators.append(generator)

    def add_renewable(self, renewable):
        self.renewables.append(renewable)

    def add_storage(self, storage):
        self.storages.append(storage)

    def get_renewable_generation(self, hour):
        """Calculate total renewable generation at given hour"""
        total = 0
        for source in self.renewables:
            total += source.get_generation(hour)
        return total
    
    def get_total_capacity(self):
        """Calculate total generation capacity of the grid"""
        total = 0
        for gen in self.generators:
            total += gen.capacity
        return total
    
    def reset(self):
        """Reset all generators and storage to initial state (all generators off, storage at 50%)"""
        for gen in self.generators:
            gen.status = 0
            gen.output = 0
            gen.hours_on = 0
            gen.hours_off = gen.min_downtime
        for storage in self.storages:
            storage.stored_energy = storage.capacity * 0.5

# Example: Create a realistic grid
def create_example_grid(demand_data, wind_data, solar_data):
    """Create a realistic energy grid with a mix of generators, renewables, and storage."""
    
    grid = EngergyGrid()
    grid.demand_profile = demand_data
    grid.wind_profile = wind_data
    grid.solar_profile = solar_data

    # Add natural gasgenerators (quick response, expensive)
    grid.add_generator(Generator(
        name = "Gas Peaker 1", 
        fuel_type = "natural_gas", 
        capacity = 200, 
        min_output = 40, 
        fuel_cost = 80, 
        startup_cost = 5000, 
        ramp_rate = 100, 
        min_uptime = 2, 
        min_downtime = 2
        ))

    grid.add_generator(Generator(
        name = "Gas Peaker 2", 
        fuel_type = "natural_gas", 
        capacity = 200, 
        min_output = 40, 
        fuel_cost = 80, 
        startup_cost = 5000, 
        ramp_rate = 100, 
        min_uptime = 2, 
        min_downtime = 2
        ))

    # Coal generator (slower response, cheaper fuel)
    grid.add_generator(Generator(
        name="Coal Plant", 
        fuel_type="coal", 
        capacity=400, 
        min_output=200, 
        fuel_cost=40, 
        startup_cost=20000, 
        ramp_rate=50, 
        min_uptime=8, 
        min_downtime=4
        ))

    # Hydro generator (fast response, low fuel cost, but limited by water availability)
    grid.add_generator(Generator(
        name="Hydro Plant", 
        fuel_type="hydro", 
        capacity=250, 
        min_output=0, #Can fully ramp down
        fuel_cost=10, # Very cheap
        startup_cost=1000, 
        ramp_rate=150
        ))

    # Renewables
    grid.add_renewable(RenewableSource(
        name="Wind", 
        capacity=600, 
        generation_profile=wind_data
        ))
    grid.add_renewable(RenewableSource(
        name="Solar",
        capacity=400,
        generation_profile=solar_data
        ))

    # Add storage
    grid.add_storage(EnergyStorage(
        name="Battery Storage",
        capacity=100, # 100 MWh total capacity
        max_charge_rate=50,
        max_discharge_rate=50
    ))

    return grid

import unittest

import numpy as np

from model import EnergyStorage, create_example_grid
from optimizer import BatteryOptimizedDispatch, OptimizedDispatch


class BatteryOptimizedDispatchTests(unittest.TestCase):
    def test_storage_efficiency_is_round_trip(self):
        storage = EnergyStorage(
            "test", capacity=100, max_charge_rate=50, max_discharge_rate=50,
            efficiency=0.9
        )
        storage.stored_energy = 0

        self.assertTrue(storage.charge(10))
        stored = storage.stored_energy
        self.assertTrue(storage.discharge(stored * storage.one_way_efficiency))
        self.assertAlmostEqual(storage.stored_energy, 0)
        self.assertAlmostEqual(10 * storage.one_way_efficiency**2, 9)

    def test_battery_respects_limits_and_reduces_unserved_energy(self):
        demand = np.full(4, 100.0)
        wind = np.array([200.0, 0.0, 0.0, 0.0])
        grid = create_example_grid(demand, wind, np.zeros(4))

        without_battery = OptimizedDispatch(
            grid, demand, wind, hours=len(demand)
        )
        self.assertTrue(without_battery.optimize())

        with_battery = BatteryOptimizedDispatch(
            grid, demand, wind, hours=len(demand)
        )
        self.assertTrue(with_battery.optimize())
        schedule = with_battery.get_dispatch_schedule()
        storage = grid.storages[0]

        charge = schedule[f"{storage.name}_charge_mw"]
        discharge = schedule[f"{storage.name}_discharge_mw"]
        energy = schedule[f"{storage.name}_energy_mwh"]

        self.assertGreater(charge.sum(), 0)
        self.assertGreater(discharge.sum(), 0)
        self.assertLessEqual(charge.max(), storage.max_charge_rate)
        self.assertLessEqual(discharge.max(), storage.max_discharge_rate)
        self.assertTrue(energy.between(0, storage.capacity).all())
        self.assertAlmostEqual(energy.iloc[-1], storage.capacity * 0.5)
        self.assertTrue(((charge == 0) | (discharge == 0)).all())
        self.assertLess(
            with_battery.results["total_cost"],
            without_battery.results["total_cost"],
        )
        self.assertLessEqual(
            sum(with_battery.results["unmet"].values()),
            sum(without_battery.results["unmet"].values()),
        )

    def test_standard_optimizer_does_not_add_battery_schedule_columns(self):
        demand = np.array([100.0])
        renewable = np.array([0.0])
        grid = create_example_grid(demand, renewable, renewable)
        optimizer = OptimizedDispatch(grid, demand, renewable, hours=1)

        self.assertTrue(optimizer.optimize())
        schedule = optimizer.get_dispatch_schedule()
        self.assertFalse(any("Battery Storage" in column for column in schedule.columns))


if __name__ == "__main__":
    unittest.main()

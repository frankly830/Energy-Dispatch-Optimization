"""Cost-minimizing electricity dispatch using SciPy's mixed-integer solver."""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class OptimizedDispatch:
    """Optimize generator dispatch with ``scipy.optimize.milp``."""

    def __init__(
        self, grid, demand, renewable_gen, hours=8760, timestamps=None,
        include_storage=False
    ):
        self.grid = grid
        self.demand = np.asarray(demand, dtype=float)
        self.renewable_gen = np.asarray(renewable_gen, dtype=float)
        self.hours = int(hours)
        self.timestamps = timestamps
        self.storages = grid.storages if include_storage else []
        self.results = None

        if self.hours < 0:
            raise ValueError("hours must be non-negative")
        if len(self.demand) < self.hours or len(self.renewable_gen) < self.hours:
            raise ValueError("Input data is shorter than the requested optimization horizon")
        if timestamps is not None and len(timestamps) < self.hours:
            raise ValueError("Timestamps are shorter than the requested optimization horizon")
        if not np.all(np.isfinite(self.demand[:self.hours])) or not np.all(np.isfinite(self.renewable_gen[:self.hours])):
            raise ValueError("Demand and renewable generation must contain finite values")

        self.solver_result = None

    def optimize(self):
        """Solve the mixed-integer unit commitment and dispatch problem."""
        generator_count = len(self.grid.generators)
        storage_count = len(self.storages)
        hours = self.hours

        # Variables are grouped into hourly blocks: generator outputs, on/off
        # statuses, startups, unmet demand, renewable curtailment, then storage.
        output_start = 0
        status_start = hours * generator_count
        startup_start = status_start + hours * generator_count
        unmet_start = startup_start + hours * generator_count
        curtail_start = unmet_start + hours
        charge_start = curtail_start + hours
        discharge_start = charge_start + hours * storage_count
        energy_start = discharge_start + hours * storage_count
        storage_mode_start = energy_start + hours * storage_count
        variable_count = storage_mode_start + hours * storage_count

        def output_idx(h, g):
            return output_start + h * generator_count + g

        def status_idx(h, g):
            return status_start + h * generator_count + g

        def startup_idx(h, g):
            return startup_start + h * generator_count + g

        def unmet_idx(h):
            return unmet_start + h

        def curtail_idx(h):
            return curtail_start + h

        def charge_idx(h, s):
            return charge_start + h * storage_count + s

        def discharge_idx(h, s):
            return discharge_start + h * storage_count + s

        def energy_idx(h, s):
            return energy_start + h * storage_count + s

        def storage_mode_idx(h, s):
            return storage_mode_start + h * storage_count + s

        objective = np.zeros(variable_count)
        lower = np.zeros(variable_count)
        upper = np.full(variable_count, np.inf)
        integrality = np.zeros(variable_count, dtype=np.int32)

        for h in range(hours):
            for g, gen in enumerate(self.grid.generators):
                oi, si, ui = output_idx(h, g), status_idx(h, g), startup_idx(h, g)
                objective[oi] = gen.fuel_cost
                objective[ui] = gen.startup_cost
                upper[oi] = gen.capacity
                upper[si] = upper[ui] = 1
                integrality[si] = integrality[ui] = 1
            objective[unmet_idx(h)] = 10000  # Strongly discourage unserved load.
            upper[curtail_idx(h)] = max(0.0, self.renewable_gen[h])
            for s, storage in enumerate(self.storages):
                upper[charge_idx(h, s)] = storage.max_charge_rate
                upper[discharge_idx(h, s)] = storage.max_discharge_rate
                upper[energy_idx(h, s)] = storage.capacity
                upper[storage_mode_idx(h, s)] = 1
                integrality[storage_mode_idx(h, s)] = 1

        # Each row is a sparse linear constraint with inclusive lower/upper bounds.
        row_indices, col_indices, values = [], [], []
        row_lower, row_upper = [], []

        def add_constraint(coefficients, lb=-np.inf, ub=np.inf):
            row = len(row_lower)
            for col, value in coefficients.items():
                if value:
                    row_indices.append(row)
                    col_indices.append(col)
                    values.append(value)
            row_lower.append(lb)
            row_upper.append(ub)

        for h in range(hours):
            # Thermal + renewable + discharge + unmet - charge - curtailment = demand.
            balance = {output_idx(h, g): 1 for g in range(generator_count)}
            balance[unmet_idx(h)] = 1
            balance[curtail_idx(h)] = -1
            for s in range(storage_count):
                balance[charge_idx(h, s)] = -1
                balance[discharge_idx(h, s)] = 1
            add_constraint(balance, self.demand[h] - self.renewable_gen[h], self.demand[h] - self.renewable_gen[h])

            for s, storage in enumerate(self.storages):
                charge = charge_idx(h, s)
                discharge = discharge_idx(h, s)
                energy = energy_idx(h, s)
                mode = storage_mode_idx(h, s)

                # The binary mode prevents simultaneous charging and discharging.
                add_constraint(
                    {charge: 1, mode: -storage.max_charge_rate},
                    ub=0,
                )
                add_constraint(
                    {discharge: 1, mode: storage.max_discharge_rate},
                    ub=storage.max_discharge_rate,
                )

                # State of charge includes charging/discharging efficiency.
                state_balance = {
                    energy: 1,
                    charge: -storage.one_way_efficiency,
                    discharge: 1 / storage.one_way_efficiency,
                }
                initial_energy = storage.stored_energy
                if h > 0:
                    state_balance[energy_idx(h - 1, s)] = -1
                    add_constraint(state_balance, 0, 0)
                else:
                    add_constraint(state_balance, initial_energy, initial_energy)

                if h == hours - 1:
                    add_constraint({energy: 1}, initial_energy, initial_energy)

            for g, gen in enumerate(self.grid.generators):
                oi, si, ui = output_idx(h, g), status_idx(h, g), startup_idx(h, g)
                add_constraint({oi: 1, si: -gen.capacity}, ub=0)
                add_constraint({oi: 1, si: -gen.min_output}, lb=0)
                # startup >= status[h] - status[h-1], or status[0] at horizon start.
                startup_row = {ui: 1, si: -1}
                if h > 0:
                    startup_row[status_idx(h - 1, g)] = 1
                    add_constraint(startup_row, lb=0)
                else:
                    add_constraint(startup_row, lb=0)
                add_constraint({ui: 1, si: -1}, ub=0)  # startup <= status

        matrix = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(row_lower), variable_count),
        ).tocsr()

        self.solver_result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, np.asarray(row_lower), np.asarray(row_upper)),
            options={"disp": False},
        )

        if not self.solver_result.success or self.solver_result.x is None:
            self.results = None
            print(f"Optimization failed: {self.solver_result.message}")
            return False

        solution = self.solver_result.x
        self.results = {
            "output": {},
            "status": {},
            "startup": {},
            "unmet": {},
            "curtailment": {},
            "charge": {},
            "discharge": {},
            "energy": {},
            "total_cost": float(self.solver_result.fun),
            "fuel_cost": 0.0,
            "startup_cost": 0.0,
        }
        for h in range(hours):
            for g in range(generator_count):
                self.results["output"][h, g] = solution[output_idx(h, g)]
                self.results["status"][h, g] = int(round(solution[status_idx(h, g)]))
                self.results["startup"][h, g] = int(round(solution[startup_idx(h, g)]))
                gen = self.grid.generators[g]
                self.results["fuel_cost"] += solution[output_idx(h, g)] * gen.fuel_cost
                self.results["startup_cost"] += solution[startup_idx(h, g)] * gen.startup_cost
            self.results["unmet"][h] = solution[unmet_idx(h)]
            self.results["curtailment"][h] = solution[curtail_idx(h)]
            for s in range(storage_count):
                self.results["charge"][h, s] = solution[charge_idx(h, s)]
                self.results["discharge"][h, s] = solution[discharge_idx(h, s)]
                self.results["energy"][h, s] = solution[energy_idx(h, s)]

        print(f"Optimal solution found. Cost: €{self.results['total_cost']:,.0f}")
        return True

    def get_dispatch_schedule(self):
        """Return the optimized hourly dispatch as a pandas DataFrame."""
        if self.results is None:
            return None

        schedule = []
        for h in range(self.hours):
            row = {
                "hour": h,
                "datetime": self.timestamps[h] if self.timestamps is not None else None,
                "unmet_mw": self.results["unmet"][h],
                "curtailed_renewable_mw": self.results["curtailment"][h],
            }
            for g, gen in enumerate(self.grid.generators):
                row[f"{gen.name}_mw"] = self.results["output"][h, g]
                row[f"{gen.name}_status"] = self.results["status"][h, g]
            for s, storage in enumerate(self.storages):
                row[f"{storage.name}_charge_mw"] = self.results["charge"][h, s]
                row[f"{storage.name}_discharge_mw"] = self.results["discharge"][h, s]
                row[f"{storage.name}_energy_mwh"] = self.results["energy"][h, s]
            schedule.append(row)
        return pd.DataFrame(schedule)


class BatteryOptimizedDispatch(OptimizedDispatch):
    """Cost-minimize dispatch while also scheduling grid battery storage."""

    def __init__(self, grid, demand, renewable_gen, hours=8760, timestamps=None):
        if not grid.storages:
            raise ValueError("Battery optimization requires at least one storage unit")
        super().__init__(
            grid, demand, renewable_gen, hours, timestamps, include_storage=True
        )

from ortools.linear_solver import pywraplp
import pandas as pd

class OptimizedDispatch:
    """Optimize dispatch using linear programming."""
    
    def __init__(self, grid, demand, renewable_gen, hours=8760, timestamps=None):
        self.grid = grid
        self.demand = demand
        self.renewable_gen = renewable_gen
        self.hours = hours
        self.timestamps = timestamps
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        
        if not self.solver:
            raise RuntimeError("Solver not available")

        if len(demand) < hours or len(renewable_gen) < hours:
            raise ValueError("Input data is shorter than the requested optimization horizon")
        
        self.results = None
    
    def optimize(self):
        """Solve the unit commitment problem."""
        
        print("Setting up optimization problem...")
        
        # Decision variables
        # output[h][g] = generation of generator g in hour h
        output = {}
        # status[h][g] = 1 if generator g is on in hour h, 0 otherwise
        status = {}
        # startup[h][g] = 1 if generator g starts up in hour h
        startup = {}
        # unmet[h] = demand that cannot be served by generation
        unmet = {}
        # curtailment[h] = renewable energy that is not used
        curtailment = {}
        
        for h in range(self.hours):
            for g_idx, gen in enumerate(self.grid.generators):
                # Generation must be between 0 and capacity
                output[h, g_idx] = self.solver.NumVar(
                    0, gen.capacity, f'output_h{h}_g{g_idx}'
                )
                # Binary status variable
                status[h, g_idx] = self.solver.IntVar(
                    0, 1, f'status_h{h}_g{g_idx}'
                )
                startup[h, g_idx] = self.solver.IntVar(
                    0, 1, f'startup_h{h}_g{g_idx}'
                )

            unmet[h] = self.solver.NumVar(0, self.solver.infinity(), f'unmet_h{h}')
            curtailment[h] = self.solver.NumVar(0, self.solver.infinity(), f'curtailment_h{h}')
        
        print("Adding constraints...")
        
        # Constraints
        for h in range(self.hours):
            renewable_available = self.renewable_gen[h]
            demand = self.demand[h]

            generation_sum = sum(output[h, g_idx] for g_idx in range(len(self.grid.generators)))

            # Balance demand exactly with optional unmet demand and renewable curtailment.
            self.solver.Add(
                generation_sum + renewable_available - curtailment[h] + unmet[h] == demand
            )
            self.solver.Add(curtailment[h] <= renewable_available)
            
            # Generator must be within min/max output
            for g_idx, gen in enumerate(self.grid.generators):
                # If on, must be at least min_output
                self.solver.Add(
                    output[h, g_idx] >= gen.min_output * status[h, g_idx]
                )
                # Cannot exceed capacity
                self.solver.Add(output[h, g_idx] <= gen.capacity * status[h, g_idx])

                # Startup is triggered when status changes from off to on.
                if h == 0:
                    self.solver.Add(startup[h, g_idx] >= status[h, g_idx])
                else:
                    self.solver.Add(startup[h, g_idx] >= status[h, g_idx] - status[h - 1, g_idx])
                self.solver.Add(startup[h, g_idx] <= status[h, g_idx])
        
        print("Setting objective function...")
        
        # Objective: Minimize total cost
        objective = 0
        unmet_penalty = 10000  # Strongly discourage blackouts.
        
        for h in range(self.hours):
            for g_idx, gen in enumerate(self.grid.generators):
                # Fuel cost
                objective += output[h, g_idx] * gen.fuel_cost
                
                # Startup cost (if turning on)
                objective += startup[h, g_idx] * gen.startup_cost

            objective += unmet[h] * unmet_penalty
        
        self.solver.Minimize(objective)
        
        print("Solving...")
        status_code = self.solver.Solve()
        
        if status_code == pywraplp.Solver.OPTIMAL:
            print(f"Optimal solution found! Cost: ${self.solver.Objective().Value():,.0f}")
            self._store_results(output, status, startup, unmet, curtailment)
            return True
        else:
            print("Optimization did not find optimal solution.")
            return False
    
    def _store_results(self, output, status, startup, unmet, curtailment):
        """Extract and store optimization results."""
        
        self.results = {
            'output': {},
            'status': {},
            'startup': {},
            'unmet': {},
            'curtailment': {},
            'total_cost': self.solver.Objective().Value()
        }
        
        for h in range(self.hours):
            for g_idx in range(len(self.grid.generators)):
                self.results['output'][h, g_idx] = output[h, g_idx].solution_value()
                self.results['status'][h, g_idx] = int(status[h, g_idx].solution_value())
                self.results['startup'][h, g_idx] = int(startup[h, g_idx].solution_value())
            self.results['unmet'][h] = unmet[h].solution_value()
            self.results['curtailment'][h] = curtailment[h].solution_value()
    
    def get_dispatch_schedule(self):
        """Return optimized dispatch as dataframe."""
        
        if self.results is None:
            return None
        
        schedule = []
        for h in range(self.hours):
            row = {
                'hour': h,
                'datetime': self.timestamps[h] if self.timestamps is not None else None,
                'unmet_mw': self.results['unmet'][h],
                'curtailed_renewable_mw': self.results['curtailment'][h],
            }
            for g_idx, gen in enumerate(self.grid.generators):
                row[f'{gen.name}_mw'] = self.results['output'][h, g_idx]
                row[f'{gen.name}_status'] = self.results['status'][h, g_idx]
            schedule.append(row)
        
        return pd.DataFrame(schedule)

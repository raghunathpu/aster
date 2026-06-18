"""
ASTER — Manpower Optimization Solver
====================================
Uses Google OR-Tools (Mixed-Integer Linear Programming) to optimally allocate
traffic officers across junctions based on predicted Event Impact Scores (EIS)
and event priority constraints.
"""
from ortools.linear_solver import pywraplp

class ManpowerOptimizer:
    def __init__(self, total_officers=30):
        self.total_officers = total_officers
        self.max_officers_per_junction = 10
        self.min_officers_high_priority = 3
        self.min_officers_low_priority = 1

    def optimize(self, events: list[dict]) -> list[dict]:
        """
        Allocate officers to events based on predicted EIS and priorities.
        Each event in 'events' should have:
          - event_id: unique identifier
          - junction: junction name or coordinates
          - predicted_eis: float (0.0 to 1.0)
          - predicted_priority: int (0 or 1)
          - requires_road_closure: int (0 or 1)
        """
        if not events:
            return []

        # Create the MIP solver with the SCIP backend
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            # Fallback to GLOP if SCIP is not available
            solver = pywraplp.Solver.CreateSolver("GLOP")
            if not solver:
                raise RuntimeError("OR-Tools solver could not be initialized.")

        n_events = len(events)
        
        # Variables: x[i] = number of officers allocated to event i
        x = {}
        for i in range(n_events):
            x[i] = solver.IntVar(0, self.max_officers_per_junction, f"x_{i}")

        # Constraints
        # 1. Total officers allocated cannot exceed capacity
        solver.Add(solver.Sum([x[i] for i in range(n_events)]) <= self.total_officers)

        # 2. Minimum allocations based on priority
        for i in range(n_events):
            priority = events[i].get("predicted_priority", 0)
            closure = events[i].get("requires_road_closure", 0)
            
            # Set minimum requirements
            min_req = self.min_officers_low_priority
            if priority == 1:
                min_req = self.min_officers_high_priority
            if closure == 1:
                min_req = max(min_req, self.min_officers_high_priority + 1)
                
            solver.Add(x[i] >= min_req)

        # Objective: Maximize total utility (weighted by predicted EIS)
        # Utility = Sum(x[i] * predicted_eis)
        objective = solver.Objective()
        for i in range(n_events):
            eis = events[i].get("predicted_eis", 0.1)
            # Give higher weight to higher EIS events
            objective.SetCoefficient(x[i], float(eis))
        objective.SetMaximization()

        # Solve the system
        status = solver.Solve()

        allocations = []
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(n_events):
                allocated_val = int(x[i].solution_value())
                events[i]["allocated_officers"] = allocated_val
                # Priority index
                events[i]["allocation_priority"] = "High" if events[i].get("predicted_priority", 0) == 1 else "Medium"
                if events[i].get("requires_road_closure", 0) == 1:
                    events[i]["allocation_priority"] = "Critical"
                allocations.append(events[i])
        else:
            # Fallback allocation if solver fails (e.g. constraints are infeasible due to too few officers)
            print("Warning: Solver failed or was infeasible. Applying heuristic fallback.")
            for i in range(n_events):
                priority = events[i].get("predicted_priority", 0)
                allocated_val = self.min_officers_high_priority if priority == 1 else self.min_officers_low_priority
                events[i]["allocated_officers"] = allocated_val
                events[i]["allocation_priority"] = "Fallback"
                allocations.append(events[i])

        return allocations

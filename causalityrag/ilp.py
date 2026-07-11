"""Small 0/1 ILP solvers for minimum evidence cuts.

The default path uses ``scipy.optimize.milp`` when it is installed on the server.
For the first token baseline, where costs are normally integer token counts, an
exact dynamic-programming fallback is enough to keep the code runnable without
extra optimizer packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isclose
from typing import Iterable


EPS = 1e-9


@dataclass(frozen=True)
class Unit:
    """One selectable evidence unit."""

    unit_id: str
    support: float
    cost: float = 1.0
    payload: dict = field(default_factory=dict)


@dataclass
class ILPResult:
    selected_ids: list[str]
    selected_indices: list[int]
    objective: float
    support_removed: float
    total_support: float
    remaining_support: float
    target_support: float
    status: str
    solver: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "selected_cut": self.selected_ids,
            "selected_indices": self.selected_indices,
            "cut_cost": round(self.objective, 6),
            "support_removed": round(self.support_removed, 6),
            "total_support": round(self.total_support, 6),
            "remaining_support": round(self.remaining_support, 6),
            "target_support": round(self.target_support, 6),
            "status": self.status,
            "solver": self.solver,
        }
        out.update(self.metadata)
        return out


def solve_min_cost_cover(
    units: Iterable[Unit],
    *,
    target_support: float | None = None,
    tau_graph: float | None = 0.2,
    solver: str = "auto",
) -> ILPResult:
    """Minimize cost while removing enough predicted answer-support flow.

    The graph proxy objective is:

        min sum_i cost_i x_i
        s.t. sum_i support_i x_i >= target_support
             x_i in {0, 1}

    When ``target_support`` is omitted, it is derived as
    ``(1 - tau_graph) * total_support``.
    """

    items = _clean(units)
    total = sum(u.support for u in items)
    if target_support is None:
        if tau_graph is None:
            raise ValueError("either target_support or tau_graph is required")
        target_support = max(0.0, (1.0 - float(tau_graph)) * total)
    target_support = float(target_support)

    if target_support <= EPS:
        return _empty(items, target_support, "optimal", "trivial")
    if total + EPS < target_support:
        return _empty(items, target_support, "infeasible", "none")

    if solver in {"auto", "scipy"}:
        result = _solve_min_cost_scipy(items, target_support)
        if result is not None or solver == "scipy":
            return result or _empty(items, target_support, "solver_failed", "scipy")
    if solver in {"auto", "dp"}:
        result = _solve_dp(items, target_support=target_support)
        if result is not None or solver == "dp":
            return result or _empty(items, target_support, "solver_failed", "dp")
    if solver not in {"auto", "branch"}:
        raise ValueError(f"unknown solver: {solver}")
    return _solve_min_cost_branch(items, target_support)


def solve_budgeted_support(
    units: Iterable[Unit],
    *,
    budget: float,
    solver: str = "auto",
) -> ILPResult:
    """Maximize removed support under a fixed edit budget."""

    items = _clean(units)
    budget = float(budget)
    if budget < -EPS:
        raise ValueError("budget must be non-negative")
    if not items or budget <= EPS:
        return _empty(items, 0.0, "optimal", "trivial", metadata={"budget": round(budget, 6)})

    if solver in {"auto", "scipy"}:
        result = _solve_budgeted_scipy(items, budget)
        if result is not None or solver == "scipy":
            return result or _empty(items, 0.0, "solver_failed", "scipy")
    if solver in {"auto", "dp"}:
        result = _solve_dp(items, budget=budget)
        if result is not None or solver == "dp":
            return result or _empty(items, 0.0, "solver_failed", "dp")
    if solver not in {"auto", "branch"}:
        raise ValueError(f"unknown solver: {solver}")
    return _solve_budgeted_branch(items, budget)


def _clean(units: Iterable[Unit]) -> list[Unit]:
    out = []
    for unit in units:
        support = float(unit.support)
        cost = float(unit.cost)
        if support <= EPS:
            continue
        if cost < -EPS:
            raise ValueError(f"negative cost for {unit.unit_id}: {cost}")
        out.append(Unit(str(unit.unit_id), support, max(0.0, cost), dict(unit.payload)))
    return out


def _result(
    units: list[Unit],
    selected: list[int],
    *,
    target_support: float,
    status: str,
    solver: str,
    metadata: dict | None = None,
) -> ILPResult:
    selected = sorted(selected)
    support = sum(units[i].support for i in selected)
    cost = sum(units[i].cost for i in selected)
    total = sum(u.support for u in units)
    return ILPResult(
        selected_ids=[units[i].unit_id for i in selected],
        selected_indices=selected,
        objective=cost,
        support_removed=support,
        total_support=total,
        remaining_support=max(0.0, total - support),
        target_support=target_support,
        status=status,
        solver=solver,
        metadata=metadata or {},
    )


def _empty(
    units: list[Unit],
    target_support: float,
    status: str,
    solver: str,
    metadata: dict | None = None,
) -> ILPResult:
    total = sum(u.support for u in units)
    return ILPResult([], [], 0.0, 0.0, total, total, target_support, status, solver, metadata or {})


def _solve_min_cost_scipy(units: list[Unit], target_support: float) -> ILPResult | None:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None

    costs = np.array([u.cost for u in units], dtype=float)
    supports = np.array([u.support for u in units], dtype=float)
    constraints = LinearConstraint(supports[None, :], lb=[target_support], ub=[np.inf])
    try:
        res = milp(
            c=costs,
            integrality=np.ones(len(units), dtype=int),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"disp": False},
        )
    except Exception:
        return None
    if not res.success:
        return None
    selected = [i for i, x in enumerate(res.x) if x >= 0.5]
    return _result(units, selected, target_support=target_support, status="optimal", solver="scipy")


def _solve_budgeted_scipy(units: list[Unit], budget: float) -> ILPResult | None:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None

    costs = np.array([u.cost for u in units], dtype=float)
    supports = np.array([u.support for u in units], dtype=float)
    constraints = LinearConstraint(costs[None, :], lb=[-np.inf], ub=[budget])
    try:
        res = milp(
            c=-supports,
            integrality=np.ones(len(units), dtype=int),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"disp": False},
        )
    except Exception:
        return None
    if not res.success:
        return None
    selected = [i for i, x in enumerate(res.x) if x >= 0.5]
    target = sum(units[i].support for i in selected)
    return _result(
        units,
        selected,
        target_support=target,
        status="optimal",
        solver="scipy",
        metadata={"budget": round(budget, 6)},
    )


def _integer_costs(units: list[Unit], limit: int = 20000) -> list[int] | None:
    costs = []
    total = 0
    for unit in units:
        rounded = int(round(unit.cost))
        if rounded < 0 or not isclose(unit.cost, rounded, abs_tol=1e-9):
            return None
        costs.append(rounded)
        total += rounded
        if total > limit:
            return None
    return costs


def _solve_dp(
    units: list[Unit],
    *,
    target_support: float | None = None,
    budget: float | None = None,
) -> ILPResult | None:
    costs = _integer_costs(units)
    if costs is None:
        return None
    max_cost = int(round(budget)) if budget is not None else sum(costs)
    if max_cost < 0:
        return None

    dp = [-inf] * (max_cost + 1)
    dp[0] = 0.0
    take = [bytearray(max_cost + 1) for _ in units]
    for i, unit in enumerate(units):
        c = costs[i]
        if c > max_cost:
            continue
        for b in range(max_cost, c - 1, -1):
            cand = dp[b - c] + unit.support
            if cand > dp[b] + EPS:
                dp[b] = cand
                take[i][b] = 1

    if target_support is not None:
        chosen_budget = next((b for b, support in enumerate(dp) if support + EPS >= target_support), None)
        if chosen_budget is None:
            return _empty(units, target_support, "infeasible", "dp")
        target = target_support
    else:
        chosen_budget = max(range(max_cost + 1), key=lambda b: (dp[b], -b))
        target = max(0.0, dp[chosen_budget])

    selected = []
    b = chosen_budget
    for i in range(len(units) - 1, -1, -1):
        if take[i][b]:
            selected.append(i)
            b -= costs[i]
    return _result(
        units,
        selected,
        target_support=target,
        status="optimal",
        solver="dp",
        metadata={"budget": round(float(budget), 6)} if budget is not None else {},
    )


def _solve_min_cost_branch(units: list[Unit], target_support: float) -> ILPResult:
    order = sorted(range(len(units)), key=lambda i: (-units[i].support / max(units[i].cost, EPS), units[i].cost))
    ordered = [units[i] for i in order]
    suffix = [0.0] * (len(ordered) + 1)
    for i in range(len(ordered) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + ordered[i].support

    best_cost, best_support, best_selected = inf, 0.0, []

    def dfs(i: int, cost: float, support: float, selected: list[int]) -> None:
        nonlocal best_cost, best_support, best_selected
        if support + EPS >= target_support:
            if cost < best_cost - EPS or (abs(cost - best_cost) <= EPS and support > best_support):
                best_cost, best_support, best_selected = cost, support, list(selected)
            return
        if i >= len(ordered) or support + suffix[i] + EPS < target_support or cost >= best_cost - EPS:
            return
        selected.append(i)
        dfs(i + 1, cost + ordered[i].cost, support + ordered[i].support, selected)
        selected.pop()
        dfs(i + 1, cost, support, selected)

    dfs(0, 0.0, 0.0, [])
    if best_cost == inf:
        return _empty(units, target_support, "infeasible", "branch")
    return _result(
        units,
        [order[i] for i in best_selected],
        target_support=target_support,
        status="optimal",
        solver="branch",
    )


def _solve_budgeted_branch(units: list[Unit], budget: float) -> ILPResult:
    order = sorted(range(len(units)), key=lambda i: (-units[i].support / max(units[i].cost, EPS), units[i].cost))
    ordered = [units[i] for i in order]
    suffix = [0.0] * (len(ordered) + 1)
    for i in range(len(ordered) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + ordered[i].support

    best_support, best_cost, best_selected = 0.0, 0.0, []

    def dfs(i: int, cost: float, support: float, selected: list[int]) -> None:
        nonlocal best_support, best_cost, best_selected
        if cost > budget + EPS:
            return
        if support > best_support + EPS or (abs(support - best_support) <= EPS and cost < best_cost):
            best_support, best_cost, best_selected = support, cost, list(selected)
        if i >= len(ordered) or support + suffix[i] <= best_support + EPS:
            return
        selected.append(i)
        dfs(i + 1, cost + ordered[i].cost, support + ordered[i].support, selected)
        selected.pop()
        dfs(i + 1, cost, support, selected)

    dfs(0, 0.0, 0.0, [])
    return _result(
        units,
        [order[i] for i in best_selected],
        target_support=best_support,
        status="optimal",
        solver="branch",
        metadata={"budget": round(budget, 6)},
    )

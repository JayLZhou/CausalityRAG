"""Experimental MILP oracle for grouped residual-flow interdiction.

The oracle is intentionally separate from the proposed min-cut algorithm.  It
is used only on restricted graphs to measure optimization gaps.
"""

from __future__ import annotations

from math import isfinite
from time import monotonic

from causalityrag.mixed_cut import (
    RawContributionNetwork,
    remaining_group_support_flow,
    remaining_support_flow,
)


def solve_exact_group_flow_blocker(
    network: RawContributionNetwork,
    *,
    flow_threshold: float,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
) -> dict:
    """Minimize selected token labels subject to residual max-flow at most B.

    Max-flow/min-cut duality turns the residual-flow constraint into the
    existence of an ``s-t`` cut of capacity at most ``flow_threshold``.  A
    binary label variable permits every layer-copy gate carrying that label to
    cross the cut for free.  The formulation is exact but NP-hard because the
    label variable is shared across copies.
    """

    if network.status != "ok":
        return _empty_result(network.status, flow_threshold)
    if flow_threshold < 0 or not isfinite(flow_threshold):
        raise ValueError("flow_threshold must be finite and non-negative")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive when provided")
    if mip_rel_gap < 0 or not isfinite(mip_rel_gap):
        raise ValueError("mip_rel_gap must be finite and non-negative")
    if network.gate_scope != "contracted_token_nodes":
        raise ValueError("exact group oracle requires explicit layer-copy gates")

    initial_flow = remaining_support_flow(network, frozenset())
    if initial_flow <= flow_threshold + 1e-12:
        return {
            **_empty_result("optimal", flow_threshold),
            "selected_ids": [],
            "n_selected": 0,
            "initial_flow": initial_flow,
            "remaining_support_flow": initial_flow,
            "objective_value": 0.0,
            "optimal": True,
            "solver_status_code": 0,
        }

    labels = sorted(set(network.selection_unit_by_gate.values()))
    if not labels:
        return {
            **_empty_result("infeasible", flow_threshold),
            "initial_flow": initial_flow,
            "solver_status_code": 2,
        }

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("SciPy with scipy.optimize.milp is required") from exc

    token_node_to_gate = {
        node: gate_id for gate_id, node in network.token_nodes_by_unit.items()
    }
    if len(token_node_to_gate) != len(network.token_nodes_by_unit):
        raise ValueError("each layer-copy gate must own a distinct graph node")

    next_variable = 0
    x_index = {}
    for label in labels:
        x_index[label] = next_variable
        next_variable += 1

    z_in_index = {}
    z_out_index = {}
    for node in sorted(network.nodes):
        if node in token_node_to_gate:
            z_in_index[node] = next_variable
            next_variable += 1
            z_out_index[node] = next_variable
            next_variable += 1
        else:
            z_in_index[node] = next_variable
            z_out_index[node] = next_variable
            next_variable += 1

    y_index = []
    for _ in network.edges:
        y_index.append(next_variable)
        next_variable += 1

    objective = np.zeros(next_variable, dtype=float)
    integrality = np.zeros(next_variable, dtype=np.uint8)
    lower = np.zeros(next_variable, dtype=float)
    upper = np.ones(next_variable, dtype=float)
    for index in x_index.values():
        objective[index] = 1.0
        integrality[index] = 1
    for index in set(z_in_index.values()) | set(z_out_index.values()):
        integrality[index] = 1

    sink_index = z_in_index[network.sink]
    lower[sink_index] = 0.0
    upper[sink_index] = 0.0
    for root_gate in network.roots_by_unit:
        root_node = network.token_nodes_by_unit[root_gate]
        index = z_in_index[root_node]
        lower[index] = 1.0
        upper[index] = 1.0

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(coefficients: dict[int, float], lb: float, ub: float) -> None:
        row = len(constraint_lower)
        for column, value in coefficients.items():
            if value:
                row_indices.append(row)
                column_indices.append(column)
                values.append(float(value))
        constraint_lower.append(lb)
        constraint_upper.append(ub)

    # A gate may separate its in/out copies only when its shared token label is
    # selected.  Registry-excluded gates have no label variable and are fixed
    # as immutable by the same inequality without x_i.
    for node, gate_id in token_node_to_gate.items():
        coefficients = {
            z_in_index[node]: 1.0,
            z_out_index[node]: -1.0,
        }
        label = network.selection_unit_by_gate.get(gate_id)
        if label is not None:
            coefficients[x_index[label]] = -1.0
        add_constraint(coefficients, -np.inf, 0.0)

    # y_e is one when a finite contribution edge crosses from the source side
    # to the sink side of the witness cut.
    for edge_index, (src, dst, _) in enumerate(network.edges):
        add_constraint(
            {
                z_out_index[src]: 1.0,
                z_in_index[dst]: -1.0,
                y_index[edge_index]: -1.0,
            },
            -np.inf,
            0.0,
        )
    add_constraint(
        {
            y_index[index]: capacity
            for index, (_, _, capacity) in enumerate(network.edges)
        },
        -np.inf,
        flow_threshold,
    )

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_lower), next_variable),
    ).tocsr()
    options = {"mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = time_limit
    started = monotonic()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower),
            np.asarray(constraint_upper),
        ),
        options=options,
    )
    elapsed = monotonic() - started
    status_names = {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    status = status_names.get(int(result.status), "solver_error")
    selected = []
    if result.x is not None:
        selected = sorted(
            label for label, index in x_index.items() if result.x[index] >= 0.5
        )
    residual = (
        remaining_group_support_flow(network, set(selected))
        if result.x is not None
        else None
    )
    return {
        "status": status,
        "selected_ids": selected,
        "n_selected": len(selected),
        "initial_flow": initial_flow,
        "flow_threshold": flow_threshold,
        "remaining_support_flow": residual,
        "objective_value": (
            float(result.fun) if result.fun is not None else None
        ),
        "optimal": int(result.status) == 0,
        "solver_status_code": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": _optional_float(getattr(result, "mip_gap", None)),
        "mip_node_count": _optional_int(
            getattr(result, "mip_node_count", None)
        ),
        "mip_dual_bound": _optional_float(
            getattr(result, "mip_dual_bound", None)
        ),
        "elapsed_seconds": elapsed,
        "variables": next_variable,
        "binary_variables": int(integrality.sum()),
        "constraints": len(constraint_lower),
        "editable_labels": len(labels),
        "layer_copy_gates": len(token_node_to_gate),
        "finite_edges": len(network.edges),
        "solver": "scipy_highs_exact_group_cut_oracle",
    }


def _empty_result(status: str, flow_threshold: float) -> dict:
    return {
        "status": status,
        "selected_ids": [],
        "n_selected": 0,
        "initial_flow": 0.0,
        "flow_threshold": flow_threshold,
        "remaining_support_flow": None,
        "objective_value": None,
        "optimal": False,
        "solver": "scipy_highs_exact_group_cut_oracle",
    }


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_int(value) -> int | None:
    return None if value is None else int(value)

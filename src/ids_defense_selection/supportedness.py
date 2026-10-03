"""Supported versus unsupported Pareto-efficient defense strategies (C6).

Weighted-sum scalarisation (Eq. 6) can only ever select *supported* efficient
points, i.e. points that maximise a non-negative weighted sum.  Efficient points
that lie below the upper convex hull of the objective set are *unsupported* and
unreachable for every preference vector.  With six discrete candidates this is
an exact linear-programming question, not an approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def supported_solutions(objectives: pd.DataFrame, *, tol: float = 1e-9) -> pd.DataFrame:
    """Classify every candidate as supported, weakly supported or unreachable.

    For candidate ``i`` we solve

        max t   s.t.  (φ_i − φ_j) · θ ≥ t  for all j ≠ i,
                      Σθ_k = 1, θ_k ≥ 0

    The optimum ``t*`` is the support margin: ``t* > 0`` means strongly
    supported, ``t* ≈ 0`` weakly supported (ties), ``t* < 0`` means no
    preference vector can make candidate ``i`` the recommendation.
    """
    if objectives.empty:
        return pd.DataFrame(columns=["model", "supported", "margin",
                                     "theta1", "theta2", "theta3", "theta4"])
    matrix = objectives.to_numpy(dtype=float)
    names = objectives.index.tolist()
    n_candidates, n_objectives = matrix.shape
    rows = []
    for index in range(n_candidates):
        a_ub, b_ub = [], []
        for other in range(n_candidates):
            if other == index:
                continue
            # -(φ_i − φ_j)·θ + t ≤ 0
            a_ub.append(list(-(matrix[index] - matrix[other])) + [1.0])
            b_ub.append(0.0)
        cost = [0.0] * n_objectives + [-1.0]  # maximise t
        result = linprog(
            cost, A_ub=a_ub, b_ub=b_ub,
            A_eq=[[1.0] * n_objectives + [0.0]], b_eq=[1.0],
            bounds=[(0.0, None)] * n_objectives + [(None, None)],
            method="highs",
        )
        if not result.success:
            rows.append({"model": names[index], "supported": None,
                         "margin": float("nan"),
                         **{f"theta{k + 1}": float("nan") for k in range(n_objectives)}})
            continue
        margin = float(-result.fun)
        theta = np.round(result.x[:n_objectives], 6)
        rows.append({
            "model": names[index],
            "supported": bool(margin >= -tol),
            "margin": margin,
            **{f"theta{k + 1}": float(theta[k]) for k in range(n_objectives)},
        })
    return pd.DataFrame(rows)

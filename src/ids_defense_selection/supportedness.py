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


def tchebycheff_scores(objectives: pd.DataFrame, theta, *, rho: float = 0.01) -> np.ndarray:
    """Augmented weighted Tchebycheff score (minimise; smaller is better).

    ``gap = (ideal - x) * theta``; the score is ``max(gap) - rho * sum(gap)``.
    Unlike the weighted sum, this scalarisation can reach unsupported efficient
    points when the front is non-convex.
    """
    matrix = objectives.to_numpy(dtype=float)
    ideal = matrix.max(axis=0)
    gap = (ideal - matrix) * np.asarray(theta, dtype=float)
    return gap.max(axis=1) - rho * gap.sum(axis=1)


def tchebycheff_selection(objectives: pd.DataFrame, theta, *, rho: float = 0.01) -> str:
    """Candidate minimising the augmented Tchebycheff score."""
    scores = tchebycheff_scores(objectives, theta, rho=rho)
    return objectives.index[int(np.argmin(scores))]


def _simplex_grid(n_objectives: int, step: float) -> list[list[float]]:
    n = int(round(1.0 / step))
    if n < 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError(f"step {step!r} must divide 1.0 exactly")
    points: list[list[float]] = []

    def walk(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            points.append([value / n for value in prefix + [remaining]])
            return
        for value in range(remaining + 1):
            walk(prefix + [value], remaining - value, slots - 1)

    walk([], n, n_objectives)
    return points


def tchebycheff_reachability(objectives: pd.DataFrame, *, step: float = 0.05,
                            rho: float = 0.01) -> pd.DataFrame:
    """Which candidates are reachable by the augmented Tchebycheff rule."""
    if objectives.empty:
        return pd.DataFrame(columns=["model", "reachable_tchebycheff", "theta_tchebycheff"])
    reachable: dict[str, list[float] | None] = {name: None for name in objectives.index}
    for theta in _simplex_grid(objectives.shape[1], step):
        winner = tchebycheff_selection(objectives, theta, rho=rho)
        if reachable[winner] is None:
            reachable[winner] = list(theta)
    return pd.DataFrame([
        {"model": name,
         "reachable_tchebycheff": theta is not None,
         "theta_tchebycheff": "" if theta is None else ",".join(f"{v:.2f}" for v in theta)}
        for name, theta in reachable.items()
    ])

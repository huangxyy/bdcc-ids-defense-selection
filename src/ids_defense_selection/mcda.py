"""Reference multi-criteria methods: NSGA-II, MOEA/D and AHP.

The decision problem has six discrete candidates and four bounded objectives,
so these methods are implemented to answer a specific question: do generic
multi-objective search (NSGA-II, MOEA/D) or preference elicitation (AHP)
change the recommendations of the Pareto + weighted-sum framework?  All three
operate on the same admissible candidate set used by the decision layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OBJECTIVES = ("phi1", "phi2", "phi3", "phi4")


# ── shared helpers ───────────────────────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_names(objectives: np.ndarray, names: list[str]) -> list[str]:
    """Enumerated Pareto front (maximisation) of a candidate set."""
    matrix = np.asarray(objectives, dtype=float)
    keep = [i for i in range(len(names))
            if not any(_dominates(matrix[j], matrix[i])
                       for j in range(len(names)) if j != i)]
    return sorted(names[i] for i in keep)


def _minmax(matrix: np.ndarray) -> np.ndarray:
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    return (matrix - lo) / (hi - lo + 1e-10)


def _pick_from_front(objectives: np.ndarray, names: list[str],
                     front: list[str], theta: tuple) -> str:
    """Select from an achieved front with the framework's weighted sum."""
    if not front:
        return "N/A"
    index = [names.index(name) for name in front]
    norm = _minmax(np.asarray(objectives, dtype=float)[index])
    return front[int(np.argmax(norm @ np.asarray(theta, dtype=float)))]


# ── NSGA-II (discrete candidate set) ─────────────────────────────────────────

def _fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    n = len(objectives)
    dominated: list[list[int]] = [[] for _ in range(n)]
    count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objectives[i], objectives[j]):
                dominated[i].append(j)
                count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominated[j].append(i)
                count[i] += 1
    fronts: list[list[int]] = []
    current = [i for i in range(n) if count[i] == 0]
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for i in current:
            for j in dominated[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def _crowding(front: list[int], objectives: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(front))
    if len(front) <= 2:
        return np.full(len(front), np.inf)
    for m in range(objectives.shape[1]):
        values = objectives[front, m]
        order = np.argsort(values)
        distance[order[0]] = distance[order[-1]] = np.inf
        span = values[order[-1]] - values[order[0]]
        if span <= 0:
            continue
        for k in range(1, len(front) - 1):
            distance[order[k]] += (values[order[k + 1]] - values[order[k - 1]]) / span
    return distance


def nsga2_front(objectives: np.ndarray, names: list[str], *,
                population: int = 24, generations: int = 40,
                seed: int = 2026) -> list[str]:
    """Run a discrete NSGA-II and return its non-dominated set.

    Each individual is an index of a candidate defense; variation samples the
    six discrete choices.  The returned set is compared against the enumerated
    front, which measures whether the search recovers all efficient points.
    """
    matrix = np.asarray(objectives, dtype=float)
    n = len(names)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    pop = list(rng.integers(0, n, size=population))
    for _ in range(generations):
        fronts = _fast_non_dominated_sort(matrix[pop])
        rank = {idx: r for r, front in enumerate(fronts) for idx in front}
        crowd: dict[int, float] = {}
        for front in fronts:
            dist = _crowding(front, matrix[pop])
            crowd.update({idx: d for idx, d in zip(front, dist)})

        def tournament() -> int:
            a, b = map(int, rng.integers(0, population, 2))
            if rank[a] != rank[b]:
                return a if rank[a] < rank[b] else b
            return a if crowd[a] >= crowd[b] else b

        offspring: list[int] = []
        while len(offspring) < population:
            child = pop[tournament()] if rng.random() < 0.5 else pop[tournament()]
            if rng.random() < 1.0 / n:
                child = int(rng.integers(0, n))
            offspring.append(child)

        combined = pop + offspring
        fronts = _fast_non_dominated_sort(matrix[combined])
        new_pop: list[int] = []
        for front in fronts:
            if len(new_pop) + len(front) <= population:
                new_pop.extend(front)
            else:
                dist = _crowding(front, matrix[combined])
                order = np.argsort(-dist)
                new_pop.extend(front[i] for i in order[:population - len(new_pop)])
                break
        pop = [combined[i] for i in new_pop]

    final_front = _fast_non_dominated_sort(matrix[pop])[0]
    return sorted({names[pop[i]] for i in final_front})


# ── MOEA/D (weighted Tchebycheff, discrete candidate set) ────────────────────

def _simplex_weights(partitions: int) -> np.ndarray:
    points = []
    for i in range(partitions + 1):
        for j in range(partitions + 1 - i):
            for k in range(partitions + 1 - i - j):
                last = partitions - i - j - k
                points.append([i / partitions, j / partitions,
                               k / partitions, last / partitions])
    return np.asarray(points, dtype=float)


def moead_front(objectives: np.ndarray, names: list[str], *,
                partitions: int = 8, neighborhood: int = 20,
                generations: int = 40, seed: int = 2026) -> list[str]:
    """Run MOEA/D with weighted Tchebycheff scalarisation.

    The population is a set of candidate indices, one per weight vector; the
    neighbourhood replacement follows the classic MOEA/D scheme.  Weighted
    Tchebycheff can reach unsupported efficient points, so the achieved front
    is compared with the enumeration as well as with the weighted sum.
    """
    matrix = np.asarray(objectives, dtype=float)
    n = len(names)
    if n == 0:
        return []
    weights = _simplex_weights(partitions)
    rng = np.random.default_rng(seed)
    pop = list(rng.integers(0, n, size=len(weights)))
    ideal = matrix.max(axis=0)
    order = np.argsort(np.linalg.norm(weights[:, None, :] - weights[None, :, :],
                                      axis=2), axis=1)
    size = min(neighborhood, len(weights))

    def tchebycheff(candidate: int, weight: np.ndarray) -> float:
        return float(np.max(weight * (ideal - matrix[candidate])))

    for _ in range(generations):
        for i, weight in enumerate(weights):
            neighbours = order[i, :size]
            a, b = (int(x) for x in rng.choice(neighbours, 2, replace=False))
            child = pop[a] if rng.random() < 0.5 else pop[b]
            if rng.random() < 1.0 / n:
                child = int(rng.integers(0, n))
            for j in neighbours:
                if tchebycheff(child, weights[j]) <= tchebycheff(pop[j], weights[j]):
                    pop[j] = child
                ideal = np.maximum(ideal, matrix[child])

    unique = sorted(set(pop))
    sub = matrix[unique]
    keep = non_dominated_names(sub, [names[i] for i in unique])
    return sorted(keep)


# ── AHP (preference elicitation from pairwise comparisons) ───────────────────

def pairwise_from_theta(theta: tuple, floor: float = 1e-6) -> np.ndarray:
    """Consistent AHP pairwise matrix implied by a preference vector."""
    weights = np.maximum(np.asarray(theta, dtype=float), floor)
    return np.outer(weights, 1.0 / weights)


def ahp_weights(pairwise: np.ndarray) -> np.ndarray:
    """Principal-eigenvector weights of an AHP pairwise comparison matrix."""
    values, vectors = np.linalg.eig(np.asarray(pairwise, dtype=float))
    principal = int(np.argmax(values.real))
    weights = np.abs(vectors[:, principal].real)
    total = weights.sum()
    return weights / total if total > 0 else weights


def ahp_select(objectives: np.ndarray, names: list[str], theta: tuple, *,
               normalise: str = "minmax") -> tuple[str, float]:
    """AHP-elicited weights, then the framework's selection rule.

    Returns ``(selected_name, max_abs_weight_error)`` where the error compares
    the eigenvector weights with the (renormalised) preference vector.
    """
    if not names:
        return "N/A", float("nan")
    matrix = np.asarray(objectives, dtype=float)
    weights = ahp_weights(pairwise_from_theta(theta))
    norm = (matrix - matrix.min(axis=0)) / (
        matrix.max(axis=0) - matrix.min(axis=0) + 1e-10) if normalise == "minmax" \
        else np.clip(matrix, 0.0, 1.0)
    target = np.asarray(theta, dtype=float)
    target = target / target.sum() if target.sum() else target
    error = float(np.max(np.abs(weights - target)))
    return names[int(np.argmax(norm @ weights))], error


# ── combined comparison table ────────────────────────────────────────────────

def compare_mcda_methods(means: pd.DataFrame, presets: dict[str, tuple], *,
                         seed: int = 2026) -> pd.DataFrame:
    """One row per preset: NSGA-II, MOEA/D and AHP against the same candidates.

    The reference methods are deterministic, so their selection is normalised
    over the *point-estimate* Pareto front.  The uncertainty-aware framework
    front may contain extra candidates (for example FT TRADES), and the
    difference between ``pareto_weighted`` and ``pareto_point_weighted`` is
    exactly the candidate-set effect that is reported separately.
    """
    names = means.index.tolist()
    matrix = means[list(OBJECTIVES)].values
    enumerated = non_dominated_names(matrix, names)
    front_index = [names.index(name) for name in enumerated]
    front_matrix = matrix[front_index]
    nsga2 = nsga2_front(matrix, names, seed=seed)
    moead = moead_front(matrix, names, seed=seed)
    rows = []
    for name, theta in presets.items():
        ahp_name, ahp_error = ahp_select(front_matrix, enumerated, theta)
        rows.append({
            "theta_name": name,
            "enumerated_front": ", ".join(enumerated),
            "nsga2_front": ", ".join(nsga2),
            "nsga2_selected": _pick_from_front(matrix, names, nsga2, theta),
            "nsga2_matches_enumeration": nsga2 == enumerated,
            "moead_front": ", ".join(moead),
            "moead_selected": _pick_from_front(matrix, names, moead, theta),
            "moead_matches_enumeration": moead == enumerated,
            "ahp_selected": ahp_name,
            "ahp_weight_error": ahp_error,
        })
    return pd.DataFrame(rows)

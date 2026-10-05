"""Reference MCDA/EMO methods: NSGA-II, MOEA/D and AHP."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ids_defense_selection.mcda import (
    ahp_select,
    ahp_weights,
    compare_mcda_methods,
    moead_front,
    non_dominated_names,
    nsga2_front,
    pairwise_from_theta,
)
from ids_defense_selection.selection import (
    OBJECTIVE_COLUMNS,
    THETA_PRESETS,
    select_defense,
)


def _objectives() -> pd.DataFrame:
    """a/b are supported efficient; mid is efficient but unsupported."""
    return pd.DataFrame(
        [[1.0, 0.0, 0.5, 0.5],
         [0.0, 1.0, 0.5, 0.5],
         [0.4, 0.4, 0.5, 0.5]],
        index=["a", "b", "mid"], columns=OBJECTIVE_COLUMNS,
    )


def test_enumeration_keeps_unsupported_efficient_point() -> None:
    frame = _objectives()
    assert non_dominated_names(frame.values, list(frame.index)) == ["a", "b", "mid"]


def test_nsga2_recovers_the_enumerated_front() -> None:
    frame = _objectives()
    enumerated = non_dominated_names(frame.values, list(frame.index))
    assert nsga2_front(frame.values, list(frame.index)) == enumerated


def test_moead_front_is_efficient_and_reaches_unsupported_point() -> None:
    frame = _objectives()
    enumerated = set(non_dominated_names(frame.values, list(frame.index)))
    front = moead_front(frame.values, list(frame.index))
    assert set(front) <= enumerated
    # Weighted Tchebycheff reaches the unsupported efficient point at equal weights.
    assert "mid" in front


def test_ahp_recovers_a_consistent_preference_vector() -> None:
    theta = (0.1, 0.6, 0.1, 0.2)
    weights = ahp_weights(pairwise_from_theta(theta))
    assert np.allclose(weights, np.asarray(theta) / sum(theta), atol=1e-9)
    frame = _objectives()
    selected, error = ahp_select(frame.values, list(frame.index), theta)
    assert error < 1e-9
    assert selected == select_defense(frame.values, list(frame.index), theta)


def test_compare_mcda_table_columns_and_matches() -> None:
    frame = _objectives()
    table = compare_mcda_methods(frame, THETA_PRESETS)
    assert len(table) == len(THETA_PRESETS)
    assert set(table.columns) >= {
        "theta_name", "nsga2_front", "nsga2_selected", "nsga2_matches_enumeration",
        "moead_front", "moead_selected", "moead_matches_enumeration",
        "ahp_selected", "ahp_weight_error",
    }
    assert table["nsga2_matches_enumeration"].all()
    assert (table["ahp_weight_error"] < 1e-9).all()


def test_reference_methods_match_deterministic_weighted_selection() -> None:
    """With a finite candidate set the reference methods recover the same choice."""
    frame = _objectives()
    theta = THETA_PRESETS["Balanced"]
    front = non_dominated_names(frame.values, list(frame.index))
    expected = select_defense(frame.loc[front].values, front, theta)
    table = compare_mcda_methods(frame, {"Balanced": theta})
    assert table.loc[0, "nsga2_selected"] == expected
    assert table.loc[0, "moead_selected"] == expected
    assert table.loc[0, "ahp_selected"] == expected

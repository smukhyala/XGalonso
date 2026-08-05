"""A known divergence between the API's prediction path and the CLI's, pinned.

**This file asserts a defect, on purpose.** Nothing here is a property the
system should have; every assertion records what the system currently does, so
that closing the gap is a deliberate, visible change rather than something that
happens by accident and nobody notices either way.

The divergence. `apps/cli/.../main.py::_objective_feature_columns` is what makes
feature discovery matter: it reads the accepted features an objective earned
from the discovery registry and appends them to the frame before a model sees
it. Its own docstring says so — "without it the registry records which features
serve which objective and nothing downstream reads it, so every objective is
scored by the same fixed catalogue". The CLI calls it from `train` and from
`_predict_all`, which is the path behind `xg plan` and `xg build-squad`.

`DecisionService._predict` does not call it, and cannot: it takes a gameweek
and nothing else, and caches on that gameweek alone. So `POST /squad/plan`
scores every objective through the same unconditioned catalogue, and the same
objective against the same data root can produce a different squad depending on
whether it was asked for over HTTP or on the command line — the *same class* of
failure the `api/service.py` module header already records once, where a
mismatched default `model_path` made the two surfaces disagree by four points
and six million.

Scheduled for closure in a later phase, and deliberately not closed here: the
fix touches the objective-conditioning path that is under active change, and a
second uncoordinated edit to it would be worse than the divergence.

When it is closed, these tests should *fail*, and the right response is to
delete this file — not to relax it.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from tests.api.conftest import DISCOVERY_OBJECTIVE
from xg_alonso.api.service import DecisionService

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


def test_the_registry_does_hold_an_accepted_feature_for_this_objective(
    api_data_root: Path,
) -> None:
    """The premise. Without this, the tests below would prove nothing.

    A divergence between two paths is only observable when there is something
    for one of them to miss. This asserts the fixture registry holds a feature
    the CLI's `_objective_feature_columns` would materialise, so the assertions
    that follow are about the API declining to read it rather than about there
    being nothing to read.
    """
    from xg_alonso.discovery.registry import DiscoveryRegistry
    from xg_alonso.storage import ParquetTableStore

    registry = DiscoveryRegistry(ParquetTableStore(api_data_root / "discovery"))
    accepted = [spec for spec, _note in registry.accepted_features(DISCOVERY_OBJECTIVE)]
    assert [spec.name for spec in accepted] == ["minutes_trend_5"]


def test_the_service_cannot_condition_a_prediction_on_an_objective() -> None:
    """`_predict(gameweek)` — there is no parameter an objective could arrive on.

    Checked on the signature rather than on the body because it is the interface
    that makes the divergence structural: the CLI threads `objective_id` and
    `data_root` down to the feature build, and this seam has neither.
    """
    parameters = list(inspect.signature(DecisionService._predict).parameters)
    assert parameters == ["self", "gameweek"]


def test_two_objectives_share_one_cached_prediction_set(
    client: TestClient, api_service: DecisionService
) -> None:
    """The behavioural consequence, and the cheapest way to see it.

    `_predictions` is keyed on the gameweek alone, so planning under two
    different objectives never rebuilds the features. If the API ever started
    conditioning on the objective, that cache would have to grow a second key
    and this assertion would fail — which is exactly the alarm it exists to be.

    A private attribute is read deliberately: the divergence is invisible on the
    wire, and a characterisation test that could only see the wire could not
    characterise it.
    """
    presets = [row["id"] for row in client.get("/objectives").json()]
    assert len(presets) >= 2, "two distinct presets are needed to observe the sharing"

    for preset in presets[:2]:
        response = client.post("/squad/plan", json={"text": "", "preset": preset})
        assert response.status_code == 200, response.text

    gameweeks = set(api_service._predictions)
    assert gameweeks == {1}, (
        "predictions are cached per gameweek and not per objective, so both "
        f"plans were scored from one unconditioned feature build: {gameweeks}"
    )

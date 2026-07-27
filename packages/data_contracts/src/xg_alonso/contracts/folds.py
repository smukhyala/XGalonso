"""The single walk-forward fold definition.

Planning produced three fold types with three key types and three embargo
semantics. This is the one.

Slice 1 does not train a model — its baseline is closed-form — so nothing
constructs folds yet. The contract is frozen now anyway, because the moment two
packages need folds they will otherwise invent them separately and diverge on
exactly the boundary condition that causes silent leakage.

**Random splits on temporal data are forbidden.** Not discouraged — forbidden.
A shuffled split lets a model see gameweek 30 while predicting gameweek 12, and
the resulting metrics are fiction. :func:`walk_forward_folds` is the only
sanctioned constructor, and it is incapable of producing a shuffled split.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import GameweekId

__all__ = [
    "WalkForwardFold",
    "walk_forward_folds",
]


class WalkForwardFold(BaseModel):
    """One train/validate split, keyed by gameweek and strictly ordered in time.

    Gameweeks are treated as a globally increasing index across seasons, so a
    fold never wraps around a season boundary backwards.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_index: int = Field(ge=0)
    train_start: GameweekId
    train_end: GameweekId = Field(description="Inclusive last training gameweek")
    validate_start: GameweekId
    validate_end: GameweekId = Field(description="Inclusive last validation gameweek")
    embargo_gameweeks: int = Field(
        ge=0,
        description=(
            "Gameweeks skipped between training and validation. Guards against "
            "rolling features whose lookback window would otherwise straddle the "
            "boundary and carry validation information into training."
        ),
    )

    @model_validator(mode="after")
    def _check_strictly_forward(self) -> WalkForwardFold:
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start")
        if self.validate_end < self.validate_start:
            raise ValueError("validate_end must not precede validate_start")
        expected_start = self.train_end + 1 + self.embargo_gameweeks
        if self.validate_start < expected_start:
            raise ValueError(
                f"validation starts at GW{self.validate_start} but training ends at "
                f"GW{self.train_end} with a {self.embargo_gameweeks}-gameweek embargo, "
                f"so validation may not start before GW{expected_start}. "
                "Validation must always lie strictly in the future of training."
            )
        return self

    @property
    def train_gameweeks(self) -> range:
        return range(self.train_start, self.train_end + 1)

    @property
    def validate_gameweeks(self) -> range:
        return range(self.validate_start, self.validate_end + 1)


def walk_forward_folds(
    *,
    gameweeks: Sequence[GameweekId],
    min_train_gameweeks: int,
    validate_gameweeks: int = 1,
    embargo_gameweeks: int = 0,
    expanding: bool = True,
) -> list[WalkForwardFold]:
    """Build walk-forward folds over an ordered gameweek sequence.

    This is the only sanctioned way to construct folds. It cannot shuffle.

    Args:
        gameweeks: Ordered, ascending, unique gameweek identifiers.
        min_train_gameweeks: Minimum training window before the first fold.
        validate_gameweeks: Validation window length.
        embargo_gameweeks: Gap between training and validation.
        expanding: ``True`` grows the training window from the start (more data
            each fold); ``False`` slides a fixed-width window (adapts faster to
            regime change, at the cost of sample size).

    Raises:
        ValueError: if ``gameweeks`` is unsorted, has duplicates, or is too short.
    """
    ordered = list(gameweeks)
    if len(ordered) != len(set(ordered)):
        raise ValueError("gameweeks must be unique")
    if ordered != sorted(ordered):
        raise ValueError(
            "gameweeks must be ascending; an unsorted sequence is the usual "
            "symptom of a shuffled split reaching this function"
        )
    if min_train_gameweeks < 1:
        raise ValueError("min_train_gameweeks must be at least 1")

    span = min_train_gameweeks + embargo_gameweeks + validate_gameweeks
    if len(ordered) < span:
        raise ValueError(
            f"need at least {span} gameweeks for one fold "
            f"({min_train_gameweeks} train + {embargo_gameweeks} embargo + "
            f"{validate_gameweeks} validate), got {len(ordered)}"
        )

    folds: list[WalkForwardFold] = []
    train_end_pos = min_train_gameweeks - 1

    while True:
        validate_start_pos = train_end_pos + 1 + embargo_gameweeks
        validate_end_pos = validate_start_pos + validate_gameweeks - 1
        if validate_end_pos >= len(ordered):
            break

        train_start_pos = 0 if expanding else train_end_pos - min_train_gameweeks + 1

        folds.append(
            WalkForwardFold(
                fold_index=len(folds),
                train_start=ordered[train_start_pos],
                train_end=ordered[train_end_pos],
                validate_start=ordered[validate_start_pos],
                validate_end=ordered[validate_end_pos],
                embargo_gameweeks=embargo_gameweeks,
            )
        )
        train_end_pos += validate_gameweeks

    return folds

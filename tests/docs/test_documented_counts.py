"""Assert that the *quantities* the documentation states are still true of the code.

**Why this exists, separately from** :mod:`tests.docs.test_docs_match_code`. That
module checks the cheapest class of claim — *does this name exist*. It says so in
its own docstring, and it is explicit that the claims block is "a curated subset,
not a derived one". Two claims slipped through exactly that gap:

1. Five documents said the model-ready frame carries **231** distinct feature
   columns. :func:`model_feature_names` returns 224, and has for some time. No
   test asserted the documented figure, so the prose and the code disagreed
   while the whole suite stayed green.
2. The README said "Twenty-six commands" and listed twenty-six rows, after
   ``xg demo`` and ``xg discover-demo`` shipped. The claims block named neither,
   so nothing noticed the surface had grown.

Both are the same failure: a number copied into prose, then left behind by the
code it described. A name that stops existing fails an import; a number that
stops being true fails nothing. This module closes that class.

**What it does not do.** It does not read prose for meaning, and it does not try
to find every number in the documentation set — most of them are illustrative
(row counts in an example, a fold count in a recorded experiment) and pinning
those to live code would be wrong. It checks the two counts that are *claims
about the current build* and that a reader would reasonably take as current.

The suite runs offline: it imports the feature schema and the CLI app, and reads
Markdown. Nothing here touches the network or ``.data``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest
import typer

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
README: Final[Path] = REPO_ROOT / "README.md"

#: Every document that states a feature-column count. Scanned rather than listed,
#: so a sixth document repeating the figure is covered the day it is written.
_DOC_ROOTS: Final[tuple[Path, ...]] = (REPO_ROOT / "docs",)

#: `CLAUDE.md` is where the stale 231 originated, and it is read as authoritative
#: by anyone — human or agent — starting work here. It is checked with the docs.
_STANDALONE_DOCS: Final[tuple[Path, ...]] = (README, REPO_ROOT / "CLAUDE.md")

#: "224 distinct feature columns", "224 distinct columns". The optional "feature"
#: is why this is one pattern rather than two.
_FEATURE_COUNT = re.compile(r"(\d+)\s+distinct\s+(?:feature\s+)?columns?")

#: "Twenty-eight commands, all under the single `xg` entry point".
_COMMAND_COUNT = re.compile(r"^([A-Za-z-]+|\d+)\s+commands,\s+all under the single", re.MULTILINE)

#: Rows of the README command table: "| `xg models list` | Every artifact ... |".
_COMMAND_ROW = re.compile(r"^\|\s*`xg ([^`]+)`\s*\|", re.MULTILINE)

#: Spelled-out counts the prose is allowed to use. A number outside this range
#: fails with a message telling the author to use a digit, which is better than
#: silently skipping the assertion.
_NUMBER_WORDS: Final[dict[str, int]] = {
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
    "thirty": 30,
    "thirty-one": 31,
    "thirty-two": 32,
    "thirty-three": 33,
    "thirty-four": 34,
    "thirty-five": 35,
}


def _markdown_files() -> list[Path]:
    found: list[Path] = [path for path in _STANDALONE_DOCS if path.is_file()]
    for root in _DOC_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            if "node_modules" in path.parts or ".next" in path.parts:
                continue
            found.append(path)
    return sorted(set(found))


def _parse_count(raw: str, *, where: str) -> int:
    if raw.isdigit():
        return int(raw)
    word = raw.lower()
    if word not in _NUMBER_WORDS:
        raise AssertionError(
            f"{where}: cannot read the command count {raw!r}. Use a digit, or add the word "
            f"to _NUMBER_WORDS in {Path(__file__).name}."
        )
    return _NUMBER_WORDS[word]


def _declared_name(value: object) -> str | None:
    """A Typer name, or ``None`` when Typer left it to be derived.

    An unnamed ``add_typer`` stores a ``DefaultPlaceholder`` rather than ``None``,
    which is truthy. Testing for that placeholder by truthiness is how a walker
    ends up with ``"<DefaultPlaceholder ...> list"`` as a command name, so the
    check here is on the concrete type instead.
    """
    return value if type(value) is str else None


def _cli_command_names() -> set[str]:
    """Every invocable ``xg`` command, including those under a group.

    Typer nests a group as a sub-app, so a flat read of ``registered_commands``
    misses ``models list`` and ``evaluate report`` — the two groups that exist
    today, and the two the README got wrong. ``app.add_typer(sub)`` is called
    without a name here, so the group's name lives on ``typer_instance.info``
    rather than on the registration.
    """
    from xg_alonso.cli.main import app

    names: set[str] = set()

    def walk(current: typer.Typer, prefix: str) -> None:
        for command in current.registered_commands:
            declared = _declared_name(command.name)
            if declared is None:
                callback = command.callback
                assert callback is not None, "a Typer command has neither name nor callback"
                declared = callback.__name__.replace("_", "-")
            names.add(f"{prefix}{declared}")
        for group in current.registered_groups:
            nested = group.typer_instance
            assert nested is not None, "a Typer group has no Typer instance"
            declared = _declared_name(group.name) or _declared_name(nested.info.name)
            assert declared is not None, "a Typer group resolves to no name"
            walk(nested, f"{prefix}{declared} ")

    walk(app, "")
    return names


class TestFeatureColumnCount:
    """The figure five documents state must be the figure the code produces."""

    def test_every_documented_feature_count_matches_the_schema(self) -> None:
        from xg_alonso.features.schema import model_feature_names

        expected = len(model_feature_names())
        wrong: list[str] = []
        for path in _markdown_files():
            text = path.read_text(encoding="utf-8")
            for match in _FEATURE_COUNT.finditer(text):
                stated = int(match.group(1))
                if stated != expected:
                    label = path.relative_to(REPO_ROOT)
                    wrong.append(f"{label}: says {stated}, schema has {expected}")
        assert not wrong, "documented feature-column counts are stale:\n  " + "\n  ".join(wrong)

    def test_the_pattern_matches_the_phrasing_the_docs_use(self) -> None:
        """Guards the regex, not the docs.

        A pattern that silently stopped matching would make the test above pass
        vacuously, which is the failure mode this whole module exists to catch.
        """
        assert _FEATURE_COUNT.search("carries **224 distinct feature columns**. Quality")
        assert _FEATURE_COUNT.search("180 catalogue specs; 224 distinct columns including")
        assert _FEATURE_COUNT.search("— 224 distinct columns — which is a ceiling")


class TestCommandCount:
    """The README calls its table the discoverable surface, so it must be complete."""

    def test_the_stated_count_matches_the_table(self) -> None:
        text = README.read_text(encoding="utf-8")
        match = _COMMAND_COUNT.search(text)
        assert match is not None, (
            "README no longer states a command count in the form "
            "'<N> commands, all under the single `xg` entry point'"
        )
        stated = _parse_count(match.group(1), where="README.md")
        rows = _COMMAND_ROW.findall(text)
        assert stated == len(rows), f"README says {stated} commands but the table lists {len(rows)}"

    def test_the_table_lists_every_command_the_cli_registers(self) -> None:
        rows = set(_COMMAND_ROW.findall(README.read_text(encoding="utf-8")))
        registered = _cli_command_names()
        undocumented = sorted(registered - rows)
        invented = sorted(rows - registered)
        assert not undocumented, f"commands exist but the README table omits them: {undocumented}"
        assert not invented, f"the README table lists commands that do not exist: {invented}"

    def test_the_walker_finds_grouped_commands(self) -> None:
        """The bug was in the groups, so assert the walker descends into them."""
        registered = _cli_command_names()
        assert "models list" in registered
        assert "evaluate report" in registered


@pytest.mark.parametrize(("word", "value"), sorted(_NUMBER_WORDS.items()))
def test_number_words_are_self_consistent(word: str, value: int) -> None:
    """A typo in the table would make a real mismatch read as a passing test."""
    assert _parse_count(word.capitalize(), where="self-test") == value

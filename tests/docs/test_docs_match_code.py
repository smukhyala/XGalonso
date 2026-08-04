"""Assert that the concrete claims a document makes are still true of the code.

**Why this exists.** A documentation sweep is a one-time fix; the rot returns the
week after. Before this file, `docs/api/01_public_api.md` documented nine HTTP
routes under a `/v1` prefix, none of which had ever existed, and four CLI
commands that were never written. Nothing failed, because nothing was checking.

**What it does not do.** It does not read prose, judge accuracy, or verify that a
described behaviour matches the implementation. It checks the cheapest class of
claim — *does this name exist* — because that is the class that rots silently and
the class a reader trusts most. A route that exists but returns the wrong shape
is a job for the API tests; a route that does not exist at all is a job for this
one.

**The opt-in convention.** A document declares its checkable claims in an HTML
comment near the top, so the block is invisible in every Markdown renderer::

    <!-- claims
    package: packages/discovery
    commands: xg discover, xg models list
    routes: GET /health, POST /squad/plan
    symbols: xg_alonso.discovery.search:beam_search, xg_alonso.discovery.llm
    -->

Every key is optional and a document with no block is skipped entirely. That is
deliberate: most of `docs/` is prose and reasoning, and forcing a block onto
those files would add maintenance cost for no coverage. Only documents that name
an API surface pay anything.

**The honest maintenance cost.** Three things, and they are real:

1. A block can go stale in the *other* direction — a document may stop mentioning
   a command it still documents elsewhere in the text, and the block will not
   notice. Nothing here parses the prose to find the claims, so the block is a
   curated subset, not a derived one. It is a floor on accuracy, not a proof.
2. Renaming a command or route now fails a test in `tests/docs/` as well as the
   test that owns the behaviour. That is the point, but it is a second edit.
3. `package:` plus a `Deferred` status is the one *semantic* rule encoded here
   (see :func:`test_deferred_status_is_not_claimed_for_a_shipped_package`), and
   it needs a human to keep the mapping right.

Deriving claims automatically from the prose was considered and rejected: a
regex over Markdown would match `xg` in ordinary sentences and route-shaped
strings inside JSON examples, and the false positives would train people to
delete the test. An explicit, opt-in block is smaller and honest about being
partial.

The suite runs offline: it imports the two app modules and reads files. Nothing
here touches the network or `.data`.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Final

import pytest
import typer

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Trees scanned for documents. `.claude/worktrees/` holds full copies of the
#: repository and is deliberately excluded — checking a snapshot of an older
#: branch against today's code proves nothing.
_DOC_ROOTS: Final[tuple[Path, ...]] = (
    REPO_ROOT / "docs",
    REPO_ROOT / "apps" / "web",
)

_STANDALONE_DOCS: Final[tuple[Path, ...]] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
)

_CLAIMS_BLOCK = re.compile(r"<!--\s*claims\s*\n(.*?)-->", re.DOTALL)

#: Recognised keys. Anything else is a typo, and a typo that silently disabled a
#: check would be worse than no check at all.
_KNOWN_KEYS: Final[frozenset[str]] = frozenset({"package", "commands", "routes", "symbols"})

_STATUS_ROW = re.compile(r"^\|\s*Status\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)
_STATUS_BOLD = re.compile(r"^\*\*Status:?\*\*\s*[`]?([^`\n]+?)[`]?\s*$", re.MULTILINE)


def _markdown_files() -> list[Path]:
    found: list[Path] = []
    for root in _DOC_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            if "node_modules" in path.parts or ".next" in path.parts:
                continue
            found.append(path)
    found.extend(path for path in _STANDALONE_DOCS if path.is_file())
    return sorted(set(found))


def _label(path: Path) -> str:
    """Repo-relative where possible; the tests exercise the parser on tmp files too."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _parse_claims(path: Path) -> dict[str, list[str]]:
    """Return the declared claims, or an empty mapping when none are declared."""
    match = _CLAIMS_BLOCK.search(path.read_text(encoding="utf-8"))
    if match is None:
        return {}
    claims: dict[str, list[str]] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key not in _KNOWN_KEYS:
            raise AssertionError(
                f"{_label(path)}: unknown claims key {key!r}; expected one of {sorted(_KNOWN_KEYS)}"
            )
        claims[key] = [item.strip() for item in value.split(",") if item.strip()]
    return claims


def _declares_claims(path: Path) -> bool:
    """Collection-time predicate. A malformed block is collected, then fails loudly.

    Raising during collection would abort the whole module and hide which file
    was at fault, so the strict parse is deferred to the per-document tests.
    """
    try:
        return bool(_parse_claims(path))
    except AssertionError:
        return True


_DOCS_WITH_CLAIMS: Final[list[Path]] = [p for p in _markdown_files() if _declares_claims(p)]


def _ids(paths: list[Path]) -> list[str]:
    return [_label(p) for p in paths]


def _declared_status(text: str) -> str | None:
    for pattern in (_STATUS_ROW, _STATUS_BOLD):
        match = pattern.search(text)
        if match is not None:
            return match.group(1).strip().strip("`")
    return None


# --- the two app surfaces, introspected once ---------------------------------


def _typer_commands() -> frozenset[str]:
    from xg_alonso.cli.main import app

    names: set[str] = set()

    def walk(group: typer.Typer, prefix: str) -> None:
        for command in group.registered_commands:
            name = command.name
            if name is None:
                assert command.callback is not None
                name = command.callback.__name__.replace("_", "-")
            names.add(f"{prefix}{name}")
        for sub in group.registered_groups:
            sub_name = sub.name or (sub.typer_instance.info.name if sub.typer_instance else None)
            assert sub_name is not None, "a Typer sub-app must be named"
            assert sub.typer_instance is not None
            walk(sub.typer_instance, f"{prefix}{sub_name} ")

    walk(app, "")
    return frozenset(names)


def _api_routes() -> frozenset[str]:
    from xg_alonso.api.main import app

    routes: set[str] = set()
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or path is None:
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.add(f"{method} {path}")
    return frozenset(routes)


# --- the checks --------------------------------------------------------------


class TestTheConventionItself:
    """The block is only useful if it is being read. Prove that it is."""

    def test_some_document_declares_claims(self) -> None:
        assert _DOCS_WITH_CLAIMS, (
            "no document declares a claims block; either the convention was "
            "removed or the parser stopped matching it"
        )

    def test_an_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        doc = tmp_path / "bad.md"
        doc.write_text("<!-- claims\nroutez: GET /health\n-->\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="unknown claims key"):
            _parse_claims(doc)


@pytest.mark.parametrize("doc", _DOCS_WITH_CLAIMS, ids=_ids(_DOCS_WITH_CLAIMS))
class TestDeclaredClaims:
    def test_every_named_cli_command_exists(self, doc: Path) -> None:
        declared = _parse_claims(doc).get("commands", [])
        if not declared:
            pytest.skip("no CLI commands declared")
        available = _typer_commands()
        for command in declared:
            assert command.startswith("xg "), (
                f"{doc.name}: a command claim is written as it is invoked, "
                f"e.g. 'xg recommend'; got {command!r}"
            )
            assert command.removeprefix("xg ") in available, (
                f"{doc.name} documents `{command}`, which the Typer app does not "
                f"define. Available: {sorted(available)}"
            )

    def test_every_named_route_exists(self, doc: Path) -> None:
        declared = _parse_claims(doc).get("routes", [])
        if not declared:
            pytest.skip("no HTTP routes declared")
        available = _api_routes()
        for route in declared:
            assert route in available, (
                f"{doc.name} documents `{route}`, which the FastAPI app does not "
                f"serve. Available: {sorted(available)}"
            )

    def test_every_named_symbol_is_importable(self, doc: Path) -> None:
        declared = _parse_claims(doc).get("symbols", [])
        if not declared:
            pytest.skip("no symbols declared")
        for symbol in declared:
            module_name, _, attribute = symbol.partition(":")
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:  # pragma: no cover - the failure message is the point
                raise AssertionError(
                    f"{doc.name} names `{symbol}`, but {module_name} is not importable: {exc}"
                ) from exc
            if attribute:
                assert hasattr(module, attribute), (
                    f"{doc.name} names `{symbol}`, but {module_name} has no attribute {attribute!r}"
                )

    def test_every_named_package_exists(self, doc: Path) -> None:
        declared = _parse_claims(doc).get("package", [])
        if not declared:
            pytest.skip("no package declared")
        for relative in declared:
            package = REPO_ROOT / relative
            assert package.is_dir(), f"{doc.name} names `{relative}`, which is not a directory"

    def test_deferred_status_is_not_claimed_for_a_shipped_package(self, doc: Path) -> None:
        """The failure mode this whole sweep was called in to fix.

        Four `docs/ml/` documents carried `Status | Deferred (Post-MVP)` while
        the capability they describe was shipping in `packages/discovery`. A
        status field is the first thing a reader checks, so a wrong one is worse
        than a wrong paragraph.
        """
        declared = _parse_claims(doc).get("package", [])
        if not declared:
            pytest.skip("no package declared")
        status = _declared_status(doc.read_text(encoding="utf-8"))
        if status is None or "deferred" not in status.lower():
            return
        for relative in declared:
            sources = list((REPO_ROOT / relative).rglob("*.py"))
            assert not sources, (
                f"{doc.name} declares status {status!r} but names `{relative}`, "
                f"which holds {len(sources)} Python source files. Either the "
                f"status is stale or the document names the wrong package."
            )

"""docs/reference/cli.md is generated from the Typer app; the committed page must match."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _generator():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "gen_cli_reference", ROOT / "scripts/gen_cli_reference.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_cli_reference_matches_the_application() -> None:
    gen = _generator()
    assert (ROOT / "docs/reference/cli.md").read_text(encoding="utf-8") == gen.render(), (
        "run `uv run python scripts/gen_cli_reference.py` and commit the page"
    )


def test_every_command_and_option_is_documented() -> None:
    import typer

    from datacopilot.cli.app import app

    page = (ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")
    gen = _generator()
    for name, command in gen.commands_of(typer.main.get_command(app)).items():
        assert f"## datacopilot {name}\n" in page
        for param in command.params:
            for opt in [*param.opts, *param.secondary_opts]:
                assert opt in page or not opt.startswith("-"), (name, opt)


def test_check_mode_reports_drift(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gen = _generator()
    monkeypatch.setattr(gen, "PAGE", tmp_path / "cli.md")
    assert gen.main(["--check"]) == 1
    assert gen.main([]) == 0 and gen.main(["--check"]) == 0
    (tmp_path / "cli.md").write_text("stale")
    assert gen.main(["--check"]) == 1
    assert gen.main(["--bogus"]) == 2

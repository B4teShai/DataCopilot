"""``dc.split``: the library form of the command, same validation, nothing printed."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_split_usecase import _env, specs, store  # noqa: F401  # shared fixtures

import datacopilot as dc
from datacopilot.store import DatasetStore


def test_split_is_public_returns_the_report_and_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "split" in dc.__all__ and "SplitReport" in dc.__all__
    source = store(tmp_path, specs())
    report = dc.split(
        source, tmp_path / "s.dcp", near_duplicates=True, embedder="hash", stratify="category"
    )
    assert isinstance(report, dc.SplitReport)
    assert report.dataset_version == DatasetStore.open(tmp_path / "s.dcp").manifest().version
    assert capsys.readouterr().out == ""


def test_split_validates_its_options(tmp_path: Path) -> None:
    source = store(tmp_path, specs())
    with pytest.raises(ValueError, match="stratify"):
        dc.split(source, tmp_path / "s.dcp", stratify="size")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        dc.split(source, tmp_path / "s.dcp", seed=-1)
    with pytest.raises(ValueError, match="unknown embedder"):
        dc.split(source, tmp_path / "s.dcp", near_duplicates=True, embedder="nope")
    assert not (tmp_path / "s.dcp").exists()

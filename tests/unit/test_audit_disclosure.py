"""What the audit may read and what its findings may reveal about the file system."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fixtures.auditdata import Img, build
from fixtures.synth import png_bytes

from datacopilot.audit.checks import duplicates, integrity
from datacopilot.audit.context import CheckContext
from datacopilot.audit.paths import resolve_image_path
from datacopilot.core.schema import Dataset, Finding

RUN = "1111aaaa-2222-4333-8444-5555bbbb5555"
SAME = png_bytes(16, 16, 7)


def ctx(root: Path, cache: Path | None = None, **kw: Any) -> CheckContext:
    return CheckContext(run_id=RUN, root=root, cache_dir=cache or root / "cache", **kw)


def dump(findings: list[Finding]) -> str:
    return json.dumps([f.model_dump(mode="json") for f in findings])


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, Path]:
    """``root`` (the dataset root) and ``outside`` (a sibling directory with a private file)."""
    root = tmp_path / "root"
    outside = tmp_path / "outside-secret-home"
    root.mkdir()
    outside.mkdir()
    return root, outside


def test_symlink_inside_root_to_an_outside_file_is_neither_hashed_nor_reported(
    world: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = world
    (outside / "private.png").write_bytes(SAME)
    ds = build(root, [Img("a.png", 16, 16, data=SAME)])
    link = root / "images" / "link.png"
    link.symlink_to(outside / "private.png")
    linked = ds.images[0].model_copy(update={"image_id": "9" * 16, "file_path": str(link)})
    ds2 = Dataset(ds.manifest, (ds.images[0], linked), ds.annotations)
    hashed: list[Path] = []
    real = duplicates.hash_file
    monkeypatch.setattr(
        duplicates, "hash_file", lambda p, m, **kw: (hashed.append(p), real(p, m, **kw))[1]
    )
    findings = duplicates.plugin.run(ds2, ctx(root, use_hash_cache=False))
    assert (
        outside / "private.png" not in hashed and (outside / "private.png").resolve() not in hashed
    )
    assert [f.payload["rule"] for f in findings] == ["images_not_hashed"]  # no duplicate group
    assert findings[0].payload["reasons"] == {"outside_root": 1}
    text = dump(findings)
    assert "outside-secret-home" not in text and "private" not in text


def test_integrity_does_not_confirm_files_outside_the_root(world: tuple[Path, Path]) -> None:
    root, outside = world
    (outside / "x.png").write_bytes(png_bytes(8, 8, 1))
    ds = build(
        root,
        [
            Img("exists.png", write=False, path=str(outside / "x.png")),
            Img("nope.png", seed=2, write=False, path=str(outside / "absent.png")),
        ],
    )
    findings = integrity.plugin.run(ds, ctx(root))
    # existing and missing outside files are indistinguishable: both only counted
    assert [f.payload["rule"] for f in findings] == ["external_not_checked"]
    assert findings[0].payload["count"] == 2 and "outside-secret-home" not in dump(findings)


def test_external_images_can_be_allowed_explicitly(world: tuple[Path, Path]) -> None:
    root, outside = world
    (outside / "x.png").write_bytes(png_bytes(8, 8, 1))
    ds = build(root, [Img("x.png", 8, 8, write=False, path=str(outside / "x.png"))])
    findings = integrity.plugin.run(ds, ctx(root, allow_external_images=True))
    assert findings == []


def test_findings_never_contain_absolute_paths_or_directory_names(
    world: tuple[Path, Path],
) -> None:
    root, _ = world
    nested = Img("deep/er/a.png", 16, 16, data=SAME)
    twin = Img("deep/er/b.png", 16, 16, "test", data=SAME, image_id="1" * 16)
    gone = Img("gone.png", seed=3, write=False)
    ds = build(root, [nested, twin, gone])
    c = ctx(root, use_hash_cache=False)
    text = dump(integrity.plugin.run(ds, c) + duplicates.plugin.run(ds, c))
    assert str(root) not in text and "/root/" not in text and "deep" not in text
    assert "a.png" in text


def test_duplicate_findings_carry_a_12_hex_prefix_never_the_full_digest(
    world: tuple[Path, Path],
) -> None:
    root, _ = world
    ds = build(
        root,
        [Img("a.png", 16, 16, data=SAME), Img("b.png", 16, 16, data=SAME, image_id="1" * 16)],
    )
    findings = duplicates.plugin.run(ds, ctx(root, use_hash_cache=False))
    full = hashlib.sha256(SAME).hexdigest()
    text = dump(findings)
    (group,) = [f for f in findings if f.payload["rule"] == "exact_duplicate"]
    assert group.payload["group_id"] == full[:12] and "sha256" not in group.payload
    assert full not in text and full[:17] not in text  # image_id is the 16-hex dataset id
    (stale,) = [f for f in findings if f.payload["rule"] == "content_changed"]
    assert len(str(stale.payload["actual"])) == 12


@pytest.mark.parametrize(
    "bad",
    [
        "C:\\Users\\me\\a.png",
        "..\\..\\a.png",
        "\\\\server\\share\\a.png",
        "../a.png",
        "a/../../a.png",
    ],
)
def test_windows_style_and_traversal_paths_are_rejected_without_being_opened(
    world: tuple[Path, Path], bad: str
) -> None:
    root, _ = world
    assert resolve_image_path(bad, root, allow_external=True).kind == "unsafe"
    ds = build(root, [Img("a.png", write=False, path=bad)])
    (finding,) = integrity.plugin.run(ds, ctx(root))
    assert finding.payload["rule"] == "unsafe_path"
    (skip,) = duplicates.plugin.run(ds, ctx(root, use_hash_cache=False))
    assert skip.payload["reasons"] == {"unsafe_path": 1}


def test_absolute_path_with_dotdot_that_escapes_is_outside_root(world: tuple[Path, Path]) -> None:
    root, outside = world
    sneaky = f"{root}/images/../../{outside.name}/x.png"
    assert resolve_image_path(sneaky, root, allow_external=False).kind == "external"


def test_resolved_path_is_a_real_path_inside_the_root(world: tuple[Path, Path]) -> None:
    root, _ = world
    (root / "real").mkdir()
    (root / "real" / "a.png").write_bytes(SAME)
    (root / "alias").symlink_to(root / "real")
    resolved = resolve_image_path(str(root / "alias" / "a.png"), root, allow_external=False)
    assert resolved.kind == "local" and resolved.path == (root / "real" / "a.png").resolve()


PRESIGNED = (
    "https://bucket.s3.amazonaws.com/private/team/photo.jpg"
    "?X-Amz-Credential=AKIAEXAMPLEKEY%2F20260101&X-Amz-Signature=deadbeef01&X-Amz-Expires=300#frag"
)


@pytest.mark.parametrize(
    ("path", "shown"),
    [
        (PRESIGNED, "photo.jpg"),
        ("s3://bucket/dir/a.png?versionId=SECRET", "a.png"),
        ("images/dir/b.jpg", "b.jpg"),
        ("C:\\data\\c.jpg", "c.jpg"),
        ("local/odd?name#1.jpg", "odd?name#1.jpg"),  # only remote paths carry queries
    ],
)
def test_display_name_never_shows_a_query_string_or_fragment_of_a_remote_path(
    path: str, shown: str
) -> None:
    from datacopilot.audit.paths import display_name

    assert display_name(path) == shown


def test_a_presigned_url_does_not_reach_any_report_format(tmp_path: Path) -> None:
    from datacopilot.audit import engine
    from datacopilot.audit.findings import make_finding
    from datacopilot.audit.paths import display_name
    from datacopilot.report import render

    ds = build(tmp_path, [Img("a.png", seed=1), Img("b.png", seed=2, path=PRESIGNED)])
    run = engine.run_audit(ds, ["integrity"], ctx(tmp_path))
    remote = next(i for i in ds.images if i.file_path == PRESIGNED)
    # a check that names the image (the embedding checks do) gets the cleaned name
    named = make_finding(
        RUN,
        "integrity",
        "INFO",
        image_id=remote.image_id,
        payload={"rule": "exif_orientation", "file_name": display_name(remote.file_path)},
    )
    run = run.model_copy(update={"findings": (*run.findings, named)})
    render(run, html=tmp_path / "r.html", json=tmp_path / "r.json", sarif=tmp_path / "r.sarif")
    for name in ("r.html", "r.json", "r.sarif"):
        text = (tmp_path / name).read_text()
        assert "photo.jpg" in text or name == "r.sarif", name
        assert "X-Amz" not in text and "AKIA" not in text and "deadbeef" not in text, name
        assert "private/team" not in text and "amazonaws" not in text, name

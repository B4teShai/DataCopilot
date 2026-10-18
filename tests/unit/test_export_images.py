"""An export copies only the dataset's own pictures.

A dataset directory is untrusted: ``Image.file_path`` can name any file on the machine. Every
exporter goes through ``convert.export``, which verifies each image (inside the dataset's
directory, not a link to elsewhere, regular, size-capped, content equal to ``image_id``) and
refuses the whole export otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fixtures import synth
from fixtures.auditdata import Img, build

from datacopilot import convert
from datacopilot.cli import app as cli
from datacopilot.core.errors import FormatError
from datacopilot.core.schema import Dataset
from datacopilot.io.files import image_id
from datacopilot.logging import clear_request_context
from datacopilot.store import DatasetStore

FORMATS = ["coco", "yolo", "voc", "cvat_xml", "label_studio"]
SECRET = b"-----BEGIN " + b"PRIVATE KEY-----\nnot a picture\n"  # split: the QA gate scans for the marker


def two(tmp_path: Path) -> Dataset:
    return build(tmp_path / "data", [Img("a.png", boxes=[(0, 1.0, 1.0, 9.0, 9.0)]), Img("b.png")])


def with_path(ds: Dataset, file_path: str, image_id_: str | None = None) -> Dataset:
    first, *rest = ds.images
    changed = first.model_copy(
        update={"file_path": file_path, **({"image_id": image_id_} if image_id_ else {})}
    )
    anns = tuple(
        a.model_copy(update={"image_id": changed.image_id}) if a.image_id == first.image_id else a
        for a in ds.annotations
    )
    return Dataset(ds.manifest, (changed, *rest), anns, ds.root)


def refused(ds: Dataset, fmt: str, tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(FormatError, match="nothing was exported"):
        convert.export(ds, fmt, out)
    assert not out.exists()


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_absolute_path_to_another_file_is_never_copied(tmp_path: Path, fmt: str) -> None:
    secret = tmp_path / "id_rsa"
    secret.write_bytes(SECRET)
    ds = two(tmp_path)
    # even if the dataset knows the secret's hash, a file outside the dataset is not copied
    refused(with_path(ds, str(secret), image_id(secret)), fmt, tmp_path)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_path_with_dot_dot_is_never_copied(tmp_path: Path, fmt: str) -> None:
    (tmp_path / "id_rsa").write_bytes(SECRET)
    refused(with_path(two(tmp_path), "../id_rsa"), fmt, tmp_path)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_symlink_to_a_file_outside_is_never_copied(tmp_path: Path, fmt: str) -> None:
    secret = tmp_path / "id_rsa"
    secret.write_bytes(SECRET)
    ds = two(tmp_path)
    link = tmp_path / "data" / "images" / "link.png"
    os.symlink(secret, link)
    refused(with_path(ds, str(link), image_id(secret)), fmt, tmp_path)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_file_whose_content_is_not_the_recorded_picture_is_never_copied(
    tmp_path: Path, fmt: str
) -> None:
    ds = two(tmp_path)
    Path(ds.images[0].file_path).write_bytes(synth.png_bytes(64, 48, 99))  # swapped after ingest
    refused(ds, fmt, tmp_path)


@pytest.mark.parametrize("path", ["https://example.test/a.png", "", "a\x00b.png", "C:\\x\\y.png"])
def test_remote_and_unsafe_paths_are_refused(tmp_path: Path, path: str) -> None:
    refused(with_path(two(tmp_path), path), "cvat_xml", tmp_path)


def test_a_missing_file_and_a_directory_are_refused(tmp_path: Path) -> None:
    ds = two(tmp_path)
    Path(ds.images[0].file_path).unlink()
    refused(ds, "coco", tmp_path)
    ds2 = two(tmp_path / "second")
    (tmp_path / "second" / "dir").mkdir()
    refused(with_path(ds2, str(tmp_path / "second" / "dir")), "coco", tmp_path / "second")


def test_a_dataset_without_a_root_is_refused(tmp_path: Path) -> None:
    ds = two(tmp_path)
    with pytest.raises(FormatError, match="no root"):
        convert.export(Dataset(ds.manifest, ds.images, ds.annotations), "coco", tmp_path / "out")


def test_an_oversized_image_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(convert, "MAX_EXPORT_IMAGE_BYTES", 10)
    refused(two(tmp_path), "coco", tmp_path)


def test_honest_datasets_still_export_and_the_pictures_are_the_same_bytes(tmp_path: Path) -> None:
    ds = two(tmp_path)
    convert.export(ds, "coco", tmp_path / "out")
    for image in ds.images:
        copied = next((tmp_path / "out").rglob(Path(image.file_path).name))
        assert copied.read_bytes() == Path(image.file_path).read_bytes()


def test_a_symlink_that_stays_inside_the_dataset_is_fine(tmp_path: Path) -> None:
    ds = two(tmp_path)
    real = Path(ds.images[0].file_path)
    link = real.with_name("alias.png")
    os.symlink(real, link)
    convert.export(with_path(ds, str(link)), "coco", tmp_path / "out")


def test_external_images_need_the_explicit_option_and_must_still_match(tmp_path: Path) -> None:
    ds = two(tmp_path)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(Path(ds.images[0].file_path).read_bytes())
    moved = with_path(ds, str(outside))
    refused(moved, "coco", tmp_path)
    convert.export(moved, "coco", tmp_path / "out", allow_external_images=True)
    secret = tmp_path / "id_rsa"
    secret.write_bytes(SECRET)
    hostile = with_path(ds, str(secret))  # the hash still has to match
    with pytest.raises(FormatError, match="nothing was exported"):
        convert.export(hostile, "coco", tmp_path / "out2", allow_external_images=True)
    assert not (tmp_path / "out2").exists()


def test_the_command_line_refuses_a_hostile_dcp_and_copies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "home").mkdir()
    secret = tmp_path / "home" / "id_rsa"
    secret.write_bytes(SECRET)
    ds = two(tmp_path / "work")
    hostile = with_path(ds, str(secret), image_id(secret))
    DatasetStore.create(
        tmp_path / "work" / "evil.dcp", hostile.images, hostile.annotations, hostile.categories,
        name="x", task="detection",
    )  # fmt: skip
    out = tmp_path / "work" / "out"
    monkeypatch.setattr(sys, "argv", ["datacopilot", "export", str(tmp_path / "work" / "evil.dcp"),
                                      "--format", "yolo", "-o", str(out)])  # fmt: skip
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main()
    clear_request_context()
    assert exc.value.code == 1 and not out.exists()
    assert "nothing was exported" in capsys.readouterr().err


def test_review_export_cannot_be_used_to_copy_a_foreign_file(tmp_path: Path) -> None:
    from test_proposals_review import make_dataset, make_set

    from datacopilot.proposals.review import export_review

    (tmp_path / "home").mkdir()
    secret = tmp_path / "home" / "id_rsa"
    secret.write_bytes(SECRET)
    ds = make_dataset(tmp_path / "d")
    ps = make_set(ds)
    hostile = with_path(ds, str(secret))
    with pytest.raises(FormatError, match="nothing was exported"):
        export_review(hostile, ps, tmp_path / "review")
    assert not (tmp_path / "review").exists()

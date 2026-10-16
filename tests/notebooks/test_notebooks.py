"""Static checks of the executed notebooks (no network, no model, no notebook library needed).

The notebooks are run for real by hand (see ``notebooks/README.md``); these tests make sure the
saved result is still one a reader can trust: executed to the end, no error, no folder of the
machine that ran it, no outsized pictures, and an edge-case table without a mismatch.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
NAMES = ("01_tutorial.ipynb", "02_edge_cases.ipynb")
MAX_BYTES = 1_500_000
MAX_PNG = 8
MAX_TEXT = 20_000
ALLOWED_OUTPUT_TYPES = {"stream", "display_data", "execute_result"}
PRIVATE_PATHS = re.compile(r"/Users/|/home/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\+Users")
SECRETS = re.compile(r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\bghp_[A-Za-z0-9]{20,}")
FORBIDDEN_WORDS = re.compile(r"claude|anthropic", re.IGNORECASE)


def load(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return document


def source(cell: dict[str, Any]) -> str:
    text = cell["source"]
    return text if isinstance(text, str) else "".join(text)


def text_of(value: Any) -> str:
    return value if isinstance(value, str) else "".join(value)


def output_texts(cell: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for out in cell.get("outputs", []):
        if out["output_type"] == "stream":
            found.append(text_of(out["text"]))
        else:
            found += [text_of(v) for k, v in out.get("data", {}).items() if k.startswith("text/")]
    return found


@pytest.fixture(params=NAMES)
def name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_the_notebooks_exist_and_nothing_else_is_here() -> None:
    present = sorted(p.name for p in NOTEBOOKS.iterdir() if p.is_file())
    assert present == sorted([*NAMES, "README.md"])


def test_readme_lists_every_notebook_and_stays_short() -> None:
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    for notebook in NAMES:
        assert notebook in readme
    assert len(readme.splitlines()) <= 10
    assert "DATACOPILOT_NOTEBOOK_DATA" in readme and "CC BY 4.0" in readme


def test_is_a_valid_version_4_notebook(name: str) -> None:
    doc = load(name)
    assert doc["nbformat"] == 4 and isinstance(doc["nbformat_minor"], int)
    assert isinstance(doc["metadata"], dict) and doc["cells"]
    for cell in doc["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], (str, list))
        assert ("outputs" in cell) == (cell["cell_type"] == "code")


def test_nbformat_accepts_it_when_the_library_is_installed(name: str) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOKS / name, as_version=4))


def test_every_code_cell_was_executed_and_saved_an_output(name: str) -> None:
    code = [c for c in load(name)["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 10
    for cell in code:
        where = source(cell).splitlines()[0]
        assert isinstance(cell["execution_count"], int), where
        assert cell["outputs"], where
    counts = [c["execution_count"] for c in code]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)  # run once, top to bottom


def test_no_cell_ended_in_an_error(name: str) -> None:
    for cell in load(name)["cells"]:
        for out in cell.get("outputs", []):
            assert out["output_type"] in ALLOWED_OUTPUT_TYPES, source(cell).splitlines()[0]
            assert out["output_type"] != "error"
            assert not (out["output_type"] == "stream" and "Traceback" in text_of(out["text"]))


def test_no_folder_of_the_machine_that_ran_it_is_saved(name: str) -> None:
    text = (NOTEBOOKS / name).read_text(encoding="utf-8")
    assert PRIVATE_PATHS.findall(text) == []
    assert SECRETS.findall(text) == []


def test_no_tool_or_vendor_name_is_mentioned(name: str) -> None:
    assert FORBIDDEN_WORDS.findall((NOTEBOOKS / name).read_text(encoding="utf-8")) == []
    assert FORBIDDEN_WORDS.findall((NOTEBOOKS / "README.md").read_text(encoding="utf-8")) == []


def test_size_pictures_and_text_stay_small(name: str) -> None:
    path = NOTEBOOKS / name
    assert path.stat().st_size < MAX_BYTES
    doc = load(name)
    pictures = sum(
        "image/png" in out.get("data", {}) for c in doc["cells"] for out in c.get("outputs", [])
    )
    assert pictures <= MAX_PNG
    for cell in doc["cells"]:
        for text in output_texts(cell):
            assert len(text) <= MAX_TEXT, source(cell).splitlines()[0]


def test_the_tutorial_explains_every_code_cell() -> None:
    cells = load("01_tutorial.ipynb")["cells"]
    for index, cell in enumerate(cells[:-1]):
        if cell["cell_type"] == "code":
            follower = cells[index + 1]
            assert follower["cell_type"] == "markdown", source(cell).splitlines()[0]
            assert "What this shows" in source(follower), source(cell).splitlines()[0]
    assert cells[-1]["cell_type"] == "markdown"  # the closing table


def test_the_tutorial_covers_every_step_and_the_licence_note() -> None:
    text = " ".join(source(c) for c in load("01_tutorial.ipynb")["cells"])
    for needle in (
        "doctor",
        "--fail-on",
        "warn",
        "critical",
        "dinov2_small",
        "leakage",
        "--strict",
        "dc.load",
        "dc.audit",
        "dc.export",
        "What you learned",
        "CC BY 4.0",
        "Flickr",
        "label_studio",
        "cvat_xml",
    ):
        assert needle in text, needle


def test_the_edge_case_notebook_has_one_headline_and_one_summary_per_case() -> None:
    cells = load("02_edge_cases.ipynb")["cells"]
    headlines = [source(c) for c in cells if source(c).lstrip().startswith("## Case")]
    assert len(headlines) >= 19
    assert all("What could go wrong" in h for h in headlines if h.startswith("## Case"))


def test_the_edge_case_table_has_no_mismatch() -> None:
    cells = load("02_edge_cases.ipynb")["cells"]
    tables = [
        text_of(out["data"]["text/markdown"])
        for c in cells
        for out in c.get("outputs", [])
        if "text/markdown" in out.get("data", {})
    ]
    assert len(tables) == 1
    rows = [line for line in tables[0].splitlines() if line.startswith("| ")][1:]
    assert len(rows) >= 25
    assert all(row.rstrip().endswith("| ok |") for row in rows)
    everything = " ".join(t for c in cells for t in output_texts(c))
    assert "MISMATCH" not in everything.replace("`MISMATCH`", "")
    assert re.search(r"\b(\d+) cases, \1 as expected, 0 mismatches", everything)

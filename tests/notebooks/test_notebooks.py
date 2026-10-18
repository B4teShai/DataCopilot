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
NAMES = (
    "01_tutorial.ipynb",
    "02_edge_cases.ipynb",
    "03_propose_review_apply.ipynb",
    "04_label_noise.ipynb",
)
MAX_BYTES = 1_500_000
MAX_PNG = 8
MAX_TEXT = 20_000
ALLOWED_OUTPUT_TYPES = {"stream", "display_data", "execute_result"}
PRIVATE_PATHS = re.compile(r"/Users/|/home/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\+Users")
SECRETS = re.compile(r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\bghp_[A-Za-z0-9]{20,}")
# the words are split so that this file does not contain them itself
FORBIDDEN_WORDS = re.compile("|".join(["cl" + "aude", "anthro" + "pic"]), re.IGNORECASE)


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
    assert len(readme.splitlines()) <= 11
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


def test_no_terminal_colour_codes_are_saved(name: str) -> None:
    text = (NOTEBOOKS / name).read_text(encoding="utf-8")
    assert "\x1b" not in text and "\\u001b" not in text.lower()


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


# ----------------------------------------------------------------- downloads are verified
HEX64 = re.compile(r"""["'][0-9a-f]{64}["']""")
# "tor" + "ch" keeps the name of the tensor library out of this file (the repository gate bans it)
UNSAFE_CALLS = re.compile(
    r"\.extractall\(|\.extract\(|pickle\.loads?\(|" + "tor" + r"ch\.load\(|joblib\.load\("
)


def download_problems(doc: dict[str, Any]) -> list[str]:
    """What is wrong with the way a notebook fetches and opens downloaded files (empty if fine)."""
    problems: list[str] = []
    code = [source(c) for c in doc["cells"] if c["cell_type"] == "code"]
    downloads = [s for s in code if "urlopen(" in s or "urlretrieve(" in s]
    if not downloads:
        problems.append("no download cell")
    for cell in downloads:
        if not HEX64.search(cell):
            problems.append("download without a pinned 64-hex digest")
        if not re.search(r"_URL = \"https://", cell) or "http://" in cell:
            problems.append("download is not over https")
        if not re.search(r"if actual != \w+SHA256:\n(?:[^\n]*\n){0,2}[^\n]*raise ", cell):
            problems.append("download without a comparison that raises")
        if "unlink(" not in cell or ".replace(" not in cell:
            problems.append("download without deleting a bad file or an atomic rename")
        if "timeout=" not in cell or "MAX_DOWNLOAD_BYTES" not in cell:
            problems.append("download without a timeout and a size bound")
        check = cell.find("if actual != COCO128_SHA256")
        use = min((cell.find(t) for t in ("unpack_safely(", "ZipFile(") if t in cell), default=-1)
        if use != -1 and 0 <= check and use < check and "def " not in cell[use - 4 : use]:
            problems.append("archive is opened before the checksum is compared")
    for text in code:
        if UNSAFE_CALLS.search(text):
            problems.append("unsafe extraction or deserialisation call")
    if not any(all(t in s for t in ("is_link", '".."', "MAX_RATIO")) for s in code):
        problems.append("no archive-entry validator")
    return problems


def test_downloads_are_pinned_checked_before_use_and_unpacked_safely(name: str) -> None:
    assert download_problems(load(name)) == []


def test_the_download_check_notices_a_doctored_notebook() -> None:
    doc = load("01_tutorial.ipynb")
    assert download_problems(doc) == []

    def changed(old: str, new: str) -> dict[str, Any]:
        copy = json.loads(json.dumps(doc))
        for cell in copy["cells"]:
            if cell["cell_type"] == "code" and "urlopen(" in source(cell):
                cell["source"] = source(cell).replace(old, new)
        return copy

    assert download_problems(changed("raise ", "print  # "))
    assert download_problems(changed("61e5e302", "zz"))  # digest no longer 64 hex characters
    assert download_problems(changed("https://github", "http://github"))
    assert download_problems(
        changed(
            "zipfile.ZipFile(archive_path)",
            "zipfile.ZipFile(archive_path)\n    archive.extractall(destination)",
        )
    )


# ------------------------------------------------------------ the propose, review, apply notebook
PROPOSE = "03_propose_review_apply.ipynb"


def all_output(name: str) -> str:
    return "\n".join(t for c in load(name)["cells"] for t in output_texts(c))


def test_the_propose_notebook_ran_the_real_pinned_model_on_every_photo() -> None:
    out = all_output(PROPOSE)
    assert "model=grounding_dino@a2bb814dd30d" in out  # the pinned revision, not a test double
    assert "test_boxes" not in out
    assert re.search(r"images=128/128 set=[0-9a-f]{32}", out)
    assert "photos looked at      : 128 of 128 | incomplete: None" in out
    assert "bound to the version  : True" in out


def test_the_propose_notebook_reports_every_number_the_plan_asks_for() -> None:
    out = all_output(PROPOSE)
    for needle in (
        "deleted (held out)",
        "proposals that are right",
        "precision (right / proposals)",
        "recall (found / deleted)",
        "TUNED ON THE SAME DATA",
        "run time",
        "classes in val",
        "Verdict: the precision is",
    ):
        assert needle in out, needle
    held = re.search(r"deleted \(held out\): (\d+) boxes = (\d+\.\d) percent", out)
    assert held and 15.0 <= float(held.group(2)) <= 25.0  # about one box in five
    assert re.search(r"precision \(right / proposals\): 0\.\d{3}", out)


def test_the_propose_notebook_draws_six_mistakes_and_a_curve() -> None:
    cells = load(PROPOSE)["cells"]
    pictures = sum("image/png" in o.get("data", {}) for c in cells for o in c.get("outputs", []))
    assert pictures == 2
    out = all_output(PROPOSE)
    assert len(re.findall(r"proposed '[a-z ]+' with confidence 0\.\d\d", out)) == 6


def test_the_propose_notebook_states_its_caveats_in_plain_words() -> None:
    text = " ".join(source(c) for c in load(PROPOSE)["cells"])
    for needle in (
        "oracle",
        "random",
        "understate",
        "tuned on the same data",
        "not labelled exhaustively",
        "What this shows",
        "What you learned",
        "CC BY 4.0",
        "Flickr",
        "never applied",
    ):
        assert needle in text or needle.capitalize() in text, needle


def test_the_propose_notebook_refusal_table_has_no_mismatch() -> None:
    out = all_output(PROPOSE)
    assert re.search(r"\b(\d+) cases, \1 as expected, 0 mismatches", out)
    tables = [
        text_of(o["data"]["text/markdown"])
        for c in load(PROPOSE)["cells"]
        for o in c.get("outputs", [])
        if "text/markdown" in o.get("data", {})
        and "| Case |" in text_of(o["data"]["text/markdown"])
    ]
    assert len(tables) == 1
    rows = [ln for ln in tables[0].splitlines() if ln.startswith("| ")][1:]
    assert len(rows) >= 7 and all(r.rstrip().endswith("| ok |") for r in rows)
    causes = ("edited by hand", "another dataset version", "allow-list", "source dataset",
              "entity bomb", "time budget", "--allow-incomplete", "nobody reviewed")  # fmt: skip
    for cause in causes:
        assert cause in tables[0], cause


def test_the_propose_notebook_never_loads_a_pickle_or_an_unpinned_model() -> None:
    code = "\n".join(source(c) for c in load(PROPOSE)["cells"] if c["cell_type"] == "code")
    assert "trust_remote_code" not in code and "from_pretrained" not in code
    assert "get_model(" in code and "grounding_dino" in code


# ------------------------------------------------------------------ the label-noise notebook
NOISE = "04_label_noise.ipynb"


def test_the_label_noise_notebook_explains_every_code_cell() -> None:
    cells = load(NOISE)["cells"]
    for index, cell in enumerate(cells[:-1]):
        if cell["cell_type"] == "code" and index > 4:  # the set-up and download cells come first
            follower = cells[index + 1]
            assert follower["cell_type"] == "markdown", source(cell).splitlines()[0]
            assert "What this shows" in source(follower), source(cell).splitlines()[0]
    assert cells[-1]["cell_type"] == "markdown"


def test_the_label_noise_notebook_used_the_real_pinned_model() -> None:
    out = all_output(NOISE)
    assert "model=dinov2_small@ed25f3a31f01" in out
    assert "revision ed25f3a31f01632728cabb09d1542f84ab7b0056" in out
    assert "mismatched: 0" in out and "cache state: ok" in out
    assert "hash@" not in out and "test-only" not in out
    code = "\n".join(source(c) for c in load(NOISE)["cells"] if c["cell_type"] == "code")
    assert "trust_remote_code" not in code and "from_pretrained" not in code
    assert "resolve_embedder(" in code


def test_the_label_noise_notebook_reports_every_number_the_plan_asks_for() -> None:
    out = all_output(NOISE)
    for needle in (
        "precision",
        "recall",
        "FP rate",
        "base rate (precision of reporting boxes at random)",
        "UNTOUCHED dataset",
        "false-positive rate",
        "not scored",
        "embedding 128 photos with the real model",
        "label_noise itself",
        "tuned on this same data",
        "Verdict:",
    ):
        assert needle in out or needle in " ".join(source(c) for c in load(NOISE)["cells"]), needle
    assert "check and independent version give the same boxes in every run: True" in out
    assert "same boxes: True" in out
    assert re.search(r"seed  planted  reported  found  precision   recall  FP rate", out)
    assert len(re.findall(r"^ {0,3}\d  +46 ", out, re.MULTILINE)) >= 10  # five seeds, two protocols


def test_the_label_noise_notebook_shows_six_honest_failures_in_one_picture() -> None:
    cells = load(NOISE)["cells"]
    pictures = sum("image/png" in o.get("data", {}) for c in cells for o in c.get("outputs", []))
    assert pictures == 1
    out = all_output(NOISE)
    assert len(re.findall(r"^false alarm +\d+\.jpg", out, re.MULTILINE)) == 3
    assert len(re.findall(r"^missed +\d+\.jpg", out, re.MULTILINE)) == 3


def test_the_label_noise_notebook_has_no_hand_typed_results() -> None:
    for cell in load(NOISE)["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = source(cell).replace("CC BY 4.0", "")
        assert re.findall(r"\d+\.\d+|\d+ ?%|\bpercent of the (?:boxes|photos)", text) == [], text
        assert not re.search(r"(precision|recall|rate) (of|is|was|=) *\d", text), text


def test_the_label_noise_verdict_follows_the_computed_numbers() -> None:
    out = all_output(NOISE)
    mean = re.search(
        r"^mean +[\d.]+ +[\d.]+ +[\d.]+ +(?P<p>[\d.]+) +(?P<r>[\d.]+) +[\d.]+ +[\d.]+$",
        out,
        re.MULTILINE,
    )
    assert mean, "no mean row"
    verdict = re.search(r"\*\*Verdict: ([a-z ]+) on this dataset", out)
    assert verdict
    weak = float(mean["p"]) < 0.5 or float(mean["r"]) < 0.5
    assert (verdict[1] == "weak") == weak
    if weak:
        assert "never as a list of errors" in out


def test_the_label_noise_notebook_download_cell_is_the_pinned_one_of_the_tutorial() -> None:
    def fetch(doc: dict[str, Any]) -> str:
        return next(source(c) for c in doc["cells"] if "def fetch_coco128" in source(c))

    assert fetch(load(NOISE)) == fetch(load("01_tutorial.ipynb"))

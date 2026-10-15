import warnings

import pytest

from datacopilot.deprecation import deprecated


@deprecated(since="0.2", removal="0.3", replacement="datacopilot.load")
def old_load(path: str, *, flag: bool = False) -> tuple[str, bool]:
    """Old docs."""
    return path, flag


def test_the_old_name_keeps_working_and_warns_naming_the_replacement() -> None:
    with pytest.warns(DeprecationWarning) as caught:
        assert old_load("x", flag=True) == ("x", True)
    message = str(caught[0].message)
    assert "old_load" in message and "since 0.2" in message and "removed in 0.3" in message
    assert "use datacopilot.load instead" in message


def test_the_warning_points_at_the_caller_not_at_the_helper() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_load("x")
    assert caught[0].filename == __file__


def test_metadata_of_the_wrapped_function_is_kept() -> None:
    assert old_load.__name__ == "old_load" and old_load.__doc__ == "Old docs."


def test_it_warns_on_every_call() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_load("a")
        old_load("b")
    assert len(caught) == 2


@pytest.mark.parametrize(
    ("since", "removal"),
    [("0.2", "0.2"), ("0.3", "0.2"), ("1.0", "0.9.9"), ("x", "0.3"), ("0.2", "v0.3"), ("", "0.3")],
)
def test_invalid_versions_are_refused_at_definition_time(since: str, removal: str) -> None:
    with pytest.raises(ValueError):
        deprecated(since=since, removal=removal, replacement="x")


def test_a_replacement_must_be_named() -> None:
    with pytest.raises(ValueError, match="replacement"):
        deprecated(since="0.2", removal="0.3", replacement="  ")


def test_a_patch_release_can_remove_a_name_deprecated_in_the_previous_minor() -> None:
    deprecated(since="0.2.1", removal="0.3.0", replacement="x")

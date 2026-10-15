import pytest

from datacopilot.settings import Settings, SettingsError, load_settings


def test_defaults() -> None:
    s = load_settings({})
    assert (s.env, s.log_level, s.effective_log_level) == ("prod", "INFO", "INFO")


def test_reads_and_normalises_case() -> None:
    s = load_settings({"DATACOPILOT_ENV": " DEV ", "DATACOPILOT_LOG_LEVEL": "debug"})
    assert (s.env, s.log_level) == ("dev", "DEBUG")


def test_debug_is_capped_to_info_outside_dev() -> None:
    assert Settings(env="prod", log_level="DEBUG").effective_log_level == "INFO"
    assert Settings(env="staging", log_level="DEBUG").effective_log_level == "INFO"
    assert Settings(env="dev", log_level="DEBUG").effective_log_level == "DEBUG"


@pytest.mark.parametrize(
    ("var", "value"),
    [("DATACOPILOT_ENV", "banana"), ("DATACOPILOT_LOG_LEVEL", "LOUD"), ("DATACOPILOT_ENV", "")],
)
def test_invalid_value_names_the_variable(var: str, value: str) -> None:
    with pytest.raises(SettingsError, match=var):
        load_settings({var: value})


def test_table_limits_have_defaults_and_can_be_overridden() -> None:
    s = load_settings({})
    assert s.max_table_rows == 5_000_000 and s.max_table_bytes == 4 * 1024**3
    s = load_settings({"DATACOPILOT_MAX_TABLE_ROWS": " 10 ", "DATACOPILOT_MAX_TABLE_BYTES": "2048"})
    assert (s.max_table_rows, s.max_table_bytes) == (10, 2048)


@pytest.mark.parametrize("var", ["DATACOPILOT_MAX_TABLE_ROWS", "DATACOPILOT_MAX_TABLE_BYTES"])
@pytest.mark.parametrize("value", ["0", "-5", "lots", "1.5", ""])
def test_invalid_table_limit_names_the_variable(var: str, value: str) -> None:
    with pytest.raises(SettingsError, match=var):
        load_settings({var: value})


def test_audit_settings_defaults() -> None:
    s = load_settings({"HOME": "/home/u"})
    assert s.workers >= 1 and s.hash_cache is True
    assert str(s.cache_dir) == "/home/u/.cache/datacopilot"


def test_cache_dir_follows_xdg_then_the_explicit_variable() -> None:
    assert str(load_settings({"XDG_CACHE_HOME": "/x", "HOME": "/h"}).cache_dir) == "/x/datacopilot"
    explicit = load_settings({"DATACOPILOT_CACHE_DIR": "/c", "XDG_CACHE_HOME": "/x"})
    assert str(explicit.cache_dir) == "/c"


def test_audit_settings_can_be_overridden() -> None:
    s = load_settings(
        {
            "DATACOPILOT_WORKERS": " 3 ",
            "DATACOPILOT_HASH_CACHE": "FALSE",
            "DATACOPILOT_CACHE_DIR": "/c",
        }
    )
    assert (s.workers, s.hash_cache) == (3, False)


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("DATACOPILOT_WORKERS", "0"),
        ("DATACOPILOT_WORKERS", "257"),
        ("DATACOPILOT_WORKERS", "many"),
        ("DATACOPILOT_WORKERS", ""),
        ("DATACOPILOT_HASH_CACHE", "maybe"),
        ("DATACOPILOT_HASH_CACHE", ""),
        ("DATACOPILOT_CACHE_DIR", ""),
        ("DATACOPILOT_CACHE_DIR", "a\x00b"),
    ],
)
def test_invalid_audit_setting_names_the_variable(var: str, value: str) -> None:
    with pytest.raises(SettingsError, match=var):
        load_settings({var: value})


def test_cache_key_and_ci_settings() -> None:
    s = load_settings({})
    assert s.cache_key is None and s.ci is False
    s = load_settings({"DATACOPILOT_CACHE_KEY": " " + "k" * 20, "CI": "true"})
    assert s.cache_key is not None and s.cache_key.get_secret_value() == "k" * 20 and s.ci is True
    assert "k" * 20 not in repr(s) and "k" * 20 not in s.model_dump_json()
    assert load_settings({"CI": "0"}).ci is False and load_settings({"CI": ""}).ci is False
    assert load_settings({"CI": "1"}).ci is True


@pytest.mark.parametrize("value", ["short", "", "a" * 5000])
def test_invalid_cache_key_names_the_variable_not_the_value(value: str) -> None:
    with pytest.raises(SettingsError) as info:
        load_settings({"DATACOPILOT_CACHE_KEY": value})
    assert (
        "DATACOPILOT_CACHE_KEY" in str(info.value) and value not in str(info.value) or value == ""
    )


# ------------------------------------------------------------ DATACOPILOT_ALLOWED_LICENSES
def test_allowed_licenses_default_is_the_permissive_set() -> None:
    assert load_settings({}).allowed_licenses == frozenset({"apache-2.0", "mit", "bsd-3-clause"})


def test_allowed_licenses_are_parsed_trimmed_and_case_folded() -> None:
    s = load_settings({"DATACOPILOT_ALLOWED_LICENSES": " MIT , Apache-2.0,SAM-License "})
    assert s.allowed_licenses == frozenset({"mit", "apache-2.0", "sam-license"})


@pytest.mark.parametrize(
    "value",
    [
        "",  # an empty allow-list is almost certainly a mistake, never "allow everything"
        "   ",
        "MIT,,Apache-2.0",
        "MIT,",
        "MIT;Apache-2.0",
        "mit\nbsd",
        "x" * 65,
        "../etc/passwd",
        ",".join(f"L{n}" for n in range(33)),  # more entries than any sane policy
    ],
)
def test_invalid_allowed_licenses_name_the_variable_not_the_value(value: str) -> None:
    with pytest.raises(SettingsError, match="DATACOPILOT_ALLOWED_LICENSES") as err:
        load_settings({"DATACOPILOT_ALLOWED_LICENSES": value})
    assert value.strip() == "" or value not in str(err.value)

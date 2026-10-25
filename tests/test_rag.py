"""Tests unitaires des fonctions pures de `rag`."""
import pytest

import rag


def test_read_required_environment_variable_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_VARIABLE", "value")

    assert rag.read_required_environment_variable("SOME_VARIABLE") == "value"


@pytest.mark.parametrize("value", [None, ""])
def test_read_required_environment_variable_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("SOME_VARIABLE", raising=False)
    else:
        monkeypatch.setenv("SOME_VARIABLE", value)

    with pytest.raises(ValueError, match="SOME_VARIABLE"):
        rag.read_required_environment_variable("SOME_VARIABLE")

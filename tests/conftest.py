"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def _iteration_based_budgets(monkeypatch):
    """Tests shrink iteration counts to stay fast. The default configuration
    stops every algorithm by evaluation budget instead, which would ignore
    those counts, so tests use the iteration rules unless they set a budget."""
    import config.config as config

    monkeypatch.setattr(config, "EVALUATION_BUDGET", None)

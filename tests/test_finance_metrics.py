import pytest

from app.finance.metrics import (
    concentration_score,
    portfolio_diagnostics,
    portfolio_expected_return,
    portfolio_volatility,
    sharpe_ratio,
)
from app.models.schemas import AssetPosition


def _positions() -> list[AssetPosition]:
    return [
        AssetPosition(ticker="AAA", allocation_pct=50, expected_return=0.08, volatility=0.15),
        AssetPosition(ticker="BBB", allocation_pct=50, expected_return=0.04, volatility=0.05),
    ]


def test_expected_return_and_volatility() -> None:
    positions = _positions()
    exp = portfolio_expected_return(positions)
    vol = portfolio_volatility(positions)
    assert exp == 0.06
    assert vol > 0


def test_sharpe_and_concentration() -> None:
    positions = _positions()
    sharpe = sharpe_ratio(0.06, 0.1, risk_free_rate=0.02)
    concentration = concentration_score(positions)
    assert sharpe == pytest.approx(0.4)
    assert concentration == 0.5


def test_portfolio_diagnostics_contains_expected_keys() -> None:
    diagnostics = portfolio_diagnostics(_positions())
    assert "expected_return" in diagnostics
    assert "volatility" in diagnostics
    assert "sharpe_ratio" in diagnostics

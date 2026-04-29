from __future__ import annotations

import numpy as np

from app.models.schemas import AssetPosition


def portfolio_expected_return(positions: list[AssetPosition]) -> float:
    if not positions:
        return 0.0
    weights = np.array([p.allocation_pct / 100.0 for p in positions], dtype=float)
    returns = np.array([p.expected_return for p in positions], dtype=float)
    return float(np.dot(weights, returns))


def portfolio_volatility(positions: list[AssetPosition]) -> float:
    if not positions:
        return 0.0
    weights = np.array([p.allocation_pct / 100.0 for p in positions], dtype=float)
    vols = np.array([p.volatility for p in positions], dtype=float)
    covariance = np.diag(vols**2)
    return float(np.sqrt(weights.T @ covariance @ weights))


def sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float = 0.02) -> float:
    if volatility <= 1e-12:
        return 0.0
    return float((expected_return - risk_free_rate) / volatility)


def concentration_score(positions: list[AssetPosition]) -> float:
    if not positions:
        return 0.0
    weights = np.array([p.allocation_pct / 100.0 for p in positions], dtype=float)
    return float(np.sum(weights**2))


def portfolio_diagnostics(positions: list[AssetPosition]) -> dict[str, float]:
    exp = portfolio_expected_return(positions)
    vol = portfolio_volatility(positions)
    sharpe = sharpe_ratio(exp, vol)
    concentration = concentration_score(positions)
    values = np.array([exp, vol, sharpe, concentration], dtype=float)
    if len(values) > 1 and float(np.std(values)) > 1e-12:
        normalized = (values - float(np.mean(values))) / float(np.std(values))
    else:
        normalized = values
    return {
        "expected_return": exp,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "concentration": concentration,
        "health_zscore_mean": float(np.mean(normalized)),
    }

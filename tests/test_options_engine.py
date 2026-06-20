"""Tier 2 tests: options_engine.py — unit tests with mock data (no FutuOpenD needed)."""
import math
from datetime import date, timedelta

import pytest
from options_engine import (
    extract_iv,
    calculate_greeks,
    compute_realized_volatility,
    compute_iv_vs_rv_percentile,
    analyze_option,
)


# ---------------------------------------------------------------------------
# Test data: realistic HSI call option
# Spot=19850, Strike=20000, 30 DTE, Premium=350, r=4%, q=3.5%
# ---------------------------------------------------------------------------

SPOT = 19850.0
STRIKE = 20000.0
T = 30.0 / 365.0
R = 0.04
Q = 0.035
PREMIUM = 350.0
EXPIRY = date.today() + timedelta(days=30)


def make_historical_closes(n: int = 500, base: float = 19850.0) -> list:
    """Generate synthetic HSI daily closes with ~20% annual vol."""
    import random
    random.seed(42)
    closes = [base]
    for _ in range(n - 1):
        daily_ret = random.gauss(0, 0.20 / math.sqrt(252))
        closes.append(closes[-1] * (1 + daily_ret))
    return closes


HISTORICAL_CLOSES = make_historical_closes()


class TestExtractIV:
    def test_normal_convergence_call(self):
        iv, err = extract_iv(SPOT, STRIKE, T, R, Q, PREMIUM, 'call')
        assert err is None
        assert 0.05 < iv < 1.0  # Reasonable IV range

    def test_normal_convergence_put(self):
        iv, err = extract_iv(SPOT, STRIKE, T, R, Q, PREMIUM, 'put')
        assert err is None
        assert 0.05 < iv < 1.0

    def test_expired_option(self):
        iv, err = extract_iv(SPOT, STRIKE, 0.0, R, Q, PREMIUM, 'call')
        assert iv is None
        assert "到期" in err

    def test_negative_time(self):
        iv, err = extract_iv(SPOT, STRIKE, -0.01, R, Q, PREMIUM, 'call')
        assert iv is None

    def test_zero_premium(self):
        iv, err = extract_iv(SPOT, STRIKE, T, R, Q, 0.0, 'call')
        assert iv is None
        assert "权利金" in err

    def test_arbitrage_violation(self):
        # Deep ITM call: spot=19850, strike=15000, intrinsic=4850
        # premium=4000 < intrinsic → arbitrage
        iv, err = extract_iv(SPOT, 15000.0, T, R, Q, 4000.0, 'call')
        assert iv is None
        assert "无套利" in err

    def test_deep_otm_still_solvable(self):
        # Very deep OTM call — should still converge with bisection
        iv, err = extract_iv(SPOT, 25000.0, T, R, Q, 5.0, 'call')
        assert err is None
        assert iv is not None


class TestCalculateGreeks:
    def test_all_greeks_returned_call(self):
        greeks, err = calculate_greeks(SPOT, STRIKE, T, R, Q, 0.25, 'call')
        assert err is None
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            assert key in greeks
            assert isinstance(greeks[key], float)

    def test_all_greeks_returned_put(self):
        greeks, err = calculate_greeks(SPOT, STRIKE, T, R, Q, 0.25, 'put')
        assert err is None
        assert greeks["delta"] < 0  # Put delta is negative

    def test_zero_time_rejected(self):
        greeks, err = calculate_greeks(SPOT, STRIKE, 0.0, R, Q, 0.25, 'call')
        assert greeks is None
        assert "时间" in err

    def test_zero_iv_rejected(self):
        greeks, err = calculate_greeks(SPOT, STRIKE, T, R, Q, 0.0, 'call')
        assert greeks is None
        assert "波动率" in err

    def test_theta_is_daily(self):
        greeks, err = calculate_greeks(SPOT, STRIKE, T, R, Q, 0.25, 'call')
        # Daily theta should be small (< 10 index points is typical)
        assert abs(greeks["theta"]) < 100


class TestRealizedVolatility:
    def test_normal_data(self):
        vols, err = compute_realized_volatility(HISTORICAL_CLOSES, window=30)
        assert err is None
        assert len(vols) > 0
        # Annualized vol should be around 20% (±10%)
        avg_vol = sum(vols) / len(vols)
        assert 0.05 < avg_vol < 0.50

    def test_insufficient_data(self):
        vols, err = compute_realized_volatility([100, 101, 102], window=30)
        assert vols is None
        assert "不足" in err


class TestIVvsRVPercentile:
    def test_iv_below_all_rv(self):
        pct, err = compute_iv_vs_rv_percentile(0.05, [0.10, 0.15, 0.20])
        assert err is None
        assert pct == pytest.approx(0.0)

    def test_iv_above_all_rv(self):
        pct, err = compute_iv_vs_rv_percentile(0.50, [0.10, 0.15, 0.20])
        assert err is None
        assert pct == pytest.approx(100.0)

    def test_iv_mid_range(self):
        pct, err = compute_iv_vs_rv_percentile(0.15, [0.10, 0.15, 0.20])
        assert err is None
        # 0.10 < 0.15 → 1/3 = 33%
        assert pct == pytest.approx(33.33, abs=1.0)

    def test_empty_vols(self):
        pct, err = compute_iv_vs_rv_percentile(0.25, [])
        assert pct is None
        assert "为空" in err

    def test_zero_iv(self):
        pct, err = compute_iv_vs_rv_percentile(0.0, [0.10, 0.15])
        assert pct is None


class TestAnalyzeOption:
    def test_full_pipeline_returns_all_keys(self):
        result = analyze_option(
            spot=SPOT, strike=STRIKE, expiry_date=EXPIRY,
            premium=PREMIUM, bid=347, ask=353,
            option_type='call', historical_closes=HISTORICAL_CLOSES,
        )
        assert result["error"] is None
        assert result["iv"] is not None
        assert result["greeks"] is not None
        assert "delta" in result["greeks"]
        assert "days_to_expiry" in result
        assert "iv_percentile" in result
        assert "spread_pct" in result

    def test_expired_option_returns_error(self):
        past = date.today() - timedelta(days=1)
        result = analyze_option(
            spot=SPOT, strike=STRIKE, expiry_date=past,
            premium=PREMIUM, bid=347, ask=353,
            option_type='call', historical_closes=HISTORICAL_CLOSES,
        )
        assert "已到期" in result["error"]

    def test_warnings_for_near_expiry(self):
        near = date.today() + timedelta(days=3)
        result = analyze_option(
            spot=SPOT, strike=STRIKE, expiry_date=near,
            premium=150.0, bid=147, ask=153,
            option_type='call', historical_closes=HISTORICAL_CLOSES,
        )
        if not result["error"]:
            assert any("临近到期" in w for w in result["warnings"])

    def test_insufficient_historical_data(self):
        result = analyze_option(
            spot=SPOT, strike=STRIKE, expiry_date=EXPIRY,
            premium=PREMIUM, bid=347, ask=353,
            option_type='call', historical_closes=[19000, 19100, 19200],
        )
        assert result["error"] is None
        # Should have a warning about insufficient data
        assert any("历史波动率" in w for w in result["warnings"])

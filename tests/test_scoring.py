"""Tier 1 tests: scoring.py — pure functions, zero dependencies."""
import pytest
from scoring import (
    ScoreWeights,
    ScoreResult,
    calculate_p_score,
    calculate_iv_score,
    calculate_rr_score,
    calculate_theta_score,
    calculate_vega_score,
    calculate_spread_score,
    calculate_oi_score,
    calculate_liquidity_score,
    calculate_composite_score,
    color_from_score,
    recommendation_from_score,
)


class TestPScore:
    # New formula: plateau 0.30-0.60 = 100, ramps down at extremes

    def test_optimal_zone_30(self):
        assert calculate_p_score(0.30) == pytest.approx(100.0)

    def test_optimal_zone_50(self):
        assert calculate_p_score(0.50) == pytest.approx(100.0)

    def test_optimal_zone_60(self):
        assert calculate_p_score(0.60) == pytest.approx(100.0)

    def test_deep_otm(self):
        # 0.05/0.30*100 = 16.7
        assert calculate_p_score(0.05) == pytest.approx(16.67, abs=0.1)

    def test_deep_itm_penalized(self):
        # 100 - (0.95-0.60)/0.35*80 = 100-80 = 20
        assert calculate_p_score(0.95) == pytest.approx(20.0)

    def test_deep_itm_put_penalized(self):
        assert calculate_p_score(-0.95) == pytest.approx(20.0)

    def test_moderate_itm(self):
        # 100 - (0.81-0.60)/0.35*80 = 100-48 = 52
        assert calculate_p_score(-0.81) == pytest.approx(52.0, abs=0.1)

    def test_ramp_up(self):
        # 0.15/0.30*100 = 50
        assert calculate_p_score(0.15) == pytest.approx(50.0)

    def test_delta_zero(self):
        assert calculate_p_score(0.0) == pytest.approx(0.0)

    def test_delta_above_one_floor(self):
        assert calculate_p_score(1.5) == pytest.approx(20.0)

    def test_negative_delta_ramp(self):
        # 0.03/0.30*100 = 10
        assert calculate_p_score(-0.03) == pytest.approx(10.0)


class TestIVScore:
    def test_high_iv_expensive(self):
        assert calculate_iv_score(85.0) == pytest.approx(15.0)

    def test_low_iv_cheap(self):
        assert calculate_iv_score(10.0) == pytest.approx(90.0)

    def test_mid_iv(self):
        assert calculate_iv_score(50.0) == pytest.approx(50.0)

    def test_boundary_zero(self):
        assert calculate_iv_score(0.0) == pytest.approx(100.0)

    def test_boundary_hundred(self):
        assert calculate_iv_score(100.0) == pytest.approx(0.0)


class TestRRScore:
    def test_call_5x_multiplier(self):
        # Call premium=350, max_profit=350*5=1750, rr=1750/350=5, score=5*10=50
        score = calculate_rr_score('CALL', 20000, 350, max_profit_multiplier=5.0)
        assert score == pytest.approx(50.0)

    def test_call_default_5x(self):
        score = calculate_rr_score('CALL', 20000, 350)
        assert score == pytest.approx(50.0)

    def test_put_in_the_money(self):
        score = calculate_rr_score('PUT', 20000, 350)
        assert score == pytest.approx(100.0)  # RR=19650/350=56, capped at 100

    def test_put_atm_ish(self):
        score = calculate_rr_score('PUT', 19850, 500)
        assert score == pytest.approx(100.0)  # max=19350, rr=38.7, capped 100

    def test_zero_premium(self):
        assert calculate_rr_score('CALL', 20000, 0) == pytest.approx(0.0)

    def test_negative_premium(self):
        assert calculate_rr_score('CALL', 20000, -100) == pytest.approx(0.0)

    def test_invalid_option_type(self):
        assert calculate_rr_score('STRADDLE', 20000, 350) == pytest.approx(0.0)


class TestThetaScore:
    def test_normal_decay(self):
        score = calculate_theta_score(theta=5.0, premium=350.0)
        assert score == pytest.approx(100.0 - (5/350)*100)  # ≈ 98.6

    def test_high_decay_near_expiry(self):
        score = calculate_theta_score(theta=50.0, premium=350.0)
        assert score == pytest.approx(100.0 - (50/350)*100)  # ≈ 85.7

    def test_extreme_decay(self):
        score = calculate_theta_score(theta=500.0, premium=350.0)
        assert score == pytest.approx(0.0)  # Floor at 0

    def test_zero_premium(self):
        assert calculate_theta_score(theta=5.0, premium=0.0) == pytest.approx(0.0)

    def test_zero_theta(self):
        score = calculate_theta_score(theta=0.0, premium=350.0)
        assert score == pytest.approx(100.0)


class TestVegaScore:
    def test_low_vega(self):
        score = calculate_vega_score(vega=10.0, premium=350.0)
        assert score == pytest.approx(100.0 - (10/350)*100)  # ≈ 97.1

    def test_warning_threshold(self):
        score = calculate_vega_score(vega=70.0, premium=350.0)
        assert score == pytest.approx(80.0)  # 20% ratio

    def test_high_vega(self):
        score = calculate_vega_score(vega=140.0, premium=350.0)
        assert score == pytest.approx(60.0)  # 40% ratio

    def test_zero_premium(self):
        assert calculate_vega_score(vega=10.0, premium=0.0) == pytest.approx(0.0)


class TestSpreadScore:
    def test_tight_spread(self):
        score = calculate_spread_score(bid=347.0, ask=353.0)
        assert score == pytest.approx(100.0 - 1.7*5, abs=1.0)

    def test_5pct_spread(self):
        score = calculate_spread_score(bid=332.5, ask=367.5)
        assert score == pytest.approx(50.0, abs=1.0)

    def test_10pct_spread(self):
        score = calculate_spread_score(bid=315.0, ask=385.0)
        assert score == pytest.approx(0.0, abs=1.0)

    def test_wide_spread_floor(self):
        score = calculate_spread_score(bid=100, ask=500)
        assert score == pytest.approx(0.0)

    def test_zero_bid(self):
        assert calculate_spread_score(bid=0, ask=350) == pytest.approx(0.0)

    def test_negative_spread_guard(self):
        score = calculate_spread_score(bid=360, ask=340)
        assert score == pytest.approx(0.0)


class TestOIScore:
    def test_zero_oi(self):
        assert calculate_oi_score(0) == pytest.approx(0.0)

    def test_high_oi(self):
        s = calculate_oi_score(1000)
        assert s > 70  # log10(1001)/4*100 ≈ 75

    def test_low_oi(self):
        s = calculate_oi_score(10)
        assert 25 < s < 40  # log10(11)/4*100 ≈ 26

    def test_negative_oi(self):
        assert calculate_oi_score(-5) == pytest.approx(0.0)


class TestLiquidityScore:
    def test_perfect(self):
        # Tight spread (100) + high OI (100) → 100
        score = calculate_liquidity_score(bid=347, ask=353, open_interest=1000)
        assert score > 80

    def test_wide_spread_no_oi(self):
        # Wide spread (0) + no OI (0) → 0
        score = calculate_liquidity_score(bid=100, ask=500, open_interest=0)
        assert score < 10

    def test_good_spread_low_oi(self):
        # Tight spread rescues low OI somewhat
        score = calculate_liquidity_score(bid=347, ask=353, open_interest=5)
        assert 60 < score < 85  # spread≈100*0.7 + oi≈20*0.3 ≈ 76


class TestCompositeScore:
    def test_typical_call_high_score(self):
        result = calculate_composite_score(
            delta=0.50, iv_percentile=20, option_type='CALL',
            strike=20000, premium=350, theta=5.0, vega=30.0,
            bid=347, ask=353, days_to_expiry=30, open_interest=500,
        )
        assert result.composite > 50
        assert result.color in ("green", "yellow")

    def test_bad_option_low_score(self):
        result = calculate_composite_score(
            delta=0.05, iv_percentile=90, option_type='CALL',
            strike=22000, premium=10, theta=2.0, vega=5.0,
            bid=8, ask=12, days_to_expiry=3, open_interest=2,
        )
        assert result.composite < 50
        assert result.color == "red"

    def test_weight_error_returns_zero(self):
        bad_weights = ScoreWeights(p_weight=0.50)
        result = calculate_composite_score(
            delta=0.50, iv_percentile=50, option_type='CALL',
            strike=20000, premium=350, theta=5.0, vega=30.0,
            bid=347, ask=353, days_to_expiry=30, open_interest=100,
            weights=bad_weights,
        )
        assert result.composite == 0
        assert "配置错误" in result.recommendation

    def test_output_dict_has_all_fields(self):
        result = calculate_composite_score(
            delta=0.50, iv_percentile=50, option_type='CALL',
            strike=20000, premium=350, theta=5.0, vega=30.0,
            bid=347, ask=353, days_to_expiry=30, open_interest=100,
        )
        d = result.to_dict()
        for key in ("p_score", "iv_score", "rr_score", "theta_score",
                     "vega_score", "spread_score", "composite", "color", "recommendation"):
            assert key in d


class TestColorFromScore:
    def test_green(self):
        assert color_from_score(85) == "green"
        assert color_from_score(80) == "green"

    def test_yellow(self):
        assert color_from_score(79) == "yellow"
        assert color_from_score(50) == "yellow"

    def test_red(self):
        assert color_from_score(49) == "red"
        assert color_from_score(0) == "red"


class TestRecommendation:
    def test_high_score_good_conditions(self):
        # iv_percentile=30 is mid-range (20-80), neither "偏高" nor "偏低" fires
        rec = recommendation_from_score(85, 30, 2.0, 30)
        assert "高" in rec
        # 30 is not <20 and not >80, so no IV comment added

    def test_low_score_bad_conditions(self):
        rec = recommendation_from_score(30, 85, 12.0, 2)
        assert "不建议" in rec
        assert "偏高" in rec
        assert "价差过大" in rec

    def test_near_expiry_warning(self):
        rec = recommendation_from_score(70, 50, 3.0, 3)
        assert "临近到期" in rec

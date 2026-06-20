"""Tests for short_scoring.py — Short options scoring engine."""
import pytest
from short_scoring import (
    ShortScoreWeights,
    ShortScoreResult,
    calculate_short_iv_score,
    calculate_short_theta_score,
    calculate_short_rr_score,
    calculate_short_delta_score,
    calculate_short_vega_score,
    estimate_margin,
    calculate_short_composite,
    short_color_from_score,
)


class TestShortIVScore:
    def test_high_iv_good_for_short(self):
        assert calculate_short_iv_score(85.0) == pytest.approx(85.0)

    def test_low_iv_bad_for_short(self):
        assert calculate_short_iv_score(15.0) == pytest.approx(15.0)

    def test_iv_boundaries(self):
        assert calculate_short_iv_score(0.0) == pytest.approx(0.0)
        assert calculate_short_iv_score(100.0) == pytest.approx(100.0)


class TestShortThetaScore:
    def test_fast_decay_good(self):
        # Theta=20, Premium=200 → 20/200*200 = 20
        score = calculate_short_theta_score(20.0, 200.0)
        assert score == pytest.approx(20.0)

    def test_slow_decay_bad(self):
        score = calculate_short_theta_score(2.0, 500.0)
        assert score < 5.0

    def test_zero_premium(self):
        assert calculate_short_theta_score(5.0, 0.0) == pytest.approx(0.0)

    def test_capped_at_100(self):
        score = calculate_short_theta_score(200.0, 200.0)
        assert score == pytest.approx(100.0)


class TestShortRRScore:
    def test_call_rr(self):
        # Naked call: Premium=500, Spot=23925 → RR=500/23925=2.09%, score=2.09*20=42
        score = calculate_short_rr_score('CALL', 24000, 500, 23925)
        assert 30 < score < 60

    def test_put_rr(self):
        # Naked put: Premium=500, Strike=24000 → max_loss=23500, RR=500/23500=2.13%, score=43
        score = calculate_short_rr_score('PUT', 24000, 500, 23925)
        assert 30 < score < 60

    def test_high_premium_good_rr(self):
        # Higher premium = better RR
        score_high = calculate_short_rr_score('PUT', 24000, 1000, 23925)
        score_low = calculate_short_rr_score('PUT', 24000, 200, 23925)
        assert score_high > score_low

    def test_zero_premium(self):
        assert calculate_short_rr_score('CALL', 24000, 0, 23925) == pytest.approx(0.0)

    def test_invalid_type(self):
        assert calculate_short_rr_score('STRADDLE', 24000, 500, 23925) == pytest.approx(0.0)


class TestShortDeltaScore:
    def test_low_delta_good(self):
        assert calculate_short_delta_score(0.10) == pytest.approx(90.0)

    def test_atm_neutral(self):
        assert calculate_short_delta_score(0.50) == pytest.approx(50.0)

    def test_high_delta_bad(self):
        assert calculate_short_delta_score(0.90) == pytest.approx(10.0)

    def test_negative_delta(self):
        assert calculate_short_delta_score(-0.30) == pytest.approx(70.0)


class TestShortVegaScore:
    def test_low_vega_good(self):
        score = calculate_short_vega_score(10.0, 350.0)
        assert score > 95

    def test_high_vega_bad(self):
        score = calculate_short_vega_score(70.0, 350.0)
        assert score == pytest.approx(80.0)

    def test_zero_premium(self):
        assert calculate_short_vega_score(10.0, 0.0) == pytest.approx(0.0)


class TestMarginEstimate:
    def test_otm_call(self):
        # OTM Call: Strike=25000, Spot=23925, OTM=1075
        margin = estimate_margin('CALL', 25000, 23925, 200, 50)
        assert margin > 0
        # Should be ~ max(20%*23925, 1075+10%*23925) * 50 ≈ max(4785, 3468)*50 ≈ 239K

    def test_otm_put(self):
        margin = estimate_margin('PUT', 23000, 23925, 200, 50)
        assert margin > 0

    def test_different_multiplier(self):
        m50 = estimate_margin('CALL', 25000, 23925, 200, 50)
        m10 = estimate_margin('CALL', 25000, 23925, 200, 10)
        assert m50 == pytest.approx(m10 * 5)


class TestShortComposite:
    def test_high_iv_good(self):
        result = calculate_short_composite(
            iv_percentile=85, theta=10.0, premium=300, option_type='PUT',
            strike=24000, spot=23925, delta=-0.30, vega=20.0,
            bid=295, ask=305, days_to_expiry=30, open_interest=500,
        )
        assert result.composite > 50  # High IV + low delta + tight spread

    def test_low_iv_bad(self):
        result = calculate_short_composite(
            iv_percentile=15, theta=2.0, premium=100, option_type='CALL',
            strike=24000, spot=23925, delta=0.60, vega=10.0,
            bid=90, ask=110, days_to_expiry=60, open_interest=10,
        )
        assert result.composite < 60  # Low IV + high delta + wide spread

    def test_return_capital(self):
        result = calculate_short_composite(
            iv_percentile=50, theta=5.0, premium=500, option_type='PUT',
            strike=24000, spot=23925, delta=-0.50, vega=30.0,
            bid=495, ask=505, days_to_expiry=30, open_interest=200,
        )
        assert result.margin_estimate > 0
        assert result.roc_pct > 0  # annualized ROC

    def test_weight_error(self):
        bad = ShortScoreWeights(iv_weight=0.50)  # sums to 0.55
        result = calculate_short_composite(
            iv_percentile=50, theta=5.0, premium=300, option_type='PUT',
            strike=24000, spot=23925, delta=-0.50, vega=30.0,
            bid=295, ask=305, days_to_expiry=30, open_interest=200,
            weights=bad,
        )
        assert result.composite == 0
        assert "配置错误" in result.recommendation

    def test_dict_output(self):
        result = calculate_short_composite(
            iv_percentile=50, theta=5.0, premium=300, option_type='PUT',
            strike=24000, spot=23925, delta=-0.50, vega=30.0,
            bid=295, ask=305, days_to_expiry=30, open_interest=200,
        )
        d = result.to_dict()
        for key in ("iv_score", "theta_score", "rr_score", "delta_score",
                     "vega_score", "liquidity_score", "composite", "color",
                     "margin_estimate", "roc_pct"):
            assert key in d


class TestShortColor:
    def test_colors(self):
        assert short_color_from_score(85) == "green"
        assert short_color_from_score(65) == "yellow"
        assert short_color_from_score(30) == "red"

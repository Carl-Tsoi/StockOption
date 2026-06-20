"""值博率 Short (卖出) scoring engine.

Short options: you receive premium upfront, bear potentially large risk.
Key differences from Long scoring:
- IV_score: direct (high IV = expensive premium = good to sell)
- Theta_score: reward (fast decay = income for seller)
- RR_score: inverted (max profit = premium, max loss varies)
- Delta_score: lower is better (less directional risk for premium seller)

All functions pure — independently testable, no I/O.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ShortScoreWeights:
    """Configurable Short scoring weights."""
    iv_weight: float = 0.25
    theta_weight: float = 0.25
    rr_weight: float = 0.15
    delta_weight: float = 0.10
    vega_weight: float = 0.10
    liquidity_weight: float = 0.15

    def validate(self) -> Optional[str]:
        total = (self.iv_weight + self.theta_weight + self.rr_weight
                 + self.delta_weight + self.vega_weight + self.liquidity_weight)
        if abs(total - 1.0) > 0.001:
            return f"Weights must sum to 1.0, got {total:.3f}"
        for name, w in [("IV", self.iv_weight), ("Theta", self.theta_weight),
                         ("RR", self.rr_weight), ("Delta", self.delta_weight),
                         ("Vega", self.vega_weight), ("Liquidity", self.liquidity_weight)]:
            if w < 0:
                return f"{name}_weight must be >= 0, got {w}"
        return None


@dataclass
class ShortScoreResult:
    """Full Short scoring output."""
    iv_score: float
    theta_score: float
    rr_score: float
    delta_score: float
    vega_score: float
    liquidity_score: float
    composite: float
    color: str
    recommendation: str
    margin_estimate: float  # HKD margin required
    roc_pct: float          # Return on Capital %

    def to_dict(self) -> dict:
        return {
            "iv_score": round(self.iv_score, 1),
            "theta_score": round(self.theta_score, 1),
            "rr_score": round(self.rr_score, 1),
            "delta_score": round(self.delta_score, 1),
            "vega_score": round(self.vega_score, 1),
            "liquidity_score": round(self.liquidity_score, 1),
            "composite": round(self.composite, 1),
            "color": self.color,
            "recommendation": self.recommendation,
            "margin_estimate": round(self.margin_estimate, 0),
            "roc_pct": round(self.roc_pct, 1),
        }


# ---------------------------------------------------------------------------
# Individual Short scoring functions
# ---------------------------------------------------------------------------

def calculate_short_iv_score(iv_percentile: float) -> float:
    """IV score for Short: high IV = expensive premium = good to sell.
    Direct use of IV percentile (no inversion).
    IV分位 80 → 80 (great to sell). IV分位 20 → 20 (not worth selling).
    """
    return max(0.0, min(100.0, iv_percentile))


def calculate_short_theta_score(theta: float, premium: float) -> float:
    """Theta score for Short: fast decay = income for seller.
    Formula: min(100, |theta/premium| * 200)
    Short wants theta decay to be fast relative to premium collected.
    """
    if premium <= 0:
        return 0.0
    decay_ratio = abs(theta) / premium
    score = min(100.0, decay_ratio * 200.0)
    return max(0.0, score)


def calculate_short_rr_score(option_type: str, strike: float, premium: float,
                              spot: float) -> float:
    """Risk/Reward for Short: MaxProfit = premium received.
    MaxLoss varies:
    - Call: theoretically unlimited. Use spot as max loss proxy.
    - Put: Strike - Premium (if underlying goes to 0).
    Score = min(100, RR * 20). Short RR naturally lower than Long.
    """
    if premium <= 0:
        return 0.0

    if option_type.upper() == 'CALL':
        max_loss = spot  # proxy for naked call risk
    elif option_type.upper() == 'PUT':
        max_loss = max(0.0, strike - premium)
    else:
        return 0.0

    if max_loss <= 0:
        return 0.0

    rr_pct = premium / max_loss * 100.0
    score = min(100.0, rr_pct * 20.0)
    return max(0.0, score)


def calculate_short_delta_score(delta: float) -> float:
    """Delta score for Short: lower |delta| = less directional risk.
    Delta=0.1 → 90 (minimal direction risk). Delta=0.5 → 50.
    Delta=0.9 → 10 (too much directional exposure for premium seller).
    """
    abs_delta = abs(delta)
    score = 100.0 - abs_delta * 100.0
    return max(0.0, score)


def calculate_short_vega_score(vega: float, premium: float) -> float:
    """Vega score for Short: high vega exposure = IV spike risk.
    Same penalty direction as Long — magnitude of exposure is what matters.
    """
    if premium <= 0:
        return 0.0
    vega_ratio = abs(vega) / premium
    score = 100.0 - min(100.0, vega_ratio * 100.0)
    return max(0.0, score)


def estimate_margin(option_type: str, strike: float, spot: float,
                    premium: float, contract_multiplier: float = 50.0) -> float:
    """Estimate initial margin for naked short option.
    Simplified formula: max(20% * Spot, OTM_amount + maintenance) * multiplier.
    This is an approximation; actual margin depends on exchange rules.
    """
    otm_amount = 0.0
    if option_type.upper() == 'CALL':
        otm_amount = max(0.0, strike - spot)
    else:
        otm_amount = max(0.0, spot - strike)

    margin_per_share = max(0.20 * spot, otm_amount + 0.10 * spot)
    return margin_per_share * contract_multiplier


# ---------------------------------------------------------------------------
# Composite Short scoring
# ---------------------------------------------------------------------------

def short_color_from_score(score: float) -> str:
    if score >= 80: return "green"
    elif score >= 50: return "yellow"
    else: return "red"


def short_recommendation(score: float, iv_percentile: float,
                          spread_pct: float, days_to_expiry: float) -> str:
    parts = []
    if score >= 80:
        parts.append("值得 Short——权利金丰厚，风险可控")
    elif score >= 50:
        parts.append("Short 一般——可考虑，但回报不算突出")
    else:
        parts.append("不建议 Short——权利金太少或风险太大")

    if iv_percentile < 20:
        parts.append("；IV 偏低，权利金薄")
    elif iv_percentile > 80:
        parts.append("；IV 偏高，权利金丰厚")

    if spread_pct > 10:
        parts.append("；买卖价差过大")
    if days_to_expiry < 5:
        parts.append("；临近到期（Gamma 风险高）")

    return "".join(parts)


def calculate_short_composite(
    iv_percentile: float,
    theta: float,
    premium: float,
    option_type: str,
    strike: float,
    spot: float,
    delta: float,
    vega: float,
    bid: float,
    ask: float,
    days_to_expiry: float,
    open_interest: int = 0,
    contract_multiplier: float = 50.0,
    weights: Optional[ShortScoreWeights] = None,
) -> ShortScoreResult:
    """Calculate composite Short 值博率 score (0-100).

    Returns ShortScoreResult with component scores, composite, margin estimate,
    and Return on Capital.
    """
    if weights is None:
        weights = ShortScoreWeights()

    err = weights.validate()
    if err:
        return ShortScoreResult(
            iv_score=0, theta_score=0, rr_score=0, delta_score=0,
            vega_score=0, liquidity_score=0, composite=0,
            color="red", recommendation=f"权重配置错误: {err}",
            margin_estimate=0, roc_pct=0,
        )

    # Reuse Long liquidity scoring (same formula, same logic)
    from scoring import calculate_liquidity_score

    iv_s = calculate_short_iv_score(iv_percentile)
    theta_s = calculate_short_theta_score(theta, premium)
    rr_s = calculate_short_rr_score(option_type, strike, premium, spot)
    delta_s = calculate_short_delta_score(delta)
    vega_s = calculate_short_vega_score(vega, premium)
    liq_s = calculate_liquidity_score(bid, ask, open_interest)

    composite = (
        weights.iv_weight * iv_s
        + weights.theta_weight * theta_s
        + weights.rr_weight * rr_s
        + weights.delta_weight * delta_s
        + weights.vega_weight * vega_s
        + weights.liquidity_weight * liq_s
    )

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else 0.0
    margin = estimate_margin(option_type, strike, spot, premium, contract_multiplier)
    roc = (premium * contract_multiplier) / margin * 100.0 if margin > 0 else 0.0
    # Annualize ROC: premium collected over DTE, annualized
    if days_to_expiry > 0 and margin > 0:
        roc = roc * (365.0 / days_to_expiry)

    color = short_color_from_score(composite)
    rec = short_recommendation(composite, iv_percentile, spread_pct, days_to_expiry)

    return ShortScoreResult(
        iv_score=iv_s,
        theta_score=theta_s,
        rr_score=rr_s,
        delta_score=delta_s,
        vega_score=vega_s,
        liquidity_score=liq_s,
        composite=composite,
        color=color,
        recommendation=rec,
        margin_estimate=margin,
        roc_pct=roc,
    )

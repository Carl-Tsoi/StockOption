"""值博率 (Value/Edge) scoring engine for HSI options.

Each factor returns a score 0-100. The composite score is a weighted sum.
All functions are pure (no side effects, no I/O) — independently testable.

Formula:
    Score = w1*P_score + w2*IV_score + w3*RR_score
          + w4*Theta_score + w5*Vega_score + w6*Spread_score

Default weights (to be calibrated with trade logs):
    P=0.20, IV=0.20, RR=0.20, Theta=0.20, Vega=0.15, Spread=0.05
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreWeights:
    """Configurable scoring weights."""
    p_weight: float = 0.20
    iv_weight: float = 0.20
    rr_weight: float = 0.20
    theta_weight: float = 0.20
    vega_weight: float = 0.15
    spread_weight: float = 0.05

    def validate(self) -> Optional[str]:
        total = (self.p_weight + self.iv_weight + self.rr_weight
                 + self.theta_weight + self.vega_weight + self.spread_weight)
        if abs(total - 1.0) > 0.001:
            return f"Weights must sum to 1.0, got {total:.3f}"
        for name, w in [("P", self.p_weight), ("IV", self.iv_weight),
                         ("RR", self.rr_weight), ("Theta", self.theta_weight),
                         ("Vega", self.vega_weight), ("Spread", self.spread_weight)]:
            if w < 0:
                return f"{name}_weight must be >= 0, got {w}"
        return None


@dataclass
class ScoreResult:
    """Full scoring output with component breakdown."""
    p_score: float
    iv_score: float
    rr_score: float
    theta_score: float
    vega_score: float
    spread_score: float
    composite: float
    color: str  # "green" | "yellow" | "red"
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "p_score": round(self.p_score, 1),
            "iv_score": round(self.iv_score, 1),
            "rr_score": round(self.rr_score, 1),
            "theta_score": round(self.theta_score, 1),
            "vega_score": round(self.vega_score, 1),
            "spread_score": round(self.spread_score, 1),
            "composite": round(self.composite, 1),
            "color": self.color,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def calculate_p_score(delta: float) -> float:
    """Directional probability score from absolute delta.
    ATM (delta≈0.50) → 100. Deep OTM (delta≈0.05) → 10.
    Uses Black-Scholes delta as a rough ITM-probability proxy.
    Note: Delta diverges from true prob for tail-risk options; this is
    a practical approximation, not a theoretical claim.
    """
    abs_delta = abs(delta)
    score = abs_delta * 200  # 0.50 → 100, 0.05 → 10
    return max(0.0, min(100.0, score))


def calculate_iv_score(iv_percentile: float) -> float:
    """IV cost score. IV_percentile = current IV rank vs historical
    realized volatility distribution (0-100).
    High IV percentile → option is expensive → low score (penalized).
    Low IV percentile → option is cheap → high score (favored).
    """
    # Invert: high percentile = expensive = bad for buyers
    return max(0.0, min(100.0, 100.0 - iv_percentile))


def calculate_rr_score(option_type: str, strike: float, premium: float,
                       max_profit_multiplier: float = 5.0) -> float:
    """Risk/Reward score based on max profit / max loss ratio.
    - Call: max profit = premium * max_profit_multiplier (configurable cap)
    - Put:  max profit = strike - premium
    - Max loss = premium (long options only)
    Ratio 1.0 (breakeven) → ~100 * 1/x based on multiplier.
    """
    if premium <= 0:
        return 0.0

    if option_type.upper() == 'CALL':
        max_profit = premium * max_profit_multiplier
    elif option_type.upper() == 'PUT':
        max_profit = max(0.0, strike - premium)
    else:
        return 0.0

    if max_profit <= 0:
        return 0.0

    rr_ratio = max_profit / premium
    score = min(100.0, rr_ratio * 10)  # RR=10 → 100 (capped)
    return max(0.0, score)


def calculate_theta_score(theta: float, premium: float) -> float:
    """Daily time-decay penalty. Theta is the daily value loss.
    Formula: 100 - min(100, |theta/premium| * 100)
    Near-expiry options with high daily theta decay get penalized.
    Fixed: No T multiplier needed — theta is already a daily value.
    """
    if premium <= 0:
        return 0.0
    decay_ratio = abs(theta) / premium
    score = 100.0 - min(100.0, decay_ratio * 100.0)
    return max(0.0, score)


def calculate_vega_score(vega: float, premium: float) -> float:
    """Vega exposure penalty. High vega relative to premium
    means the position is sensitive to IV changes.
    If vega > 20% of premium, warning threshold triggered.
    """
    if premium <= 0:
        return 0.0
    vega_ratio = vega / premium
    score = 100.0 - min(100.0, vega_ratio * 100.0)
    return max(0.0, score)


def calculate_spread_score(bid: float, ask: float) -> float:
    """Bid-ask spread penalty. Wide spreads = high hidden cost for retail.
    Spread = (Ask - Bid) / Mid * 100%
    Spread 5% → score 75, Spread 10% → score 50, Spread ≥20% → score 0.
    """
    mid = (bid + ask) / 2.0
    if mid <= 0 or bid <= 0 or ask <= 0:
        return 0.0
    spread_pct = (ask - bid) / mid * 100.0
    if spread_pct < 0:
        return 0.0
    # Scale: 5% → 25pt penalty, 10% → 50pt, 20% → 100pt (floor 0)
    score = 100.0 - min(100.0, spread_pct * 5.0)
    return max(0.0, score)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def color_from_score(score: float) -> str:
    """Map composite score to color code."""
    if score >= 80:
        return "green"  # 值博率高
    elif score >= 50:
        return "yellow"  # 一般
    else:
        return "red"  # 值博率低


def recommendation_from_score(score: float, iv_percentile: float,
                               spread_pct: float, days_to_expiry: float) -> str:
    """Generate a one-line Chinese recommendation based on the score."""
    parts = []

    if score >= 80:
        parts.append("值博率高")
    elif score >= 50:
        parts.append("值博率一般")
    else:
        parts.append("值博率低，不建议买入")

    if iv_percentile > 80:
        parts.append("；IV 偏高(vs 历史实现波动率)")
    elif iv_percentile < 20:
        parts.append("；IV 偏低(vs 历史实现波动率)")

    if spread_pct > 10:
        parts.append("；买卖价差过大")
    elif spread_pct > 5:
        parts.append("；买卖价差偏大")

    if days_to_expiry < 5:
        parts.append("；临近到期，时间损耗加速")

    return "".join(parts)


def calculate_composite_score(
    delta: float,
    iv_percentile: float,
    option_type: str,
    strike: float,
    premium: float,
    theta: float,
    vega: float,
    bid: float,
    ask: float,
    days_to_expiry: float,
    weights: Optional[ScoreWeights] = None,
    max_profit_multiplier: float = 5.0,
) -> ScoreResult:
    """Calculate the composite 值博率 score (0-100).

    Returns a ScoreResult with all component scores, composite, color, and
    Chinese recommendation string.

    All inputs should be in consistent units:
    - delta: absolute (0 to 1)
    - iv_percentile: 0-100 (percentile vs historical realized vol)
    - strike, premium: index points
    - theta: daily index points
    - vega: index points per 1% IV change
    - bid, ask: index points
    - days_to_expiry: float
    """
    if weights is None:
        weights = ScoreWeights()

    err = weights.validate()
    if err:
        # Return a zero-score result with error indication
        return ScoreResult(
            p_score=0, iv_score=0, rr_score=0,
            theta_score=0, vega_score=0, spread_score=0,
            composite=0, color="red",
            recommendation=f"评分权重配置错误: {err}",
        )

    p_score = calculate_p_score(delta)
    iv_score = calculate_iv_score(iv_percentile)
    rr_score = calculate_rr_score(option_type, strike, premium, max_profit_multiplier)
    theta_score = calculate_theta_score(theta, premium)
    vega_score = calculate_vega_score(vega, premium)
    spread_score = calculate_spread_score(bid, ask)

    composite = (
        weights.p_weight * p_score
        + weights.iv_weight * iv_score
        + weights.rr_weight * rr_score
        + weights.theta_weight * theta_score
        + weights.vega_weight * vega_score
        + weights.spread_weight * spread_score
    )

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else 0.0
    color = color_from_score(composite)
    recommendation = recommendation_from_score(
        composite, iv_percentile, spread_pct, days_to_expiry
    )

    return ScoreResult(
        p_score=p_score,
        iv_score=iv_score,
        rr_score=rr_score,
        theta_score=theta_score,
        vega_score=vega_score,
        spread_score=spread_score,
        composite=composite,
        color=color,
        recommendation=recommendation,
    )

"""Options pricing engine: IV extraction, Greeks, IV-vs-RV percentile.

Uses vollib for Black-Scholes (European options, matching HSI spec).
Dividend yield handled via S_adj = S * exp(-q * T) — standard Merton adjustment.
All functions return (result, error) tuples per the architecture decision.

Note: vollib does not accept a dividend yield parameter. We use the
dividend-adjusted spot (S * e^(-qT)) for pricing and Greeks. Delta is
further adjusted by e^(-qT). Other Greeks use the adjusted spot directly
(small approximation for Phase 1; cross-validated against Futu data).
"""

import math
from datetime import date
from functools import lru_cache
from typing import Optional, Tuple, List

import numpy as np
from vollib.black_scholes import black_scholes as bs_price
from vollib.black_scholes.greeks.analytical import (
    delta as bs_delta,
    gamma as bs_gamma,
    theta as bs_theta,
    vega as bs_vega,
    rho as bs_rho,
)

# Cache realized volatility computation — same for all options sharing historical data
_rv_cache: dict[int, List[float]] = {}


# ---------------------------------------------------------------------------
# Dividend adjustment helper
# ---------------------------------------------------------------------------

def _adj_spot(S: float, T: float, q: float) -> float:
    """Apply continuous dividend yield adjustment to spot price.
    S_adj = S * exp(-q * T)
    """
    if T <= 0:
        return S
    return S * math.exp(-q * T)


# ---------------------------------------------------------------------------
# IV Extraction
# ---------------------------------------------------------------------------

def extract_iv(
    S: float, K: float, T: float, r: float, q: float,
    market_premium: float, option_type: str,
    max_iter: int = 100, precision: float = 1e-6,
) -> Tuple[Optional[float], Optional[str]]:
    """Extract implied volatility from market premium using Newton-Raphson.

    Falls back to bisection if NR doesn't converge.
    Uses dividend-adjusted spot (S * e^(-qT)) for vollib compatibility.
    Returns (iv, None) on success, (None, error_description) on failure.
    """
    if T <= 0:
        return None, "合约已到期或到期日无效"
    if market_premium <= 0:
        return None, "权利金必须大于 0"
    if S <= 0 or K <= 0:
        return None, "标的价格和行权价必须大于 0"

    flag = 'c' if option_type.lower() == 'call' else 'p'
    S_adj = _adj_spot(S, T, q)

    # Check arbitrage bounds for European options
    intrinsic = max(0.0, K - S_adj) if flag == 'p' else max(0.0, S_adj - K)
    if market_premium < intrinsic * 0.99:
        return None, "权利金低于内在价值，可能违反无套利边界"

    # Newton-Raphson
    sigma = 0.30
    for _ in range(max_iter):
        try:
            price = bs_price(flag, S_adj, K, T, r, sigma)
            vega_val = bs_vega(flag, S_adj, K, T, r, sigma)
        except Exception:
            break

        diff = price - market_premium
        if abs(diff) < precision:
            return sigma, None

        if abs(vega_val) < 1e-10:
            break

        sigma = sigma - diff / vega_val
        if sigma <= 0:
            sigma = 0.01

    # Bisection fallback
    return _extract_iv_bisection(S_adj, K, T, r, market_premium, flag, precision)


def _extract_iv_bisection(
    S_adj: float, K: float, T: float, r: float,
    market_premium: float, flag: str, precision: float,
    max_iter: int = 200,
) -> Tuple[Optional[float], Optional[str]]:
    """Bisection fallback for IV extraction."""
    lo, hi = 0.001, 5.0

    try:
        price_lo = bs_price(flag, S_adj, K, T, r, lo)
        price_hi = bs_price(flag, S_adj, K, T, r, hi)
    except Exception:
        return None, "Black-Scholes 计算异常，请检查输入参数"

    if not (price_lo <= market_premium <= price_hi or
            price_hi <= market_premium <= price_lo):
        return None, "权利金超出隐含波动率范围 (0.1%-500%)，无法提取 IV"

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        try:
            price_mid = bs_price(flag, S_adj, K, T, r, mid)
        except Exception:
            return None, "BS 计算异常"

        if abs(price_mid - market_premium) < precision:
            return mid, None

        if (price_mid - market_premium) * (price_lo - market_premium) > 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0, None


# ---------------------------------------------------------------------------
# Greeks Calculation
# ---------------------------------------------------------------------------

def calculate_greeks(
    S: float, K: float, T: float, r: float, q: float,
    iv: float, option_type: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """Calculate all 5 Greeks for a European option.

    Uses dividend-adjusted spot. Delta is further adjusted by e^(-qT).
    Theta returned as DAILY value (÷365). Vega/Rho returned per 1 unit change.

    Returns ({delta, gamma, theta, vega, rho}, None) on success.
    """
    if T <= 0:
        return None, "距到期时间必须大于 0"
    if iv <= 0:
        return None, "隐含波动率必须大于 0"

    flag = 'c' if option_type.lower() == 'call' else 'p'
    S_adj = _adj_spot(S, T, q)
    adj_factor = math.exp(-q * T)

    try:
        d = bs_delta(flag, S_adj, K, T, r, iv) * adj_factor
        g = bs_gamma(flag, S_adj, K, T, r, iv) * adj_factor
        t = bs_theta(flag, S_adj, K, T, r, iv) / 365.0
        v = bs_vega(flag, S_adj, K, T, r, iv) / 100.0
        rh = bs_rho(flag, S_adj, K, T, r, iv) / 100.0
    except Exception as e:
        return None, f"Greeks 计算出错: {e}"

    return {
        "delta": d,
        "gamma": g,
        "theta": t,
        "vega": v,
        "rho": rh,
    }, None


# ---------------------------------------------------------------------------
# Historical Volatility & IV-vs-RV Percentile
# ---------------------------------------------------------------------------

def compute_realized_volatility(
    closing_prices: List[float],
    window: int = 30,
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Compute rolling realized volatility from daily closing prices.
    Cached by data identity — identical historical data returns cached result.

    Returns (list of annualized vol values, None).
    Each value = std(log returns) * sqrt(252) over the window.
    """
    if len(closing_prices) < window + 1:
        return None, f"数据不足：需要至少 {window + 1} 个交易日，当前 {len(closing_prices)} 个"

    # Cache key: hash of first+last price + length + window (fast identity check)
    cache_key = (len(closing_prices), window, closing_prices[0], closing_prices[-1])
    if cache_key in _rv_cache:
        return _rv_cache[cache_key], None

    log_returns = [
        math.log(closing_prices[i] / closing_prices[i - 1])
        for i in range(1, len(closing_prices))
    ]

    vols = []
    for i in range(len(log_returns) - window + 1):
        window_returns = log_returns[i:i + window]
        if len(window_returns) < 2:
            continue
        std = np.std(window_returns, ddof=1)
        vols.append(std * math.sqrt(252))

    _rv_cache[cache_key] = vols
    return vols, None


def compute_iv_vs_rv_percentile(
    current_iv: float,
    historical_vols: List[float],
) -> Tuple[Optional[float], Optional[str]]:
    """Where current IV sits in historical realized vol distribution.

    Percentile = % of historical RV values BELOW current IV.
    0 = IV at historic lows. 100 = IV at historic highs.

    NOTE: This compares IMPLIED vol against REALIZED vol, not historical IV.
    Label as "IV vs RV Percentile", NOT "IV Rank". Phase 2 upgrade to true IV Rank.
    """
    if not historical_vols:
        return None, "历史波动率数据为空"
    if current_iv <= 0:
        return None, "当前隐含波动率必须大于 0"

    count_below = sum(1 for v in historical_vols if v < current_iv)
    percentile = (count_below / len(historical_vols)) * 100.0
    return percentile, None


# ---------------------------------------------------------------------------
# Convenience: full pipeline for a single option
# ---------------------------------------------------------------------------

def analyze_option(
    spot: float,
    strike: float,
    expiry_date: date,
    premium: float,
    bid: float,
    ask: float,
    option_type: str,
    historical_closes: List[float],
    r: float = 0.04,
    q: float = 0.035,
    contract_multiplier: float = 50.0,
) -> dict:
    """Full analysis pipeline for one option contract.

    Returns a dict with all outputs ready for Streamlit rendering.
    Error states surfaced as 'error' and 'warnings' keys.
    """
    result = {
        "spot": spot, "strike": strike, "premium": premium,
        "bid": bid, "ask": ask, "option_type": option_type,
        "contract_multiplier": contract_multiplier,
        "error": None, "warnings": [],
    }

    today = date.today()
    days = (expiry_date - today).days
    if days <= 0:
        result["error"] = "此合约已到期"
        return result
    T = days / 365.0
    result["days_to_expiry"] = days
    result["T"] = T

    iv, iv_err = extract_iv(spot, strike, T, r, q, premium, option_type)
    if iv_err:
        result["error"] = f"IV 提取失败: {iv_err}"
        return result
    result["iv"] = iv

    greeks, greeks_err = calculate_greeks(spot, strike, T, r, q, iv, option_type)
    if greeks_err:
        result["error"] = f"Greeks 计算失败: {greeks_err}"
        return result
    result["greeks"] = greeks

    rv_vols, rv_err = compute_realized_volatility(historical_closes)
    if rv_err:
        result["warnings"].append(f"历史波动率计算: {rv_err}")
        iv_percentile = 50.0
    else:
        iv_percentile, pct_err = compute_iv_vs_rv_percentile(iv, rv_vols)
        if pct_err:
            result["warnings"].append(f"IV 分位计算: {pct_err}")
            iv_percentile = 50.0
    result["iv_percentile"] = iv_percentile

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else 0.0
    result["spread_pct"] = spread_pct

    # Warnings
    if greeks["vega"] / premium > 0.15:
        vega_impact = greeks["vega"] * contract_multiplier
        result["warnings"].append(
            f"Vega 敞口偏高: IV 下降 1% 将损失约 HK${vega_impact:.0f}"
        )
    if days < 5:
        daily_theta_hkd = abs(greeks["theta"]) * contract_multiplier
        result["warnings"].append(
            f"临近到期（{days}天），时间损耗加速。每日 Theta 约 HK${daily_theta_hkd:.1f}"
        )
    if spread_pct > 10:
        result["warnings"].append(
            f"买卖价差过大（{spread_pct:.1f}%）。此合约流动性差，退出成本高"
        )
    elif spread_pct > 5:
        result["warnings"].append(
            f"买卖价差偏大（{spread_pct:.1f}%）。"
            f"实际买入成本比中间价高约 {spread_pct/2:.1f}%"
        )

    return result

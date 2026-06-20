"""FutuOpenD API wrapper for HSI options data.

All functions return (result, error) tuples per architecture decision.
Requires FutuOpenD running on localhost:11111 with logged-in account.
"""

import socket
from typing import Optional, Tuple, List
from datetime import date, timedelta

from futu import OpenQuoteContext, RET_OK, KLType


def _check_port(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, Optional[str]]:
    """Fast pre-check: is the port reachable before attempting OpenQuoteContext?"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result != 0:
            return False, f"端口 {host}:{port} 无法连接——FutuOpenD 可能未启动"
        return True, None
    except Exception as e:
        return False, f"网络检查失败: {e}"


def connect(host: str = '127.0.0.1', port: int = 11111) -> Tuple[Optional[OpenQuoteContext], Optional[str]]:
    """Connect to FutuOpenD gateway. Returns (context, None) or (None, error).

    Performs a fast socket pre-check before attempting OpenQuoteContext,
    which avoids the long internal retry loop when FutuOpenD is down.
    """
    reachable, port_err = _check_port(host, port)
    if not reachable:
        return None, f"无法连接 FutuOpenD ({host}:{port}): {port_err}"

    try:
        ctx = OpenQuoteContext(host=host, port=port)
        return ctx, None
    except Exception as e:
        return None, f"无法连接 FutuOpenD ({host}:{port}): {e}"


def get_hsi_snapshot(ctx: OpenQuoteContext) -> Tuple[Optional[dict], Optional[str]]:
    """Get HSI index snapshot. Returns ({last_price, ...}, None) or (None, error)."""
    ret, data = ctx.get_market_snapshot(['HK.800000'])
    if ret != RET_OK:
        return None, f"获取恒指快照失败: {data}"
    row = data.iloc[0]
    return {
        "last_price": float(row['last_price']),
        "open_price": float(row.get('open_price', 0)),
        "high_price": float(row.get('high_price', 0)),
        "low_price": float(row.get('low_price', 0)),
        "update_time": str(row.get('update_time', '')),
    }, None


def get_option_chain(
    ctx: OpenQuoteContext,
    code: str = 'HK.800000',
    start_date: str = '',
    end_date: str = '',
) -> Tuple[Optional[list], Optional[str]]:
    """Get HSI option chain. Returns (list of option dicts, None) or (None, error).

    Futu API limits date range to 30 days max. If no dates given, defaults
    to today → +30 days.

    Each dict: {code, strike_price, option_type, expiry_date, bid_price, ask_price, ...}
    """
    if not start_date:
        start_date = date.today().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')

    ret, chain = ctx.get_option_chain(
        code=code,
        start=start_date,
        end=end_date,
        option_type='ALL',
        option_cond_type='ALL',
    )
    if ret != RET_OK:
        return None, f"获取期权链失败: {chain}"

    options = []
    for _, row in chain.iterrows():
        options.append({
            "code": str(row.get('code', '')),
            "strike_price": float(row.get('strike_price', 0)),
            "option_type": str(row.get('option_type', '')),
            "expiry_date": str(row.get('strike_time', '')),
            "bid_price": float(row.get('bid_price', 0)),
            "ask_price": float(row.get('ask_price', 0)),
            "last_price": float(row.get('last_price', 0)),
            "volume": int(row.get('volume', 0)),
            "open_interest": int(row.get('open_interest', 0)),
        })
    return options, None


def get_history_kline(
    ctx: OpenQuoteContext,
    code: str = 'HK.800000',
    days: int = 500,
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Get HSI historical daily closing prices.

    Returns (list of closes, None) or (None, error).
    """
    ret, kline, _ = ctx.request_history_kline(
        code=code,
        ktype=KLType.K_DAY,
        max_count=days,
    )
    if ret != RET_OK:
        return None, f"获取历史 K 线失败: {kline}"

    closes = [float(v) for v in kline['close'].tolist()]
    return closes, None


def close(ctx: OpenQuoteContext) -> None:
    """Close the FutuOpenD connection."""
    try:
        ctx.close()
    except Exception:
        pass


def get_option_expiration_dates(
    ctx: OpenQuoteContext,
    code: str = 'HK.800000',
) -> Tuple[Optional[list], Optional[str]]:
    """Get all available option expiration dates for an underlying.

    No 30-day limit — returns ALL expiries available (weekly + monthly).
    Returns (list of date strings 'YYYY-MM-DD', None) or (None, error).
    """
    ret, data = ctx.get_option_expiration_date(code)
    if ret != RET_OK:
        return None, f"获取到期日失败: {data}"

    dates = []
    for _, row in data.iterrows():
        ts = str(row.get('strike_time', ''))
        if ts:
            dates.append(ts[:10])
    return sorted(set(dates)), None

"""Phase 2: Daily option chain scanner + Telegram push.

Scans ALL available HSI option expiries, scores every contract,
and pushes a ranked report to Telegram.

Usage:
    python daily_scan.py                          # print report to stdout
    python daily_scan.py --push                   # push to Telegram
    python daily_scan.py --min-score 80           # filter threshold
    python daily_scan.py --max-spread 10          # max bid-ask spread %

Environment variables:
    TELEGRAM_BOT_TOKEN    # from @BotFather
    TELEGRAM_CHAT_ID      # your chat ID (get from @userinfobot)
"""

import json
import os
import sys
from datetime import date, datetime, timedelta

from futu import OpenQuoteContext, RET_OK, KLType
from options_engine import analyze_option
from scoring import calculate_composite_score, ScoreWeights

HSI_CODE = 'HK.800000'
CACHE_FILE = os.path.join(os.path.dirname(__file__), "hsi_history_cache.json")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

    # HSI spot
    ret, snap = ctx.get_market_snapshot([HSI_CODE])
    spot = float(snap.iloc[0]['last_price']) if ret == RET_OK else 19850.0

    # Historical closes
    ret, kline, _ = ctx.request_history_kline(code=HSI_CODE, ktype=KLType.K_DAY, max_count=500)
    if ret == RET_OK:
        closes = [float(v) for v in kline['close'].tolist()]
        with open(CACHE_FILE, 'w') as f:
            json.dump({"updated": str(date.today()), "closes": closes}, f)
    elif os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            closes = json.load(f).get("closes", [])
    else:
        closes = []

    # All expiration dates
    ret, exp_data = ctx.get_option_expiration_date(HSI_CODE)
    expiries = []
    if ret == RET_OK:
        today_str = date.today().strftime('%Y-%m-%d')
        for _, row in exp_data.iterrows():
            ts = str(row['strike_time'])[:10]
            if ts >= today_str:
                expiries.append(ts)
    expiries = sorted(set(expiries))

    return ctx, spot, closes, expiries


def scan_expiry(ctx, spot, closes, expiry, r, q, mult, weights, max_profit_mult):
    """Scan all Call+Put options for one expiry. Returns scored results."""
    ret, chain = ctx.get_option_chain(
        code=HSI_CODE, start=expiry, end=expiry,
        option_type='ALL', option_cond_type='ALL',
    )
    if ret != RET_OK:
        return []

    if len(chain) == 0:
        return []

    # Batch-fetch market snapshots for all option codes at once
    codes = list(chain['code'].tolist())
    all_snaps = {}
    batch_size = 400
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        ret_s, snap_data = ctx.get_market_snapshot(batch)
        if ret_s == RET_OK:
            for _, snap_row in snap_data.iterrows():
                all_snaps[snap_row['code']] = snap_row

    results = []
    for _, row in chain.iterrows():
        ot = str(row.get('option_type', ''))
        strike = float(row.get('strike_price', 0))
        code = str(row.get('code', ''))

        snap = all_snaps.get(code)
        if snap is None:
            continue

        bid = float(snap.get('bid_price', 0))
        ask = float(snap.get('ask_price', 0))
        last = float(snap.get('last_price', 0))
        vol = int(snap.get('volume', 0))

        if last <= 0 and bid <= 0 and ask <= 0:
            continue  # No market data at all
        if last <= 0:
            last = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        if last <= 0:
            continue
        if bid <= 0:
            bid = last * 0.95
        if ask <= 0:
            ask = last * 1.05

        # Parse expiry from option code
        code = str(row.get('code', ''))
        try:
            exp_date = datetime.strptime(code[6:12], '%y%m%d').date()
        except (ValueError, IndexError):
            exp_date = datetime.strptime(expiry, '%Y-%m-%d').date()

        analysis = analyze_option(
            spot=spot, strike=strike, expiry_date=exp_date,
            premium=last, bid=bid, ask=ask, option_type=ot,
            historical_closes=closes, r=r, q=q, contract_multiplier=mult,
        )
        if analysis["error"]:
            continue

        g = analysis["greeks"]
        score = calculate_composite_score(
            delta=g["delta"], iv_percentile=analysis["iv_percentile"],
            option_type=ot, strike=strike, premium=last,
            theta=g["theta"], vega=g["vega"],
            bid=bid, ask=ask, days_to_expiry=analysis["days_to_expiry"],
            weights=weights, max_profit_multiplier=max_profit_mult,
        )

        spread_pct = (ask - bid) / ((bid + ask) / 2) * 100 if (bid + ask) > 0 else 0
        results.append({
            "expiry": expiry,
            "type": ot,
            "strike": strike,
            "score": score.composite,
            "color": score.color,
            "iv_pct": analysis["iv"] * 100,
            "iv_pctile": analysis["iv_percentile"],
            "spread_pct": spread_pct,
            "delta": g["delta"],
            "theta": g["theta"],
            "bid": bid,
            "ask": ask,
            "premium": last,
            "volume": vol,
            "cost_hkd": ask * mult,
        })

    return results


def format_report(all_results, spot, min_score, max_spread):
    """Format results as Markdown report."""
    # Filter
    filtered = [r for r in all_results
                if r["score"] >= min_score and r["spread_pct"] <= max_spread]
    filtered.sort(key=lambda r: r["score"], reverse=True)

    lines = [
        f"📊 *恒指期权每日扫描*",
        f"📅 {date.today().strftime('%Y-%m-%d')} | 恒指 {spot:,.0f}",
        f"",
    ]

    if not filtered:
        lines.append(f"今日无符合条件的期权（值博率≥{min_score}，价差≤{max_spread}%）")
        return "\n".join(lines)

    green = [r for r in filtered if r["color"] == "green"]
    yellow = [r for r in filtered if r["color"] == "yellow"]

    lines.append(f"🎯 值博率≥{min_score}，价差≤{max_spread}%: *{len(filtered)} 张*")
    if green:
        lines.append(f"   🟢 高值博率: {len(green)} 张")
    if yellow:
        lines.append(f"   🟡 一般: {len(yellow)} 张")
    lines.append("")

    # Top 5
    lines.append("*🏆 TOP 5:*")
    for i, r in enumerate(filtered[:5], 1):
        emoji = "🟢" if r["color"] == "green" else "🟡"
        lines.append(
            f"{i}. {emoji} *{r['type']} {r['strike']:.0f}* "
            f"({r['expiry']}) — 评分 *{r['score']:.0f}*"
        )
        lines.append(
            f"   IV {r['iv_pct']:.1f}% | Δ {r['delta']:.2f} | "
            f"价差 {r['spread_pct']:.1f}% | 成本 HK${r['cost_hkd']:,.0f}"
        )

    # Summary table
    lines.append("")
    lines.append("*📋 全部符合条件的合约:*")
    lines.append(f"```")
    lines.append(f"{'类型':4s} {'行权价':>8s} {'到期':>10s} {'评分':>5s} {'IV%':>6s} {'Δ':>6s} {'价差%':>6s} {'成本':>10s}")
    lines.append("-" * 65)
    for r in filtered[:20]:
        lines.append(
            f"{r['type']:4s} {r['strike']:>8.0f} {r['expiry']:>10s} "
            f"{r['score']:>5.0f} {r['iv_pct']:>6.1f} {r['delta']:>6.2f} "
            f"{r['spread_pct']:>6.1f} {r['cost_hkd']:>10,.0f}"
        )
    if len(filtered) > 20:
        lines.append(f"... 还有 {len(filtered) - 20} 张")
    lines.append("```")
    lines.append("")
    lines.append("📝 *记录交易：* 打开 Streamlit → 交易日志 → 填入合约和盈亏")
    lines.append("   或回复此消息: `/log CALL 24000 +1200 5% 方向对了`")

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Send message via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("ERROR: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        return False

    import urllib.request
    import urllib.parse

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }).encode()

    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="HSI Option Daily Scanner")
    parser.add_argument("--push", action="store_true", help="Send report to Telegram")
    parser.add_argument("--min-score", type=float, default=80, help="Min 值博率 score (default: 80)")
    parser.add_argument("--max-spread", type=float, default=10, help="Max spread %% (default: 10)")
    parser.add_argument("--all", action="store_true", help="Show all results (no filter)")
    args = parser.parse_args()

    if args.all:
        args.min_score = 0
        args.max_spread = 200

    # Settings (match Streamlit defaults)
    r = 0.04
    q = 0.035
    mult = 50.0
    weights = ScoreWeights()
    max_profit_mult = 5.0

    print("=" * 60)
    print("HSI Option Daily Scanner")
    print("=" * 60)

    # Load data
    print("Connecting to FutuOpenD...")
    ctx, spot, closes, expiries = load_data()
    print(f"HSI Spot: {spot:,.0f}")
    print(f"Historical data: {len(closes)} days")
    print(f"Expiries to scan: {len(expiries)}")
    print()

    # Scan all expiries
    all_results = []
    for i, expiry in enumerate(expiries):
        print(f"[{i+1}/{len(expiries)}] Scanning {expiry}...", end=" ", flush=True)
        results = scan_expiry(ctx, spot, closes, expiry, r, q, mult, weights, max_profit_mult)
        all_results.extend(results)
        green_count = len([r for r in results if r["color"] == "green"])
        print(f"{len(results)} options, {green_count} green")

    ctx.close()

    # Format and output
    report = format_report(all_results, spot, args.min_score, args.max_spread)
    print()
    print(report)

    # Push to Telegram
    if args.push:
        print("\nSending to Telegram...")
        ok = send_telegram(report)
        if ok:
            print("✅ Sent!")
        else:
            print("❌ Failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            sys.exit(1)

    # Summary
    filtered = [r for r in all_results
                if r["score"] >= args.min_score and r["spread_pct"] <= args.max_spread]
    green_total = len([r for r in filtered if r["color"] == "green"])
    print(f"\nDone. {len(filtered)} contracts matched, {green_total} green.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

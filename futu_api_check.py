"""T1: futu-api connection validation script.
Run after starting FutuOpenD: python3 futu_api_check.py
"""
import sys
from futu import OpenQuoteContext, RET_OK

def check_futu_connection():
    """Verify futu-api can connect, authenticate, and retrieve HSI option chain."""
    print("=" * 50)
    print("T1: futu-api Connection Validation")
    print("=" * 50)

    # Step 1: Connect to FutuOpenD
    print("\n[1/4] Connecting to FutuOpenD (localhost:11111)...")
    try:
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    except Exception as e:
        print(f"  FAIL: Cannot connect — {e}")
        print("  → Ensure FutuOpenD is running and logged in.")
        return False
    print("  OK: Connected.")

    # Step 2: Get HSI spot price
    print("\n[2/4] Fetching HSI spot price (HK.800000)...")
    ret, data = quote_ctx.get_market_snapshot(['HK.800000'])
    if ret != RET_OK:
        print(f"  FAIL: {data}")
        quote_ctx.close()
        return False
    spot = data.iloc[0]
    print(f"  OK: HSI spot = {spot['last_price']} (updated: {spot['update_time']})")

    # Step 3: Get HSI option chain (within 30-day limit)
    from datetime import date, timedelta
    today = date.today()
    start = today.strftime('%Y-%m-%d')
    end = (today + timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"\n[3/4] Fetching HSI option chain ({start} to {end})...")
    ret, chain = quote_ctx.get_option_chain(
        code='HK.800000',
        start=start,
        end=end,
        option_type='ALL',
        option_cond_type='ALL',
    )
    if ret != RET_OK:
        print(f"  FAIL: {chain}")
        quote_ctx.close()
        return False
    print(f"  OK: {len(chain)} options found.")
    if len(chain) > 0:
        sample = chain.iloc[0]
        print(f"  Sample: {sample.get('code', 'N/A')} "
              f"strike={sample.get('strike_price', 'N/A')} "
              f"bid={sample.get('bid_price', 'N/A')} "
              f"ask={sample.get('ask_price', 'N/A')}")

    # Step 4: Get historical K-line
    print("\n[4/4] Fetching HSI historical K-line (2 years daily)...")
    ret, kline, _ = quote_ctx.request_history_kline(
        code='HK.800000',
        ktype='K_DAY',
        max_count=500,
    )
    if ret != RET_OK:
        print(f"  FAIL: {kline}")
        quote_ctx.close()
        return False
    print(f"  OK: {len(kline)} daily bars retrieved.")
    print(f"  Range: {kline['time_key'].iloc[0]} → {kline['time_key'].iloc[-1]}")

    quote_ctx.close()
    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED — futu-api is ready.")
    print("=" * 50)
    return True


if __name__ == '__main__':
    success = check_futu_connection()
    sys.exit(0 if success else 1)

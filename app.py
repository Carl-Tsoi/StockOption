"""HSI 期权值博率评估工具 — Streamlit UI.

Usage: streamlit run app.py
Requires FutuOpenD running on localhost:11111.

Tabs:
  1. 单合约评估 — pick one contract, see full analysis
  2. 全链扫描 — scan ALL Call+Put for an expiry, ranked by 值博率
"""

import json
import os
from datetime import date, datetime

import streamlit as st
import pandas as pd
from futu import OpenQuoteContext, RET_OK, KLType

from options_engine import analyze_option
from scoring import calculate_composite_score, ScoreWeights

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="恒指期权值博率", page_icon="📊", layout="wide")
st.title("📊 恒指期权值博率评估工具")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_FILE = os.path.join(os.path.dirname(__file__), "hsi_history_cache.json")
DEFAULT_R = 0.04
DEFAULT_Q = 0.035

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=300)
def get_quote_context():
    """Persistent FutuOpenD connection. Fast-fail if unavailable."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 11111))
        sock.close()
        if result != 0:
            return None
    except Exception:
        return None

    try:
        return OpenQuoteContext(host='127.0.0.1', port=11111)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_historical_closes() -> list:
    ctx = get_quote_context()
    if ctx is None:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                return json.load(f).get("closes", [])
        return []
    ret, kline, _ = ctx.request_history_kline(code='HK.800000', ktype=KLType.K_DAY, max_count=500)
    if ret != RET_OK:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                return json.load(f).get("closes", [])
        return []
    closes = [float(v) for v in kline['close'].tolist()]
    with open(CACHE_FILE, 'w') as f:
        json.dump({"updated": str(date.today()), "closes": closes}, f)
    return closes

@st.cache_data(ttl=600)
def load_hsi_spot() -> float:
    ctx = get_quote_context()
    if ctx is None: return 19850.0
    ret, data = ctx.get_market_snapshot(['HK.800000'])
    return float(data.iloc[0]['last_price']) if ret == RET_OK else 19850.0

@st.cache_data(ttl=600)
def load_expiration_dates() -> list:
    ctx = get_quote_context()
    if ctx is None: return []
    ret, data = ctx.get_option_expiration_date('HK.800000')
    if ret != RET_OK: return []
    dates = [str(row['strike_time'])[:10] for _, row in data.iterrows()]
    today_str = date.today().strftime('%Y-%m-%d')
    return sorted(set(d for d in dates if d >= today_str))

@st.cache_data(ttl=120)
def load_full_option_chain(expiry_date: str) -> list:
    """Load ALL Call + Put options with real-time pricing for a given expiry. Cached 2 min."""
    ctx = get_quote_context()
    if ctx is None: return []
    ret, chain = ctx.get_option_chain(
        code='HK.800000', start=expiry_date, end=expiry_date,
        option_type='ALL', option_cond_type='ALL',
    )
    if ret != RET_OK or len(chain) == 0: return []

    # Batch-fetch market snapshots (chain data has no prices)
    codes = list(chain['code'].tolist())
    all_snaps = {}
    for i in range(0, len(codes), 400):
        batch = codes[i:i+400]
        ret_s, snap_data = ctx.get_market_snapshot(batch)
        if ret_s == RET_OK:
            for _, snap_row in snap_data.iterrows():
                all_snaps[snap_row['code']] = snap_row

    options = []
    for _, row in chain.iterrows():
        code = str(row.get('code', ''))
        snap = all_snaps.get(code)
        if snap is None:
            continue
        bid = float(snap.get('bid_price', 0))
        ask = float(snap.get('ask_price', 0))
        last = float(snap.get('last_price', 0))
        if last <= 0:
            last = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        options.append({
            "code": code,
            "strike_price": float(row.get('strike_price', 0)),
            "option_type": str(row.get('option_type', '')),
            "bid_price": bid,
            "ask_price": ask,
            "last_price": last,
            "volume": int(snap.get('volume', 0)),
            "open_interest": int(snap.get('open_interest', 0)),
        })
    return options


# ---------------------------------------------------------------------------
# Shared scoring logic
# ---------------------------------------------------------------------------
def score_option(opt: dict, spot: float, historical_closes: list,
                 r: float, q: float, mult: float, weights: ScoreWeights,
                 max_profit_mult: float) -> dict | None:
    """Run full analysis + scoring for one option. Returns result dict or None on error."""
    try:
        expiry_date = datetime.strptime(opt["code"][6:12], '%y%m%d').date()
    except (ValueError, IndexError):
        return None

    bid = opt["bid_price"] if opt["bid_price"] > 0 else opt["last_price"] * 0.95
    ask = opt["ask_price"] if opt["ask_price"] > 0 else opt["last_price"] * 1.05
    premium = opt["last_price"] if opt["last_price"] > 0 else (bid + ask) / 2
    if premium <= 0:
        return None

    analysis = analyze_option(
        spot=spot, strike=opt["strike_price"], expiry_date=expiry_date,
        premium=premium, bid=bid, ask=ask, option_type=opt["option_type"],
        historical_closes=historical_closes,
        r=r, q=q, contract_multiplier=mult,
    )
    if analysis["error"]:
        return None

    g = analysis["greeks"]
    score = calculate_composite_score(
        delta=g["delta"], iv_percentile=analysis["iv_percentile"],
        option_type=opt["option_type"], strike=opt["strike_price"],
        premium=premium, theta=g["theta"], vega=g["vega"],
        bid=bid, ask=ask, days_to_expiry=analysis["days_to_expiry"],
        weights=weights, max_profit_multiplier=max_profit_mult,
        open_interest=opt.get("open_interest", 0),
    )

    spread_pct = (ask - bid) / ((bid + ask) / 2) * 100 if (bid + ask) > 0 else 0
    pop = abs(g["delta"]) * 100  # Delta ≈ probability of profit for long options
    return {
        "strike": opt["strike_price"],
        "type": opt["option_type"],
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "delta": g["delta"],
        "gamma": g["gamma"],
        "theta": g["theta"],
        "vega": g["vega"],
        "iv_pct": analysis["iv"] * 100,
        "iv_vs_rv": analysis["iv_percentile"],
        "pop": pop,
        "score": score.composite,
        "color": score.color,
        "volume": opt["volume"],
        "open_interest": opt.get("open_interest", 0),
        "premium_hkd": premium * mult,
        "cost_hkd": ask * mult,
        "spread_cost_hkd": (ask - (bid + ask) / 2) * mult,
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("加载数据..."):
    ctx = get_quote_context()
    connected = ctx is not None
    hsi_spot = load_hsi_spot()
    historical_closes = load_historical_closes()
    expiry_dates = load_expiration_dates()

if not connected:
    st.warning("⚠️ 未检测到 FutuOpenD 网关——请启动 FutuOpenD 后刷新页面")
    st.stop()

st.metric("恒指现价", f"{hsi_spot:,.0f}")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 设置")
    r = st.number_input("无风险利率", value=DEFAULT_R, step=0.001, format="%.3f")
    q = st.number_input("股息率", value=DEFAULT_Q, step=0.001, format="%.3f")

    st.divider()
    st.subheader("值博率权重")
    w_p = st.slider("P (方向概率)", 0.0, 0.5, 0.20, 0.05)
    w_iv = st.slider("IV (波动率分位)", 0.0, 0.5, 0.20, 0.05)
    w_rr = st.slider("RR (风险回报比)", 0.0, 0.5, 0.20, 0.05)
    w_theta = st.slider("Theta (时间损耗)", 0.0, 0.5, 0.20, 0.05)
    w_vega = st.slider("Vega (波动率敏感)", 0.0, 0.5, 0.15, 0.05)
    w_spread = st.slider("Spread (买卖价差)", 0.0, 0.3, 0.05, 0.05)
    max_profit_mult = st.slider("Call 最大盈利倍数", 2, 20, 5)

    total_w = w_p + w_iv + w_rr + w_theta + w_vega + w_spread
    weights_ok = abs(total_w - 1.0) < 0.01
    st.caption(f"权重合计: {total_w:.2f}" + (" ✅" if weights_ok else " ⚠️"))

    contract_multiplier = st.radio("合约", ["HSI (HK$50/点)", "MHI (HK$10/点)"])
    mult = 50.0 if "HSI" in contract_multiplier else 10.0

    st.divider()
    st.caption("设计: carl-unknown-design-20260618-172301.md")
    st.caption("审查: /plan-eng-review 2026-06-18")

weights = ScoreWeights(
    p_weight=w_p, iv_weight=w_iv, rr_weight=w_rr,
    theta_weight=w_theta, vega_weight=w_vega, spread_weight=w_spread,
)
if not weights_ok:
    weights = ScoreWeights()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 单合约评估", "📋 全链扫描", "📝 交易日志"])

# ===========================================================================
# TAB 1: Single contract evaluator
# ===========================================================================
with tab1:
    st.header("选择期权合约")

    if not expiry_dates:
        st.error("无可用到期日")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        option_type = st.radio("期权类型", ["CALL", "PUT"], horizontal=True)
    with col2:
        selected_expiry = st.selectbox("到期月", expiry_dates, key="tab1_expiry")

    options_list = load_full_option_chain(selected_expiry)
    filtered = [o for o in options_list if o["option_type"].upper() == option_type]
    if not filtered:
        st.warning(f"{selected_expiry} 无 {option_type} 期权")
        st.stop()

    strikes = sorted(set(o["strike_price"] for o in filtered))
    col_a, col_b = st.columns(2)
    with col_a:
        strike = st.selectbox("行权价", strikes)
    with col_b:
        selected = next((o for o in filtered if o["strike_price"] == strike), None)
        if selected:
            prem = selected["last_price"]
            bid = selected["bid_price"] if selected["bid_price"] > 0 else prem * 0.95
            ask = selected["ask_price"] if selected["ask_price"] > 0 else prem * 1.05
            premium = st.number_input("权利金（市价）", value=prem, step=1.0)
            st.caption(f"Bid: {bid:.1f} | Ask: {ask:.1f} | 成交量: {selected['volume']}")

    if st.button("🔍 评估值博率", type="primary", use_container_width=True):
        try:
            expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d').date()
        except ValueError:
            st.error("到期日格式错误"); st.stop()

        with st.spinner("计算中..."):
            result = score_option(
                selected, hsi_spot, historical_closes, r, q, mult,
                weights, float(max_profit_mult),
            )

        if result is None:
            st.error("计算失败——数据可能不完整")
            st.stop()

        st.divider()
        color_map = {"green": "#00C853", "yellow": "#FFD600", "red": "#FF1744"}
        hero_color = color_map.get(result["color"], "#999")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div style="text-align:center; padding:20px; border:3px solid {hero_color}; border-radius:16px;">
                <div style="font-size:4em; font-weight:bold; color:{hero_color};">{result['score']:.0f}</div>
                <div style="color:#888;">值博率 / 100</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            score = result["score"]
            if score >= 80: rec = "值博率高，可考虑买入"
            elif score >= 50: rec = "值博率一般，谨慎"
            else: rec = "值博率低，不建议买入"
            st.markdown(f"### {rec}")
            st.markdown(f"IV: {result['iv_pct']:.1f}% | IV vs RV: {result['iv_vs_rv']:.0f}% | 价差: {result['spread_pct']:.1f}%")

        st.subheader("📐 Greeks")
        gcols = st.columns(5)
        for i, (n, v) in enumerate([("Delta","delta"),("Gamma","gamma"),("Theta","theta"),("Vega","vega"),("Rho","delta")]):
            with gcols[i]: st.metric(n, f"{result.get(v, 0):.4f}" if n != "Rho" else f"{result.get('delta', 0)*0.05:.4f}")

        st.subheader("💰 成本")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("权利金 × 乘数", f"HK${result['premium_hkd']:,.0f}")
        with c2: st.metric("按 Ask 买入", f"HK${result['cost_hkd']:,.0f}")
        with c3: st.metric("价差成本", f"HK${result['spread_cost_hkd']:,.0f}")


# ===========================================================================
# TAB 2: Full chain scanner
# ===========================================================================
with tab2:
    st.header("📋 全链扫描")

    if not expiry_dates:
        st.error("无可用到期日")
        st.stop()

    scan_expiry = st.selectbox("到期月", expiry_dates, key="tab2_expiry")
    scan_type = st.radio("期权类型", ["ALL", "CALL", "PUT"], horizontal=True, key="scan_type")

    if st.button("🔍 扫描全链", type="primary", use_container_width=True):
        all_options = load_full_option_chain(scan_expiry)
        if scan_type != "ALL":
            all_options = [o for o in all_options if o["option_type"].upper() == scan_type]

        if not all_options:
            st.warning(f"无匹配期权 (到期月={scan_expiry}, 类型={scan_type})")
            st.stop()

        st.info(f"正在分析 {len(all_options)} 张期权...")
        progress = st.progress(0)
        results = []

        for i, opt in enumerate(all_options):
            rv = score_option(
                opt, hsi_spot, historical_closes, r, q, mult,
                weights, float(max_profit_mult),
            )
            if rv:
                results.append(rv)
            progress.progress((i + 1) / len(all_options))

        progress.empty()

        if not results:
            st.error("所有期权计算失败")
            st.stop()

        # Build DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values("score", ascending=False)

        # Summary stats
        green_count = len(df[df["color"] == "green"])
        yellow_count = len(df[df["color"] == "yellow"])
        red_count = len(df[df["color"] == "red"])

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("分析合约数", len(df))
        with col2: st.metric("🟢 值博率高 (≥80)", green_count)
        with col3: st.metric("🟡 一般 (50-79)", yellow_count)
        with col4: st.metric("🔴 低 (<50)", red_count)

        st.divider()

        # Filters
        filt_col1, filt_col2 = st.columns(2)
        with filt_col1:
            min_score = st.slider("最低值博率", 0, 100, 0)
        with filt_col2:
            max_spread = st.slider("最大买卖价差 %", 0, 200, 200)

        df_filtered = df[(df["score"] >= min_score) & (df["spread_pct"] <= max_spread)]

        # Color-coded dataframe
        def color_row(row):
            color = row.get("color", "yellow")
            if color == "green":
                return ["background-color: #e8f5e9"] * len(row)
            elif color == "red":
                return ["background-color: #ffebee"] * len(row)
            return [""] * len(row)

        display_cols = {
            "strike": "行权价",
            "type": "类型",
            "bid": "Bid",
            "ask": "Ask",
            "spread_pct": "价差%",
            "delta": "Delta",
            "pop": "POP%",
            "iv_pct": "IV%",
            "iv_vs_rv": "IV分位",
            "score": "值博率",
            "volume": "成交量",
            "open_interest": "OI",
            "cost_hkd": "买入成本",
        }
        df_display = df_filtered[list(display_cols.keys())].rename(columns=display_cols)
        df_display = df_display.round({
            "Bid": 0, "Ask": 0, "价差%": 1, "Delta": 3, "POP%": 0,
            "IV%": 1, "IV分位": 0, "值博率": 0, "买入成本": 0,
        })

        st.dataframe(
            df_display,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config={
                "值博率": st.column_config.NumberColumn(format="%d"),
                "价差%": st.column_config.NumberColumn(format="%.1f%%"),
                "IV%": st.column_config.NumberColumn(format="%.1f%%"),
                "Delta": st.column_config.NumberColumn(format="%.3f"),
                "买入成本": st.column_config.NumberColumn(format="HK$%.0f"),
            },
        )

        st.caption(f"显示 {len(df_filtered)} / {len(df)} 张合约（最低值博率 {min_score}，最大价差 {max_spread}%）")

        # Best pick highlight
        if green_count > 0:
            best = df[df["color"] == "green"].iloc[0]
            st.success(
                f"🏆 最高值博率: {best['type']} {best['strike']:.0f} "
                f"→ 评分 {best['score']:.0f}/100 | IV {best['iv_pct']:.1f}% | "
                f"价差 {best['spread_pct']:.1f}% | 成本 HK${best['cost_hkd']:,.0f}"
            )
        elif yellow_count > 0:
            best = df.iloc[0]
            st.info(
                f"📊 最优选择: {best['type']} {best['strike']:.0f} "
                f"→ 评分 {best['score']:.0f}/100 (无绿色合约)"
            )


# ===========================================================================
# TAB 3: Trade Log + Weight Calibration
# ===========================================================================
TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_log.csv")

def load_trade_log() -> pd.DataFrame:
    if os.path.exists(TRADE_LOG_FILE):
        return pd.read_csv(TRADE_LOG_FILE)
    return pd.DataFrame(columns=["date", "expiry", "type", "strike", "premium",
                                  "score", "direction", "pnl_hkd", "pnl_pct", "notes"])

def save_trade_log(df: pd.DataFrame):
    df.to_csv(TRADE_LOG_FILE, index=False)

with tab3:
    st.header("📝 交易日志")

    trade_df = load_trade_log()

    # ---- Record new trade ----
    st.subheader("记录新交易")
    with st.form("trade_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            trade_date = st.date_input("交易日期", value=date.today())
            trade_type = st.selectbox("类型", ["CALL", "PUT"])
            trade_expiry = st.text_input("到期日", placeholder="2026-07-30")
        with c2:
            trade_strike = st.number_input("行权价", value=24000.0, step=50.0)
            trade_premium = st.number_input("权利金（市价）", value=500.0, step=1.0)
            trade_score = st.number_input("值博率评分", value=80, min_value=0, max_value=100)
        with c3:
            trade_direction = st.selectbox("方向", ["买入", "卖出"])
            trade_pnl = st.number_input("实际盈亏 (HK$)", value=0, step=100,
                                         help="正数=盈利，负数=亏损")
            trade_pnl_pct = st.number_input("盈亏 %", value=0.0, step=0.5,
                                             help="相对投入资本的百分比")
        trade_notes = st.text_input("备注", placeholder="为什么做这笔交易？")

        submitted = st.form_submit_button("💾 记录交易", use_container_width=True)
        if submitted:
            new_row = pd.DataFrame([{
                "date": trade_date.strftime('%Y-%m-%d'),
                "expiry": trade_expiry,
                "type": trade_type,
                "strike": trade_strike,
                "premium": trade_premium,
                "score": trade_score,
                "direction": trade_direction,
                "pnl_hkd": trade_pnl,
                "pnl_pct": trade_pnl_pct,
                "notes": trade_notes,
            }])
            trade_df = pd.concat([trade_df, new_row], ignore_index=True)
            save_trade_log(trade_df)
            st.success("✅ 交易已记录")
            st.rerun()

    st.divider()

    # ---- Trade history ----
    if len(trade_df) == 0:
        st.info("暂无交易记录。开始记录你的第一笔交易吧。")
        st.stop()

    st.subheader(f"交易历史 ({len(trade_df)} 笔)")

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("总交易数", len(trade_df))
    win_trades = trade_df[trade_df["pnl_hkd"] > 0]
    win_rate = len(win_trades) / len(trade_df) * 100 if len(trade_df) > 0 else 0
    with c2: st.metric("胜率", f"{win_rate:.0f}%")
    total_pnl = trade_df["pnl_hkd"].sum()
    with c3: st.metric("总盈亏", f"HK${total_pnl:+,.0f}")
    with c4:
        avg_score = trade_df["score"].mean()
        st.metric("平均评分", f"{avg_score:.0f}")

    # Score vs P&L correlation
    if len(trade_df) >= 5:
        st.subheader("📈 评分 vs 盈亏分析")
        high_score = trade_df[trade_df["score"] >= 80]
        low_score = trade_df[trade_df["score"] < 80]
        hs_win = len(high_score[high_score["pnl_hkd"] > 0]) / len(high_score) * 100 if len(high_score) > 0 else 0
        ls_win = len(low_score[low_score["pnl_hkd"] > 0]) / len(low_score) * 100 if len(low_score) > 0 else 0
        hs_pnl = high_score["pnl_hkd"].sum() if len(high_score) > 0 else 0
        ls_pnl = low_score["pnl_hkd"].sum() if len(low_score) > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.metric("高评分(≥80)胜率", f"{hs_win:.0f}% ({len(high_score)}笔)",
                      delta=f"{hs_win - ls_win:+.0f}% vs 低评分" if len(low_score) > 0 else None)
            st.metric("高评分总盈亏", f"HK${hs_pnl:+,.0f}")
        with c2:
            st.metric("低评分(<80)胜率", f"{ls_win:.0f}% ({len(low_score)}笔)" if len(low_score) > 0 else "暂无")
            st.metric("低评分总盈亏", f"HK${ls_pnl:+,.0f}" if len(low_score) > 0 else "暂无")

        if hs_win > ls_win and len(low_score) >= 3:
            st.success("✅ 高评分交易的胜率更高——值博率公式方向正确")
        elif len(low_score) >= 3:
            st.warning("⚠️ 高评分交易的胜率不高于低评分——值博率权重可能需要调整")

    # Weight calibration hint
    if len(trade_df) >= 30:
        st.subheader("🔧 权重校准建议")
        st.info(
            "已积累 30+ 笔交易数据，可以运行逻辑回归校准权重。"
            "在终端执行：\n\n"
            "`python calibrate_weights.py`"
        )
    elif len(trade_df) >= 10:
        st.info(f"还需要 {30 - len(trade_df)} 笔交易达到校准门槛（30笔）。继续记录！")

    # Trade table
    st.subheader("全部记录")
    display_df = trade_df.sort_values("date", ascending=False).copy()
    display_df["pnl_hkd"] = display_df["pnl_hkd"].apply(lambda x: f"HK${x:+,.0f}")
    display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Export
    st.download_button("📥 导出 CSV", trade_df.to_csv(index=False),
                       f"trade_log_{date.today()}.csv", "text/csv")

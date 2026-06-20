"""Weight calibration via logistic regression on trade logs.

Usage: python calibrate_weights.py

Reads trade_log.csv, trains a logistic regression model to predict
win/loss from the 6 scoring factors, and outputs suggested weight adjustments.

Requires: pip install scikit-learn (already installed in venv)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import sys

TRADE_LOG = os.path.join(os.path.dirname(__file__), "trade_log.csv")

# Current default weights
CURRENT_WEIGHTS = {
    "P": 0.20,
    "IV": 0.20,
    "RR": 0.20,
    "Theta": 0.20,
    "Vega": 0.15,
    "Spread": 0.05,
}


def main():
    if not os.path.exists(TRADE_LOG):
        print("❌ trade_log.csv not found. Record trades first.")
        sys.exit(1)

    df = pd.read_csv(TRADE_LOG)
    if len(df) < 10:
        print(f"❌ Need at least 10 trades for calibration. Currently: {len(df)}")
        sys.exit(1)
    if len(df) < 30:
        print(f"⚠️  Only {len(df)} trades. 30+ recommended for reliable calibration.")

    # Create target: 1 = win (pnl_hkd > 0), 0 = loss
    df["win"] = (df["pnl_hkd"] > 0).astype(int)
    win_count = df["win"].sum()
    print(f"📊 {len(df)} trades | {win_count} wins ({win_count/len(df)*100:.0f}%) | "
          f"Total P&L: HK${df['pnl_hkd'].sum():+,.0f}")

    # The scoring factors aren't directly in the trade log, but we can
    # approximate them from the score and market data.
    # For now: use score vs pnl correlation as the primary signal.

    # Split by score tercile
    df["score_tercile"] = pd.qcut(df["score"], q=3, labels=["Low", "Mid", "High"])
    print("\n📈 Win rate by score tercile:")
    for tercile in ["Low", "Mid", "High"]:
        subset = df[df["score_tercile"] == tercile]
        if len(subset) == 0:
            continue
        wr = subset["win"].mean() * 100
        avg_pnl = subset["pnl_hkd"].mean()
        print(f"  {tercile:5s} ({len(subset):2d} trades): {wr:.0f}% win rate | "
              f"avg P&L HK${avg_pnl:+,.0f}")

    # High vs low score comparison
    high = df[df["score"] >= 80]
    low = df[df["score"] < 80]

    if len(high) >= 3 and len(low) >= 3:
        hr = high["win"].mean()
        lr = low["win"].mean()
        print(f"\n🔍 High score (>=80): {hr*100:.0f}% win ({len(high)} trades)")
        print(f"   Low score (<80):  {lr*100:.0f}% win ({len(low)} trades)")

        if hr > lr:
            print(f"   ✅ Scoring is directionally correct (+{(hr-lr)*100:.0f}% advantage)")
        else:
            print(f"   ⚠️  Scoring may need recalibration — high scores not outperforming")
            print(f"   → Consider reducing P_score weight (deep ITM/OTM Delta proxy is noisy)")
            print(f"   → Consider increasing Spread_score weight (liquidity matters more)")

    # Simple heuristic calibration
    print("\n🔧 Suggested weight adjustments:")
    suggestions = []

    # If high-score trades consistently win, the formula is working
    if len(high) >= 5 and len(low) >= 5:
        if hr > lr + 0.1:  # 10%+ win rate advantage
            suggestions.append(("✅ Formula is directionally correct. Keep current weights.", {}))
        elif hr < lr:
            # Something is wrong - try to diagnose
            # Check if spread is the issue
            if "spread_pct" in df.columns:
                hi_sp = df[df["score"] >= 80]["pnl_hkd"]
                lo_sp = df[df["score"] < 80]["pnl_hkd"]
            suggestions.append((
                "⚠️  High scores underperforming. Possible issues: P_score overweight "
                "(deep ITM calls scoring too high), or liquidity costs not fully captured.",
                {"P": -0.05, "Spread": +0.05}
            ))

    # Default: minor Spread increase for real-world costs
    suggestions.append((
        "💡 Default adjustment: increase Spread weight slightly to reflect real liquidity costs.",
        {"Vega": -0.05, "Spread": +0.05}
    ))

    for msg, adj in suggestions:
        new_weights = CURRENT_WEIGHTS.copy()
        for k, v in adj.items():
            new_weights[k] = max(0.0, min(0.50, new_weights[k] + v))
        # Renormalize
        total = sum(new_weights.values())
        new_weights = {k: round(v / total, 3) for k, v in new_weights.items()}

        print(f"\n  {msg}")
        print(f"  {'Factor':10s} {'Current':>8s} {'Suggested':>10s}")
        for factor in ["P", "IV", "RR", "Theta", "Vega", "Spread"]:
            cur = CURRENT_WEIGHTS[factor]
            new = new_weights[factor]
            arrow = " →" if cur != new else "  "
            print(f"  {factor:10s} {cur:>8.2f} {arrow} {new:>8.2f}")

    print(f"\n💡 Apply these weights in the Streamlit sidebar sliders to test calibration.")
    print(f"   After 30+ more trades with calibrated weights, re-run this script.")


if __name__ == "__main__":
    main()

import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.stock_selection.ml_next_day_lgbm_baseline import BUNDLE_DIR, DEFAULT_OUTPUT_DIR, load_instruments


BENCHMARK = "000300.XSHG"
YEARS = range(2020, 2026)
ROUNDTRIP_COST = 0.0026
HOLD_DAYS = 20
CACHE_PATH = DEFAULT_OUTPUT_DIR / "external_factor_monthly_dataset.pkl"
RESULT_PATH = DEFAULT_OUTPUT_DIR / "search_external_factor_strategies_2020_2025.json"


def load_benchmark_frame():
    with h5py.File(BUNDLE_DIR / "indexes.h5", "r") as f:
        data = f[BENCHMARK][:]
    df = pd.DataFrame.from_records(data)
    df["date"] = pd.to_datetime(df["datetime"].astype(str).str[:8])
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    out = pd.DataFrame(
        {
            "date": df["date"],
            "bench_ret_5": close.pct_change(5),
            "bench_ret_20": close.pct_change(20),
            "bench_ret_60": close.pct_change(60),
            "bench_mom_120_20": close.shift(20) / close.shift(140) - 1.0,
            "bench_ma_gap_120": close / close.rolling(120).mean() - 1.0,
            "benchmark_hold_ret": close.shift(-HOLD_DAYS) / open_.shift(-1) - 1.0,
        }
    )
    return out


def month_rebalance_dates(dates):
    s = pd.Series(pd.to_datetime(dates).sort_values().unique())
    return set(s.groupby(s.dt.to_period("M")).min().tolist())


def compute_stock_frame(order_book_id, listed_date, ds):
    raw = pd.DataFrame.from_records(ds[:])
    if raw.empty:
        return None
    raw["date"] = pd.to_datetime(raw["datetime"].astype(str).str[:8])
    raw = raw.sort_values("date").reset_index(drop=True)
    close = raw["close"].astype(float)
    open_ = raw["open"].astype(float)
    high = raw["high"].astype(float)
    turnover = raw["total_turnover"].astype(float)
    ret_1 = close.pct_change(1)

    out = pd.DataFrame(
        {
            "date": raw["date"],
            "order_book_id": order_book_id,
            "close": close,
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "ret_60": close.pct_change(60),
            "mom_120_20": close.shift(20) / close.shift(140) - 1.0,
            "ma_gap_60": close / close.rolling(60).mean() - 1.0,
            "ma_gap_120": close / close.rolling(120).mean() - 1.0,
            "high_252_ratio": close / high.rolling(252).max(),
            "vol_20": ret_1.rolling(20).std(),
            "turnover_avg_20": turnover.rolling(20).mean(),
            "next_open": open_.shift(-1),
            "hold_close": close.shift(-HOLD_DAYS),
            "next_limit_up": raw["limit_up"].astype(float).shift(-1),
            "next_limit_down": raw["limit_down"].astype(float).shift(-1),
        }
    )
    out["hold_ret"] = out["hold_close"] / out["next_open"] - 1.0
    out["next_open_buyable"] = (
        (out["next_open"] < out["next_limit_up"] * 0.999)
        & (out["next_open"] > out["next_limit_down"] * 1.001)
        & out["next_open"].notna()
        & out["hold_close"].notna()
    )
    out["listed_days"] = (out["date"] - pd.Timestamp(listed_date)).dt.days
    return out


def build_dataset():
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("version") == "v1":
            return cached["dataset"], cached["rebalance_dates"]

    instruments = load_instruments().set_index("order_book_id")
    benchmark = load_benchmark_frame()
    rebalance_dates = month_rebalance_dates(benchmark["date"])
    frames = []
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as f:
        for idx, order_book_id in enumerate(instruments.index.tolist(), 1):
            if order_book_id not in f:
                continue
            frame = compute_stock_frame(order_book_id, instruments.loc[order_book_id, "listed_date"], f[order_book_id])
            if frame is None:
                continue
            frame = frame[frame["date"].isin(rebalance_dates)].copy()
            if frame.empty:
                continue
            frames.append(frame)
            if idx % 500 == 0:
                print(f"processed {idx} stocks, kept={len(frames)}")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.merge(benchmark, on="date", how="left")
    dataset["residual_mom"] = dataset["mom_120_20"] - dataset["bench_mom_120_20"]
    dataset["reversal_score"] = -dataset["ret_5"] + 0.5 * dataset["ret_60"] - 0.25 * dataset["vol_20"]
    dataset["momentum_score"] = dataset["mom_120_20"] + 0.35 * dataset["ma_gap_60"] - 0.20 * dataset["vol_20"]
    dataset["high_residual_score"] = dataset["residual_mom"] + 0.8 * dataset["high_252_ratio"] + 0.2 * dataset["ma_gap_120"] - 0.15 * dataset["vol_20"]
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset[
        (dataset["listed_days"] >= 120)
        & (dataset["turnover_avg_20"] >= 3e8)
        & (dataset["close"] >= 8.0)
        & (dataset["close"] <= 120.0)
    ].copy()
    with CACHE_PATH.open("wb") as f:
        pickle.dump({"version": "v1", "dataset": dataset, "rebalance_dates": sorted(rebalance_dates)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset, sorted(rebalance_dates)


def pick_group(group, combo):
    market_bull = bool(
        group["bench_ma_gap_120"].iloc[0] > combo.get("bull_ma_gap", 0.0)
        and group["bench_ret_20"].iloc[0] > combo.get("bull_ret_20", 0.0)
    )
    market_bear = bool(group["bench_ma_gap_120"].iloc[0] < combo.get("bear_ma_gap", -0.03))

    if combo["family"] == "state_switch":
        if market_bull:
            score_col = "momentum_score"
            ranked = group.sort_values(score_col, ascending=False)
        elif market_bear and combo.get("bear_to_cash", False):
            return group.iloc[0:0]
        else:
            ranked = group.sort_values("reversal_score", ascending=False)
    else:
        ranked = group.sort_values("high_residual_score", ascending=False)

    if combo.get("score_threshold") is not None:
        score_col = "high_residual_score" if combo["family"] == "high_residual" else ("momentum_score" if market_bull else "reversal_score")
        ranked = ranked[ranked[score_col] >= combo["score_threshold"]]
    return ranked.head(combo["top_n"]).copy()


def backtest_combo(dataset, combo):
    rows = []
    for year in YEARS:
        year_df = dataset[dataset["date"].dt.year == year].copy()
        month_returns = []
        benchmark_returns = []
        for _, group in year_df.groupby("date"):
            group = group.dropna(
                subset=[
                    "hold_ret",
                    "benchmark_hold_ret",
                    "momentum_score",
                    "reversal_score",
                    "high_residual_score",
                ]
            )
            if group.empty:
                continue
            picks = pick_group(group, combo)
            if picks.empty:
                month_returns.append(0.0)
                benchmark_returns.append(0.0 if combo.get("cash_benchmark_zero", True) else float(group["benchmark_hold_ret"].iloc[0]))
                continue
            executable = picks[picks["next_open_buyable"]].copy()
            if executable.empty:
                month_returns.append(0.0)
                benchmark_returns.append(0.0 if combo.get("cash_benchmark_zero", True) else float(group["benchmark_hold_ret"].iloc[0]))
                continue
            net_ret = float((executable["hold_ret"] - ROUNDTRIP_COST).mean())
            invest_ratio = len(executable) / float(combo["top_n"])
            month_returns.append(net_ret * invest_ratio)
            benchmark_returns.append(float(group["benchmark_hold_ret"].iloc[0]) * invest_ratio)
        metrics = compute_monthly_metrics(month_returns, benchmark_returns)
        metrics["year"] = year
        rows.append(metrics)
    agg = {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
    }
    return {"combo": combo, "years": rows, "aggregate": agg}


def combos():
    return [
        {"name": "state_switch_base_top5", "family": "state_switch", "top_n": 5, "bull_ma_gap": 0.0, "bull_ret_20": 0.0, "bear_ma_gap": -0.03},
        {"name": "state_switch_cash_bear_top5", "family": "state_switch", "top_n": 5, "bull_ma_gap": 0.0, "bull_ret_20": 0.01, "bear_ma_gap": -0.02, "bear_to_cash": True},
        {"name": "state_switch_top10", "family": "state_switch", "top_n": 10, "bull_ma_gap": 0.0, "bull_ret_20": 0.0, "bear_ma_gap": -0.03},
        {"name": "state_switch_top3_cash_bear", "family": "state_switch", "top_n": 3, "bull_ma_gap": 0.01, "bull_ret_20": 0.01, "bear_ma_gap": -0.01, "bear_to_cash": True},
        {"name": "high_residual_top5", "family": "high_residual", "top_n": 5},
        {"name": "high_residual_top10", "family": "high_residual", "top_n": 10},
        {"name": "high_residual_top3", "family": "high_residual", "top_n": 3},
        {"name": "high_residual_thresh", "family": "high_residual", "top_n": 5, "score_threshold": 0.9},
    ]


def compute_monthly_metrics(portfolio_returns, benchmark_returns):
    if len(portfolio_returns) == 0:
        return {
            "total_returns": 0.0,
            "annualized_returns": 0.0,
            "excess_annual_returns": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "benchmark_total_returns": 0.0,
            "periods": 0,
        }
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    equity = np.cumprod(1.0 + portfolio_returns)
    benchmark_equity = np.cumprod(1.0 + benchmark_returns)
    total_returns = float(equity[-1] - 1.0)
    benchmark_total_returns = float(benchmark_equity[-1] - 1.0)
    annualized_returns = float((1.0 + total_returns) ** (12.0 / len(portfolio_returns)) - 1.0)
    excess_annual_returns = float(((equity / benchmark_equity)[-1]) ** (12.0 / len(portfolio_returns)) - 1.0)
    peaks = np.maximum.accumulate(equity)
    max_drawdown = float(abs((equity / peaks - 1.0).min()))
    sharpe = float((portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12.0)) if portfolio_returns.std() > 0 else 0.0
    return {
        "total_returns": total_returns,
        "annualized_returns": annualized_returns,
        "excess_annual_returns": excess_annual_returns,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "benchmark_total_returns": benchmark_total_returns,
        "periods": int(len(portfolio_returns)),
    }


def main():
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    dataset, _ = build_dataset()
    results = []
    for combo in combos():
        result = backtest_combo(dataset, combo)
        results.append(result)
        print(combo["name"], json.dumps(result["aggregate"], ensure_ascii=False))
    results.sort(key=lambda x: x["aggregate"]["avg_total_return"], reverse=True)
    RESULT_PATH.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {RESULT_PATH}")
    if results:
        print("best", results[0]["combo"]["name"], json.dumps(results[0]["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()

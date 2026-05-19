from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.stock_selection.ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset
from tools.search_over_50_returns import (
    RESULT_PATH as OVER50_RESULT_PATH,
    ROUNDTRIP_COST,
    YEARS,
    add_forward_returns,
    apply_market_filter,
    predict_scores,
    train_model,
)


RESULT_PATH = DEFAULT_OUTPUT_DIR / "search_top1_trend_rotation_2020_2025.json"


MODEL_SPECS = [
    {"name": "reg_ret", "model_kind": "reg", "label_mode": "ret"},
    {"name": "reg_excess", "model_kind": "reg", "label_mode": "excess"},
    {"name": "rank_ret", "model_kind": "rank", "label_mode": "ret"},
    {"name": "rank_excess", "model_kind": "rank", "label_mode": "excess"},
]

HOLD_DAYS = (3, 5, 10, 20)
MARKET_FILTERS = ("none", "bull20", "soft")

STOCK_FILTERS: dict[str, dict] = {
    "none": {},
    "trend20": {"ret_20_min": 0.0, "ma_gap_20_min": 0.0},
    "trend20_strong": {"ret_20_min": 0.05, "ma_gap_20_min": 0.0, "breakout_20_min": -0.02},
    "trend60_strong": {"ret_20_min": 0.05, "ret_60_min": 0.10, "ma_gap_20_min": 0.0, "ma_gap_60_min": 0.0},
    "breakout60": {"ret_20_min": 0.02, "ma_gap_20_min": 0.0, "breakout_60_min": -0.02},
    "low_vol_trend": {"ret_20_min": 0.03, "ma_gap_20_min": 0.0, "vol_20_max": 0.045},
    "anti_crash": {"ret_20_min": 0.03, "ma_gap_20_min": 0.0, "dd_20_min": -0.12},
}


def apply_stock_filter(group: pd.DataFrame, filter_spec: dict) -> pd.DataFrame:
    filtered = group
    if filter_spec.get("ret_20_min") is not None:
        filtered = filtered[filtered["ret_20"] >= filter_spec["ret_20_min"]]
    if filter_spec.get("ret_60_min") is not None:
        filtered = filtered[filtered["ret_60"] >= filter_spec["ret_60_min"]]
    if filter_spec.get("ma_gap_20_min") is not None:
        filtered = filtered[filtered["ma_gap_20"] >= filter_spec["ma_gap_20_min"]]
    if filter_spec.get("ma_gap_60_min") is not None:
        filtered = filtered[filtered["ma_gap_60"] >= filter_spec["ma_gap_60_min"]]
    if filter_spec.get("breakout_20_min") is not None:
        filtered = filtered[filtered["breakout_20"] >= filter_spec["breakout_20_min"]]
    if filter_spec.get("breakout_60_min") is not None:
        filtered = filtered[filtered["breakout_60"] >= filter_spec["breakout_60_min"]]
    if filter_spec.get("vol_20_max") is not None:
        filtered = filtered[filtered["vol_20"] <= filter_spec["vol_20_max"]]
    if filter_spec.get("dd_20_min") is not None:
        filtered = filtered[filtered["dd_20"] >= filter_spec["dd_20_min"]]
    return filtered


def compute_trade_metrics(period_returns: list[float], benchmark_returns: list[float], hold_days: int) -> dict:
    if not period_returns:
        return {
            "total_returns": 0.0,
            "annualized_returns": 0.0,
            "excess_annual_returns": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "benchmark_total_returns": 0.0,
            "days": 0,
        }

    pr = np.asarray(period_returns, dtype=float)
    br = np.asarray(benchmark_returns, dtype=float)
    equity = np.cumprod(1.0 + pr)
    bench = np.cumprod(1.0 + br)
    total_returns = float(equity[-1] - 1.0)
    benchmark_total_returns = float(bench[-1] - 1.0)
    approx_days = max(len(pr) * hold_days, 1)
    annualized_returns = float((1.0 + total_returns) ** (252.0 / approx_days) - 1.0) if total_returns > -1.0 else -1.0
    excess_curve = equity / np.maximum(bench, 1e-12)
    excess_annual_returns = float(excess_curve[-1] ** (252.0 / approx_days) - 1.0)
    peaks = np.maximum.accumulate(equity)
    max_drawdown = float(abs((equity / peaks - 1.0).min()))
    sharpe = float((pr.mean() / pr.std()) * math.sqrt(252.0 / hold_days)) if pr.std() > 0 else 0.0
    return {
        "total_returns": total_returns,
        "annualized_returns": annualized_returns,
        "excess_annual_returns": excess_annual_returns,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "benchmark_total_returns": benchmark_total_returns,
        "days": approx_days,
    }


def backtest_top1_rotation(
    scored: pd.DataFrame,
    hold_days: int,
    market_filter: str,
    stock_filter_name: str,
) -> dict:
    stock_filter = STOCK_FILTERS[stock_filter_name]
    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    all_dates = sorted(pd.to_datetime(scored["date"].drop_duplicates()))
    rebalance_dates = all_dates[::hold_days]
    by_date = {pd.Timestamp(date): group.copy() for date, group in scored.groupby("date")}

    period_returns = []
    benchmark_returns = []
    picks = []
    invested = 0

    for current_date in rebalance_dates:
        group = by_date[current_date]
        group = apply_market_filter(group, market_filter)
        group = apply_stock_filter(group, stock_filter)
        group = group.sort_values("pred", ascending=False)
        benchmark_hold_ret = float(group[f"benchmark_hold_{hold_days}d_ret"].iloc[0]) if not group.empty else 0.0
        pick = group[group[f"hold_{hold_days}d_executable"]].head(1).copy()
        if pick.empty:
            period_returns.append(0.0)
            benchmark_returns.append(0.0)
            picks.append({"date": current_date.date().isoformat(), "in_cash": True})
            continue

        row = pick.iloc[0]
        net_ret = float(row[f"hold_{hold_days}d_ret"] - ROUNDTRIP_COST)
        period_returns.append(net_ret)
        benchmark_returns.append(benchmark_hold_ret)
        invested += 1
        picks.append(
            {
                "date": current_date.date().isoformat(),
                "order_book_id": row["order_book_id"],
                "symbol": row["symbol"],
                "pred": float(row["pred"]),
                "hold_ret": float(row[f"hold_{hold_days}d_ret"]),
                "net_ret": net_ret,
                "ret_20": float(row["ret_20"]),
                "ret_60": float(row["ret_60"]),
                "ma_gap_20": float(row["ma_gap_20"]),
                "ma_gap_60": float(row["ma_gap_60"]),
            }
        )

    metrics = compute_trade_metrics(period_returns, benchmark_returns, hold_days)
    metrics["avg_invested_ratio"] = invested / float(len(rebalance_dates)) if rebalance_dates else 0.0
    metrics["avg_names_bought"] = metrics["avg_invested_ratio"]
    metrics["trade_count"] = invested
    metrics["rebalance_count"] = len(rebalance_dates)
    return {"metrics": metrics, "picks": picks}


def aggregate(rows: list[dict]) -> dict:
    return {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
        "avg_invested_ratio": sum(r["avg_invested_ratio"] for r in rows) / len(rows),
        "return_2024": next((r["total_returns"] for r in rows if r["year"] == 2024), None),
        "return_2025": next((r["total_returns"] for r in rows if r["year"] == 2025), None),
    }


def build_scored_cache(dataset: pd.DataFrame, feature_cols: list[str]) -> dict[tuple[str, int, int], pd.DataFrame]:
    cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    for hold_days in HOLD_DAYS:
        for spec in MODEL_SPECS:
            spec_name = spec["name"]
            label_col = f"target_{spec['label_mode']}_{hold_days}d"
            for year in YEARS:
                train_end = pd.Timestamp(f"{year - 1}-12-31")
                test_start = pd.Timestamp(f"{year}-01-01")
                test_end = pd.Timestamp(f"{year}-12-31")
                needed = [label_col, f"hold_{hold_days}d_ret", f"benchmark_hold_{hold_days}d_ret"]
                train_df = dataset[(dataset["date"] <= train_end)].dropna(subset=needed + feature_cols).copy()
                test_df = dataset[(dataset["date"] >= test_start) & (dataset["date"] <= test_end)].dropna(
                    subset=needed + feature_cols
                ).copy()
                if train_df.empty or test_df.empty:
                    continue
                model = train_model(train_df, feature_cols, label_col, spec["model_kind"])
                scored = predict_scores(model, test_df, feature_cols, spec["model_kind"])
                cache[(spec_name, hold_days, year)] = scored
                print(f"trained spec={spec_name} hold={hold_days} year={year}", flush=True)
    return cache


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = build_dataset(limit_stocks=None, cache_path=DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl")
    dataset, feature_cols = add_forward_returns(dataset)
    scored_cache = build_scored_cache(dataset, feature_cols)

    results = []
    for spec in MODEL_SPECS:
        for hold_days in HOLD_DAYS:
            for market_filter in MARKET_FILTERS:
                for stock_filter_name in STOCK_FILTERS:
                    rows = []
                    sample_picks = {}
                    for year in YEARS:
                        scored = scored_cache.get((spec["name"], hold_days, year))
                        if scored is None:
                            continue
                        backtest = backtest_top1_rotation(scored, hold_days, market_filter, stock_filter_name)
                        metrics = backtest["metrics"]
                        metrics["year"] = year
                        rows.append(metrics)
                        if year == 2025:
                            sample_picks["2025"] = backtest["picks"][:20]
                    if not rows:
                        continue
                    item = {
                        "combo": {
                            "name": f"{spec['name']}_h{hold_days}_{market_filter}_{stock_filter_name}",
                            "spec": spec,
                            "hold_days": hold_days,
                            "top_n": 1,
                            "market_filter": market_filter,
                            "stock_filter": stock_filter_name,
                        },
                        "years": rows,
                        "aggregate": aggregate(rows),
                        "sample_picks": sample_picks,
                    }
                    results.append(item)
                    agg = item["aggregate"]
                    print(
                        item["combo"]["name"],
                        f"avg={agg['avg_total_return']:.2%}",
                        f"best={agg['best_year_return']:.2%}",
                        f"worst={agg['worst_year_return']:.2%}",
                        f"mdd={agg['worst_max_drawdown']:.2%}",
                        f"invested={agg['avg_invested_ratio']:.2%}",
                        flush=True,
                    )

    results.sort(key=lambda x: (x["aggregate"]["avg_total_return"], x["aggregate"]["worst_year_return"]), reverse=True)
    payload = {
        "period": "2020-2025",
        "baseline_reference": str(OVER50_RESULT_PATH),
        "roundtrip_cost": ROUNDTRIP_COST,
        "results": results,
        "best_result": results[0] if results else None,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {RESULT_PATH}")
    if results:
        print("best_name", results[0]["combo"]["name"])
        print("best_aggregate", json.dumps(results[0]["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()

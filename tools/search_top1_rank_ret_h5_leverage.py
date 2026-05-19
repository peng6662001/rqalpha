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
from tools.search_over_50_returns import ROUNDTRIP_COST, YEARS, add_forward_returns, predict_scores, train_model
from tools.search_top1_trend_rotation import compute_trade_metrics


RESULT_PATH = DEFAULT_OUTPUT_DIR / "search_top1_rank_ret_h5_leverage_2020_2025.json"
HOLD_DAYS = 5


def collect_top1_trades(scored: pd.DataFrame) -> list[dict]:
    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    all_dates = sorted(pd.to_datetime(scored["date"].drop_duplicates()))
    rebalance_dates = all_dates[::HOLD_DAYS]
    by_date = {pd.Timestamp(date): group.copy() for date, group in scored.groupby("date")}
    trades = []

    for current_date in rebalance_dates:
        group = by_date[current_date].sort_values("pred", ascending=False)
        benchmark_ret = float(group[f"benchmark_hold_{HOLD_DAYS}d_ret"].iloc[0]) if not group.empty else 0.0
        if group.empty:
            trades.append(
                {
                    "date": current_date.date().isoformat(),
                    "in_cash": True,
                    "pred": None,
                    "second_pred": None,
                    "pred_gap": None,
                    "base_net_ret": 0.0,
                    "benchmark_ret": 0.0,
                }
            )
            continue

        top2 = group.head(2).copy()
        top1 = top2.iloc[0]
        second_pred = float(top2.iloc[1]["pred"]) if len(top2) > 1 else None
        pred_gap = float(top1["pred"] - second_pred) if second_pred is not None else None
        pick = group[group[f"hold_{HOLD_DAYS}d_executable"]].head(1).copy()
        if pick.empty:
            trades.append(
                {
                    "date": current_date.date().isoformat(),
                    "in_cash": True,
                    "pred": float(top1["pred"]),
                    "second_pred": second_pred,
                    "pred_gap": pred_gap,
                    "base_net_ret": 0.0,
                    "benchmark_ret": 0.0,
                }
            )
            continue

        row = pick.iloc[0]
        trades.append(
            {
                "date": current_date.date().isoformat(),
                "in_cash": False,
                "order_book_id": row["order_book_id"],
                "symbol": row["symbol"],
                "pred": float(row["pred"]),
                "second_pred": second_pred,
                "pred_gap": pred_gap,
                "base_hold_ret": float(row[f"hold_{HOLD_DAYS}d_ret"]),
                "base_net_ret": float(row[f"hold_{HOLD_DAYS}d_ret"] - ROUNDTRIP_COST),
                "benchmark_ret": benchmark_ret,
            }
        )
    return trades


def leverage_profiles() -> list[dict]:
    profiles: list[dict] = []
    for leverage in (1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50):
        profiles.append({"name": f"fixed_{leverage:.2f}", "kind": "fixed", "leverage": leverage})
    profiles.extend(
        [
            {"name": "pred_090_hi_1.30", "kind": "pred", "base": 1.0, "pred_cut": 0.90, "high": 1.30},
            {"name": "pred_095_hi_1.40", "kind": "pred", "base": 1.0, "pred_cut": 0.95, "high": 1.40},
            {"name": "pred_100_hi_1.50", "kind": "pred", "base": 1.0, "pred_cut": 1.00, "high": 1.50},
            {
                "name": "predgap_005_hi_1.35",
                "kind": "gap",
                "base": 1.0,
                "gap_cut": 0.05,
                "high": 1.35,
            },
            {
                "name": "predgap_010_hi_1.50",
                "kind": "gap",
                "base": 1.0,
                "gap_cut": 0.10,
                "high": 1.50,
            },
            {
                "name": "combo_pred095_gap005_hi150",
                "kind": "combo",
                "base": 1.0,
                "pred_cut": 0.95,
                "gap_cut": 0.05,
                "high": 1.50,
            },
            {
                "name": "combo_pred100_gap010_hi160",
                "kind": "combo",
                "base": 1.0,
                "pred_cut": 1.00,
                "gap_cut": 0.10,
                "high": 1.60,
            },
        ]
    )
    return profiles


def trade_leverage(profile: dict, trade: dict) -> float:
    if trade["in_cash"]:
        return 0.0
    kind = profile["kind"]
    if kind == "fixed":
        return float(profile["leverage"])
    if kind == "pred":
        return float(profile["high"] if trade["pred"] is not None and trade["pred"] >= profile["pred_cut"] else profile["base"])
    if kind == "gap":
        gap = trade["pred_gap"]
        return float(profile["high"] if gap is not None and gap >= profile["gap_cut"] else profile["base"])
    if kind == "combo":
        pred_ok = trade["pred"] is not None and trade["pred"] >= profile["pred_cut"]
        gap = trade["pred_gap"]
        gap_ok = gap is not None and gap >= profile["gap_cut"]
        return float(profile["high"] if pred_ok and gap_ok else profile["base"])
    raise ValueError(f"unsupported profile kind: {kind}")


def simulate_year(trades: list[dict], profile: dict) -> tuple[dict, list[dict]]:
    period_returns = []
    benchmark_returns = []
    details = []
    leverage_sum = 0.0
    active_count = 0
    for trade in trades:
        leverage = trade_leverage(profile, trade)
        base_net_ret = float(trade["base_net_ret"])
        leveraged_ret = leverage * base_net_ret
        period_returns.append(leveraged_ret)
        benchmark_returns.append(float(trade["benchmark_ret"]))
        if leverage > 0:
            leverage_sum += leverage
            active_count += 1
        details.append(
            {
                "date": trade["date"],
                "symbol": trade.get("symbol"),
                "pred": trade["pred"],
                "pred_gap": trade["pred_gap"],
                "base_net_ret": base_net_ret,
                "leverage": leverage,
                "leveraged_ret": leveraged_ret,
            }
        )

    metrics = compute_trade_metrics(period_returns, benchmark_returns, HOLD_DAYS)
    metrics["trade_count"] = sum(1 for trade in trades if not trade["in_cash"])
    metrics["rebalance_count"] = len(trades)
    metrics["avg_leverage_when_active"] = leverage_sum / active_count if active_count else 0.0
    metrics["avg_exposure"] = leverage_sum / len(trades) if trades else 0.0
    return metrics, details


def aggregate(rows: list[dict]) -> dict:
    return {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
        "avg_exposure": sum(r["avg_exposure"] for r in rows) / len(rows),
        "avg_leverage_when_active": sum(r["avg_leverage_when_active"] for r in rows) / len(rows),
        "return_2024": next((r["total_returns"] for r in rows if r["year"] == 2024), None),
        "return_2025": next((r["total_returns"] for r in rows if r["year"] == 2025), None),
    }


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = build_dataset(limit_stocks=None, cache_path=DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl")
    dataset, feature_cols = add_forward_returns(dataset)

    scored_by_year: dict[int, pd.DataFrame] = {}
    label_col = f"target_ret_{HOLD_DAYS}d"
    for year in YEARS:
        train_df = dataset[dataset["date"] <= pd.Timestamp(f"{year - 1}-12-31")].dropna(
            subset=[label_col, f"hold_{HOLD_DAYS}d_ret", f"benchmark_hold_{HOLD_DAYS}d_ret"] + feature_cols
        )
        test_df = dataset[
            (dataset["date"] >= pd.Timestamp(f"{year}-01-01")) & (dataset["date"] <= pd.Timestamp(f"{year}-12-31"))
        ].dropna(subset=[label_col, f"hold_{HOLD_DAYS}d_ret", f"benchmark_hold_{HOLD_DAYS}d_ret"] + feature_cols)
        model = train_model(train_df, feature_cols, label_col, "rank")
        scored_by_year[year] = predict_scores(model, test_df, feature_cols, "rank")
        print(f"trained champion year={year}", flush=True)

    trades_by_year = {year: collect_top1_trades(scored) for year, scored in scored_by_year.items()}

    results = []
    for profile in leverage_profiles():
        rows = []
        sample = {}
        for year in YEARS:
            metrics, details = simulate_year(trades_by_year[year], profile)
            metrics["year"] = year
            rows.append(metrics)
            if year == 2025:
                sample["2025"] = details[:20]
        item = {
            "profile": profile,
            "years": rows,
            "aggregate": aggregate(rows),
            "sample_trades": sample,
        }
        results.append(item)
        agg = item["aggregate"]
        print(
            profile["name"],
            f"avg={agg['avg_total_return']:.2%}",
            f"worst={agg['worst_year_return']:.2%}",
            f"best={agg['best_year_return']:.2%}",
            f"mdd={agg['worst_max_drawdown']:.2%}",
            f"avg_exp={agg['avg_exposure']:.2f}",
            flush=True,
        )

    results.sort(key=lambda x: (x["aggregate"]["avg_total_return"], -x["aggregate"]["worst_max_drawdown"]), reverse=True)
    payload = {
        "base_strategy": "rank_ret_h5_none_none",
        "hold_days": HOLD_DAYS,
        "roundtrip_cost": ROUNDTRIP_COST,
        "results": results,
        "best_result": results[0] if results else None,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {RESULT_PATH}")
    if results:
        print("best_profile", results[0]["profile"]["name"])
        print("best_aggregate", json.dumps(results[0]["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()

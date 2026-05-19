from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "strategies" / "stock_selection").resolve()))
sys.path.insert(0, str((ROOT / "tools").resolve()))

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset
from search_low_turnover_lgbm_targets import (
    BUY_COST,
    SCENARIO,
    SELL_COST,
    add_open_forward_labels,
    select_with_buffer,
    train_predict_year,
)


BEST_SPEC = {"label_horizon": 10, "label_mode": "ret", "model_kind": "rank"}
BEST_PARAMS = {"rebalance_step": 5, "top_n": 5, "hold_buffer": 20}
PROFILE_FILTERS = {
    "risk": {"min_ret20": None, "max_ret20": None, "max_vol20": 0.05, "min_dd20": None},
    "risk2": {"min_ret20": None, "max_ret20": None, "max_vol20": 0.05, "min_dd20": None},
    "risk3": {"min_ret20": None, "max_ret20": None, "max_vol20": 0.05, "min_dd20": None},
    "base": {"min_ret20": None, "max_ret20": None, "max_vol20": None, "min_dd20": None},
    "steady": {"min_ret20": None, "max_ret20": None, "max_vol20": 0.05, "min_dd20": None},
    "steady2": {"min_ret20": None, "max_ret20": None, "max_vol20": 0.05, "min_dd20": None},
}
PROFILE_EXPOSURE = {
    "base": None,
    "risk": None,
    "risk2": {
        "ma60_lo": -0.08,
        "ma60_hi": -0.02,
        "ret20_lo": -0.12,
        "ret20_hi": 0.00,
        "ret60_lo": -0.08,
        "ret60_hi": 0.08,
        "ret5_gate_min": -0.09,
        "expo_hi": 1.0,
        "expo_mid": 0.65,
        "expo_lo": 0.35,
        "expo_off": 0.0,
    },
    "risk3": {
        "ma60_lo": -0.08,
        "ma60_hi": -0.02,
        "ret20_lo": -0.10,
        "ret20_hi": 0.00,
        "ret60_lo": -0.04,
        "ret60_hi": 0.09,
        "ret5_gate_min": -0.06,
        "expo_hi": 1.0,
        "expo_mid": 0.55,
        "expo_lo": 0.0,
        "expo_off": 0.0,
    },
    "steady": {
        "ma60_lo": -0.06,
        "ma60_hi": -0.02,
        "ret20_lo": -0.10,
        "ret20_hi": 0.00,
        "ret60_lo": -0.02,
        "ret60_hi": 0.09,
        "ret5_gate_min": -0.06,
        "expo_hi": 1.0,
        "expo_mid": 0.3,
        "expo_lo": 0.0,
        "expo_off": 0.0,
    },
    "steady2": {
        "ma60_lo": -0.06,
        "ma60_hi": -0.02,
        "ret20_lo": -0.12,
        "ret20_hi": -0.02,
        "ret60_lo": -0.02,
        "ret60_hi": 0.09,
        "ret5_gate_min": -0.06,
        "expo_hi": 1.0,
        "expo_mid": 0.2,
        "expo_lo": 0.0,
        "expo_off": 0.0,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="复现低换手 LightGBM Ranker 达标策略并输出候选名单")
    parser.add_argument("--years", default="2021,2023,2025", help="逗号分隔年份，默认 2021,2023,2025")
    parser.add_argument("--predict-date", default="2025-12-31", help="输出该日期的候选名单；若无该日数据则使用此前最近交易日")
    parser.add_argument("--top", type=int, default=20, help="候选名单展示数量，默认 20")
    parser.add_argument("--profile", choices=sorted(PROFILE_FILTERS), default="risk", help="risk 为默认降回撤版；base 为原始达标版")
    parser.add_argument("--score-cache", default="", help="可选 pickle 路径；存在则复用年度打分，不存在则训练后写入")
    parser.add_argument("--output", default="", help="JSON 输出路径，默认写入 .temp")
    args = parser.parse_args()
    risk_filter = PROFILE_FILTERS[args.profile]
    exposure_rule = PROFILE_EXPOSURE[args.profile]

    years = [int(item.strip()) for item in args.years.split(",") if item.strip()]
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"

    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)
    dataset = add_open_forward_labels(dataset)

    yearly_rows = []
    score_cache_path = Path(args.score_cache) if args.score_cache else None
    scored_by_year = load_or_train_scores(dataset, feature_cols, BEST_SPEC, years, score_cache_path)
    for year in years:
        scored = scored_by_year[year]
        metrics, trace = backtest_with_trace(scored, risk_filter=risk_filter, exposure_rule=exposure_rule, **BEST_PARAMS)
        metrics["year"] = year
        metrics["drawdown"] = trace["drawdown"]
        metrics["worst_rebalance_windows"] = trace["worst_rebalance_windows"]
        yearly_rows.append(metrics)

    predict_date = pd.Timestamp(args.predict_date)
    predict_year = predict_date.year
    if predict_year not in scored_by_year:
        scored_by_year[predict_year] = train_predict_year(dataset, feature_cols, BEST_SPEC, predict_year)
    picks, used_date = latest_picks(scored_by_year[predict_year], predict_date, args.top, risk_filter)

    payload = {
        "strategy": "low_turnover_lgbm_ranker_h10_step5_top5_buffer20",
        "spec": BEST_SPEC,
        "params": BEST_PARAMS,
        "profile": args.profile,
        "risk_filter": risk_filter,
        "exposure_rule": exposure_rule,
        "cost_assumptions": cost_summary(),
        "years": yearly_rows,
        "aggregate": aggregate(yearly_rows),
        "latest": {
            "requested_date": predict_date.date().isoformat(),
            "used_date": used_date.date().isoformat(),
            "top": picks,
        },
    }

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"low_turnover_lgbm_ranker_report_{args.profile}_{'_'.join(map(str, years))}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(payload, output_path)
    return 0


def load_or_train_scores(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    spec: dict,
    years: list[int],
    score_cache_path: Path | None,
) -> dict[int, pd.DataFrame]:
    cached = None
    if score_cache_path and score_cache_path.exists():
        with score_cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("spec") != spec or cached.get("years") != years:
            cached = None

    scored_by_year: dict[int, pd.DataFrame] = {}
    if cached is not None:
        raw_scores = cached.get("scores", {})
        for year in years:
            frame = raw_scores.get(year)
            if frame is not None:
                scored_by_year[year] = frame

    missing_years = [year for year in years if year not in scored_by_year]
    for year in missing_years:
        scored_by_year[year] = train_predict_year(dataset, feature_cols, spec, year)

    if score_cache_path and missing_years:
        score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spec": spec,
            "years": years,
            "scores": {year: scored_by_year[year] for year in years},
        }
        with score_cache_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return scored_by_year


def latest_picks(scored: pd.DataFrame, predict_date: pd.Timestamp, top_n: int, risk_filter: dict) -> tuple[list[dict], pd.Timestamp]:
    eligible_dates = scored.loc[scored["date"] <= predict_date, "date"]
    if eligible_dates.empty:
        raise RuntimeError(f"没有找到 {predict_date.date().isoformat()} 或更早的打分数据")
    used_date = pd.Timestamp(eligible_dates.max())
    cols = [
        "order_book_id",
        "symbol",
        "pred",
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_60",
        "ma_gap_20",
        "breakout_60",
        "vol_20",
        "turnover_avg_20",
    ]
    picks = (
        apply_risk_filter(scored[scored["date"] == used_date], risk_filter)
        .sort_values("pred", ascending=False)
        .head(top_n)[cols]
        .copy()
    )
    for col in ["pred", "ret_5", "ret_10", "ret_20", "ret_60", "ma_gap_20", "breakout_60", "vol_20", "turnover_avg_20"]:
        picks[col] = picks[col].astype(float)
    return picks.to_dict(orient="records"), used_date


def backtest_with_trace(
    scored: pd.DataFrame,
    top_n: int,
    rebalance_step: int,
    hold_buffer: int,
    risk_filter: dict,
    exposure_rule: dict | None,
) -> tuple[dict, dict]:
    from ml_next_day_lgbm_baseline import compute_metrics

    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    all_dates = sorted(pd.to_datetime(scored["date"].drop_duplicates()))
    rebalance_dates = set(all_dates[::rebalance_step])
    by_date = {pd.Timestamp(date): group.copy() for date, group in scored.groupby("date")}

    holdings = {}
    cash = 1.0
    daily_returns = []
    benchmark_returns = []
    turnover_values = []
    equity_rows = []
    rebalance_rows = []
    equity = 1.0
    exposure_values = []

    for current_date in all_dates:
        group = by_date[current_date]
        if current_date in rebalance_dates:
            ranked = apply_risk_filter(group.sort_values("pred", ascending=False), risk_filter)
            ranked_ids = ranked["order_book_id"].tolist()
            rank_map = {order_book_id: rank + 1 for rank, order_book_id in enumerate(ranked_ids)}
            target_ids = select_with_buffer(ranked_ids, holdings.keys(), top_n, hold_buffer, rank_map)
            buyable = set(ranked.loc[ranked["next_open_buyable"], "order_book_id"])
            target_ids = [order_book_id for order_book_id in target_ids if order_book_id in buyable]
            target_exposure = exposure_for_group(group, exposure_rule)
            target_weight = target_exposure / len(target_ids) if target_ids else 0.0
            targets = {order_book_id: target_weight for order_book_id in target_ids}

            sell_turnover = sum(
                max(holdings.get(order_book_id, 0.0) - targets.get(order_book_id, 0.0), 0.0)
                for order_book_id in set(holdings) | set(targets)
            )
            buy_turnover = sum(
                max(targets.get(order_book_id, 0.0) - holdings.get(order_book_id, 0.0), 0.0)
                for order_book_id in set(holdings) | set(targets)
            )
            cost = min(sell_turnover * SELL_COST + buy_turnover * BUY_COST, 0.99)
            if targets:
                holdings = {order_book_id: weight * (1.0 - cost) for order_book_id, weight in targets.items()}
                cash = max((1.0 - target_exposure) * (1.0 - cost), 0.0)
            else:
                holdings = {}
                cash = max((1.0 - target_exposure) * (1.0 - cost), 0.0)
            turnover = sell_turnover + buy_turnover
            turnover_values.append(turnover)
            rebalance_rows.append(
                {
                    "date": current_date.date().isoformat(),
                    "turnover": float(turnover),
                    "cost": float(cost),
                    "target_exposure": float(target_exposure),
                    "holdings": enrich_holdings(ranked, holdings),
                }
            )

        next_returns = group.set_index("order_book_id")["next_ret_1"].to_dict()
        next_holdings = {}
        exposure_values.append(float(sum(holdings.values())))
        for order_book_id, weight in holdings.items():
            ret = next_returns.get(order_book_id)
            if ret is None or pd.isna(ret):
                next_holdings[order_book_id] = weight
                continue
            next_holdings[order_book_id] = weight * (1.0 + float(ret))
        gross_equity = cash + sum(next_holdings.values())
        if gross_equity <= 0:
            break
        daily_return = gross_equity - 1.0
        equity *= 1.0 + daily_return
        holdings = {order_book_id: weight / gross_equity for order_book_id, weight in next_holdings.items()}
        cash = cash / gross_equity
        daily_returns.append(float(daily_return))
        benchmark_returns.append(float(group["benchmark_next_ret_1"].iloc[0]))
        equity_rows.append({"date": current_date.date().isoformat(), "equity": float(equity), "daily_return": float(daily_return)})

    metrics = compute_metrics(daily_returns, benchmark_returns)
    if metrics is None:
        metrics = {
            "total_returns": 0.0,
            "annualized_returns": 0.0,
            "excess_annual_returns": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "benchmark_total_returns": 0.0,
            "days": 0,
        }
    metrics["avg_turnover"] = float(sum(turnover_values) / len(turnover_values)) if turnover_values else 0.0
    metrics["rebalance_count"] = len(turnover_values)
    metrics["avg_exposure"] = float(sum(exposure_values) / len(exposure_values)) if exposure_values else 0.0
    return metrics, {
        "drawdown": drawdown_summary(equity_rows),
        "worst_rebalance_windows": worst_rebalances(rebalance_rows, equity_rows),
    }


def apply_risk_filter(frame: pd.DataFrame, risk_filter: dict) -> pd.DataFrame:
    filtered = frame
    if risk_filter.get("min_ret20") is not None:
        filtered = filtered[filtered["ret_20"] >= risk_filter["min_ret20"]]
    if risk_filter.get("max_ret20") is not None:
        filtered = filtered[filtered["ret_20"] <= risk_filter["max_ret20"]]
    if risk_filter.get("max_vol20") is not None:
        filtered = filtered[filtered["vol_20"] <= risk_filter["max_vol20"]]
    if risk_filter.get("min_dd20") is not None:
        filtered = filtered[filtered["dd_20"] >= risk_filter["min_dd20"]]
    return filtered


def enrich_holdings(ranked: pd.DataFrame, holdings: dict[str, float]) -> list[dict]:
    meta = ranked.set_index("order_book_id")
    rows = []
    for order_book_id, weight in sorted(holdings.items(), key=lambda item: item[1], reverse=True):
        if order_book_id not in meta.index:
            continue
        row = meta.loc[order_book_id]
        rows.append(
            {
                "order_book_id": order_book_id,
                "symbol": row["symbol"],
                "weight": float(weight),
                "pred": float(row["pred"]),
                "ret_10": float(row["ret_10"]),
                "ret_20": float(row["ret_20"]),
            }
        )
    return rows


def exposure_for_group(group: pd.DataFrame, exposure_rule: dict | None) -> float:
    if exposure_rule is None:
        return 1.0
    row = group.iloc[0]
    if exposure_rule.get("ret5_gate_min") is not None and float(row["mkt_ret_5"]) < exposure_rule["ret5_gate_min"]:
        return exposure_rule["expo_off"]
    score = 0.0
    if float(row["mkt_ma_gap_60"]) >= exposure_rule["ma60_hi"]:
        score += 1.0
    elif float(row["mkt_ma_gap_60"]) >= exposure_rule["ma60_lo"]:
        score += 0.5
    if float(row["mkt_ret_20"]) >= exposure_rule["ret20_hi"]:
        score += 1.0
    elif float(row["mkt_ret_20"]) >= exposure_rule["ret20_lo"]:
        score += 0.5
    if float(row["mkt_ret_60"]) >= exposure_rule["ret60_hi"]:
        score += 1.0
    elif float(row["mkt_ret_60"]) >= exposure_rule["ret60_lo"]:
        score += 0.5
    if score >= 2.5:
        return exposure_rule["expo_hi"]
    if score >= 1.5:
        return exposure_rule["expo_mid"]
    if score >= 0.5:
        return exposure_rule["expo_lo"]
    return exposure_rule["expo_off"]


def drawdown_summary(equity_rows: list[dict]) -> dict:
    peak = -1.0
    peak_date = None
    worst = {"max_drawdown": 0.0, "peak_date": None, "trough_date": None, "trough_equity": None, "peak_equity": None}
    for row in equity_rows:
        equity = row["equity"]
        if equity > peak:
            peak = equity
            peak_date = row["date"]
        if peak > 0:
            drawdown = equity / peak - 1.0
            if abs(drawdown) > worst["max_drawdown"]:
                worst = {
                    "max_drawdown": float(abs(drawdown)),
                    "peak_date": peak_date,
                    "trough_date": row["date"],
                    "peak_equity": float(peak),
                    "trough_equity": float(equity),
                }
    return worst


def worst_rebalances(rebalance_rows: list[dict], equity_rows: list[dict]) -> list[dict]:
    equity_by_date = {row["date"]: row["equity"] for row in equity_rows}
    rows = []
    for idx, row in enumerate(rebalance_rows[:-1]):
        next_date = rebalance_rows[idx + 1]["date"]
        start_equity = equity_by_date.get(row["date"])
        end_equity = equity_by_date.get(next_date)
        if start_equity is None or end_equity is None:
            continue
        item = dict(row)
        item["next_rebalance_date"] = next_date
        item["window_return"] = float(end_equity / start_equity - 1.0)
        rows.append(item)
    return sorted(rows, key=lambda item: item["window_return"])[:5]


def aggregate(rows: list[dict]) -> dict:
    return {
        "min_total_return": min(row["total_returns"] for row in rows),
        "avg_total_return": sum(row["total_returns"] for row in rows) / len(rows),
        "worst_max_drawdown": max(row["max_drawdown"] for row in rows),
        "avg_turnover": sum(row.get("avg_turnover", 0.0) for row in rows) / len(rows),
        "avg_exposure": sum(row.get("avg_exposure", 0.0) for row in rows) / len(rows),
        "all_years_gt_50": all(row["total_returns"] > 0.5 for row in rows),
    }


def print_report(payload: dict, output_path: Path) -> None:
    print()
    print("低换手 LightGBM Ranker 策略报告")
    print("参数: 10日收益排序 / 每5个交易日调仓 / Top5 等权 / Top20 缓冲 / 无杠杆")
    costs = payload["cost_assumptions"]
    print(
        "成本: "
        f"买佣金 {_pct(costs['buy_commission'])} + 买滑点 {_pct(costs['buy_slippage'])} = 买入总成本 {_pct(costs['buy_total'])}; "
        f"卖佣金 {_pct(costs['sell_commission'])} + 印花税 {_pct(costs['sell_tax'])} + 卖滑点 {_pct(costs['sell_slippage'])} = 卖出总成本 {_pct(costs['sell_total'])}; "
        f"完整买卖一轮总成本 {_pct(costs['roundtrip_total'])}"
    )
    print(f"版本: {payload['profile']} / 风控: {payload['risk_filter']}")
    if payload["exposure_rule"] is not None:
        print(f"仓位规则: {payload['exposure_rule']}")
    print()
    for row in payload["years"]:
        print(
            f"{row['year']}: 收益 {_pct(row['total_returns'])}, "
            f"年化 {_pct(row['annualized_returns'])}, "
            f"最大回撤 {_pct(row['max_drawdown'])}, "
            f"Sharpe {row['sharpe']:.2f}, "
            f"平均单次调仓换手 {row['avg_turnover']:.2f}, "
            f"平均仓位 {_pct(row['avg_exposure'])}"
        )
        dd = row["drawdown"]
        print(
            f"    最大回撤区间: {dd['peak_date']} -> {dd['trough_date']} "
            f"({_pct(dd['max_drawdown'])})"
        )
    agg = payload["aggregate"]
    print()
    print(
        f"汇总: 最弱年份 {_pct(agg['min_total_return'])}, "
        f"平均收益 {_pct(agg['avg_total_return'])}, "
        f"最差回撤 {_pct(agg['worst_max_drawdown'])}, "
        f"平均仓位 {_pct(agg['avg_exposure'])}, "
        f"三年均超50%: {agg['all_years_gt_50']}"
    )
    latest = payload["latest"]
    print()
    print(f"候选名单日期: {latest['used_date']}，前 {len(latest['top'])}：")
    for idx, row in enumerate(latest["top"], 1):
        print(
            f"{idx:02d}. {row['symbol']} {row['order_book_id']} "
            f"score={row['pred']:.4f} ret10={_pct(row['ret_10'])} ret20={_pct(row['ret_20'])} "
            f"vol20={_pct(row['vol_20'])}"
        )
    print()
    print(f"完整 JSON: {output_path}")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def cost_summary() -> dict:
    return {
        **SCENARIO,
        "buy_total": SCENARIO["buy_commission"] + SCENARIO["buy_slippage"],
        "sell_total": SCENARIO["sell_commission"] + SCENARIO["sell_tax"] + SCENARIO["sell_slippage"],
        "roundtrip_total": sum(SCENARIO.values()),
    }


if __name__ == "__main__":
    raise SystemExit(main())

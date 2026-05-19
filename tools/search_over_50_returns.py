import json
import math
import pickle
import sys
from pathlib import Path

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.stock_selection.ml_next_day_lgbm_baseline import (
    BENCHMARK,
    BUNDLE_DIR,
    DEFAULT_OUTPUT_DIR,
    build_dataset,
    compute_metrics,
)


YEARS = range(2020, 2026)
HOLD_DAYS = (3, 5, 10, 20)
SCENARIO = {
    "buy_commission": 0.0003,
    "sell_commission": 0.0003,
    "sell_tax": 0.0010,
    "buy_slippage": 0.0010,
    "sell_slippage": 0.0010,
}
ROUNDTRIP_COST = sum(SCENARIO.values())
DATASET_CACHE = DEFAULT_OUTPUT_DIR / "multihorizon_return_search_dataset.pkl"
RESULT_PATH = DEFAULT_OUTPUT_DIR / "search_over_50_returns_2020_2025.json"


def build_benchmark_holds():
    with h5py.File(BUNDLE_DIR / "indexes.h5", "r") as f:
        data = f[BENCHMARK][:]
    df = pd.DataFrame.from_records(data)
    df["date"] = pd.to_datetime(df["datetime"].astype(str).str[:8])
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    out = pd.DataFrame({"date": df["date"]})
    for hold_days in HOLD_DAYS:
        out[f"benchmark_hold_{hold_days}d_ret"] = close.shift(-hold_days) / open_.shift(-1) - 1.0
    return out


def add_forward_returns(dataset):
    if DATASET_CACHE.exists():
        with DATASET_CACHE.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("version") == "v1":
            return cached["dataset"], cached["feature_cols"]

    dataset, feature_cols = dataset
    stock_ids = dataset["order_book_id"].drop_duplicates().tolist()
    merge_cols = ["order_book_id", "date", "next_open_buyable"]
    for hold_days in HOLD_DAYS:
        merge_cols.extend(
            [
                f"day_{hold_days}_date",
                f"hold_{hold_days}d_close",
                f"hold_{hold_days}d_ret",
                f"hold_{hold_days}d_executable",
            ]
        )

    forward_frames = []
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as f:
        for idx, order_book_id in enumerate(stock_ids, 1):
            if order_book_id not in f:
                continue
            raw = pd.DataFrame.from_records(f[order_book_id][:])
            if raw.empty:
                continue
            raw["date"] = pd.to_datetime(raw["datetime"].astype(str).str[:8])
            raw = raw.sort_values("date").reset_index(drop=True)
            frame = pd.DataFrame(
                {
                    "order_book_id": order_book_id,
                    "date": raw["date"],
                    "next_open": raw["open"].astype(float).shift(-1),
                    "next_close": raw["close"].astype(float).shift(-1),
                    "next_limit_up": raw["limit_up"].astype(float).shift(-1),
                    "next_limit_down": raw["limit_down"].astype(float).shift(-1),
                }
            )
            frame["next_open_buyable"] = (
                (frame["next_open"] < frame["next_limit_up"] * 0.999)
                & (frame["next_open"] > frame["next_limit_down"] * 1.001)
                & frame["next_open"].notna()
                & frame["next_close"].notna()
            )
            for hold_days in HOLD_DAYS:
                frame[f"day_{hold_days}_date"] = frame["date"].shift(-hold_days)
                frame[f"hold_{hold_days}d_close"] = raw["close"].astype(float).shift(-hold_days)
                frame[f"hold_{hold_days}d_ret"] = frame[f"hold_{hold_days}d_close"] / frame["next_open"] - 1.0
                frame[f"hold_{hold_days}d_executable"] = frame["next_open_buyable"] & frame[f"hold_{hold_days}d_close"].notna()
            forward_frames.append(frame[merge_cols])
            if idx % 500 == 0:
                print(f"forward return processed {idx} stocks")

    forward_df = pd.concat(forward_frames, ignore_index=True)
    extra_cols = [col for col in merge_cols if col not in ("order_book_id", "date") and col in dataset.columns]
    if extra_cols:
        dataset = dataset.drop(columns=extra_cols)
    dataset = dataset.merge(forward_df, on=["order_book_id", "date"], how="left")
    benchmark = build_benchmark_holds()
    dataset = dataset.merge(benchmark, on="date", how="left")

    for hold_days in HOLD_DAYS:
        dataset[f"target_ret_{hold_days}d"] = dataset[f"hold_{hold_days}d_ret"]
        dataset[f"target_excess_{hold_days}d"] = dataset[f"hold_{hold_days}d_ret"] - dataset[f"benchmark_hold_{hold_days}d_ret"]
        dataset[f"target_cls_{hold_days}d"] = (
            dataset[f"hold_{hold_days}d_ret"] > (ROUNDTRIP_COST + 0.01)
        ).astype(int)

    with DATASET_CACHE.open("wb") as f:
        pickle.dump(
            {"version": "v1", "dataset": dataset, "feature_cols": feature_cols},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return dataset, feature_cols


def train_model(train_df, feature_cols, label_col, model_kind):
    if model_kind == "reg":
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_df[feature_cols], train_df[label_col])
        return model

    if model_kind == "cls":
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_df[feature_cols], train_df[label_col])
        return model

    if model_kind == "rank":
        grouped = train_df.sort_values(["date", "order_book_id"]).copy()
        grouped["_rank_label"] = (
            grouped.groupby("date")[label_col]
            .rank(method="first", pct=True)
            .mul(9)
            .round()
            .astype(int)
        )
        group_sizes = grouped.groupby("date").size().tolist()
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(grouped[feature_cols], grouped["_rank_label"], group=group_sizes)
        return model

    raise ValueError(f"unsupported model kind: {model_kind}")


def predict_scores(model, test_df, feature_cols, model_kind):
    scored = test_df.copy()
    if model_kind == "cls":
        scored["pred"] = model.predict_proba(scored[feature_cols])[:, 1]
    else:
        scored["pred"] = model.predict(scored[feature_cols])
    return scored


def apply_market_filter(group, market_filter):
    if market_filter == "none":
        return group
    if market_filter == "bull20":
        return group[group["mkt_ma_gap_20"] > 0]
    if market_filter == "soft":
        return group[(group["mkt_ma_gap_20"] > -0.01) & (group["mkt_ret_20"] > -0.05)]
    raise ValueError(f"unsupported market filter: {market_filter}")


def backtest(scored, hold_days, top_n, pred_threshold, market_filter):
    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    bucket_events = {}
    benchmark_events = {}
    invested_ratios = []
    names_bought = []

    for _, group in scored.groupby("date"):
        eligible = apply_market_filter(group, market_filter)
        if pred_threshold is not None:
            eligible = eligible[eligible["pred"] >= pred_threshold]
        picks = eligible.head(top_n).copy()
        bench_hold_ret = float(group[f"benchmark_hold_{hold_days}d_ret"].iloc[0])
        if math.isnan(bench_hold_ret):
            continue

        if picks.empty:
            invested_ratio = 0.0
            names = 0
            daily_piece = 0.0
        else:
            executable = picks[picks[f"hold_{hold_days}d_executable"]].copy()
            invested_ratio = len(executable) / float(top_n)
            names = len(executable)
            sleeve_allocation = invested_ratio / float(hold_days)
            if executable.empty:
                daily_piece = 0.0
            else:
                net_ret = executable[f"hold_{hold_days}d_ret"] - ROUNDTRIP_COST
                daily_piece = float(net_ret.mean() * sleeve_allocation / hold_days)

        invested_ratios.append(invested_ratio)
        names_bought.append(names)

        if picks.empty:
            daily_piece = 0.0

        for offset in range(hold_days):
            day = pd.Timestamp(group["date"].iloc[0]) + pd.offsets.BDay(offset + 1)
            bucket_events.setdefault(day, []).append(daily_piece)
            benchmark_events.setdefault(day, []).append(bench_hold_ret / float(hold_days * hold_days))

    all_dates = sorted(set(bucket_events.keys()) | set(benchmark_events.keys()))
    portfolio_daily = [sum(bucket_events.get(d, [])) for d in all_dates]
    benchmark_daily = [sum(benchmark_events.get(d, [])) for d in all_dates]
    metrics = compute_metrics(portfolio_daily, benchmark_daily)
    if metrics is None:
        return None
    metrics["avg_invested_ratio"] = sum(invested_ratios) / len(invested_ratios) if invested_ratios else 0.0
    metrics["avg_names_bought"] = sum(names_bought) / len(names_bought) if names_bought else 0.0
    metrics["hold_days"] = hold_days
    metrics["top_n"] = top_n
    metrics["pred_threshold"] = pred_threshold
    metrics["market_filter"] = market_filter
    return metrics


def aggregate(rows):
    return {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
        "avg_invested_ratio": sum(r["avg_invested_ratio"] for r in rows) / len(rows),
        "avg_names_bought": sum(r["avg_names_bought"] for r in rows) / len(rows),
        "return_2024": next((r["total_returns"] for r in rows if r["year"] == 2024), None),
        "return_2025": next((r["total_returns"] for r in rows if r["year"] == 2025), None),
    }


def run_combo(dataset, feature_cols, combo):
    hold_days = combo["hold_days"]
    label_mode = combo["label_mode"]
    model_kind = combo["model_kind"]
    top_n = combo["top_n"]
    pred_threshold = combo["pred_threshold"]
    market_filter = combo["market_filter"]

    label_col = f"target_{label_mode}_{hold_days}d"
    rows = []
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
        model = train_model(train_df, feature_cols, label_col, model_kind)
        scored = predict_scores(model, test_df, feature_cols, model_kind)
        metrics = backtest(scored, hold_days, top_n, pred_threshold, market_filter)
        if metrics is None:
            continue
        metrics["year"] = year
        rows.append(metrics)
    if not rows:
        return None
    return {
        "combo": combo,
        "years": rows,
        "aggregate": aggregate(rows),
    }


def main():
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = build_dataset(limit_stocks=None, cache_path=DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl")
    dataset, feature_cols = add_forward_returns(dataset)

    combos = [
        {"name": "reg_ret_3_top1", "model_kind": "reg", "label_mode": "ret", "hold_days": 3, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_ret_3_top3", "model_kind": "reg", "label_mode": "ret", "hold_days": 3, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_excess_3_top1", "model_kind": "reg", "label_mode": "excess", "hold_days": 3, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_excess_3_top3", "model_kind": "reg", "label_mode": "excess", "hold_days": 3, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "rank_ret_3_top1", "model_kind": "rank", "label_mode": "ret", "hold_days": 3, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "rank_ret_3_top3", "model_kind": "rank", "label_mode": "ret", "hold_days": 3, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_ret_5_top1", "model_kind": "reg", "label_mode": "ret", "hold_days": 5, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_ret_5_top3", "model_kind": "reg", "label_mode": "ret", "hold_days": 5, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_excess_5_top1", "model_kind": "reg", "label_mode": "excess", "hold_days": 5, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_excess_5_top3", "model_kind": "reg", "label_mode": "excess", "hold_days": 5, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "rank_ret_5_top1", "model_kind": "rank", "label_mode": "ret", "hold_days": 5, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "rank_ret_5_top3", "model_kind": "rank", "label_mode": "ret", "hold_days": 5, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_ret_5_top1_bull20", "model_kind": "reg", "label_mode": "ret", "hold_days": 5, "top_n": 1, "pred_threshold": None, "market_filter": "bull20"},
        {"name": "reg_ret_5_top3_bull20", "model_kind": "reg", "label_mode": "ret", "hold_days": 5, "top_n": 3, "pred_threshold": None, "market_filter": "bull20"},
        {"name": "reg_excess_5_top1_soft", "model_kind": "reg", "label_mode": "excess", "hold_days": 5, "top_n": 1, "pred_threshold": None, "market_filter": "soft"},
        {"name": "rank_ret_5_top1_soft", "model_kind": "rank", "label_mode": "ret", "hold_days": 5, "top_n": 1, "pred_threshold": None, "market_filter": "soft"},
        {"name": "reg_ret_10_top1", "model_kind": "reg", "label_mode": "ret", "hold_days": 10, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_ret_10_top3", "model_kind": "reg", "label_mode": "ret", "hold_days": 10, "top_n": 3, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_excess_10_top1", "model_kind": "reg", "label_mode": "excess", "hold_days": 10, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "rank_ret_10_top1", "model_kind": "rank", "label_mode": "ret", "hold_days": 10, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "cls_10_top1", "model_kind": "cls", "label_mode": "cls", "hold_days": 10, "top_n": 1, "pred_threshold": 0.55, "market_filter": "none"},
        {"name": "cls_10_top3", "model_kind": "cls", "label_mode": "cls", "hold_days": 10, "top_n": 3, "pred_threshold": 0.55, "market_filter": "none"},
        {"name": "reg_ret_20_top1", "model_kind": "reg", "label_mode": "ret", "hold_days": 20, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
        {"name": "reg_excess_20_top1", "model_kind": "reg", "label_mode": "excess", "hold_days": 20, "top_n": 1, "pred_threshold": None, "market_filter": "none"},
    ]

    results = []
    for combo in combos:
        print(f"running {combo['name']}")
        try:
            result = run_combo(dataset, feature_cols, combo)
        except Exception as exc:
            result = {
                "combo": combo,
                "error": repr(exc),
            }
            print(combo["name"], "ERROR", repr(exc))
            results.append(result)
            RESULT_PATH.write_text(
                json.dumps(
                    {
                        "period": "2020-2025",
                        "assumptions": SCENARIO,
                        "results": results,
                        "best_result": None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        if result is None:
            continue
        results.append(result)
        if "aggregate" in result:
            print(combo["name"], json.dumps(result["aggregate"], ensure_ascii=False))
        RESULT_PATH.write_text(
            json.dumps(
                {
                    "period": "2020-2025",
                    "assumptions": SCENARIO,
                    "results": results,
                    "best_result": None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    ranked_results = [r for r in results if "aggregate" in r]
    ranked_results.sort(key=lambda x: x["aggregate"]["avg_total_return"], reverse=True)
    payload = {
        "period": "2020-2025",
        "assumptions": SCENARIO,
        "results": results,
        "best_result": ranked_results[0] if ranked_results else None,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {RESULT_PATH}")
    if ranked_results:
        print("best", json.dumps(ranked_results[0]["aggregate"], ensure_ascii=False))
        print("best_name", ranked_results[0]["combo"]["name"])


if __name__ == "__main__":
    main()

import json
import h5py

import pandas as pd

from ml_next_day_lgbm_baseline import BUNDLE_DIR, DEFAULT_OUTPUT_DIR, build_dataset, compute_metrics, lgb


TOP_N = 5
YEARS = range(2020, 2026)
SCENARIO = {
    "buy_commission": 0.0003,
    "sell_commission": 0.0003,
    "sell_tax": 0.0010,
    "buy_slippage": 0.0010,
    "sell_slippage": 0.0010,
}
ROUNDTRIP_COST = sum(SCENARIO.values())
HOLD_PERIODS = (2, 3)


def train_predict_year(dataset, feature_cols, year):
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")

    labeled = dataset.dropna(subset=["next_ret_1", "benchmark_next_ret_1"]).copy()
    train_df = labeled[labeled["date"] <= train_end].copy()
    test_df = labeled[(labeled["date"] >= test_start) & (labeled["date"] <= test_end)].copy()
    if train_df.empty or test_df.empty:
        return None

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
    model.fit(train_df[feature_cols], train_df["next_ret_1"])
    test_df["pred"] = model.predict(test_df[feature_cols])
    return test_df


def add_forward_returns(dataset):
    merge_cols = ["order_book_id", "date", "next_open", "next_open_buyable"]
    for hold_days in HOLD_PERIODS:
        merge_cols.extend([f"hold_{hold_days}d_close", f"hold_{hold_days}d_ret", f"hold_{hold_days}d_executable"])

    forward_frames = []
    stock_ids = dataset["order_book_id"].drop_duplicates().tolist()
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as f:
        for order_book_id in stock_ids:
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
            for hold_days in HOLD_PERIODS:
                frame[f"hold_{hold_days}d_close"] = raw["close"].astype(float).shift(-hold_days)
                frame[f"hold_{hold_days}d_ret"] = frame[f"hold_{hold_days}d_close"] / frame["next_open"] - 1.0
                frame[f"hold_{hold_days}d_executable"] = frame["next_open_buyable"] & frame[f"hold_{hold_days}d_close"].notna()
            forward_frames.append(frame[merge_cols])

    forward_df = pd.concat(forward_frames, ignore_index=True)
    dataset = dataset.drop(
        columns=[
            col
            for col in ["next_open", "next_open_buyable"] + [f"hold_{d}d_close" for d in HOLD_PERIODS] + [f"hold_{d}d_ret" for d in HOLD_PERIODS] + [f"hold_{d}d_executable" for d in HOLD_PERIODS]
            if col in dataset.columns
        ]
    )
    return dataset.merge(forward_df, on=["order_book_id", "date"], how="left")


def load_benchmark_close_series(dataset):
    benchmark_by_date = dataset.groupby("date")["benchmark_next_ret_1"].first().sort_index()
    bench_frame = pd.DataFrame({"date": benchmark_by_date.index, "benchmark_next_ret_1": benchmark_by_date.values})
    return bench_frame


def build_benchmark_hold_returns(bench_frame, hold_days):
    bench = bench_frame.sort_values("date").copy()
    bench["synthetic_close"] = (1.0 + bench["benchmark_next_ret_1"]).cumprod()
    bench["buy_open_proxy"] = bench["synthetic_close"].shift(0)
    bench["sell_close_proxy"] = bench["synthetic_close"].shift(hold_days)
    bench[f"benchmark_hold_{hold_days}d_ret"] = bench["sell_close_proxy"] / bench["buy_open_proxy"] - 1.0
    return bench[["date", f"benchmark_hold_{hold_days}d_ret"]]


def backtest_hold_period_with_costs(scored, hold_days, benchmark_ret_map, top_n=TOP_N):
    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    groups = list(scored.groupby("date"))
    bucket_events = {}
    benchmark_events = {}
    invested_ratios = []
    names_bought = []

    for _, group in groups:
        picks = group.head(top_n).copy()
        if picks.empty:
            continue
        executable = picks[picks[f"hold_{hold_days}d_executable"]].copy()
        invested_ratio = len(executable) / float(top_n)
        invested_ratios.append(invested_ratio)
        names_bought.append(len(executable))

        event_date = pd.Timestamp(group["date"].iloc[0])
        bench_hold_ret = benchmark_ret_map.get(event_date)
        if bench_hold_ret is None or pd.isna(bench_hold_ret):
            continue

        if executable.empty:
            daily_piece = 0.0
        else:
            net_hold_ret = executable[f"hold_{hold_days}d_ret"] - ROUNDTRIP_COST
            daily_piece = float(net_hold_ret.mean() * invested_ratio / hold_days)

        for offset in range(hold_days):
            day = event_date + pd.offsets.BDay(offset + 1)
            bucket_events.setdefault(day, []).append(daily_piece)
            benchmark_events.setdefault(day, []).append(float(bench_hold_ret) / hold_days)

    all_dates = sorted(set(bucket_events.keys()) | set(benchmark_events.keys()))
    portfolio_daily = [sum(bucket_events.get(d, [])) for d in all_dates]
    benchmark_daily = [sum(benchmark_events.get(d, [])) for d in all_dates]
    metrics = compute_metrics(portfolio_daily, benchmark_daily)
    metrics["avg_invested_ratio"] = sum(invested_ratios) / len(invested_ratios) if invested_ratios else 0.0
    metrics["avg_names_bought"] = sum(names_bought) / len(names_bought) if names_bought else 0.0
    metrics["hold_days"] = hold_days
    return metrics


def aggregate_metrics(rows):
    return {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
        "avg_invested_ratio": sum(r["avg_invested_ratio"] for r in rows) / len(rows),
        "avg_names_bought": sum(r["avg_names_bought"] for r in rows) / len(rows),
    }


def main():
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)
    dataset = add_forward_returns(dataset)
    bench_frame = load_benchmark_close_series(dataset)

    scored_by_year = {}
    for year in YEARS:
        scored = train_predict_year(dataset, feature_cols, year)
        if scored is not None:
            scored_by_year[year] = scored

    summary = {
        "strategy": "ml_next_day_lgbm_hold_period_cost_audit",
        "period": "2020-2025",
        "assumptions": SCENARIO,
        "top_n": TOP_N,
        "holds": {},
    }

    for hold_days in HOLD_PERIODS:
        bench_hold = build_benchmark_hold_returns(bench_frame, hold_days)
        benchmark_ret_map = dict(zip(pd.to_datetime(bench_hold["date"]), bench_hold[f"benchmark_hold_{hold_days}d_ret"]))
        rows = []
        for year, scored in scored_by_year.items():
            metrics = backtest_hold_period_with_costs(scored.copy(), hold_days, benchmark_ret_map, top_n=TOP_N)
            metrics["year"] = year
            rows.append(metrics)
            print(f"hold_{hold_days}d", year, json.dumps(metrics, ensure_ascii=False))
        summary["holds"][str(hold_days)] = {
            "years": rows,
            "aggregate": aggregate_metrics(rows),
        }
        print(f"hold_{hold_days}d", json.dumps(summary["holds"][str(hold_days)]["aggregate"], ensure_ascii=False))

    out = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_hold_period_cost_audit_2020_2025.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

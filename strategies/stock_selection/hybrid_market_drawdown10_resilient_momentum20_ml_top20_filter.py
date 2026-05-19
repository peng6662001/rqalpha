import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, r"F:\Stock\rqalpha")

from strategies.stock_selection import ml_next_day_lgbm_baseline as ml


OUTPUT_DIR = Path(r"F:\Stock\rqalpha\.temp")
BENCHMARK = "000300.XSHG"


def compute_metrics(portfolio_returns, benchmark_returns):
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    equity = np.cumprod(1.0 + portfolio_returns)
    benchmark_equity = np.cumprod(1.0 + benchmark_returns)
    total_returns = float(equity[-1] - 1.0)
    benchmark_total_returns = float(benchmark_equity[-1] - 1.0)
    annualized_returns = float((1.0 + total_returns) ** (252.0 / len(portfolio_returns)) - 1.0)
    excess_annual_returns = float(((equity / benchmark_equity)[-1]) ** (252.0 / len(portfolio_returns)) - 1.0)
    peaks = np.maximum.accumulate(equity)
    max_drawdown = float(abs((equity / peaks - 1.0).min()))
    sharpe = float((portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252.0)) if portfolio_returns.std() > 0 else 0.0
    return {
        "total_returns": total_returns,
        "annualized_returns": annualized_returns,
        "excess_annual_returns": excess_annual_returns,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "benchmark_total_returns": benchmark_total_returns,
        "days": int(len(portfolio_returns)),
    }


def prepare_dataset():
    dataset, feature_cols = ml.build_dataset(limit_stocks=None, cache_path=None)
    index_path = ml.BUNDLE_DIR / "indexes.h5"
    with h5py.File(index_path, "r") as f:
        bench_data = f[BENCHMARK][:]
    benchmark = pd.DataFrame.from_records(bench_data)
    benchmark["date"] = pd.to_datetime(benchmark["datetime"].astype(str).str[:8])
    benchmark["benchmark_close"] = benchmark["close"].astype(float)
    benchmark["benchmark_next_ret_1"] = (
        benchmark["benchmark_close"].shift(-1) / benchmark["open"].astype(float).shift(-1) - 1.0
    )
    benchmark = benchmark[["date", "benchmark_close", "benchmark_next_ret_1"]].copy()
    return dataset, feature_cols, benchmark


def add_signal_columns(dataset):
    ds = dataset.copy()
    ds["resilient_ok"] = (
        (ds["close"] >= 5.0)
        & (ds["close"] <= 120.0)
        & (ds["turnover_avg_20"] >= 2e8)
        & (ds["ret_20"] >= 0.03)
    )
    ds = ds.sort_values(["order_book_id", "date"]).copy()
    ds["rolling_max_120"] = ds.groupby("order_book_id")["close"].transform(lambda s: s.rolling(120, min_periods=120).max())
    ds["stock_drawdown_120"] = 1.0 - ds["close"] / ds["rolling_max_120"]
    ds["resilient_ok"] = ds["resilient_ok"] & (ds["stock_drawdown_120"] <= 0.10)
    return ds


def run_hybrid_year(dataset, feature_cols, benchmark_df, year):
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")

    labeled = dataset.dropna(subset=["next_ret_1", "benchmark_next_ret_1"]).copy()
    train_df = labeled[labeled["date"] <= train_end].copy()
    test_df = labeled[(labeled["date"] >= test_start) & (labeled["date"] <= test_end)].copy()
    if train_df.empty or test_df.empty:
        return None

    model = ml.lgb.LGBMRegressor(
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
    test_df["hybrid_score"] = (
        test_df["ret_20"] + 0.35 * test_df["ret_10"] + 0.10 * test_df["ret_60"] - 0.20 * test_df["stock_drawdown_120"]
    )

    bench_year = benchmark_df[(benchmark_df["date"] >= test_start) & (benchmark_df["date"] <= test_end)].copy()
    bench_year["market_peak"] = bench_year["benchmark_close"].cummax()
    bench_year["market_drawdown"] = 1.0 - bench_year["benchmark_close"] / bench_year["market_peak"]
    bench_map = bench_year.set_index("date")

    positions = {}
    portfolio_daily = []
    benchmark_daily = []

    for date in sorted(test_df["date"].unique()):
        day_rows = test_df[test_df["date"] == date].copy()
        if date not in bench_map.index:
            continue
        market_drawdown = float(bench_map.loc[date, "market_drawdown"])

        for order_book_id, pos in list(positions.items()):
            row = day_rows[day_rows["order_book_id"] == order_book_id]
            if row.empty:
                continue
            close_price = float(row["close"].iloc[0])
            profit = close_price / pos["entry_price"] - 1.0
            if profit >= 0.50 or market_drawdown <= 0.06:
                del positions[order_book_id]

        if market_drawdown >= 0.10:
            candidates = day_rows[day_rows["resilient_ok"]].copy()
            if not candidates.empty:
                pred_cutoff = candidates["pred"].quantile(0.8)
                candidates = candidates[candidates["pred"] >= pred_cutoff].copy()
            candidates = candidates.sort_values(["hybrid_score", "pred"], ascending=False)
            selected = candidates.head(5)
            selected_ids = set(selected["order_book_id"].tolist())
            for oid in list(positions.keys()):
                if oid not in selected_ids:
                    del positions[oid]
            for _, row in selected.iterrows():
                oid = row["order_book_id"]
                if oid not in positions:
                    positions[oid] = {"entry_price": float(row["close"])}

        weight = 1.0 / 5.0
        realized = 0.0
        for oid in positions:
            row = day_rows[day_rows["order_book_id"] == oid]
            if row.empty:
                continue
            realized += float(row["next_ret_1"].iloc[0]) * weight
        portfolio_daily.append(realized)
        benchmark_daily.append(float(day_rows["benchmark_next_ret_1"].iloc[0]))

    metrics = compute_metrics(portfolio_daily, benchmark_daily)
    metrics["year"] = year
    return metrics


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset, feature_cols, benchmark_df = prepare_dataset()
    dataset = add_signal_columns(dataset)

    rows = []
    for year in range(2020, 2026):
        metrics = run_hybrid_year(dataset, feature_cols, benchmark_df, year)
        if metrics is None:
            continue
        rows.append(metrics)
        print(year, json.dumps(metrics, ensure_ascii=False))

    summary = {
        "years": rows,
        "aggregate": {
            "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
            "worst_year_return": min(r["total_returns"] for r in rows),
            "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
            "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
            "return_2024": next(r["total_returns"] for r in rows if r["year"] == 2024),
            "return_2025": next(r["total_returns"] for r in rows if r["year"] == 2025),
        },
    }
    out = OUTPUT_DIR / "hybrid_market_drawdown10_resilient_momentum20_ml_top20_filter_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

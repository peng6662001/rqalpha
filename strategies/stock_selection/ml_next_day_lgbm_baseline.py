import argparse
import json
import os
import pickle
from pathlib import Path

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd


BUNDLE_DIR = Path(os.path.expanduser(r"~/.rqalpha/bundle"))
DEFAULT_OUTPUT_DIR = Path.cwd() / ".temp"
BENCHMARK = "000300.XSHG"
CACHE_VERSION = "v3_open_close_exec"


def load_instruments():
    path = BUNDLE_DIR / "instruments.pk"
    with path.open("rb") as f:
        instruments = pickle.load(f)
    df = pd.DataFrame(instruments)
    df = df[(df["type"] == "CS") & (df["exchange"].isin(["XSHG", "XSHE"]))]
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce", format="%Y-%m-%d")
    df["de_listed_date"] = pd.to_datetime(df["de_listed_date"], errors="coerce", format="%Y-%m-%d")
    df.loc[df["de_listed_date"].isna(), "de_listed_date"] = pd.Timestamp("2099-12-31")
    return df[["order_book_id", "symbol", "listed_date", "de_listed_date"]]


def load_benchmark_features():
    path = BUNDLE_DIR / "indexes.h5"
    with h5py.File(path, "r") as f:
        data = f[BENCHMARK][:]
    df = pd.DataFrame.from_records(data)
    df["date"] = pd.to_datetime(df["datetime"].astype(str).str[:8])
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    df["benchmark_next_ret_1"] = close.shift(-1) / open_.shift(-1) - 1.0
    df["mkt_ret_5"] = close.pct_change(5)
    df["mkt_ret_20"] = close.pct_change(20)
    df["mkt_ret_60"] = close.pct_change(60)
    df["mkt_ma_gap_20"] = close / close.rolling(20).mean() - 1.0
    df["mkt_ma_gap_60"] = close / close.rolling(60).mean() - 1.0
    return df[
        ["date", "benchmark_next_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_ret_60", "mkt_ma_gap_20", "mkt_ma_gap_60"]
    ]


def compute_limit_hits(df, window=10):
    prev_close = df["prev_close"].replace(0, np.nan)
    up_hit = ((df["high"] / prev_close - 1.0) >= 0.097).astype(float)
    down_hit = ((df["low"] / prev_close - 1.0) <= -0.097).astype(float)
    return (up_hit + down_hit).rolling(window).sum()


def compute_features(order_book_id, symbol, listed_date, ds):
    df = pd.DataFrame.from_records(ds[:])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["datetime"].astype(str).str[:8])
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    turnover = df["total_turnover"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    ret_1 = close.pct_change(1)

    out = pd.DataFrame(
        {
            "date": df["date"],
            "order_book_id": order_book_id,
            "symbol": symbol,
            "open": df["open"].astype(float),
            "close": close,
            "limit_up": df["limit_up"].astype(float),
            "limit_down": df["limit_down"].astype(float),
            "ret_1": ret_1,
            "ret_5": close.pct_change(5),
            "ret_10": close.pct_change(10),
            "ret_20": close.pct_change(20),
            "ret_60": close.pct_change(60),
            "ma_gap_5": close / close.rolling(5).mean() - 1.0,
            "ma_gap_10": close / close.rolling(10).mean() - 1.0,
            "ma_gap_20": close / close.rolling(20).mean() - 1.0,
            "ma_gap_60": close / close.rolling(60).mean() - 1.0,
            "vol_5": ret_1.rolling(5).std(),
            "vol_20": ret_1.rolling(20).std(),
            "turnover_avg_5": turnover.rolling(5).mean(),
            "turnover_avg_20": turnover.rolling(20).mean(),
            "turnover_ratio_5_20": turnover.rolling(5).mean() / turnover.rolling(20).mean() - 1.0,
            "volume_ratio_5_20": volume.rolling(5).mean() / volume.rolling(20).mean() - 1.0,
            "breakout_20": (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min()),
            "breakout_60": (close - low.rolling(60).min()) / (high.rolling(60).max() - low.rolling(60).min()),
            "dd_20": close / close.rolling(20).max() - 1.0,
            "dd_60": close / close.rolling(60).max() - 1.0,
            "limit_hits_10": compute_limit_hits(df, 10),
        }
    )
    out["listed_days"] = (out["date"] - pd.Timestamp(listed_date)).dt.days
    out["next_open"] = out["open"].shift(-1)
    out["next_close"] = out["close"].shift(-1)
    out["next_limit_up"] = out["limit_up"].shift(-1)
    out["next_limit_down"] = out["limit_down"].shift(-1)
    out["next_open_to_close_ret"] = out["next_close"] / out["next_open"] - 1.0
    out["next_open_buyable"] = (
        (out["next_open"] < out["next_limit_up"] * 0.999)
        & (out["next_open"] > out["next_limit_down"] * 1.001)
        & out["next_open"].notna()
        & out["next_close"].notna()
    )
    out["next_ret_1"] = close.shift(-1) / out["open"].shift(-1) - 1.0
    out["target_up"] = (out["next_ret_1"] > 0).astype(int)

    filtered = out[
        (out["listed_days"] >= 120)
        & (out["turnover_avg_20"] >= 3e8)
        & (out["close"] >= 8.0)
        & (out["close"] <= 120.0)
        & (out["limit_hits_10"] <= 1.0)
    ].copy()
    return filtered


def build_dataset(limit_stocks=None, cache_path=None):
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        with cache_file.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("cache_version") == CACHE_VERSION:
            return cached["dataset"], cached["feature_cols"]

    instruments = load_instruments()
    benchmark_features = load_benchmark_features()
    path = BUNDLE_DIR / "stocks.h5"
    frames = []
    with h5py.File(path, "r") as f:
        order_book_ids = instruments["order_book_id"].tolist()
        if limit_stocks:
            order_book_ids = order_book_ids[:limit_stocks]
        meta = instruments.set_index("order_book_id")
        for idx, order_book_id in enumerate(order_book_ids, 1):
            if order_book_id not in f:
                continue
            row = meta.loc[order_book_id]
            frame = compute_features(order_book_id, row["symbol"], row["listed_date"], f[order_book_id])
            if frame is None or frame.empty:
                continue
            frames.append(frame)
            if idx % 500 == 0:
                print(f"processed {idx} stocks, frames={len(frames)}")
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.merge(benchmark_features, on="date", how="left")
    feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_60",
        "ma_gap_5",
        "ma_gap_10",
        "ma_gap_20",
        "ma_gap_60",
        "vol_5",
        "vol_20",
        "turnover_avg_20",
        "turnover_ratio_5_20",
        "volume_ratio_5_20",
        "breakout_20",
        "breakout_60",
        "dd_20",
        "dd_60",
        "mkt_ret_5",
        "mkt_ret_20",
        "mkt_ret_60",
        "mkt_ma_gap_20",
        "mkt_ma_gap_60",
    ]
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=feature_cols)
    if cache_file:
        try:
            with cache_file.open("wb") as f:
                pickle.dump(
                    {"cache_version": CACHE_VERSION, "dataset": dataset, "feature_cols": feature_cols},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except OSError as exc:
            print(f"cache write skipped: {exc}")
    return dataset, feature_cols


def compute_metrics(portfolio_returns, benchmark_returns):
    if len(portfolio_returns) == 0:
        return None
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


def backtest_topn(scored, top_n=5):
    daily = []
    benchmark = []
    scored = scored.sort_values(["date", "pred"], ascending=[True, False])
    for date, group in scored.groupby("date"):
        picks = group.head(top_n).copy()
        if picks.empty:
            continue
        executable = picks[picks["next_open_buyable"]].copy()
        if executable.empty:
            daily.append(0.0)
            benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]))
            continue
        cash_weight = 1.0 - len(executable) / float(top_n)
        realized = executable["next_open_to_close_ret"].mean() * (1.0 - cash_weight)
        daily.append(float(realized))
        benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]))
    return compute_metrics(daily, benchmark)


def train_and_backtest(dataset, feature_cols, train_end, test_start, test_end):
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    labeled = dataset.dropna(subset=["next_ret_1", "benchmark_next_ret_1"]).copy()
    train_df = labeled[labeled["date"] <= train_end].copy()
    test_df = labeled[(labeled["date"] >= test_start) & (labeled["date"] <= test_end)].copy()

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
    metrics = backtest_topn(test_df, top_n=5)
    return model, test_df, metrics


def score_latest(model, dataset, feature_cols, predict_date):
    predict_date = pd.Timestamp(predict_date)
    latest = dataset[dataset["date"] == predict_date].copy()
    used_date = predict_date
    if latest.empty:
        eligible_dates = dataset.loc[dataset["date"] <= predict_date, "date"]
        if eligible_dates.empty:
            return None, None
        used_date = eligible_dates.max()
        latest = dataset[dataset["date"] == used_date].copy()
    if latest.empty:
        return None, None
    latest["pred"] = model.predict(latest[feature_cols])
    latest = latest.sort_values("pred", ascending=False)
    cols = ["order_book_id", "symbol", "pred", "next_ret_1", "ret_20", "ret_60", "ma_gap_20", "breakout_60", "vol_20"]
    return latest[cols], used_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--predict-date", default="2026-04-30")
    parser.add_argument("--limit-stocks", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=args.limit_stocks, cache_path=cache_path)
    model, scored_test, metrics = train_and_backtest(
        dataset,
        feature_cols,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    latest, used_date = score_latest(model, dataset, feature_cols, args.predict_date)
    if latest is None:
        raise RuntimeError("No eligible rows found for latest scoring date or any earlier date.")
    latest_output = latest.head(10).copy()
    latest_output["pred"] = latest_output["pred"].astype(float)
    latest_output["next_ret_1"] = latest_output["next_ret_1"].astype(float)

    suffix = f"_{args.tag}" if args.tag else ""
    model_path = DEFAULT_OUTPUT_DIR / f"ml_next_day_lgbm_baseline{suffix}.pkl"
    metrics_path = DEFAULT_OUTPUT_DIR / f"ml_next_day_lgbm_baseline_metrics{suffix}.json"
    latest_path = DEFAULT_OUTPUT_DIR / f"ml_next_day_lgbm_picks_{used_date.date().isoformat()}{suffix}.json"

    with model_path.open("wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(latest_output.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print("metrics:", json.dumps(metrics, ensure_ascii=False))
    print("latest_scored_date:", used_date.date().isoformat())
    print("latest_top5:")
    print(latest_output.head(5).to_string(index=False))
    print(f"saved model: {model_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved picks: {latest_path}")


if __name__ == "__main__":
    main()

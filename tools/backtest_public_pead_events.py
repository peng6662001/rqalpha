from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.stock_selection.ml_next_day_lgbm_baseline import BUNDLE_DIR, BENCHMARK, compute_metrics


EVENTS_PATH = ROOT / ".temp" / "public_pead" / "public_pead_events_2020_2025.csv"
OUTPUT_PATH = ROOT / ".temp" / "public_pead" / "public_pead_backtest_2020_2025.json"
HORIZONS = [1, 5, 20, 60]
ROUNDTRIP_COST = 0.0026


def normalize_public_order_book_id(order_book_id: str) -> str:
    text = str(order_book_id)
    if text.endswith(".SZ"):
        return text[:-3] + ".XSHE"
    if text.endswith(".SH"):
        return text[:-3] + ".XSHG"
    return text


def load_trading_calendar() -> pd.DatetimeIndex:
    with h5py.File(BUNDLE_DIR / "indexes.h5", "r") as f:
        data = pd.DataFrame.from_records(f[BENCHMARK][:])
    dates = pd.to_datetime(data["datetime"].astype(str).str[:8])
    return pd.DatetimeIndex(dates.sort_values().unique())


def load_benchmark_returns(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    with h5py.File(BUNDLE_DIR / "indexes.h5", "r") as f:
        df = pd.DataFrame.from_records(f[BENCHMARK][:])
    df["date"] = pd.to_datetime(df["datetime"].astype(str).str[:8])
    df = df.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    for h in HORIZONS:
        out[f"bench_ret_{h}"] = close.shift(-h) / open_.shift(-1) - 1.0
    out["daily_close_ret"] = close.pct_change().fillna(0.0)
    return out.set_index("date").reindex(calendar)


def assign_score_buckets(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
    out["bundle_order_book_id"] = out["order_book_id"].map(normalize_public_order_book_id)
    out["score_bucket"] = np.nan
    out["score_rank_label"] = "nan"
    valid = out["surprise_score"].notna()
    for family, family_df in out[valid].groupby("event_family"):
        idx = family_df.index
        ranks = family_df["surprise_score"].rank(method="first", pct=True)
        bucket = np.ceil(ranks * 5).clip(1, 5).astype(int)
        out.loc[idx, "score_bucket"] = bucket
        out.loc[idx, "score_rank_label"] = bucket.map({1: "q1", 2: "q2", 3: "q3", 4: "q4", 5: "q5"})
    return out


def load_stock_forward_returns(order_book_ids: list[str]) -> pd.DataFrame:
    rows = []
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as f:
        for i, order_book_id in enumerate(order_book_ids, 1):
            if order_book_id not in f:
                continue
            raw = pd.DataFrame.from_records(f[order_book_id][:])
            if raw.empty:
                continue
            raw["date"] = pd.to_datetime(raw["datetime"].astype(str).str[:8])
            raw = raw.sort_values("date").reset_index(drop=True)
            open_ = raw["open"].astype(float)
            close = raw["close"].astype(float)
            limit_up = raw["limit_up"].astype(float)
            limit_down = raw["limit_down"].astype(float)
            next_open = open_.shift(-1)
            next_limit_up = limit_up.shift(-1)
            next_limit_down = limit_down.shift(-1)
            out = pd.DataFrame({"bundle_order_book_id": order_book_id, "anchor_date": raw["date"]})
            out["next_open_buyable"] = (
                (next_open < next_limit_up * 0.999)
                & (next_open > next_limit_down * 1.001)
                & next_open.notna()
            )
            for h in HORIZONS:
                out[f"ret_{h}"] = close.shift(-h) / next_open - 1.0
            rows.append(out)
            if i % 1000 == 0:
                print(f"loaded forward returns for {i} stocks")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_event_return_frame(events: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    order_book_ids = sorted(events["bundle_order_book_id"].dropna().astype(str).unique())
    stock_returns = load_stock_forward_returns(order_book_ids)
    merged = events.merge(stock_returns, on=["bundle_order_book_id", "anchor_date"], how="left")
    benchmark_frame = benchmark.reset_index().rename(columns={"index": "anchor_date", "date": "anchor_date"})
    merged = merged.merge(benchmark_frame, on="anchor_date", how="left")
    for h in HORIZONS:
        merged[f"excess_ret_{h}"] = merged[f"ret_{h}"] - merged[f"bench_ret_{h}"]
    return merged


def summarize_event_study(df: pd.DataFrame) -> dict:
    out = {}
    grouped = df.groupby(["event_family", "score_rank_label"], dropna=False)
    for (family, label), group in grouped:
        stats = {"count": int(len(group))}
        for h in HORIZONS:
            stats[f"avg_ret_{h}"] = float(group[f"ret_{h}"].dropna().mean()) if group[f"ret_{h}"].notna().any() else None
            stats[f"avg_excess_ret_{h}"] = float(group[f"excess_ret_{h}"].dropna().mean()) if group[f"excess_ret_{h}"].notna().any() else None
            stats[f"hit_rate_{h}"] = float((group[f"ret_{h}"] > 0).mean()) if group[f"ret_{h}"].notna().any() else None
        out[f"{family}|{label}"] = stats
    return out


def summarize_forecast_subtypes(df: pd.DataFrame) -> dict:
    forecasts = df[df["event_family"] == "earnings_forecast"].copy()
    forecasts["forecast_type"] = forecasts["forecast_type"].fillna("unknown").astype(str)
    out = {}
    for forecast_type, group in forecasts.groupby("forecast_type"):
        stats = {"count": int(len(group))}
        for h in HORIZONS:
            ret = group[f"ret_{h}"].dropna()
            ex = group[f"excess_ret_{h}"].dropna()
            stats[f"avg_ret_{h}"] = float(ret.mean()) if len(ret) else None
            stats[f"avg_excess_ret_{h}"] = float(ex.mean()) if len(ex) else None
            stats[f"hit_rate_{h}"] = float((ret > 0).mean()) if len(ret) else None
        q5 = group[group["score_rank_label"] == "q5"]
        q1 = group[group["score_rank_label"] == "q1"]
        subtype_spread = {}
        for h in HORIZONS:
            q5r = q5[f"ret_{h}"].dropna()
            q1r = q1[f"ret_{h}"].dropna()
            q5x = q5[f"excess_ret_{h}"].dropna()
            q1x = q1[f"excess_ret_{h}"].dropna()
            subtype_spread[f"h{h}"] = {
                "q5_minus_q1_ret": float(q5r.mean() - q1r.mean()) if len(q5r) and len(q1r) else None,
                "q5_minus_q1_excess_ret": float(q5x.mean() - q1x.mean()) if len(q5x) and len(q1x) else None,
                "q5_count": int(len(q5r)),
                "q1_count": int(len(q1r)),
            }
        stats["q5_q1_spread"] = subtype_spread
        out[forecast_type] = stats
    return out


def aggregate_daily_portfolio(signal_events: pd.DataFrame, benchmark: pd.DataFrame, horizon: int) -> dict:
    active = signal_events[
        signal_events["next_open_buyable"].fillna(False)
        & signal_events[f"ret_{horizon}"].notna()
    ].copy()
    if active.empty:
        return {"metrics": None, "daily_rows": 0}

    active["entry_date"] = active["event_date"].map(lambda d: next_trading_date(benchmark.index, d))
    active["exit_date"] = active["entry_date"].map(lambda d: nth_trading_date(benchmark.index, d, horizon - 1))
    active = active[active["entry_date"].notna() & active["exit_date"].notna()].copy()
    if active.empty:
        return {"metrics": None, "daily_rows": 0}

    bclose = benchmark["daily_close_ret"].fillna(0.0)
    contributions = []
    for _, row in active.iterrows():
        entry = row["entry_date"]
        exit_ = row["exit_date"]
        if pd.isna(entry) or pd.isna(exit_):
            continue
        path = bclose.loc[(bclose.index >= entry) & (bclose.index <= exit_)]
        if len(path) != horizon:
            continue
        total_ret = float(row[f"ret_{horizon}"] - ROUNDTRIP_COST)
        leg_daily = distribute_total_return(total_ret, horizon)
        for date, daily_ret in zip(path.index, leg_daily):
            contributions.append({"date": date, "daily_ret": daily_ret})
    if not contributions:
        return {"metrics": None, "daily_rows": 0}
    cdf = pd.DataFrame(contributions)
    portfolio = cdf.groupby("date")["daily_ret"].mean().sort_index()
    aligned_bench = bclose.reindex(portfolio.index).fillna(0.0)
    metrics = compute_metrics(portfolio.values, aligned_bench.values)
    return {
        "metrics": metrics,
        "daily_rows": int(len(portfolio)),
        "avg_concurrent_positions": float(cdf.groupby("date").size().mean()),
    }


def distribute_total_return(total_ret: float, horizon: int) -> np.ndarray:
    return np.repeat((1.0 + total_ret) ** (1.0 / horizon) - 1.0, horizon)


def next_trading_date(calendar: pd.DatetimeIndex, date: pd.Timestamp):
    idx = calendar.searchsorted(pd.Timestamp(date), side="right")
    if idx >= len(calendar):
        return pd.NaT
    return calendar[idx]


def prev_or_same_trading_date(calendar: pd.DatetimeIndex, date: pd.Timestamp):
    idx = calendar.searchsorted(pd.Timestamp(date), side="right") - 1
    if idx < 0:
        return pd.NaT
    return calendar[idx]


def nth_trading_date(calendar: pd.DatetimeIndex, start_date: pd.Timestamp, offset: int):
    idx = calendar.searchsorted(pd.Timestamp(start_date), side="left")
    target = idx + offset
    if target >= len(calendar):
        return pd.NaT
    return calendar[target]


def build_long_short_summary(df: pd.DataFrame) -> dict:
    out = {}
    for family, group in df.groupby("event_family"):
        fam = {}
        for h in HORIZONS:
            q5 = group[group["score_rank_label"] == "q5"][f"ret_{h}"].dropna()
            q1 = group[group["score_rank_label"] == "q1"][f"ret_{h}"].dropna()
            q5x = group[group["score_rank_label"] == "q5"][f"excess_ret_{h}"].dropna()
            q1x = group[group["score_rank_label"] == "q1"][f"excess_ret_{h}"].dropna()
            fam[f"h{h}"] = {
                "q5_minus_q1_ret": float(q5.mean() - q1.mean()) if len(q5) and len(q1) else None,
                "q5_minus_q1_excess_ret": float(q5x.mean() - q1x.mean()) if len(q5x) and len(q1x) else None,
                "q5_count": int(len(q5)),
                "q1_count": int(len(q1)),
            }
        out[family] = fam
    return out


def build_portfolio_backtests(df: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    out = {}
    for family, group in df.groupby("event_family"):
        fam = {}
        for h in HORIZONS:
            fam[f"q5_h{h}"] = aggregate_daily_portfolio(group[group["score_rank_label"] == "q5"], benchmark, h)
            fam[f"q1_h{h}"] = aggregate_daily_portfolio(group[group["score_rank_label"] == "q1"], benchmark, h)
        out[family] = fam
    return out


def main() -> None:
    events = pd.read_csv(EVENTS_PATH, low_memory=False)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    events["report_date"] = pd.to_datetime(events["report_date"], errors="coerce").dt.normalize()
    calendar = load_trading_calendar()
    events = events[(events["report_date"] >= pd.Timestamp("2020-01-01")) & (events["report_date"] <= pd.Timestamp("2025-12-31"))].copy()
    events = assign_score_buckets(events)
    events["anchor_date"] = events["event_date"].map(lambda d: prev_or_same_trading_date(calendar, d))
    events = events[events["anchor_date"].notna()].copy()
    benchmark = load_benchmark_returns(calendar)
    frame = build_event_return_frame(events, benchmark)

    summary = {
        "event_rows": int(len(frame)),
        "tradable_ratio": float(frame["next_open_buyable"].fillna(False).mean()),
        "families": frame["event_family"].value_counts().to_dict(),
        "event_study": summarize_event_study(frame),
        "forecast_subtypes": summarize_forecast_subtypes(frame),
        "long_short_summary": build_long_short_summary(frame),
        "portfolio_backtests": build_portfolio_backtests(frame, benchmark),
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT_PATH}")
    print(json.dumps({
        "event_rows": summary["event_rows"],
        "tradable_ratio": summary["tradable_ratio"],
        "long_short_summary": summary["long_short_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_public_pead_events import (
    EVENTS_PATH,
    HORIZONS,
    ROUNDTRIP_COST,
    aggregate_daily_portfolio,
    assign_score_buckets,
    build_event_return_frame,
    load_benchmark_returns,
    load_trading_calendar,
    next_trading_date,
    nth_trading_date,
    prev_or_same_trading_date,
)
from strategies.stock_selection.ml_next_day_lgbm_baseline import compute_metrics


OUTPUT_PATH = ROOT / ".temp" / "public_pead" / "public_pead_forecast_refined_2020_2025.json"
FOCUS_HORIZONS = [5, 20, 60]
COHORTS = {
    "blend_core": ["预增", "略增", "扭亏", "减亏"],
    "blend_growth_turn": ["预增", "略增", "扭亏"],
    "blend_recovery": ["扭亏", "减亏", "续亏"],
    "single_预增": ["预增"],
    "single_略增": ["略增"],
    "single_扭亏": ["扭亏"],
    "single_减亏": ["减亏"],
    "single_续亏": ["续亏"],
    "single_续盈": ["续盈"],
}


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENTS_PATH, low_memory=False)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    events["report_date"] = pd.to_datetime(events["report_date"], errors="coerce").dt.normalize()
    events = events[
        (events["event_family"] == "earnings_forecast")
        & (events["report_date"] >= pd.Timestamp("2020-01-01"))
        & (events["report_date"] <= pd.Timestamp("2025-12-31"))
    ].copy()
    calendar = load_trading_calendar()
    events = assign_score_buckets(events)
    events["anchor_date"] = events["event_date"].map(lambda d: prev_or_same_trading_date(calendar, d))
    events = events[events["anchor_date"].notna()].copy()
    benchmark = load_benchmark_returns(calendar)
    frame = build_event_return_frame(events, benchmark)
    return frame, benchmark


def aggregate_daily_long_short(signal_events: pd.DataFrame, benchmark: pd.DataFrame, horizon: int) -> dict:
    long_leg = build_daily_leg(signal_events[signal_events["score_rank_label"] == "q5"], benchmark, horizon, side=1.0)
    short_leg = build_daily_leg(signal_events[signal_events["score_rank_label"] == "q1"], benchmark, horizon, side=-1.0)
    if long_leg.empty and short_leg.empty:
        return {"metrics": None, "daily_rows": 0}
    merged = pd.concat([long_leg, short_leg], ignore_index=True)
    portfolio = merged.groupby("date")["daily_ret"].mean().sort_index()
    aligned_bench = benchmark["daily_close_ret"].reindex(portfolio.index).fillna(0.0)
    metrics = compute_metrics(portfolio.values, aligned_bench.values)
    return {
        "metrics": metrics,
        "daily_rows": int(len(portfolio)),
        "avg_concurrent_positions": float(merged.groupby("date").size().mean()),
    }


def build_daily_leg(signal_events: pd.DataFrame, benchmark: pd.DataFrame, horizon: int, side: float) -> pd.DataFrame:
    active = signal_events[
        signal_events["next_open_buyable"].fillna(False)
        & signal_events[f"ret_{horizon}"].notna()
    ].copy()
    if active.empty:
        return pd.DataFrame(columns=["date", "daily_ret"])
    active["entry_date"] = active["event_date"].map(lambda d: next_trading_date(benchmark.index, d))
    active["exit_date"] = active["entry_date"].map(lambda d: nth_trading_date(benchmark.index, d, horizon - 1))
    active = active[active["entry_date"].notna() & active["exit_date"].notna()].copy()
    if active.empty:
        return pd.DataFrame(columns=["date", "daily_ret"])

    rows = []
    for _, row in active.iterrows():
        total_ret = float(row[f"ret_{horizon}"])
        if side < 0:
            total_ret = -total_ret
        total_ret -= ROUNDTRIP_COST
        daily_ret = np.repeat(safe_daily_return(total_ret, horizon), horizon)
        dates = benchmark.index[(benchmark.index >= row["entry_date"]) & (benchmark.index <= row["exit_date"])]
        if len(dates) != horizon:
            continue
        for date, ret in zip(dates, daily_ret):
            rows.append({"date": date, "daily_ret": ret})
    return pd.DataFrame(rows)


def safe_daily_return(total_ret: float, horizon: int) -> float:
    clipped = max(total_ret, -0.95)
    return float(np.expm1(np.log1p(clipped) / horizon))


def summarize_cohort_events(group: pd.DataFrame) -> dict:
    out = {"count": int(len(group))}
    for h in HORIZONS:
        ex = group[f"excess_ret_{h}"].dropna()
        out[f"avg_excess_ret_{h}"] = float(ex.mean()) if len(ex) else None
    return out


def run() -> dict:
    frame, benchmark = load_frame()
    summary = {
        "source_rows": int(len(frame)),
        "cohorts": {},
    }
    for name, forecast_types in COHORTS.items():
        group = frame[frame["forecast_type"].isin(forecast_types)].copy()
        cohort = {
            "forecast_types": forecast_types,
            "event_summary": summarize_cohort_events(group),
            "q5_portfolios": {},
            "long_short_portfolios": {},
        }
        for h in FOCUS_HORIZONS:
            cohort["q5_portfolios"][f"h{h}"] = aggregate_daily_portfolio(group[group["score_rank_label"] == "q5"], benchmark, h)
            cohort["long_short_portfolios"][f"h{h}"] = aggregate_daily_long_short(group, benchmark, h)
        summary["cohorts"][name] = cohort
    return summary


def main() -> None:
    summary = run()
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT_PATH}")
    compact = {}
    for name, cohort in summary["cohorts"].items():
        compact[name] = {}
        for h in FOCUS_HORIZONS:
            q5 = cohort["q5_portfolios"][f"h{h}"]["metrics"]
            ls = cohort["long_short_portfolios"][f"h{h}"]["metrics"]
            compact[name][f"h{h}"] = {
                "q5_excess_ann": None if not q5 else round(q5["excess_annual_returns"], 4),
                "ls_excess_ann": None if not ls else round(ls["excess_annual_returns"], 4),
            }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

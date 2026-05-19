from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rqalpha.apis import history_bars, instruments, is_st_stock, is_suspended

from ss_common import (
    breakout_position,
    build_stock_pool,
    close_history,
    daily_returns,
    high_low_close_history,
    max_drawdown,
    moving_average_gap,
)


DEFAULT_EVENT_PATH = Path.cwd() / ".temp" / "public_pead" / "public_pead_events_2020_2025.csv"
DEFAULT_POOL_SIZE = 300
DEFAULT_TOP_N = 5
DEFAULT_HOLD_BUFFER_RANK = 10
DEFAULT_CASH_BUFFER = 0.08


@lru_cache(maxsize=2)
def load_public_event_table(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()

    events = pd.read_csv(path, low_memory=False)
    if events.empty:
        return events

    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    if "report_date" in events.columns:
        events["report_date"] = pd.to_datetime(events["report_date"], errors="coerce")
    for column in ("express_profit_yoy", "surprise_score"):
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    events["bundle_order_book_id"] = events["order_book_id"].map(_normalize_public_order_book_id)
    events = events[events["bundle_order_book_id"].notna()].copy()
    return events


def _normalize_public_order_book_id(order_book_id: str) -> str | None:
    text = str(order_book_id)
    if text.endswith(".SZ"):
        return text[:-3] + ".XSHE"
    if text.endswith(".SH"):
        return text[:-3] + ".XSHG"
    if text.endswith(".XSHE") or text.endswith(".XSHG"):
        return text
    return None


def average_turnover(order_book_id: str, bars: int) -> float | None:
    turnover = history_bars(order_book_id, bars, "1d", "total_turnover")
    if turnover is None or len(turnover) < bars:
        return None
    turnover = np.asarray(turnover, dtype=float)
    if np.isnan(turnover).any() or np.any(turnover <= 0):
        return None
    return float(turnover.mean())


def recent_limit_hits(order_book_id: str, bars: int) -> int | None:
    data = history_bars(order_book_id, bars + 1, "1d", ["close", "high", "low"])
    if data is None or len(data) < bars + 1:
        return None

    close = np.asarray(data["close"], dtype=float)
    high = np.asarray(data["high"], dtype=float)
    low = np.asarray(data["low"], dtype=float)
    if np.isnan(close).any() or np.isnan(high).any() or np.isnan(low).any():
        return None

    hit_count = 0
    for idx in range(1, len(close)):
        prev_close = close[idx - 1]
        if prev_close <= 0:
            return None
        up_ratio = high[idx] / prev_close - 1.0
        down_ratio = low[idx] / prev_close - 1.0
        if up_ratio >= 0.097 or down_ratio <= -0.097:
            hit_count += 1
    return hit_count


def volatility(prices: np.ndarray) -> float | None:
    returns = daily_returns(prices)
    if len(returns) == 0:
        return None
    return float(np.std(returns))


def latest_event_snapshot(
    trading_date,
    order_book_ids: list[str],
    path_text: str,
    lookback_days: int = 180,
) -> pd.DataFrame:
    events = load_public_event_table(path_text)
    if events.empty:
        return pd.DataFrame(columns=["bundle_order_book_id"])

    end_ts = pd.Timestamp(trading_date)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    window = events[
        (events["bundle_order_book_id"].isin(order_book_ids))
        & (events["event_date"] <= end_ts)
        & (events["event_date"] >= start_ts)
    ].copy()
    if window.empty:
        return pd.DataFrame(columns=["bundle_order_book_id"])

    window = window.sort_values(
        ["bundle_order_book_id", "event_date", "surprise_score"],
        ascending=[True, False, False],
    )
    latest = window.groupby("bundle_order_book_id", as_index=False).head(1).copy()
    latest["days_since_event"] = (end_ts - latest["event_date"]).dt.days
    latest["fundamental_proxy_raw"] = latest.apply(_event_quality_score, axis=1)
    return latest


def _event_quality_score(row: pd.Series) -> float:
    family = str(row.get("event_family", ""))
    forecast_type = str(row.get("forecast_type", ""))
    surprise_score = float(row.get("surprise_score", 0.0) or 0.0)
    express_profit_yoy = row.get("express_profit_yoy")
    days_since = float(row.get("days_since_event", 180) or 180)

    family_base = {
        "earnings_forecast": 0.58,
        "performance_express": 0.52,
    }.get(family, 0.45)
    forecast_bonus = {
        "预增": 0.18,
        "略增": 0.14,
        "扭亏": 0.12,
        "续盈": 0.08,
        "减亏": 0.04,
    }.get(forecast_type, 0.0)
    surprise_bonus = float(np.clip(surprise_score * 12.0, -0.18, 0.20))
    if pd.notna(express_profit_yoy):
        express_bonus = float(np.clip(float(express_profit_yoy) / 200.0, -0.12, 0.20))
    else:
        express_bonus = 0.0
    recency_bonus = float(np.clip((180.0 - days_since) / 180.0, 0.0, 1.0) * 0.08)
    return float(np.clip(family_base + forecast_bonus + surprise_bonus + express_bonus + recency_bonus, 0.0, 1.0))


def collect_candidate_frame(
    trading_date,
    pool_size: int = DEFAULT_POOL_SIZE,
    event_path: Path | str = DEFAULT_EVENT_PATH,
) -> pd.DataFrame:
    stock_pool = build_stock_pool(pool_size)
    rows = []
    for order_book_id in stock_pool:
        row = _collect_single_stock_metrics(order_book_id)
        if row is not None:
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    event_frame = latest_event_snapshot(trading_date, frame["order_book_id"].tolist(), str(event_path))
    if not event_frame.empty:
        event_frame = event_frame[
            [
                "bundle_order_book_id",
                "event_family",
                "forecast_type",
                "event_date",
                "surprise_score",
                "express_profit_yoy",
                "days_since_event",
                "fundamental_proxy_raw",
            ]
        ].rename(columns={"bundle_order_book_id": "order_book_id"})
        frame = frame.merge(event_frame, on="order_book_id", how="left")
    else:
        frame["event_family"] = ""
        frame["forecast_type"] = ""
        frame["event_date"] = pd.NaT
        frame["surprise_score"] = np.nan
        frame["express_profit_yoy"] = np.nan
        frame["days_since_event"] = 999.0
        frame["fundamental_proxy_raw"] = np.nan

    for column, default_value in (
        ("event_family", ""),
        ("forecast_type", ""),
        ("days_since_event", 999.0),
        ("fundamental_proxy_raw", 0.42),
    ):
        if column not in frame.columns:
            frame[column] = default_value
        else:
            frame[column] = frame[column].fillna(default_value)
    frame["sector_strength_raw"] = frame.groupby("industry_name")["ret_60"].transform("mean")
    frame["sector_breadth_raw"] = frame.groupby("industry_name")["ma_gap_60"].transform(lambda s: float((s > 0).mean()))
    frame["prosperity_raw"] = 0.65 * frame["sector_strength_raw"] + 0.35 * frame["sector_breadth_raw"]
    frame["rebound_raw"] = frame["ret_20"] / (frame["max_dd_60"] + 0.05)
    frame["trend_raw"] = (
        0.30 * _percentile(frame["ret_20"])
        + 0.25 * _percentile(frame["ret_60"])
        + 0.15 * _percentile(frame["ma_gap_20"])
        + 0.10 * _percentile(frame["ma_gap_60"])
        + 0.10 * _percentile(frame["ma_gap_120"])
        + 0.10 * _percentile(frame["near_high_252"])
    )
    frame["risk_raw"] = (
        0.45 * _percentile(-frame["vol_20"])
        + 0.30 * _percentile(-frame["max_dd_60"])
        + 0.15 * _percentile(-frame["max_dd_120"])
        + 0.10 * _percentile(-frame["limit_hits_10"])
    )
    frame["fundamental_score"] = _percentile(frame["fundamental_proxy_raw"])
    frame["prosperity_score"] = _percentile(frame["prosperity_raw"])
    frame["trend_score"] = frame["trend_raw"]
    frame["risk_score"] = frame["risk_raw"]
    frame["composite_score"] = (
        0.35 * frame["fundamental_score"]
        + 0.20 * frame["prosperity_score"]
        + 0.30 * frame["trend_score"]
        + 0.15 * frame["risk_score"]
    )
    frame["tradeable"] = frame.apply(_is_tradeable_candidate, axis=1)
    frame = frame.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return frame


def _collect_single_stock_metrics(order_book_id: str) -> dict | None:
    try:
        if is_st_stock(order_book_id) or is_suspended(order_book_id):
            return None
    except Exception:
        return None

    prices_20 = close_history(order_book_id, 20)
    prices_60 = close_history(order_book_id, 60)
    prices_120 = close_history(order_book_id, 120)
    prices_252 = close_history(order_book_id, 252)
    high, low, close = high_low_close_history(order_book_id, 60)
    if prices_20 is None or prices_60 is None or prices_120 is None or prices_252 is None:
        return None
    if high is None or low is None or close is None:
        return None

    avg_turnover_20 = average_turnover(order_book_id, 20)
    limit_hits_10 = recent_limit_hits(order_book_id, 10)
    if avg_turnover_20 is None or limit_hits_10 is None:
        return None

    ma_gap_20 = moving_average_gap(prices_20, 20)
    ma_gap_60 = moving_average_gap(prices_60, 60)
    ma_gap_120 = moving_average_gap(prices_120, 120)
    breakout_60 = breakout_position(high, low, close)
    vol_20 = volatility(prices_20)
    if (
        ma_gap_20 is None
        or ma_gap_60 is None
        or ma_gap_120 is None
        or breakout_60 is None
        or vol_20 is None
    ):
        return None

    try:
        inst = instruments(order_book_id)
        symbol = inst.symbol
        industry_name = getattr(inst, "industry_name", None) or getattr(inst, "sector_code_name", None) or "Unknown"
        board_type = getattr(inst, "board_type", None) or ""
    except Exception:
        symbol = order_book_id
        industry_name = "Unknown"
        board_type = ""

    return {
        "order_book_id": order_book_id,
        "symbol": symbol,
        "industry_name": industry_name,
        "board_type": board_type,
        "last_price": float(prices_20[-1]),
        "avg_turnover_20": float(avg_turnover_20),
        "ret_20": float(prices_20[-1] / prices_20[0] - 1.0),
        "ret_60": float(prices_60[-1] / prices_60[0] - 1.0),
        "ret_120": float(prices_120[-1] / prices_120[0] - 1.0),
        "ma_gap_20": float(ma_gap_20),
        "ma_gap_60": float(ma_gap_60),
        "ma_gap_120": float(ma_gap_120),
        "breakout_60": float(breakout_60),
        "vol_20": float(vol_20),
        "max_dd_60": float(max_drawdown(prices_60)),
        "max_dd_120": float(max_drawdown(prices_120)),
        "near_high_252": float(prices_252[-1] / np.max(prices_252)),
        "limit_hits_10": int(limit_hits_10),
    }


def _is_tradeable_candidate(row: pd.Series) -> bool:
    if row["avg_turnover_20"] < 200000000.0:
        return False
    if row["last_price"] < 5.0 or row["last_price"] > 150.0:
        return False
    if row["limit_hits_10"] > 1:
        return False
    if row["vol_20"] > 0.055:
        return False
    if row["max_dd_120"] > 0.35:
        return False
    if row["ma_gap_120"] <= -0.02:
        return False
    if row["ret_60"] <= 0.02:
        return False
    if row["breakout_60"] < 0.55:
        return False
    return True


def _percentile(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    return series.rank(method="average", pct=True)

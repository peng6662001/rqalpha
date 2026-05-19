from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rqalpha.apis import *


BENCHMARK = "000300.XSHG"
ROOT = Path.cwd().resolve()
EVENTS_PATH = ROOT / ".temp" / "public_pead" / "public_pead_events_2020_2025.csv"
DEFAULT_ALLOWED_TYPES = ("略增", "预增", "扭亏")

__config__ = {
    "base": {
        "benchmark": BENCHMARK,
        "accounts": {
            "stock": 1000000,
        },
    }
}


def init(context):
    context.benchmark = BENCHMARK
    context.event_csv_path = EVENTS_PATH
    context.allowed_types = set(_env_types("RQALPHA_PUBLIC_PEAD_TYPES", DEFAULT_ALLOWED_TYPES))
    context.hold_days = _env_int("RQALPHA_PUBLIC_PEAD_HOLD_DAYS", 20)
    context.max_positions = _env_int("RQALPHA_PUBLIC_PEAD_MAX_POSITIONS", 20)
    context.position_weight_cap = _env_float("RQALPHA_PUBLIC_PEAD_WEIGHT_CAP", 0.08)
    context.cash_buffer = _env_float("RQALPHA_PUBLIC_PEAD_CASH_BUFFER", 0.02)
    context.min_surprise_score = _env_float("RQALPHA_PUBLIC_PEAD_MIN_SURPRISE_SCORE", 0.0)
    context.entry_by_date = {}
    context.exit_by_date = {}
    context.active_metadata = {}
    context.pending_entries = []
    _load_event_schedule(context)
    update_universe([context.benchmark] + sorted(_scheduled_universe(context)))


def before_trading(context):
    today = context.now.date()
    context.pending_entries = list(context.entry_by_date.get(today, []))


def open_auction(context, bar_dict):
    today = context.now.date()
    _process_scheduled_exits(context, today)
    _process_scheduled_entries(context, today, bar_dict)
    _rebalance_active_positions(context)


def _load_event_schedule(context):
    if not context.event_csv_path.exists():
        raise RuntimeError(f"missing event table: {context.event_csv_path}")

    events = pd.read_csv(context.event_csv_path, low_memory=False)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    events["report_date"] = pd.to_datetime(events["report_date"], errors="coerce").dt.normalize()
    events["first_appoint_date"] = pd.to_datetime(events.get("first_appoint_date"), errors="coerce").dt.normalize()
    events = events[
        (events["event_family"] == "earnings_forecast")
        & (events["forecast_type"].isin(context.allowed_types))
        & (events["surprise_score"].fillna(-999.0) >= context.min_surprise_score)
        & (events["report_date"] >= pd.Timestamp("2020-01-01"))
        & (events["report_date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    if events.empty:
        return

    events["bundle_order_book_id"] = events["order_book_id"].map(_normalize_public_order_book_id)
    events = events[events["bundle_order_book_id"].notna()].copy()
    events["score_bucket"] = _assign_quintiles(events["surprise_score"])
    events = events[events["score_bucket"] == 5].copy()
    events = events.sort_values(
        ["event_date", "surprise_score", "days_from_first_appoint_to_event", "bundle_order_book_id"],
        ascending=[True, False, True, True],
    )

    benchmark_dates = sorted(_load_benchmark_dates())
    for _, row in events.iterrows():
        event_date = row["event_date"]
        if pd.isna(event_date):
            continue
        entry_date = _next_trading_date(benchmark_dates, event_date.date())
        exit_date = _nth_trading_date(benchmark_dates, entry_date, context.hold_days - 1) if entry_date else None
        if entry_date is None or exit_date is None:
            continue
        order_book_id = row["bundle_order_book_id"]
        payload = {
            "order_book_id": order_book_id,
            "forecast_type": str(row["forecast_type"]),
            "surprise_score": float(row["surprise_score"]),
            "event_date": event_date.date().isoformat(),
            "report_date": row["report_date"].date().isoformat() if pd.notna(row["report_date"]) else "",
            "exit_date": exit_date.isoformat(),
        }
        context.entry_by_date.setdefault(entry_date, []).append(payload)
        context.exit_by_date.setdefault(exit_date, []).append(order_book_id)

    for entry_date, rows in context.entry_by_date.items():
        rows.sort(key=lambda item: (item["surprise_score"], item["forecast_type"]), reverse=True)


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_types(name, default_values):
    raw = os.environ.get(name)
    if not raw:
        return list(default_values)
    out = [part.strip() for part in raw.split(",") if part.strip()]
    return out or list(default_values)


def _load_benchmark_dates():
    import os
    from pathlib import Path

    import h5py

    bundle_dir = Path(os.path.expanduser(r"~/.rqalpha/bundle"))
    with h5py.File(bundle_dir / "indexes.h5", "r") as f:
        data = pd.DataFrame.from_records(f[BENCHMARK][:])
    dates = pd.to_datetime(data["datetime"].astype(str).str[:8], errors="coerce")
    return [d.date() for d in dates.dropna().sort_values().unique()]


def _assign_quintiles(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    return pd.Series(ranks.mul(5).apply(lambda x: min(5, max(1, int(-(-x // 1))))), index=series.index)


def _next_trading_date(calendar_dates, date_value):
    for item in calendar_dates:
        if item > date_value:
            return item
    return None


def _nth_trading_date(calendar_dates, start_date, offset):
    try:
        idx = calendar_dates.index(start_date)
    except ValueError:
        return None
    target = idx + offset
    if target >= len(calendar_dates):
        return None
    return calendar_dates[target]


def _normalize_public_order_book_id(order_book_id):
    text = str(order_book_id)
    if text.endswith(".SZ"):
        return text[:-3] + ".XSHE"
    if text.endswith(".SH"):
        return text[:-3] + ".XSHG"
    if text.endswith(".XSHE") or text.endswith(".XSHG"):
        return text
    return None


def _scheduled_universe(context):
    symbols = set()
    for items in context.entry_by_date.values():
        for item in items:
            symbols.add(item["order_book_id"])
    return symbols


def _process_scheduled_exits(context, today):
    for order_book_id in context.exit_by_date.get(today, []):
        if order_book_id in context.active_metadata:
            order_target_percent(order_book_id, 0.0)
            context.active_metadata.pop(order_book_id, None)


def _process_scheduled_entries(context, today, bar_dict):
    candidates = []
    active = {
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    }
    free_slots = max(context.max_positions - len(active), 0)
    if free_slots <= 0:
        return

    for item in context.pending_entries:
        order_book_id = item["order_book_id"]
        if order_book_id in active or order_book_id in context.active_metadata:
            continue
        if not _has_bar(order_book_id, bar_dict):
            continue
        if not _is_tradeable(order_book_id, bar_dict):
            continue
        candidates.append(item)

    for item in candidates[:free_slots]:
        context.active_metadata[item["order_book_id"]] = item


def _is_tradeable(order_book_id, bar_dict):
    if is_suspended(order_book_id) or is_st_stock(order_book_id):
        return False
    bar = bar_dict[order_book_id]
    if getattr(bar, "open", None) is None or getattr(bar, "limit_up", None) is None or getattr(bar, "limit_down", None) is None:
        return False
    if bar.open <= 0:
        return False
    if bar.open >= bar.limit_up * 0.999:
        return False
    if bar.open <= bar.limit_down * 1.001:
        return False
    return True


def _rebalance_active_positions(context):
    live_positions = [
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    ]
    tracked = sorted(set(live_positions) | set(context.active_metadata.keys()))
    if not tracked:
        return

    raw_weight = (1.0 - context.cash_buffer) / max(len(tracked), 1)
    target_weight = min(raw_weight, context.position_weight_cap)
    for order_book_id in tracked:
        if order_book_id not in live_positions and not _has_bar(order_book_id, None):
            context.active_metadata.pop(order_book_id, None)
            continue
        try:
            if order_book_id in context.active_metadata:
                order_target_percent(order_book_id, target_weight)
            else:
                order_target_percent(order_book_id, 0.0)
        except Exception:
            context.active_metadata.pop(order_book_id, None)


def _has_bar(order_book_id, bar_dict):
    try:
        if bar_dict is not None:
            bar_dict[order_book_id]
            return True
        instruments(order_book_id)
        return True
    except Exception:
        return False

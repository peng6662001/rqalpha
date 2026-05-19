from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rqalpha.apis import *


ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STRATEGY_DIR = (ROOT / "strategies" / "stock_selection").resolve()
TOOLS_DIR = ROOT / "tools"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from ml_next_day_lgbm_baseline import BENCHMARK, DEFAULT_OUTPUT_DIR, build_dataset
from search_over_50_returns import add_forward_returns, predict_scores, train_model


MODEL_KIND = "rank"
LABEL_MODE = "ret"
HOLD_DAYS = 5
TOP_N = 1
STOP_LOSS_PCT = 0.07
WEAK_EXIT_HOLD_DAYS = 3
WEAK_EXIT_MIN_RETURN = 0.0


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
    context.top_n = TOP_N
    context.cash_buffer = 0.0
    context.hold_days = HOLD_DAYS
    context.model_kind = MODEL_KIND
    context.label_mode = LABEL_MODE
    context.stop_loss_pct = STOP_LOSS_PCT
    context.weak_exit_hold_days = WEAK_EXIT_HOLD_DAYS
    context.weak_exit_min_return = WEAK_EXIT_MIN_RETURN
    context.dataset_cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    raw_dataset = build_dataset(limit_stocks=None, cache_path=context.dataset_cache_path)
    context.dataset, context.feature_cols = add_forward_returns(raw_dataset)
    context.scored_by_year = {}
    context.signal_index_by_year = {}
    context.last_selection_date = None
    context.entry_date_by_order_book_id = {}
    calendar_dates = sorted(pd.to_datetime(context.dataset["date"].drop_duplicates()).dt.date.tolist())
    context.trading_date_index = {date: idx for idx, date in enumerate(calendar_dates)}
    update_universe([context.benchmark])


def _label_col(context):
    return f"target_{context.label_mode}_{context.hold_days}d"


def _train_year_if_needed(context, year):
    if year in context.scored_by_year:
        return

    label_col = _label_col(context)
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")

    train_df = context.dataset[context.dataset["date"] <= train_end].dropna(subset=[label_col] + context.feature_cols).copy()
    test_df = context.dataset[
        (context.dataset["date"] >= test_start) & (context.dataset["date"] <= test_end)
    ].dropna(subset=context.feature_cols).copy()

    if train_df.empty or test_df.empty:
        context.scored_by_year[year] = pd.DataFrame()
        context.signal_index_by_year[year] = {}
        logger.info(f"top1-defensive: skipped training for {year} due to empty train/test window")
        return

    model = train_model(train_df, context.feature_cols, label_col, context.model_kind)
    scored = predict_scores(model, test_df, context.feature_cols, context.model_kind)
    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    signal_dates = sorted(pd.to_datetime(scored["date"].drop_duplicates()))
    context.scored_by_year[year] = scored
    context.signal_index_by_year[year] = {date: idx for idx, date in enumerate(signal_dates)}
    logger.info(f"top1-defensive: trained {context.model_kind} model for {year} with {len(signal_dates)} signal dates")


def _selection_date_for_open(context):
    current_open_date = pd.Timestamp(context.now.date())
    year = current_open_date.year
    _train_year_if_needed(context, year)
    scored = context.scored_by_year.get(year)
    if scored is None or scored.empty:
        return None

    signal_dates = [date for date in context.signal_index_by_year[year] if date < current_open_date]
    if not signal_dates:
        return None
    return max(signal_dates)


def _is_tradeable_now(order_book_id, bar_dict):
    if is_suspended(order_book_id) or is_st_stock(order_book_id):
        return False
    try:
        bar = bar_dict[order_book_id]
    except Exception:
        return False
    if getattr(bar, "open", None) is None or getattr(bar, "limit_up", None) is None or getattr(bar, "limit_down", None) is None:
        return False
    if bar.open <= 0:
        return False
    if bar.open >= bar.limit_up * 0.999:
        return False
    if bar.open <= bar.limit_down * 1.001:
        return False
    return True


def _pick_for_selection_date(context, year, selection_date, bar_dict):
    scored = context.scored_by_year[year]
    group = scored[scored["date"] == selection_date].sort_values("pred", ascending=False)
    if group.empty:
        return None
    for _, row in group.iterrows():
        if _is_tradeable_now(row["order_book_id"], bar_dict):
            return row
    return None


def _held_trading_days(context, order_book_id, current_date):
    entry_date = context.entry_date_by_order_book_id.get(order_book_id)
    if entry_date is None:
        return 0
    entry_idx = context.trading_date_index.get(entry_date)
    current_idx = context.trading_date_index.get(current_date)
    if entry_idx is None or current_idx is None or current_idx < entry_idx:
        return 0
    return current_idx - entry_idx + 1


def _process_risk_exits(context, bar_dict):
    current_date = context.now.date()
    triggered = False
    for position in get_positions():
        order_book_id = position.order_book_id
        quantity = getattr(position, "quantity", 0)
        avg_price = getattr(position, "avg_price", 0.0)
        if quantity <= 0 or avg_price <= 0 or not _is_tradeable_now(order_book_id, bar_dict):
            continue
        bar = bar_dict[order_book_id]
        open_ret = float(bar.open / avg_price - 1.0)
        held_days = _held_trading_days(context, order_book_id, current_date)
        reason = None
        if open_ret <= -context.stop_loss_pct:
            reason = "stop_loss"
        elif held_days >= context.weak_exit_hold_days and open_ret <= context.weak_exit_min_return:
            reason = "weak_exit"
        if reason is None:
            continue
        order_target_percent(order_book_id, 0.0)
        context.entry_date_by_order_book_id.pop(order_book_id, None)
        triggered = True
        logger.info(
            f"top1-defensive: exit {order_book_id} reason={reason} "
            f"held_days={held_days} open_ret={open_ret:.4f}"
        )
    return triggered


def before_trading(context):
    _train_year_if_needed(context, context.now.year)


def open_auction(context, bar_dict):
    if _process_risk_exits(context, bar_dict):
        return

    selection_date = _selection_date_for_open(context)
    if selection_date is None:
        return

    year = pd.Timestamp(selection_date).year
    signal_index = context.signal_index_by_year[year].get(selection_date)
    if signal_index is None or signal_index % context.hold_days != 0:
        return
    if context.last_selection_date == selection_date:
        return

    row = _pick_for_selection_date(context, year, selection_date, bar_dict)
    selected = [row["order_book_id"]] if row is not None else []

    current_positions = {
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    }
    selected_set = set(selected)
    trade_universe = sorted(current_positions | selected_set)
    update_universe(trade_universe + [context.benchmark])

    target_weight = (1.0 - context.cash_buffer) / max(len(selected_set), 1)
    for order_book_id in trade_universe:
        if order_book_id in selected_set:
            order_target_percent(order_book_id, target_weight)
        else:
            order_target_percent(order_book_id, 0.0)
            context.entry_date_by_order_book_id.pop(order_book_id, None)

    current_date = context.now.date()
    for order_book_id in selected_set:
        if order_book_id not in current_positions:
            context.entry_date_by_order_book_id[order_book_id] = current_date

    context.last_selection_date = selection_date
    if row is None:
        logger.info(f"top1-defensive: {selection_date.date().isoformat()} -> cash")
    else:
        logger.info(
            f"top1-defensive: {selection_date.date().isoformat()} -> "
            f"{row['order_book_id']} {row['symbol']} pred={float(row['pred']):.4f}"
        )

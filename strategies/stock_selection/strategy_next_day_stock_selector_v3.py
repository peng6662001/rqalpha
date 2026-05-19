import json
import sys
from pathlib import Path

import numpy as np
from rqalpha.apis import *

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import (
    BENCHMARK,
    breakout_position,
    build_stock_pool,
    close_history,
    daily_returns,
    high_low_close_history,
    max_drawdown,
    moving_average_gap,
)


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
    context.pool_size = 300
    context.stock_pool = []
    context.top_n = 10
    context.recommended_n = 5
    context.backup_n = 5
    context.max_vol_20 = 0.050
    context.max_dd_60 = 0.24
    context.min_ret_20 = 0.08
    context.min_ret_60 = 0.12
    context.min_breakout_60 = 0.70
    context.min_avg_turnover_20 = 300000000.0
    context.min_price = 8.0
    context.max_price = 120.0
    context.max_limit_hits_10 = 1
    context.generated = False
    update_universe([context.benchmark])


def _volatility(prices):
    returns = daily_returns(prices)
    if len(returns) == 0:
        return None
    return float(np.std(returns))


def _avg_turnover(order_book_id, bars):
    turnover = history_bars(order_book_id, bars, "1d", "total_turnover")
    if turnover is None or len(turnover) < bars:
        return None
    turnover = np.asarray(turnover, dtype=float)
    if np.isnan(turnover).any() or np.any(turnover <= 0):
        return None
    return float(turnover.mean())


def _recent_limit_hits(order_book_id, bars):
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


def _score_stock(order_book_id):
    prices_20 = close_history(order_book_id, 20)
    prices_60 = close_history(order_book_id, 60)
    prices_120 = close_history(order_book_id, 120)
    high, low, close = high_low_close_history(order_book_id, 60)
    if prices_20 is None or prices_60 is None or prices_120 is None:
        return None
    if high is None or low is None or close is None:
        return None

    ret_20 = float(prices_20[-1] / prices_20[0] - 1.0)
    ret_60 = float(prices_60[-1] / prices_60[0] - 1.0)
    ma_gap_20 = moving_average_gap(prices_20, 20)
    ma_gap_60 = moving_average_gap(prices_60, 60)
    breakout = breakout_position(high, low, close)
    vol_20 = _volatility(prices_20)
    dd_60 = max_drawdown(prices_60)
    dd_120 = max_drawdown(prices_120)

    if (
        ma_gap_20 is None
        or ma_gap_60 is None
        or breakout is None
        or vol_20 is None
        or dd_60 is None
        or dd_120 is None
    ):
        return None

    score = (
        0.32 * ret_20
        + 0.28 * ret_60
        + 0.18 * ma_gap_20
        + 0.10 * ma_gap_60
        + 0.12 * breakout
        - 0.20 * vol_20
        - 0.10 * dd_60
        - 0.05 * dd_120
    )
    return {
        "score": float(score),
        "last_price": float(prices_20[-1]),
        "ret_20": ret_20,
        "ret_60": ret_60,
        "ma_gap_20": float(ma_gap_20),
        "ma_gap_60": float(ma_gap_60),
        "breakout_60": float(breakout),
        "vol_20": float(vol_20),
        "max_dd_60": float(dd_60),
        "max_dd_120": float(dd_120),
    }


def _is_tradeable_candidate(context, order_book_id, metrics):
    if is_st_stock(order_book_id) or is_suspended(order_book_id):
        return False
    avg_turnover_20 = _avg_turnover(order_book_id, 20)
    if avg_turnover_20 is None or avg_turnover_20 < context.min_avg_turnover_20:
        return False
    if metrics["last_price"] < context.min_price or metrics["last_price"] > context.max_price:
        return False
    limit_hits_10 = _recent_limit_hits(order_book_id, 10)
    if limit_hits_10 is None or limit_hits_10 > context.max_limit_hits_10:
        return False
    if metrics["vol_20"] > context.max_vol_20:
        return False
    if metrics["max_dd_60"] > context.max_dd_60:
        return False
    if metrics["ret_20"] < context.min_ret_20:
        return False
    if metrics["ret_60"] < context.min_ret_60:
        return False
    if metrics["breakout_60"] < context.min_breakout_60:
        return False
    return True


def after_trading(context):
    if context.generated:
        return

    context.stock_pool = build_stock_pool(context.pool_size)
    update_universe(context.stock_pool + [context.benchmark])

    results = []
    for order_book_id in context.stock_pool:
        metrics = _score_stock(order_book_id)
        if metrics is None:
            continue
        try:
            inst = instruments(order_book_id)
            symbol = inst.symbol if inst is not None else order_book_id
        except Exception:
            symbol = order_book_id
        item = {"order_book_id": order_book_id, "symbol": symbol}
        item.update(metrics)
        item["tradeable"] = _is_tradeable_candidate(context, order_book_id, metrics)
        results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    selected = [item for item in results if item["tradeable"]][: context.top_n]
    raw_top = results[: context.top_n]
    recommended = selected[: context.recommended_n]
    backup = selected[context.recommended_n : context.recommended_n + context.backup_n]

    selection_date = context.now.date().isoformat()
    try:
        next_date = get_next_trading_date(context.now.date())
        next_trading_date = next_date.isoformat()
    except Exception:
        next_trading_date = None

    payload = {
        "selection_date": selection_date,
        "next_trading_date": next_trading_date,
        "benchmark": context.benchmark,
        "pool_size": len(context.stock_pool),
        "top_n": context.top_n,
        "filters": {
            "exclude_st": True,
            "max_vol_20": context.max_vol_20,
            "max_dd_60": context.max_dd_60,
            "min_ret_20": context.min_ret_20,
            "min_ret_60": context.min_ret_60,
            "min_breakout_60": context.min_breakout_60,
            "min_avg_turnover_20": context.min_avg_turnover_20,
            "min_price": context.min_price,
            "max_price": context.max_price,
            "max_limit_hits_10": context.max_limit_hits_10,
        },
        "raw_top_candidates": raw_top,
        "recommended": recommended,
        "backup": backup,
        "candidates": selected,
    }

    output_path = Path.cwd() / ".temp" / f"next_day_selection_v3_{selection_date}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"next-day selector v3 wrote: {output_path}")
    for idx, item in enumerate(recommended, 1):
        logger.info(
            f"recommended rank={idx} {item['order_book_id']} {item['symbol']} "
            f"score={item['score']:.4f} ret20={item['ret_20']:.4f} "
            f"ret60={item['ret_60']:.4f} gap20={item['ma_gap_20']:.4f}"
        )
    for idx, item in enumerate(backup, 1):
        logger.info(
            f"backup rank={idx} {item['order_book_id']} {item['symbol']} "
            f"score={item['score']:.4f} ret20={item['ret_20']:.4f} "
            f"ret60={item['ret_60']:.4f} gap20={item['ma_gap_20']:.4f}"
        )

    context.generated = True

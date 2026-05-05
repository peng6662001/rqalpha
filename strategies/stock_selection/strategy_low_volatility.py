import sys
import os
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import close_history, moving_average_gap, run_monthly_selection, setup_context, total_return, volatility


def init(context):
    lookback = int(os.environ.get("RQALPHA_LOW_VOL_LOOKBACK", "80"))
    top_n = int(os.environ.get("RQALPHA_LOW_VOL_TOP_N", "5"))
    cash_buffer = float(os.environ.get("RQALPHA_LOW_VOL_CASH_BUFFER", "0.02"))
    rebalance_frequency = os.environ.get("RQALPHA_LOW_VOL_REBALANCE_FREQUENCY", "daily")
    hold_buffer_rank = os.environ.get("RQALPHA_LOW_VOL_HOLD_BUFFER_RANK", "18")
    hold_buffer_rank = int(hold_buffer_rank) if hold_buffer_rank else None
    context.vol_window = int(os.environ.get("RQALPHA_LOW_VOL_VOL_WINDOW", "60"))
    context.trend_filter = os.environ.get("RQALPHA_LOW_VOL_TREND_FILTER", "off")
    context.trend_window = int(os.environ.get("RQALPHA_LOW_VOL_TREND_WINDOW", "60"))
    context.ma_window = int(os.environ.get("RQALPHA_LOW_VOL_MA_WINDOW", "120"))
    setup_context(
        context,
        lookback=lookback,
        top_n=top_n,
        cash_buffer=cash_buffer,
        defensive_filter=True,
        rebalance_frequency=rebalance_frequency,
        hold_buffer_rank=hold_buffer_rank,
    )


def score_stock(context, order_book_id):
    bars = context.vol_window
    if context.trend_filter in ("positive_return", "both"):
        bars = max(bars, context.trend_window + 1)
    if context.trend_filter in ("above_ma", "both"):
        bars = max(bars, context.ma_window)
    prices = close_history(order_book_id, bars)
    if prices is None:
        return None
    if context.trend_filter in ("positive_return", "both"):
        if total_return(prices[-(context.trend_window + 1):]) <= 0:
            return None
    if context.trend_filter in ("above_ma", "both"):
        ma_gap = moving_average_gap(prices, context.ma_window)
        if ma_gap is None or ma_gap <= 0:
            return None
    vol = volatility(prices[-context.vol_window:])
    if vol is None:
        return None
    return -vol


def handle_bar(context, bar_dict):
    run_monthly_selection(context, score_stock)

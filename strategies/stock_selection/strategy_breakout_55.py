import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import breakout_position, high_low_close_history, run_monthly_selection, setup_context


def init(context):
    setup_context(context, lookback=70, top_n=5, defensive_filter=True)


def score_stock(context, order_book_id):
    high, low, close = high_low_close_history(order_book_id, 55)
    if high is None:
        return None
    return breakout_position(high, low, close)


def handle_bar(context, bar_dict):
    run_monthly_selection(context, score_stock)

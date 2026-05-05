import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import close_history, rsi, run_monthly_selection, setup_context


def init(context):
    setup_context(context, lookback=25, top_n=5, defensive_filter=False)


def score_stock(context, order_book_id):
    prices = close_history(order_book_id, 20)
    if prices is None:
        return None
    value = rsi(prices, period=14)
    if value is None:
        return None
    return 100.0 - value


def handle_bar(context, bar_dict):
    run_monthly_selection(context, score_stock)

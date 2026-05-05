import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import close_history, run_monthly_selection, setup_context, total_return


def init(context):
    setup_context(context, lookback=20, top_n=5, defensive_filter=False)


def score_stock(context, order_book_id):
    prices = close_history(order_book_id, 5)
    if prices is None:
        return None
    return -total_return(prices)


def handle_bar(context, bar_dict):
    run_monthly_selection(context, score_stock)

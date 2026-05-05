import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import close_history, downside_volatility, max_drawdown, run_monthly_selection, setup_context, total_return


def init(context):
    setup_context(context, lookback=140, top_n=5, defensive_filter=True)


def score_stock(context, order_book_id):
    prices = close_history(order_book_id, 120)
    if prices is None:
        return None
    trend = total_return(prices)
    dd = max_drawdown(prices)
    downside = downside_volatility(prices)
    penalty = max(dd, 0.03) + max(downside, 0.01)
    return trend / penalty


def handle_bar(context, bar_dict):
    run_monthly_selection(context, score_stock)

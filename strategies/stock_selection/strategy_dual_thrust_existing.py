from rqalpha.apis import history_bars, order_target_percent, subscribe


def init(context):
    context.s1 = "000905.XSHG"
    subscribe(context.s1)


def handle_bar(context, bar_dict):
    his = history_bars(context.s1, 10, "1d", "close")
    if his[9] / his[8] < 0.97:
        if len(context.portfolio.positions) > 0:
            for stock in list(context.portfolio.positions.keys()):
                order_target_percent(stock, 0)
        return

    position = context.portfolio.positions[context.s1].quantity
    if position < 100:
        high = history_bars(context.s1, 3, "1d", "high")
        low = history_bars(context.s1, 3, "1d", "low")
        close = history_bars(context.s1, 3, "1d", "close")
        opening = history_bars(context.s1, 3, "1d", "open")

        hh = max(high[:2])
        lc = min(close[:2])
        hc = max(close[:2])
        ll = min(low[:2])
        open_price = opening[2]
        current_price = close[2]
        price_range = max((hh - lc), (hc - ll))
        buy_line = open_price + 0.9 * price_range
        if current_price > buy_line:
            order_target_percent(context.s1, 1)

    hist = history_bars(context.s1, 3, "1d", "close")
    case1 = (1 - hist[2] / hist[0]) >= 0.06
    case2 = hist[1] / hist[0] <= 0.92
    if case1 or case2:
        order_target_percent(context.s1, 0)

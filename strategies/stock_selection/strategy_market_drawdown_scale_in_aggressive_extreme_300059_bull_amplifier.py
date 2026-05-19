from rqalpha.apis import *


__config__ = {
    "base": {
        "benchmark": "000300.XSHG",
        "accounts": {
            "stock": 1000000,
        },
    }
}


def init(context):
    context.benchmark = "000300.XSHG"
    context.target = "300059.XSHE"
    context.market_peak = 0.0
    context.entry_drawdown = 0.02
    context.reset_drawdown = 0.015
    context.wait_for_reset = False
    context.armed_for_entry = False
    context.position_peak_price = 0.0
    context.trend_ma_period = 40
    context.bull_gap = 0.025
    context.super_bull_gap = 0.07
    context.profit_trim_1 = 0.65
    context.profit_trim_2 = 1.00
    context.profit_exit = 1.60
    context.trailing_activation_profit = 0.80
    context.trailing_stop_drawdown = 0.22
    context.trailing_exit_drawdown = 0.34
    context.lock1_profit = 0.90
    context.lock1_cap = 0.90
    context.lock2_profit = 1.30
    context.lock2_cap = 0.75
    context.lock3_profit = 1.70
    context.lock3_cap = 0.45
    update_universe([context.benchmark, context.target])


def _target_percent_for_drawdown(drawdown):
    if drawdown < 0.02:
        return 0.0
    if drawdown < 0.05:
        return 0.50
    if drawdown < 0.08:
        return 0.80
    return 1.00


def _trend_gap(context, price):
    history = history_bars(context.target, context.trend_ma_period, "1d", "close")
    if history is None or len(history) < context.trend_ma_period:
        return None
    moving_average = history.mean()
    if moving_average <= 0:
        return None
    return price / moving_average - 1.0


def _profit_based_cap(context, profit, trend_gap):
    if trend_gap is not None and trend_gap >= context.super_bull_gap:
        if profit >= context.profit_exit:
            return 0.0
        if profit >= context.profit_trim_2:
            return 0.45
        if profit >= context.profit_trim_1:
            return 0.80
        return 1.0
    if profit >= context.profit_exit:
        return 0.0
    if profit >= context.profit_trim_2:
        return 0.25
    if profit >= context.profit_trim_1:
        return 0.60
    return 1.0


def _profit_lock_cap(context, profit):
    if profit >= context.lock3_profit:
        return context.lock3_cap
    if profit >= context.lock2_profit:
        return context.lock2_cap
    if profit >= context.lock1_profit:
        return context.lock1_cap
    return 1.0


def handle_bar(context, bar_dict):
    benchmark_bar = bar_dict[context.benchmark]
    target_bar = bar_dict[context.target]

    if benchmark_bar.isnan or target_bar.isnan:
        return

    benchmark_price = benchmark_bar.close
    target_price = target_bar.close
    if benchmark_price <= 0 or target_price <= 0:
        return

    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return

    drawdown = 1.0 - benchmark_price / context.market_peak
    trend_gap = _trend_gap(context, target_price)
    plot("market_drawdown", drawdown)
    plot("trend_gap", 0 if trend_gap is None else trend_gap)

    if not context.armed_for_entry and drawdown >= context.entry_drawdown:
        context.armed_for_entry = True
        logger.info(f"entry armed: drawdown={drawdown:.4f}")

    position = get_position(context.target)
    if position.quantity > 0 and position.avg_price > 0:
        context.position_peak_price = max(context.position_peak_price, target_price)
        profit = target_price / position.avg_price - 1.0
        peak_profit = context.position_peak_price / position.avg_price - 1.0
        trailing_drawdown = 1.0 - target_price / context.position_peak_price
        plot("target_profit", profit)
    else:
        profit = None
        peak_profit = None
        trailing_drawdown = None
        context.position_peak_price = 0.0

    if context.wait_for_reset:
        if drawdown <= context.reset_drawdown:
            context.wait_for_reset = False
            context.armed_for_entry = False
        else:
            return

    current_percent = 0.0
    if context.portfolio.total_value > 0:
        current_percent = position.market_value / context.portfolio.total_value

    target_percent = current_percent
    if position.quantity <= 0:
        target_percent = _target_percent_for_drawdown(drawdown) if context.armed_for_entry else 0.0
    elif drawdown >= context.entry_drawdown:
        target_percent = max(current_percent, _target_percent_for_drawdown(drawdown))

    if trend_gap is not None and trend_gap < 0:
        target_percent = min(target_percent, 0.35)
    elif trend_gap is not None and trend_gap >= context.super_bull_gap:
        target_percent = max(target_percent, 1.0)

    if profit is not None:
        profit_cap = _profit_based_cap(context, profit, trend_gap)
        lock_cap = _profit_lock_cap(context, profit)
        target_percent = min(target_percent, lock_cap)
        if profit_cap < 1.0:
            target_percent = min(target_percent, profit_cap)
            logger.info(
                f"profit trim: drawdown={drawdown:.4f} profit={profit:.4f} cap={profit_cap:.2f}"
            )
            if profit_cap == 0.0:
                context.wait_for_reset = True
                context.armed_for_entry = False

        if (
            peak_profit is not None
            and trailing_drawdown is not None
            and peak_profit >= context.trailing_activation_profit
        ):
            stop_drawdown = context.trailing_stop_drawdown
            exit_drawdown = context.trailing_exit_drawdown
            if trend_gap is not None and trend_gap >= context.super_bull_gap:
                stop_drawdown += 0.03
                exit_drawdown += 0.04
            if trailing_drawdown >= exit_drawdown:
                target_percent = 0.0
                context.wait_for_reset = True
                context.armed_for_entry = False
                logger.info(
                    f"trailing exit: peak_profit={peak_profit:.4f} trailing_drawdown={trailing_drawdown:.4f}"
                )
            elif trailing_drawdown >= stop_drawdown:
                trim_cap = 0.50 if trend_gap is not None and trend_gap >= context.super_bull_gap else 0.35
                target_percent = min(target_percent, trim_cap)
                logger.info(
                    f"trailing trim: peak_profit={peak_profit:.4f} trailing_drawdown={trailing_drawdown:.4f}"
                )

    plot("target_percent", target_percent)
    if abs(target_percent - current_percent) >= 0.05:
        logger.info(
            f"rebalance: drawdown={drawdown:.4f} current={current_percent:.4f} target={target_percent:.4f}"
        )
        order_target_percent(context.target, target_percent)

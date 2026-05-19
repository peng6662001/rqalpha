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
    context.trailing_activation_profit = 0.55
    context.trailing_stop_drawdown = 0.16
    context.trailing_exit_drawdown = 0.24
    context.target_ma_period = 30
    context.target_bull_gap = 0.012
    context.target_recovery_gap = 0.002
    context.target_bear_cap = 0.70
    context.deep_drawdown_trigger = 0.12
    context.deep_drawdown_bear_cap = 0.30
    update_universe([context.benchmark, context.target])


def _target_percent_for_drawdown(drawdown):
    if drawdown < 0.02:
        return 0.0
    if drawdown < 0.05:
        return 0.50
    if drawdown < 0.08:
        return 0.80
    return 1.00


def _profit_based_cap(profit):
    if profit >= 1.10:
        return 0.0
    if profit >= 0.70:
        return 0.25
    if profit >= 0.45:
        return 0.60
    return 1.00


def _target_mode(context, target_price):
    history = history_bars(context.target, context.target_ma_period, "1d", "close")
    if history is None or len(history) < context.target_ma_period:
        return "bear"
    moving_average = history.mean()
    if moving_average <= 0:
        return "bear"
    gap = target_price / moving_average - 1.0
    if gap >= context.target_bull_gap:
        return "bull"
    if gap >= context.target_recovery_gap:
        return "recovery"
    return "bear"


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
    mode = _target_mode(context, target_price)
    plot("market_drawdown", drawdown)
    plot("mode_flag", 1 if mode == "bull" else 0.5 if mode == "recovery" else 0)

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

    if mode == "recovery":
        target_percent = min(target_percent, 0.70)
    elif mode == "bear":
        target_percent = min(target_percent, context.target_bear_cap)
    if drawdown >= context.deep_drawdown_trigger and mode != "bull":
        target_percent = min(target_percent, context.deep_drawdown_bear_cap)

    if profit is not None:
        profit_cap = _profit_based_cap(profit)
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
            if trailing_drawdown >= context.trailing_exit_drawdown:
                target_percent = 0.0
                context.wait_for_reset = True
                context.armed_for_entry = False
                logger.info(
                    f"trailing exit: peak_profit={peak_profit:.4f} trailing_drawdown={trailing_drawdown:.4f}"
                )
            elif trailing_drawdown >= context.trailing_stop_drawdown:
                target_percent = min(target_percent, 0.35)
                logger.info(
                    f"trailing trim: peak_profit={peak_profit:.4f} trailing_drawdown={trailing_drawdown:.4f}"
                )

    plot("target_percent", target_percent)
    if abs(target_percent - current_percent) >= 0.05:
        logger.info(
            f"rebalance: drawdown={drawdown:.4f} mode={mode} current={current_percent:.4f} target={target_percent:.4f}"
        )
        order_target_percent(context.target, target_percent)

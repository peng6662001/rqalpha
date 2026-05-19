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
    context.target = "510300.XSHG"
    context.market_peak = 0.0
    context.entry_drawdown = 0.10
    context.reset_drawdown = 0.05
    context.take_profit = 0.30
    context.wait_for_reset = False
    update_universe([context.benchmark, context.target])


def _target_percent_for_drawdown(drawdown):
    if drawdown < 0.10:
        return 0.0
    if drawdown < 0.15:
        return 0.20
    if drawdown < 0.20:
        return 0.40
    if drawdown < 0.25:
        return 0.70
    return 1.00


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
    plot("market_drawdown", drawdown)

    position = get_position(context.target)
    if position.quantity > 0 and position.avg_price > 0:
        profit = target_price / position.avg_price - 1.0
        plot("target_profit", profit)
        if profit >= context.take_profit:
            logger.info(
                "take profit: market_drawdown=%.4f target_profit=%.4f",
                drawdown,
                profit,
            )
            order_target_percent(context.target, 0)
            context.wait_for_reset = True
            return

    if context.wait_for_reset:
        if drawdown <= context.reset_drawdown:
            logger.info("market recovered, re-arm buying logic")
            context.wait_for_reset = False
        else:
            return

    target_percent = _target_percent_for_drawdown(drawdown)
    current_percent = 0.0
    if context.portfolio.total_value > 0:
        current_percent = position.market_value / context.portfolio.total_value

    plot("target_percent", target_percent)
    if abs(target_percent - current_percent) >= 0.05:
        logger.info(
            "rebalance: drawdown=%.4f current=%.4f target=%.4f",
            drawdown,
            current_percent,
            target_percent,
        )
        order_target_percent(context.target, target_percent)

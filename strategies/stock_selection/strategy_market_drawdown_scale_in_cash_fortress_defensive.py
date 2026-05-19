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
    context.target = "512000.XSHG"
    context.defensive_target = "510300.XSHG"

    context.market_peak = 0.0
    context.entry_drawdown = 0.055
    context.reset_drawdown = 0.03

    context.wait_for_reset = False
    context.armed_for_entry = False

    context.position_peak_price = 0.0
    context.trailing_activation_profit = 0.18
    context.trailing_stop_drawdown = 0.07
    context.trailing_exit_drawdown = 0.10

    context.max_hold_days = 55
    context.max_hold_profit_floor = 0.06
    context.time_stop_trim = 0.10
    context.time_stop_ma_period = 20

    context.last_position_quantity = 0
    context.entry_date = None

    context.benchmark_ma_period = 90
    context.benchmark_bull_gap = 0.0
    context.benchmark_recovery_gap = -0.01
    context.target_bull_gap = 0.0
    context.target_recovery_gap = -0.002
    context.cash_drawdown_trigger = 0.07
    context.deep_cash_drawdown = 0.095
    context.recovery_cap = 0.15
    context.panic_target_gap = -0.02
    context.max_exposure = 0.70

    context.defensive_cap = 0.20
    context.defensive_ma_period = 60
    context.defensive_gap_floor = -0.01

    update_universe([context.benchmark, context.target, context.defensive_target])


def _target_percent_for_drawdown(drawdown):
    if drawdown < 0.055:
        return 0.0
    if drawdown < 0.085:
        return 0.20
    if drawdown < 0.125:
        return 0.35
    if drawdown < 0.17:
        return 0.55
    return 0.75


def _profit_based_cap(profit, mode):
    if mode == "recovery":
        if profit >= 0.18:
            return 0.0
        if profit >= 0.12:
            return 0.15
        if profit >= 0.08:
            return 0.25
        return 1.0

    if profit >= 0.40:
        return 0.0
    if profit >= 0.28:
        return 0.25
    if profit >= 0.18:
        return 0.45
    return 1.0


def _holding_days(context):
    if context.entry_date is None:
        return 0
    return (context.now.date() - context.entry_date).days


def _moving_average(order_book_id, period):
    history = history_bars(order_book_id, period, "1d", "close")
    if history is None or len(history) < period:
        return None
    moving_average = history.mean()
    if moving_average <= 0:
        return None
    return moving_average


def _market_mode(context, benchmark_price, target_price, drawdown):
    benchmark_ma = _moving_average(context.benchmark, context.benchmark_ma_period)
    target_ma = _moving_average(context.target, 20)
    if benchmark_ma is None or target_ma is None:
        return "cash", None, None

    benchmark_gap = benchmark_price / benchmark_ma - 1.0
    target_gap = target_price / target_ma - 1.0

    if target_gap <= context.panic_target_gap:
        return "cash", benchmark_gap, target_gap

    if (
        benchmark_gap >= context.benchmark_bull_gap
        and drawdown <= context.cash_drawdown_trigger
        and target_gap >= context.target_bull_gap
    ):
        return "bull", benchmark_gap, target_gap

    if (
        benchmark_gap >= context.benchmark_recovery_gap
        and drawdown <= context.deep_cash_drawdown
        and target_gap >= context.target_recovery_gap
    ):
        return "recovery", benchmark_gap, target_gap

    return "cash", benchmark_gap, target_gap


def _time_stop_triggered(context, price, profit):
    moving_average = _moving_average(context.target, context.time_stop_ma_period)
    if moving_average is None or profit >= context.max_hold_profit_floor:
        return False
    return price < moving_average


def _defensive_percent(context, mode, drawdown, defensive_price):
    if mode != "recovery":
        return 0.0

    defensive_ma = _moving_average(context.defensive_target, context.defensive_ma_period)
    if defensive_ma is None or defensive_price <= 0:
        return 0.0

    defensive_gap = defensive_price / defensive_ma - 1.0
    if defensive_gap < context.defensive_gap_floor:
        return 0.0

    target_percent = context.defensive_cap
    if drawdown >= context.deep_cash_drawdown - 0.01:
        target_percent = min(target_percent, context.defensive_cap * 0.6)
    return target_percent


def handle_bar(context, bar_dict):
    benchmark_bar = bar_dict[context.benchmark]
    target_bar = bar_dict[context.target]
    defensive_bar = bar_dict[context.defensive_target]

    if benchmark_bar.isnan or target_bar.isnan or defensive_bar.isnan:
        return

    benchmark_price = benchmark_bar.close
    target_price = target_bar.close
    defensive_price = defensive_bar.close
    if benchmark_price <= 0 or target_price <= 0 or defensive_price <= 0:
        return

    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return

    drawdown = 1.0 - benchmark_price / context.market_peak
    mode, benchmark_gap, target_gap = _market_mode(context, benchmark_price, target_price, drawdown)
    plot("market_drawdown", drawdown)
    plot("mode_flag", 1 if mode == "bull" else 0.5 if mode == "recovery" else 0.0)

    if not context.armed_for_entry and drawdown >= context.entry_drawdown:
        context.armed_for_entry = True
        logger.info(f"entry armed: drawdown={drawdown:.4f}")

    attack_position = get_position(context.target)
    defense_position = get_position(context.defensive_target)
    if attack_position.quantity > 0 and context.last_position_quantity <= 0:
        context.entry_date = context.now.date()
    elif attack_position.quantity <= 0:
        context.entry_date = None

    if attack_position.quantity > 0 and attack_position.avg_price > 0:
        context.position_peak_price = max(context.position_peak_price, target_price)
        profit = target_price / attack_position.avg_price - 1.0
        peak_profit = context.position_peak_price / attack_position.avg_price - 1.0
        trailing_drawdown = 1.0 - target_price / context.position_peak_price
        holding_days = _holding_days(context)
        plot("target_profit", profit)
    else:
        profit = None
        peak_profit = None
        trailing_drawdown = None
        holding_days = 0
        context.position_peak_price = 0.0

    if context.wait_for_reset:
        if drawdown <= context.reset_drawdown:
            logger.info("market recovered, re-arm buying logic")
            context.wait_for_reset = False
            context.armed_for_entry = False
        else:
            context.last_position_quantity = attack_position.quantity
            return

    attack_current = 0.0
    defense_current = 0.0
    if context.portfolio.total_value > 0:
        attack_current = attack_position.market_value / context.portfolio.total_value
        defense_current = defense_position.market_value / context.portfolio.total_value

    defense_target_percent = _defensive_percent(context, mode, drawdown, defensive_price)

    if mode == "cash":
        attack_target_percent = 0.0
        defense_target_percent = 0.0
    else:
        if attack_position.quantity <= 0:
            attack_target_percent = _target_percent_for_drawdown(drawdown) if context.armed_for_entry else 0.0
        elif drawdown >= context.entry_drawdown:
            attack_target_percent = max(attack_current, _target_percent_for_drawdown(drawdown))
        else:
            attack_target_percent = attack_current

        if mode == "recovery":
            attack_target_percent = min(attack_target_percent, context.recovery_cap)
        else:
            attack_target_percent = min(attack_target_percent, context.max_exposure)

        if target_gap is not None and target_gap < 0:
            attack_target_percent = min(attack_target_percent, 0.20 if mode == "recovery" else 0.35)

    if profit is not None:
        profit_cap = _profit_based_cap(profit, mode)
        if profit_cap < 1.0:
            attack_target_percent = min(attack_target_percent, profit_cap)
            logger.info(f"profit trim: mode={mode} profit={profit:.4f} cap={profit_cap:.2f}")
            if profit_cap == 0.0:
                context.wait_for_reset = True
                context.armed_for_entry = False

        if holding_days >= context.max_hold_days and _time_stop_triggered(context, target_price, profit):
            attack_target_percent = min(attack_target_percent, context.time_stop_trim)
            logger.info(
                f"timed trend trim: holding_days={holding_days} profit={profit:.4f} target={attack_target_percent:.2f}"
            )

        if (
            peak_profit is not None
            and trailing_drawdown is not None
            and peak_profit >= context.trailing_activation_profit
        ):
            exit_drawdown = context.trailing_exit_drawdown
            stop_drawdown = context.trailing_stop_drawdown
            if mode == "recovery":
                exit_drawdown -= 0.02
                stop_drawdown -= 0.02
            if trailing_drawdown >= exit_drawdown:
                attack_target_percent = 0.0
                context.wait_for_reset = True
                context.armed_for_entry = False
                logger.info(
                    f"trailing exit: peak_profit={peak_profit:.4f} trailing_drawdown={trailing_drawdown:.4f}"
                )
            elif trailing_drawdown >= stop_drawdown:
                attack_target_percent = min(attack_target_percent, 0.12 if mode == "recovery" else 0.25)
                logger.info(
                    f"trailing trim: peak_profit={peak_profit:.4f} trailing_drawdown={trailing_drawdown:.4f}"
                )

    total_target = attack_target_percent + defense_target_percent
    if total_target > 0.90:
        scale = 0.90 / total_target
        attack_target_percent *= scale
        defense_target_percent *= scale

    plot("attack_target_percent", attack_target_percent)
    plot("defense_target_percent", defense_target_percent)
    if abs(attack_target_percent - attack_current) >= 0.05:
        logger.info(
            f"attack rebalance: mode={mode} current={attack_current:.4f} target={attack_target_percent:.4f}"
        )
        order_target_percent(context.target, attack_target_percent)
    if abs(defense_target_percent - defense_current) >= 0.05:
        logger.info(
            f"defense rebalance: mode={mode} current={defense_current:.4f} target={defense_target_percent:.4f}"
        )
        order_target_percent(context.defensive_target, defense_target_percent)

    context.last_position_quantity = attack_position.quantity

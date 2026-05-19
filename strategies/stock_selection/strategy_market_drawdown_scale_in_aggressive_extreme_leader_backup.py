import os

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
    def env_float(name, default):
        return float(os.environ.get(name, default))

    def env_int(name, default):
        return int(os.environ.get(name, default))

    context.benchmark = "000300.XSHG"
    context.primary = os.environ.get("RQALPHA_PRIMARY", "300059.XSHE")
    context.backup = os.environ.get("RQALPHA_BACKUP", "159949.XSHE")
    tertiary = os.environ.get("RQALPHA_TERTIARY")
    context.targets = [context.primary, context.backup]
    if tertiary:
        context.targets.append(tertiary)
    context.targets = list(dict.fromkeys(context.targets))
    context.active_target = context.primary
    context.market_peak = 0.0
    context.entry_drawdown = env_float("RQALPHA_ENTRY_DRAWDOWN", 0.02)
    context.reset_drawdown = env_float("RQALPHA_RESET_DRAWDOWN", 0.015)
    context.wait_for_reset = False
    context.armed_for_entry = False
    context.position_peak_price = {target: 0.0 for target in context.targets}
    context.lookback = env_int("RQALPHA_LOOKBACK", 30)
    context.bull_gap = env_float("RQALPHA_BULL_GAP", 0.02)
    context.super_bull_gap = env_float("RQALPHA_SUPER_BULL_GAP", 0.06)
    context.trend_down_cap = env_float("RQALPHA_TREND_DOWN_CAP", 0.35)
    context.recovery_gap = env_float("RQALPHA_RECOVERY_GAP", 0.0)
    context.recovery_cap = env_float("RQALPHA_RECOVERY_CAP", 0.35)
    context.rebound_drawdown = env_float("RQALPHA_REBOUND_DRAWDOWN", 1.0)
    context.rebound_fast_return = env_float("RQALPHA_REBOUND_FAST_RETURN", 1.0)
    context.deep_drawdown_threshold = env_float("RQALPHA_DEEP_DRAWDOWN", 1.0)
    context.deep_recovery_gap = env_float("RQALPHA_DEEP_RECOVERY_GAP", -1.0)
    context.deep_drawdown_cap = env_float("RQALPHA_DEEP_DRAWDOWN_CAP", context.trend_down_cap)
    context.switch_delta = env_float("RQALPHA_SWITCH_DELTA", 0.03)
    context.level1_drawdown = env_float("RQALPHA_LEVEL1_DRAWDOWN", 0.02)
    context.level2_drawdown = env_float("RQALPHA_LEVEL2_DRAWDOWN", 0.05)
    context.level3_drawdown = env_float("RQALPHA_LEVEL3_DRAWDOWN", 0.08)
    context.level1_target = env_float("RQALPHA_LEVEL1_TARGET", 0.50)
    context.level2_target = env_float("RQALPHA_LEVEL2_TARGET", 0.80)
    context.level3_target = env_float("RQALPHA_LEVEL3_TARGET", 1.00)
    context.profit_trim_1 = env_float("RQALPHA_PROFIT_TRIM_1", 0.60)
    context.profit_trim_2 = env_float("RQALPHA_PROFIT_TRIM_2", 0.95)
    context.profit_exit = env_float("RQALPHA_PROFIT_EXIT", 1.50)
    context.trailing_activation_profit = env_float("RQALPHA_TRAILING_ACTIVATION", 0.75)
    context.trailing_stop_drawdown = env_float("RQALPHA_TRAILING_STOP", 0.21)
    context.trailing_exit_drawdown = env_float("RQALPHA_TRAILING_EXIT", 0.32)
    context.lock1_profit = env_float("RQALPHA_LOCK1_PROFIT", 0.85)
    context.lock1_cap = env_float("RQALPHA_LOCK1_CAP", 0.92)
    context.lock2_profit = env_float("RQALPHA_LOCK2_PROFIT", 1.20)
    context.lock2_cap = env_float("RQALPHA_LOCK2_CAP", 0.78)
    context.lock3_profit = env_float("RQALPHA_LOCK3_PROFIT", 1.60)
    context.lock3_cap = env_float("RQALPHA_LOCK3_CAP", 0.50)
    update_universe([context.benchmark] + context.targets)


def _target_percent_for_drawdown(drawdown):
    context = _target_percent_for_drawdown.context
    if drawdown < context.level1_drawdown:
        return 0.0
    if drawdown < context.level2_drawdown:
        return context.level1_target
    if drawdown < context.level3_drawdown:
        return context.level2_target
    return context.level3_target


def _trend_gap(context, target, price):
    history = history_bars(target, context.lookback, "1d", "close")
    if history is None or len(history) < context.lookback:
        return None
    moving_average = history.mean()
    if moving_average <= 0:
        return None
    return price / moving_average - 1.0


def _momentum_score(context, target, price):
    gap = _trend_gap(context, target, price)
    if gap is None:
        return None
    fast = history_bars(target, 5, "1d", "close")
    if fast is None or len(fast) < 5 or fast[0] <= 0:
        return gap
    return gap + 0.35 * (fast[-1] / fast[0] - 1.0)


def _fast_return(target, window):
    history = history_bars(target, window, "1d", "close")
    if history is None or len(history) < window or history[0] <= 0:
        return None
    return history[-1] / history[0] - 1.0


def _pick_target(context, bar_dict):
    best_target = context.primary
    best_score = None
    for target in context.targets:
        price = bar_dict[target].close
        if price <= 0:
            continue
        score = _momentum_score(context, target, price)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_target = target
    return best_target


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
    _target_percent_for_drawdown.context = context
    benchmark_bar = bar_dict[context.benchmark]
    if benchmark_bar.isnan:
        return
    benchmark_price = benchmark_bar.close
    if benchmark_price <= 0:
        return

    for target in context.targets:
        if bar_dict[target].isnan or bar_dict[target].close <= 0:
            return

    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return

    drawdown = 1.0 - benchmark_price / context.market_peak
    plot("market_drawdown", drawdown)

    chosen_target = _pick_target(context, bar_dict)
    active_price = bar_dict[context.active_target].close
    chosen_price = bar_dict[chosen_target].close
    active_score = _momentum_score(context, context.active_target, active_price)
    chosen_score = _momentum_score(context, chosen_target, chosen_price)

    if (
        chosen_target != context.active_target
        and chosen_score is not None
        and active_score is not None
        and chosen_score - active_score >= context.switch_delta
    ):
        active_position = get_position(context.active_target)
        active_percent = 0.0
        if context.portfolio.total_value > 0:
            active_percent = active_position.market_value / context.portfolio.total_value
        if active_percent > 0.01:
            order_target_percent(context.active_target, 0.0)
            return
        context.active_target = chosen_target

    plot("active_target_flag", 1 if context.active_target == context.primary else 0)
    target = context.active_target
    target_price = bar_dict[target].close
    trend_gap = _trend_gap(context, target, target_price)

    if not context.armed_for_entry and drawdown >= context.entry_drawdown:
        context.armed_for_entry = True
        logger.info(f"entry armed: drawdown={drawdown:.4f}")

    position = get_position(target)
    if position.quantity > 0 and position.avg_price > 0:
        context.position_peak_price[target] = max(context.position_peak_price[target], target_price)
        profit = target_price / position.avg_price - 1.0
        peak_profit = context.position_peak_price[target] / position.avg_price - 1.0
        trailing_drawdown = 1.0 - target_price / context.position_peak_price[target]
        plot("target_profit", profit)
    else:
        profit = None
        peak_profit = None
        trailing_drawdown = None
        context.position_peak_price[target] = 0.0

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
        trend_cap = context.trend_down_cap
        fast_return = _fast_return(target, 5)
        if (
            drawdown >= context.rebound_drawdown
            and trend_gap >= context.recovery_gap
            and fast_return is not None
            and fast_return >= context.rebound_fast_return
        ):
            trend_cap = max(trend_cap, context.recovery_cap)
        if drawdown >= context.deep_drawdown_threshold and trend_gap < context.deep_recovery_gap:
            trend_cap = min(trend_cap, context.deep_drawdown_cap)
        target_percent = min(target_percent, trend_cap)
    elif trend_gap is not None and trend_gap >= context.super_bull_gap:
        target_percent = max(target_percent, 1.0)

    if profit is not None:
        target_percent = min(target_percent, _profit_lock_cap(context, profit))
        profit_cap = _profit_based_cap(context, profit, trend_gap)
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
            f"rebalance: drawdown={drawdown:.4f} target={target} current={current_percent:.4f} desired={target_percent:.4f}"
        )
        order_target_percent(target, target_percent)

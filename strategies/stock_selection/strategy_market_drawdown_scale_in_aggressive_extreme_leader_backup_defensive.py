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
    context.primary = "300059.XSHE"
    context.backup = "159949.XSHE"
    context.defensive_target = "510300.XSHG"
    context.targets = [context.primary, context.backup]
    context.active_target = context.primary
    context.market_peak = 0.0
    context.entry_drawdown = 0.02
    context.reset_drawdown = 0.015
    context.wait_for_reset = False
    context.armed_for_entry = False
    context.position_peak_price = {context.primary: 0.0, context.backup: 0.0}
    context.lookback = 30
    context.bull_gap = 0.02
    context.super_bull_gap = 0.06
    context.switch_delta = 0.03
    context.benchmark_ma_period = 120
    context.benchmark_bull_gap = 0.005
    context.benchmark_recovery_gap = -0.02
    context.cash_drawdown_trigger = 0.09
    context.recovery_cap = 0.45
    context.deep_cash_drawdown = 0.13
    context.defensive_entry_drawdown = 0.02
    context.defensive_ma_period = 60
    context.defensive_trim_gap = -0.02
    context.defensive_full_cap = 0.45
    context.defensive_trim_cap = 0.20
    context.profit_trim_1 = 0.60
    context.profit_trim_2 = 0.95
    context.profit_exit = 1.50
    context.trailing_activation_profit = 0.75
    context.trailing_stop_drawdown = 0.21
    context.trailing_exit_drawdown = 0.32
    context.lock1_profit = 0.85
    context.lock1_cap = 0.92
    context.lock2_profit = 1.20
    context.lock2_cap = 0.78
    context.lock3_profit = 1.60
    context.lock3_cap = 0.50
    update_universe([context.benchmark, context.defensive_target] + context.targets)


def _target_percent_for_drawdown(drawdown):
    if drawdown < 0.02:
        return 0.0
    if drawdown < 0.05:
        return 0.50
    if drawdown < 0.08:
        return 0.80
    return 1.00


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


def _benchmark_mode(context, benchmark_price, drawdown):
    history = history_bars(context.benchmark, context.benchmark_ma_period, "1d", "close")
    if history is None or len(history) < context.benchmark_ma_period:
        return "bear"
    moving_average = history.mean()
    if moving_average <= 0:
        return "bear"
    gap = benchmark_price / moving_average - 1.0
    if gap >= context.benchmark_bull_gap and drawdown <= context.cash_drawdown_trigger:
        return "bull"
    if gap >= context.benchmark_recovery_gap and drawdown <= context.deep_cash_drawdown:
        return "recovery"
    return "bear"


def _defensive_target_percent(context, drawdown, defensive_price):
    if drawdown < context.defensive_entry_drawdown:
        return 0.0
    history = history_bars(context.defensive_target, context.defensive_ma_period, "1d", "close")
    if history is None or len(history) < context.defensive_ma_period:
        return context.defensive_full_cap
    moving_average = history.mean()
    if moving_average <= 0:
        return context.defensive_full_cap
    gap = defensive_price / moving_average - 1.0
    if gap <= context.defensive_trim_gap:
        return context.defensive_trim_cap
    return context.defensive_full_cap


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
    defensive_bar = bar_dict[context.defensive_target]
    if benchmark_bar.isnan or defensive_bar.isnan:
        return
    benchmark_price = benchmark_bar.close
    defensive_price = defensive_bar.close
    if benchmark_price <= 0 or defensive_price <= 0:
        return

    for target in context.targets:
        if bar_dict[target].isnan or bar_dict[target].close <= 0:
            return

    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return

    drawdown = 1.0 - benchmark_price / context.market_peak
    benchmark_mode = _benchmark_mode(context, benchmark_price, drawdown)
    plot("market_drawdown", drawdown)
    plot("benchmark_mode", 1 if benchmark_mode == "bull" else 0.5 if benchmark_mode == "recovery" else 0)

    # Only hold one risk bucket at a time: attack, defense ETF, or cash.
    defensive_position = get_position(context.defensive_target)
    if benchmark_mode != "bull":
        for target in context.targets:
            attack_position = get_position(target)
            attack_percent = 0.0
            if context.portfolio.total_value > 0:
                attack_percent = attack_position.market_value / context.portfolio.total_value
            if attack_percent > 0.01:
                order_target_percent(target, 0.0)
                return
    if benchmark_mode == "bull":
        defensive_percent = 0.0
        if context.portfolio.total_value > 0:
            defensive_percent = defensive_position.market_value / context.portfolio.total_value
        if defensive_percent > 0.01:
            order_target_percent(context.defensive_target, 0.0)
            return

    if benchmark_mode == "bear":
        if context.portfolio.total_value > 0:
            defensive_percent = defensive_position.market_value / context.portfolio.total_value
        else:
            defensive_percent = 0.0
        if defensive_percent > 0.01:
            order_target_percent(context.defensive_target, 0.0)
        return

    if benchmark_mode == "recovery":
        target_percent = _defensive_target_percent(context, drawdown, defensive_price)
        current_percent = 0.0
        if context.portfolio.total_value > 0:
            current_percent = defensive_position.market_value / context.portfolio.total_value
        plot("target_percent", target_percent)
        if abs(target_percent - current_percent) >= 0.05:
            order_target_percent(context.defensive_target, target_percent)
        return

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

    target = context.active_target
    target_price = bar_dict[target].close
    trend_gap = _trend_gap(context, target, target_price)
    plot("active_target_flag", 1 if target == context.primary else 0)

    if not context.armed_for_entry and drawdown >= context.entry_drawdown:
        context.armed_for_entry = True

    position = get_position(target)
    if position.quantity > 0 and position.avg_price > 0:
        context.position_peak_price[target] = max(context.position_peak_price[target], target_price)
        profit = target_price / position.avg_price - 1.0
        peak_profit = context.position_peak_price[target] / position.avg_price - 1.0
        trailing_drawdown = 1.0 - target_price / context.position_peak_price[target]
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
        target_percent = min(target_percent, 0.25)
    elif trend_gap is not None and trend_gap >= context.super_bull_gap:
        target_percent = max(target_percent, 1.0)

    if profit is not None:
        target_percent = min(target_percent, _profit_lock_cap(context, profit))
        profit_cap = _profit_based_cap(context, profit, trend_gap)
        if profit_cap < 1.0:
            target_percent = min(target_percent, profit_cap)
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
            elif trailing_drawdown >= stop_drawdown:
                trim_cap = 0.50 if trend_gap is not None and trend_gap >= context.super_bull_gap else 0.35
                target_percent = min(target_percent, trim_cap)

    plot("target_percent", target_percent)
    if abs(target_percent - current_percent) >= 0.05:
        order_target_percent(target, target_percent)

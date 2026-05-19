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
    context.core_target = "512000.XSHG"
    context.satellite_target = "159949.XSHE"
    context.market_peak = 0.0
    context.last_position_quantity = {}
    context.entry_date = {}
    context.position_peak_price = {}
    context.wait_for_reset = {}
    context.armed_for_entry = {}
    context.ma_period = 120
    context.satellite_max_alloc = 0.15
    context.satellite_drawdown_limit = 0.12
    context.satellite_trend_gap = 0.012
    context.target_ma_period = 35
    context.target_gap_floor = 0.022
    context.deep_drawdown_cutoff = 0.085
    context.deep_drawdown_multiplier = 0.45
    for target in [context.core_target, context.satellite_target]:
        context.last_position_quantity[target] = 0
        context.entry_date[target] = None
        context.position_peak_price[target] = 0.0
        context.wait_for_reset[target] = False
        context.armed_for_entry[target] = False
    update_universe([context.benchmark, context.core_target, context.satellite_target])


def _holding_days(context, target):
    if context.entry_date[target] is None:
        return 0
    return (context.now.date() - context.entry_date[target]).days


def _target_config(context, target):
    if target == context.satellite_target:
        return {
            "entry_drawdown": 0.03,
            "reset_drawdown": 0.02,
            "trailing_activation_profit": 0.35,
            "trailing_stop_drawdown": 0.12,
            "trailing_exit_drawdown": 0.18,
            "time_stop_enabled": False,
        }
    return {
        "entry_drawdown": 0.055,
        "reset_drawdown": 0.03,
        "trailing_activation_profit": 0.22,
        "trailing_stop_drawdown": 0.09,
        "trailing_exit_drawdown": 0.13,
        "time_stop_enabled": True,
        "max_hold_days": 80,
        "max_hold_profit_floor": 0.10,
        "time_stop_trim": 0.25,
        "time_stop_ma_period": 20,
    }


def _target_percent_for_drawdown(target, drawdown):
    if target == "159949.XSHE":
        if drawdown < 0.03:
            return 0.0
        if drawdown < 0.06:
            return 0.35
        if drawdown < 0.10:
            return 0.60
        if drawdown < 0.14:
            return 0.85
        return 1.0
    if drawdown < 0.055:
        return 0.0
    if drawdown < 0.085:
        return 0.25
    if drawdown < 0.125:
        return 0.45
    if drawdown < 0.17:
        return 0.70
    return 0.90


def _profit_based_cap(target, profit):
    if target == "159949.XSHE":
        if profit >= 0.70:
            return 0.0
        if profit >= 0.50:
            return 0.25
        if profit >= 0.35:
            return 0.50
        return 1.0
    if profit >= 0.48:
        return 0.0
    if profit >= 0.38:
        return 0.35
    if profit >= 0.28:
        return 0.60
    return 1.0


def _time_stop_triggered(context, target, price, profit, cfg):
    history = history_bars(target, cfg["time_stop_ma_period"], "1d", "close")
    if history is None or len(history) < cfg["time_stop_ma_period"]:
        return False
    moving_average = history.mean()
    if moving_average <= 0 or profit >= cfg["max_hold_profit_floor"]:
        return False
    return price < moving_average


def _sync_entry_state(context, target, position):
    if position.quantity > 0 and context.last_position_quantity[target] <= 0:
        context.entry_date[target] = context.now.date()
    elif position.quantity <= 0:
        context.entry_date[target] = None
        context.position_peak_price[target] = 0.0


def _base_signal_percent(context, target, drawdown, bar_dict):
    position = get_position(target)
    _sync_entry_state(context, target, position)
    target_price = bar_dict[target].close
    if target_price <= 0:
        return 0.0
    cfg = _target_config(context, target)

    if not context.armed_for_entry[target] and drawdown >= cfg["entry_drawdown"]:
        context.armed_for_entry[target] = True

    if position.quantity > 0 and position.avg_price > 0:
        context.position_peak_price[target] = max(context.position_peak_price[target], target_price)
        profit = target_price / position.avg_price - 1.0
        peak_profit = context.position_peak_price[target] / position.avg_price - 1.0
        trailing_drawdown = 1.0 - target_price / context.position_peak_price[target]
        holding_days = _holding_days(context, target)
    else:
        profit = None
        peak_profit = None
        trailing_drawdown = None
        holding_days = 0

    if context.wait_for_reset[target]:
        if drawdown <= cfg["reset_drawdown"]:
            context.wait_for_reset[target] = False
            context.armed_for_entry[target] = False
        else:
            return 0.0

    current_percent = 0.0
    if context.portfolio.total_value > 0:
        current_percent = position.market_value / context.portfolio.total_value

    target_percent = current_percent
    if position.quantity <= 0:
        target_percent = _target_percent_for_drawdown(target, drawdown) if context.armed_for_entry[target] else 0.0
    elif drawdown >= cfg["entry_drawdown"]:
        target_percent = max(current_percent, _target_percent_for_drawdown(target, drawdown))

    if profit is not None:
        profit_cap = _profit_based_cap(target, profit)
        if profit_cap < 1.0:
            target_percent = min(target_percent, profit_cap)
            if profit_cap == 0.0:
                context.wait_for_reset[target] = True
                context.armed_for_entry[target] = False

        if cfg.get("time_stop_enabled") and holding_days >= cfg["max_hold_days"]:
            if _time_stop_triggered(context, target, target_price, profit, cfg):
                target_percent = min(target_percent, cfg["time_stop_trim"])

        if (
            peak_profit is not None
            and trailing_drawdown is not None
            and peak_profit >= cfg["trailing_activation_profit"]
        ):
            if trailing_drawdown >= cfg["trailing_exit_drawdown"]:
                target_percent = 0.0
                context.wait_for_reset[target] = True
                context.armed_for_entry[target] = False
            elif trailing_drawdown >= cfg["trailing_stop_drawdown"]:
                stop_trim = 0.25 if target == context.satellite_target else 0.35
                target_percent = min(target_percent, stop_trim)

    return target_percent


def _satellite_gate(context, benchmark_price, drawdown, satellite_price):
    bench_history = history_bars(context.benchmark, context.ma_period, "1d", "close")
    target_history = history_bars(context.satellite_target, context.target_ma_period, "1d", "close")
    if (
        bench_history is None
        or len(bench_history) < context.ma_period
        or target_history is None
        or len(target_history) < context.target_ma_period
    ):
        return 0.0

    bench_ma = bench_history.mean()
    target_ma = target_history.mean()
    if bench_ma <= 0 or target_ma <= 0:
        return 0.0

    bench_gap = benchmark_price / bench_ma - 1.0
    target_gap = satellite_price / target_ma - 1.0
    if drawdown > context.satellite_drawdown_limit:
        return 0.0
    if bench_gap < context.satellite_trend_gap:
        return 0.0
    if target_gap < context.target_gap_floor:
        return 0.0

    gate = 0.40
    gate += 10.0 * max(bench_gap - context.satellite_trend_gap, 0.0)
    gate += 5.0 * max(target_gap - context.target_gap_floor, 0.0)
    gate -= 3.5 * max(drawdown - 0.05, 0.0)
    if drawdown >= context.deep_drawdown_cutoff:
        gate *= context.deep_drawdown_multiplier
    return max(0.0, min(1.0, gate))


def handle_bar(context, bar_dict):
    benchmark_bar = bar_dict[context.benchmark]
    core_bar = bar_dict[context.core_target]
    satellite_bar = bar_dict[context.satellite_target]
    if benchmark_bar.isnan or core_bar.isnan or satellite_bar.isnan:
        return

    benchmark_price = benchmark_bar.close
    core_price = core_bar.close
    satellite_price = satellite_bar.close
    if benchmark_price <= 0 or core_price <= 0 or satellite_price <= 0:
        return

    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return

    drawdown = 1.0 - benchmark_price / context.market_peak
    core_signal = _base_signal_percent(context, context.core_target, drawdown, bar_dict)
    satellite_signal = _base_signal_percent(context, context.satellite_target, drawdown, bar_dict)
    satellite_gate = _satellite_gate(context, benchmark_price, drawdown, satellite_price)

    desired_core = min(0.90, core_signal)
    desired_satellite = min(
        context.satellite_max_alloc,
        context.satellite_max_alloc * satellite_signal * satellite_gate,
    )

    total_desired = desired_core + desired_satellite
    if total_desired > 1.0:
        desired_core /= total_desired
        desired_satellite /= total_desired

    core_position = get_position(context.core_target)
    satellite_position = get_position(context.satellite_target)
    core_current = 0.0
    satellite_current = 0.0
    if context.portfolio.total_value > 0:
        core_current = core_position.market_value / context.portfolio.total_value
        satellite_current = satellite_position.market_value / context.portfolio.total_value

    plot("market_drawdown", drawdown)
    plot("core_target_percent", desired_core)
    plot("satellite_target_percent", desired_satellite)
    plot("satellite_gate", satellite_gate)

    if abs(desired_core - core_current) >= 0.05:
        order_target_percent(context.core_target, desired_core)
    if abs(desired_satellite - satellite_current) >= 0.05:
        order_target_percent(context.satellite_target, desired_satellite)

    context.last_position_quantity[context.core_target] = core_position.quantity
    context.last_position_quantity[context.satellite_target] = satellite_position.quantity

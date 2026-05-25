from itertools import product

import pandas as pd

from backtest_engine import BacktestResult, empty_signal, run_signal_backtest
from indicators import bbands, crossed_above, crossed_below, kdj, macd, rsi, sma


def moving_average_strategy(df: pd.DataFrame, short_window: int = 5, long_window: int = 20, quantity: int = 1, stop_loss: float = 0) -> BacktestResult:
    detail = df.copy()
    detail["ma_short"] = sma(detail["close"], short_window)
    detail["ma_long"] = sma(detail["close"], long_window)
    long_entry = crossed_above(detail["ma_short"], detail["ma_long"])
    short_entry = crossed_below(detail["ma_short"], detail["ma_long"])
    result = run_signal_backtest(detail, long_entry, short_entry, quantity=quantity, stop_loss=stop_loss)
    result.detail = detail
    return result


def rsi_trend_strategy(df: pd.DataFrame, short_period: int = 5, long_period: int = 10, quantity: int = 1, stop_loss: float = 0) -> BacktestResult:
    detail = df.copy()
    detail["rsi_short"] = rsi(detail["close"], short_period)
    detail["rsi_long"] = rsi(detail["close"], long_period)
    long_entry = crossed_above(detail["rsi_short"], detail["rsi_long"])
    short_entry = crossed_below(detail["rsi_short"], detail["rsi_long"])
    result = run_signal_backtest(detail, long_entry, short_entry, quantity=quantity, stop_loss=stop_loss)
    result.detail = detail
    return result


def rsi_reversal_strategy(df: pd.DataFrame, period: int = 5, lower: int = 20, upper: int = 80, quantity: int = 1, stop_loss: float = 0) -> BacktestResult:
    detail = df.copy()
    detail["rsi"] = rsi(detail["close"], period)
    lower_line = pd.Series(lower, index=detail.index)
    upper_line = pd.Series(upper, index=detail.index)
    long_entry = crossed_above(detail["rsi"], lower_line)
    short_entry = crossed_below(detail["rsi"], upper_line)
    long_exit = crossed_above(detail["rsi"], upper_line)
    short_exit = crossed_below(detail["rsi"], lower_line)
    result = run_signal_backtest(detail, long_entry, short_entry, long_exit, short_exit, quantity=quantity, stop_loss=stop_loss)
    result.detail = detail
    return result


def bbands_strategy(df: pd.DataFrame, period: int = 20, num_std: float = 2.0, quantity: int = 1, stop_loss: float = 0) -> BacktestResult:
    detail = df.copy()
    detail["bb_upper"], detail["bb_middle"], detail["bb_lower"] = bbands(detail["close"], period, num_std)
    long_entry = crossed_above(detail["close"], detail["bb_upper"])
    short_entry = crossed_below(detail["close"], detail["bb_lower"])
    long_exit = crossed_below(detail["close"], detail["bb_middle"])
    short_exit = crossed_above(detail["close"], detail["bb_middle"])
    result = run_signal_backtest(detail, long_entry, short_entry, long_exit, short_exit, quantity=quantity, stop_loss=stop_loss)
    result.detail = detail
    return result


def macd_strategy(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, zero_filter: bool = True, quantity: int = 1, stop_loss: float = 0) -> BacktestResult:
    detail = df.copy()
    detail["dif"], detail["dea"], detail["macd_hist"] = macd(detail["close"], fast_period, slow_period, signal_period)
    long_entry = crossed_above(detail["dif"], detail["dea"])
    short_entry = crossed_below(detail["dif"], detail["dea"])
    if zero_filter:
        long_entry = long_entry & (detail["dif"] > 0) & (detail["dea"] > 0)
        short_entry = short_entry & (detail["dif"] < 0) & (detail["dea"] < 0)
    result = run_signal_backtest(detail, long_entry, short_entry, quantity=quantity, stop_loss=stop_loss)
    result.detail = detail
    return result


def kdj_strategy(df: pd.DataFrame, rsv_period: int = 9, k_period: int = 3, d_period: int = 3, use_bounds: bool = True, lower: int = 20, upper: int = 80, quantity: int = 1, stop_loss: float = 0) -> BacktestResult:
    detail = df.copy()
    detail["k"], detail["d"], detail["j"] = kdj(detail["high"], detail["low"], detail["close"], rsv_period, k_period, d_period)
    long_entry = crossed_above(detail["k"], detail["d"])
    short_entry = crossed_below(detail["k"], detail["d"])
    if use_bounds:
        long_entry = long_entry & (detail["k"] < lower)
        short_entry = short_entry & (detail["k"] > upper)
    result = run_signal_backtest(detail, long_entry, short_entry, quantity=quantity, stop_loss=stop_loss)
    result.detail = detail
    return result


STRATEGIES = {
    "移動平均線策略": moving_average_strategy,
    "RSI 順勢策略": rsi_trend_strategy,
    "RSI 逆勢策略": rsi_reversal_strategy,
    "布林通道策略": bbands_strategy,
    "MACD 策略": macd_strategy,
    "KDJ 策略": kdj_strategy,
}


STRATEGY_DESCRIPTIONS = {
    "移動平均線策略": "短期均線向上突破長期均線時做多，向下跌破時做空或反向。適合觀察趨勢轉折，但盤整時容易反覆交易。",
    "RSI 順勢策略": "短週期 RSI 向上突破長週期 RSI 視為多方動能增強，向下跌破視為空方動能增強。",
    "RSI 逆勢策略": "RSI 從低檔區向上突破視為超賣反彈，從高檔區向下跌破視為超買回落。",
    "布林通道策略": "價格突破上軌視為強勢追價，跌破下軌視為弱勢放空，回到中軌附近出場。",
    "MACD 策略": "DIF 向上突破 DEA 為黃金交叉，向下跌破 DEA 為死亡交叉；預設加入零軸濾網降低雜訊。",
    "KDJ 策略": "K 值向上突破 D 值為買進訊號，向下跌破為賣出訊號；預設加入超買超賣區間濾網。",
}


OPTIMIZATION_GRIDS = {
    "移動平均線策略": {"short_window": [3, 5, 8, 10], "long_window": [15, 20, 30, 40]},
    "RSI 順勢策略": {"short_period": [3, 5, 7], "long_period": [10, 14, 21]},
    "RSI 逆勢策略": {"period": [5, 7, 10, 14], "lower": [20, 25, 30], "upper": [70, 75, 80]},
    "布林通道策略": {"period": [10, 20, 30, 60], "num_std": [1.5, 2.0, 2.5]},
    "MACD 策略": {"fast_period": [8, 12, 15], "slow_period": [20, 26, 35], "signal_period": [7, 9, 12]},
    "KDJ 策略": {"rsv_period": [5, 9, 14], "k_period": [3, 5], "d_period": [3, 5]},
}


def optimize_strategy(name: str, df: pd.DataFrame, base_params: dict, max_runs: int = 100) -> pd.DataFrame:
    grid = OPTIMIZATION_GRIDS[name]
    keys = list(grid.keys())
    rows = []
    runs = 0
    for values in product(*[grid[key] for key in keys]):
        params = dict(base_params)
        params.update(dict(zip(keys, values)))
        if name == "移動平均線策略" and params["short_window"] >= params["long_window"]:
            continue
        if name == "RSI 順勢策略" and params["short_period"] >= params["long_period"]:
            continue
        if name == "MACD 策略" and params["fast_period"] >= params["slow_period"]:
            continue

        result = STRATEGIES[name](df, **params)
        row = {"strategy": name, **{key: params[key] for key in keys}, **result.metrics}
        rows.append(row)
        runs += 1
        if runs >= max_runs:
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "total_profit"], ascending=False).reset_index(drop=True)


def build_best_parameter_summary(opt_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    metric_cols = {
        "total_profit",
        "trade_count",
        "win_rate",
        "avg_profit",
        "max_drawdown",
        "profit_factor",
        "return_drawdown_ratio",
        "sharpe_like",
        "expectancy",
        "score",
    }
    for strategy, opt_df in opt_results.items():
        if opt_df.empty:
            continue
        best = opt_df.iloc[0].to_dict()
        params = {key: value for key, value in best.items() if key not in metric_cols and key != "strategy"}
        rows.append(
            {
                "strategy": strategy,
                "best_params": str(params),
                "total_profit": best["total_profit"],
                "win_rate": best["win_rate"],
                "max_drawdown": best["max_drawdown"],
                "return_drawdown_ratio": best["return_drawdown_ratio"],
                "sharpe_like": best["sharpe_like"],
                "score": best["score"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "total_profit"], ascending=False).reset_index(drop=True)


def buy_and_hold(df: pd.DataFrame, quantity: int = 1) -> BacktestResult:
    long_entry = pd.Series([True] + [False] * (len(df) - 1), index=df.index)
    return run_signal_backtest(df.copy(), long_entry, empty_signal(df), quantity=quantity, allow_short=False)

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict
    detail: pd.DataFrame


def load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def resample_kbars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1min":
        return df.copy()
    working = df.set_index("time")
    grouped = working.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "product": "last",
        }
    )
    return grouped.dropna(subset=["open", "high", "low", "close"]).reset_index()


def run_signal_backtest(
    df: pd.DataFrame,
    long_entry: pd.Series,
    short_entry: pd.Series,
    long_exit: pd.Series | None = None,
    short_exit: pd.Series | None = None,
    quantity: int = 1,
    stop_loss: float = 0.0,
    allow_short: bool = True,
) -> BacktestResult:
    long_exit = long_exit if long_exit is not None else short_entry
    short_exit = short_exit if short_exit is not None else long_entry

    position = 0
    entry_price = 0.0
    entry_time = None
    entry_side = ""
    trades = []
    equity_points = []
    realized_profit = 0.0

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        price_close = float(row["close"])

        if position > 0:
            stop_hit = stop_loss > 0 and price_close <= entry_price - stop_loss
            exit_hit = bool(long_exit.iloc[i]) or stop_hit
            if exit_hit:
                exit_price = price_close if stop_hit else float(next_row["open"])
                profit = (exit_price - entry_price) * quantity
                realized_profit += profit
                trades.append(_trade(entry_side, entry_time, entry_price, row["time"] if stop_hit else next_row["time"], exit_price, quantity, profit))
                position = 0
        elif position < 0:
            stop_hit = stop_loss > 0 and price_close >= entry_price + stop_loss
            exit_hit = bool(short_exit.iloc[i]) or stop_hit
            if exit_hit:
                exit_price = price_close if stop_hit else float(next_row["open"])
                profit = (entry_price - exit_price) * quantity
                realized_profit += profit
                trades.append(_trade(entry_side, entry_time, entry_price, row["time"] if stop_hit else next_row["time"], exit_price, quantity, profit))
                position = 0

        if position == 0:
            if bool(long_entry.iloc[i]):
                position = quantity
                entry_price = float(next_row["open"])
                entry_time = next_row["time"]
                entry_side = "Buy"
            elif allow_short and bool(short_entry.iloc[i]):
                position = -quantity
                entry_price = float(next_row["open"])
                entry_time = next_row["time"]
                entry_side = "Sell"

        unrealized = 0.0
        if position > 0:
            unrealized = (price_close - entry_price) * quantity
        elif position < 0:
            unrealized = (entry_price - price_close) * quantity
        equity_points.append({"time": row["time"], "equity": realized_profit + unrealized, "realized_profit": realized_profit})

    if position != 0 and len(df) > 0:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        profit = ((exit_price - entry_price) if position > 0 else (entry_price - exit_price)) * quantity
        realized_profit += profit
        trades.append(_trade(entry_side, entry_time, entry_price, last["time"], exit_price, quantity, profit))
        equity_points.append({"time": last["time"], "equity": realized_profit, "realized_profit": realized_profit})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)
    return BacktestResult(trades_df, equity_df, calculate_metrics(trades_df, equity_df), df)


def _trade(side: str, entry_time, entry_price: float, exit_time, exit_price: float, quantity: int, profit: float) -> dict:
    return {
        "side": side,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "quantity": quantity,
        "profit": profit,
        "return_pct": profit / (entry_price * quantity) if entry_price and quantity else 0,
    }


def calculate_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    base = {
        "total_profit": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "return_drawdown_ratio": 0.0,
        "sharpe_like": 0.0,
        "expectancy": 0.0,
        "score": 0.0,
    }
    if trades.empty:
        return base

    profits = trades["profit"].astype(float)
    returns = trades["return_pct"].astype(float)
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    total_profit = float(profits.sum())
    win_rate = float((profits > 0).mean())
    avg_profit = float(profits.mean())

    if equity.empty:
        max_drawdown = 0.0
    else:
        peak = equity["equity"].cummax()
        max_drawdown = float((peak - equity["equity"]).max())

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss else (gross_profit if gross_profit else 0.0)
    return_drawdown_ratio = total_profit / max_drawdown if max_drawdown else (total_profit if total_profit > 0 else 0.0)
    sharpe_like = float(returns.mean() / returns.std(ddof=0)) if len(returns) > 1 and returns.std(ddof=0) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    score = total_profit - 0.5 * max_drawdown + 100 * (win_rate - 0.5) + 10 * return_drawdown_ratio

    base.update(
        {
            "total_profit": total_profit,
            "trade_count": int(len(trades)),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "return_drawdown_ratio": float(return_drawdown_ratio),
            "sharpe_like": float(sharpe_like),
            "expectancy": float(expectancy),
            "score": float(score),
        }
    )
    return base


def metrics_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {"strategy": name}
        row.update(result.metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "total_profit"], ascending=False)


def empty_signal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

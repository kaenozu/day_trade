#!/usr/bin/env python3
"""
計算処理最適化

テクニカル指標の計算やバックテスト処理のベクトル化と最適化を提供します。
"""

import time
from typing import Any, Callable, Dict, List

import pandas as pd


class CalculationOptimizer:
    """計算処理の最適化クラス"""

    @staticmethod
    def vectorize_technical_indicators(
        data: pd.DataFrame, indicators: List[str], **params
    ) -> pd.DataFrame:
        """テクニカル指標計算のベクトル化"""
        result = data.copy()

        # 並列でベクトル化計算
        for indicator in indicators:
            if indicator == "sma":
                period = params.get("sma_period", 20)
                result[f"sma_{period}"] = (
                    data["close"].rolling(window=period).mean()
                )

            elif indicator == "ema":
                period = params.get("ema_period", 12)
                result[f"ema_{period}"] = data["close"].ewm(span=period).mean()

            elif indicator == "rsi":
                period = params.get("rsi_period", 14)
                delta = data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

            elif indicator == "bollinger":
                period = params.get("bb_period", 20)
                std_dev = params.get("bb_std", 2)
                sma = data["close"].rolling(window=period).mean()
                std = data["close"].rolling(window=period).std()
                result[f"bb_upper_{period}"] = sma + (std * std_dev)
                result[f"bb_lower_{period}"] = sma - (std * std_dev)
                result[f"bb_middle_{period}"] = sma

        return result

    @staticmethod
    def optimize_backtest_calculation(
        data: pd.DataFrame,
        strategy_func: Callable,
        initial_capital: float = 100000,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """バックテスト計算の最適化"""
        start_time = time.perf_counter()

        # データをチャンク分割して処理
        chunks = [
            data[i: i + chunk_size] for i in range(0, len(data), chunk_size)
        ]

        portfolio_value = initial_capital
        positions = {}
        trades = []

        for chunk in chunks:
            # チャンク単位でベクトル化処理
            signals = strategy_func(chunk)

            # 取引処理（ベクトル化）
            for idx, signal in signals.iterrows():
                if signal.get("action") == "buy":
                    # 買い処理
                    shares = portfolio_value * 0.1 / signal["price"]  # 10%投資
                    positions[signal["symbol"]] = (
                        positions.get(signal["symbol"], 0) + shares
                    )
                    portfolio_value -= shares * signal["price"]
                    trades.append(
                        {
                            "date": idx,
                            "symbol": signal["symbol"],
                            "action": "buy",
                            "shares": shares,
                            "price": signal["price"],
                        }
                    )
                elif (
                    signal.get("action") == "sell"
                    and signal["symbol"] in positions
                ):
                    # 売り処理
                    shares = positions[signal["symbol"]]
                    portfolio_value += shares * signal["price"]
                    del positions[signal["symbol"]]
                    trades.append(
                        {
                            "date": idx,
                            "symbol": signal["symbol"],
                            "action": "sell",
                            "shares": shares,
                            "price": signal["price"],
                        }
                    )

        execution_time = time.perf_counter() - start_time

        return {
            "final_portfolio_value": portfolio_value,
            "total_trades": len(trades),
            "execution_time": execution_time,
            "trades_per_second": (
                len(trades) / execution_time if execution_time > 0 else 0
            ),
            "return_percentage": (portfolio_value - initial_capital)
            / initial_capital
            * 100,
        }

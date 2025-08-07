"""
バックテスト戦略実装

各種取引戦略の実装を提供
"""

from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal


def simple_sma_strategy(
    symbols: List[str],
    date: datetime,
    historical_data: Dict[str, pd.DataFrame],
    short_window: int = 20,
    long_window: int = 50,
) -> List[TradingSignal]:
    """
    シンプルなSMAクロス戦略

    Args:
        symbols: 対象銘柄リスト
        date: 現在の日付
        historical_data: 履歴データ
        short_window: 短期移動平均の期間
        long_window: 長期移動平均の期間

    Returns:
        取引シグナルリスト
    """
    signals = []

    for symbol in symbols:
        if symbol not in historical_data:
            continue

        data = historical_data[symbol]

        # 日付を文字列形式で検索
        date_str = date.strftime("%Y-%m-%d")

        try:
            # 現在日付以前のデータを取得
            current_data = data[data.index <= date_str]

            if len(current_data) < long_window:
                continue

            # 移動平均を計算
            short_sma = current_data["Close"].rolling(window=short_window).mean()
            long_sma = current_data["Close"].rolling(window=long_window).mean()

            # 最新の移動平均値を取得
            if len(short_sma) < 2 or len(long_sma) < 2:
                continue

            current_short = short_sma.iloc[-1]
            current_long = long_sma.iloc[-1]
            prev_short = short_sma.iloc[-2]
            prev_long = long_sma.iloc[-2]

            # ゴールデンクロス（買いシグナル）
            if prev_short <= prev_long and current_short > current_long:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=float(current_data["Close"].iloc[-1]),
                        timestamp=date,
                        indicators={
                            "sma_short": current_short,
                            "sma_long": current_long,
                        },
                        confidence=0.7,
                    )
                )

            # デッドクロス（売りシグナル）
            elif prev_short >= prev_long and current_short < current_long:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=float(current_data["Close"].iloc[-1]),
                        timestamp=date,
                        indicators={
                            "sma_short": current_short,
                            "sma_long": current_long,
                        },
                        confidence=0.7,
                    )
                )

        except Exception:
            # データエラーの場合はスキップ
            continue

    return signals


def rsi_strategy(
    symbols: List[str],
    date: datetime,
    historical_data: Dict[str, pd.DataFrame],
    rsi_period: int = 14,
    oversold_threshold: float = 30.0,
    overbought_threshold: float = 70.0,
) -> List[TradingSignal]:
    """
    RSI戦略

    Args:
        symbols: 対象銘柄リスト
        date: 現在の日付
        historical_data: 履歴データ
        rsi_period: RSI計算期間
        oversold_threshold: 売られすぎ閾値
        overbought_threshold: 買われすぎ閾値

    Returns:
        取引シグナルリスト
    """
    signals = []

    for symbol in symbols:
        if symbol not in historical_data:
            continue

        data = historical_data[symbol]
        date_str = date.strftime("%Y-%m-%d")

        try:
            current_data = data[data.index <= date_str]

            if len(current_data) < rsi_period + 1:
                continue

            # RSIを計算（簡易版）
            delta = current_data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]
            current_price = current_data["Close"].iloc[-1]

            # 売られすぎからの反発（買いシグナル）
            if current_rsi < oversold_threshold:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=float(current_price),
                        timestamp=date,
                        indicators={"rsi": current_rsi},
                        confidence=0.6,
                    )
                )

            # 買われすぎからの反落（売りシグナル）
            elif current_rsi > overbought_threshold:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=float(current_price),
                        timestamp=date,
                        indicators={"rsi": current_rsi},
                        confidence=0.6,
                    )
                )

        except Exception:
            continue

    return signals


def bollinger_band_strategy(
    symbols: List[str],
    date: datetime,
    historical_data: Dict[str, pd.DataFrame],
    period: int = 20,
    std_dev: float = 2.0,
) -> List[TradingSignal]:
    """
    ボリンジャーバンド戦略

    Args:
        symbols: 対象銘柄リスト
        date: 現在の日付
        historical_data: 履歴データ
        period: 移動平均期間
        std_dev: 標準偏差倍率

    Returns:
        取引シグナルリスト
    """
    signals = []

    for symbol in symbols:
        if symbol not in historical_data:
            continue

        data = historical_data[symbol]
        date_str = date.strftime("%Y-%m-%d")

        try:
            current_data = data[data.index <= date_str]

            if len(current_data) < period:
                continue

            # ボリンジャーバンドを計算
            sma = current_data["Close"].rolling(window=period).mean()
            std = current_data["Close"].rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            current_price = current_data["Close"].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_sma = sma.iloc[-1]

            # 下限バンドタッチ（買いシグナル）
            if current_price <= current_lower:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=float(current_price),
                        timestamp=date,
                        indicators={
                            "bb_upper": current_upper,
                            "bb_lower": current_lower,
                            "bb_middle": current_sma,
                        },
                        confidence=0.65,
                    )
                )

            # 上限バンドタッチ（売りシグナル）
            elif current_price >= current_upper:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=float(current_price),
                        timestamp=date,
                        indicators={
                            "bb_upper": current_upper,
                            "bb_lower": current_lower,
                            "bb_middle": current_sma,
                        },
                        confidence=0.65,
                    )
                )

        except Exception:
            continue

    return signals

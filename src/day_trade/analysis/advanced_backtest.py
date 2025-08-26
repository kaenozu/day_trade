"""
高度なバックテストエンジン - メインエントリーポイント

後方互換性を維持しつつ、新しいモジュラー構造にリダイレクト。
使用例とデモコードも含む。
"""

# 後方互換性のため、全ての主要クラスをここからエクスポート
from .advanced_backtest import *

# 使用例とデモ
if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd
    from day_trade.utils.logging_config import get_context_logger
    
    logger = get_context_logger(__name__)

    # サンプルデータ生成
    ticker = "7203.T"
    data = yf.download(ticker, period="2y")

    # 簡易戦略シグナル生成
    signals = pd.DataFrame(index=data.index)
    signals["signal"] = "hold"
    signals["confidence"] = 50.0

    # 簡易移動平均クロス戦略
    ma_short = data["Close"].rolling(20).mean()
    ma_long = data["Close"].rolling(50).mean()

    buy_signals = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    sell_signals = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

    signals.loc[buy_signals, "signal"] = "buy"
    signals.loc[buy_signals, "confidence"] = 70.0
    signals.loc[sell_signals, "signal"] = "sell"
    signals.loc[sell_signals, "confidence"] = 70.0

    # バックテストエンジン設定
    trading_costs = TradingCosts(
        commission_rate=0.001, 
        bid_ask_spread_rate=0.001, 
        slippage_rate=0.0005
    )

    backtest_engine = AdvancedBacktestEngine(
        initial_capital=1000000,
        trading_costs=trading_costs,
        position_sizing="percent",
        max_position_size=0.2,
        realistic_execution=True,
    )

    # バックテスト実行
    performance = backtest_engine.run_backtest(data, signals)

    logger.info(
        "高度バックテストデモ完了",
        section="demo",
        total_return=performance.total_return,
        sharpe_ratio=performance.sharpe_ratio,
        max_drawdown=performance.max_drawdown,
        total_trades=performance.total_trades,
        win_rate=performance.win_rate,
    )
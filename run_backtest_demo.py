import sys
import os

# プロジェクトルートディレクトリ (day_trade_sub) を直接sys.pathに追加
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

import day_trade.analysis.advanced_backtest as advanced_backtest
from day_trade.analysis.events import Order, OrderType, OrderStatus, TradeRecord
import pandas as pd
import yfinance as yf
from datetime import datetime

import day_trade.analysis.events as events

# ロガー設定 (advanced_backtest.py内で設定されているが、ここでは追加で明示的に設定)
from src.day_trade.utils.logging_config import get_context_logger
logger = get_context_logger(__name__)

def run_demo():
    logger.info("バックテストデモ開始")

    tickers = ["7203.T", "8031.T", "9984.T"] # 複数ティッカーに変更
    all_data = {}
    for ticker in tickers:
        data_single = yf.download(ticker, start="2020-01-01", end="2022-01-01")
        # === インデックスの整合性確保 ===
        data_single.index = pd.to_datetime(data_single.index, errors='coerce')
        data_single = data_single[data_single.index.notna()]
        data_single.index = pd.DatetimeIndex(data_single.index)
        data_single.index.name = None # インデックス名をクリア
        all_data[ticker] = data_single

    # MultiIndex DataFrameの作成
    data = pd.concat(all_data, axis=1, keys=tickers)

    # MultiIndex の signals DataFrame を作成
    # カラムは (Ticker, 'signal'), (Ticker, 'confidence') となる
    signals_cols = pd.MultiIndex.from_product([tickers, ['signal', 'confidence', 'price_target']], names=['Ticker', 'Attribute'])
    signals = pd.DataFrame(index=data.index, columns=signals_cols)
    signals.loc[:, (slice(None), 'signal')] = 'hold'
    signals.loc[:, (slice(None), 'confidence')] = 50.0

    # 各シンボルに対してシグナルを生成
    for ticker in tickers:
        if (ticker, 'Close') in data.columns:
            ma_short = data[(ticker, 'Close')].rolling(20).mean()
            ma_long = data[(ticker, 'Close')].rolling(50).mean()

            buy_signals_ticker = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
            sell_signals_ticker = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

            # シグナルを適切な位置に設定
            signals.loc[buy_signals_ticker[buy_signals_ticker].index, (ticker, 'signal')] = 'buy'
            signals.loc[buy_signals_ticker[buy_signals_ticker].index, (ticker, 'confidence')] = 70.0
            signals.loc[sell_signals_ticker[sell_signals_ticker].index, (ticker, 'signal')] = 'sell'
            signals.loc[sell_signals_ticker[sell_signals_ticker].index, (ticker, 'confidence')] = 70.0

    # バックテストエンジン設定
    trading_costs = advanced_backtest.TradingCosts(
        commission_rate=0.001, bid_ask_spread_rate=0.001, slippage_rate=0.0005
    )

    backtest_engine = advanced_backtest.AdvancedBacktestEngine(
        initial_capital=1000000,
        trading_costs=trading_costs,
        position_sizing="percent",
        max_position_size=0.2,
        realistic_execution=True,
    )

    performance = backtest_engine.run_backtest(data, signals)

    logger.info(
        "高度イベント駆動型バックテストデモ完了",
        section="demo",
        total_return=performance.total_return,
        sharpe_ratio=performance.sharpe_ratio,
        max_drawdown=performance.max_drawdown,
        total_trades=performance.total_trades,
        win_rate=performance.win_rate,
    )

if __name__ == "__main__":
    run_demo()
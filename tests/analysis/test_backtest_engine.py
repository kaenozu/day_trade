#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Engine Tests
バックテストエンジンテスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# モックデータ構造
class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    """取引データ"""
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    trade_id: Optional[str] = None

@dataclass
class Position:
    """ポジションデータ"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class BacktestResult:
    """バックテスト結果"""
    initial_capital: float
    final_capital: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[Trade]
    equity_curve: pd.Series
    performance_metrics: Dict[str, Any]

class MockBacktestEngine:
    """バックテストエンジンモック"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[float] = [initial_capital]
        self.commission_rate = 0.001  # 0.1%
        self.slippage_rate = 0.0005   # 0.05%
        
    def add_strategy(self, strategy):
        """戦略追加"""
        self.strategy = strategy
        
    def set_data(self, data: pd.DataFrame):
        """データ設定"""
        self.data = data.copy()
        self.current_bar = 0
        
    def run_backtest(self) -> BacktestResult:
        """バックテスト実行"""
        if not hasattr(self, 'data') or not hasattr(self, 'strategy'):
            raise ValueError("Data and strategy must be set before running backtest")
        
        # 各バーでの処理
        for i in range(len(self.data)):
            self.current_bar = i
            current_data = self.data.iloc[:i+1]  # 現在までのデータ
            
            if len(current_data) < 20:  # 最小データ数
                continue
                
            # 戦略からシグナル取得
            signals = self.strategy.generate_signals(current_data)
            
            # シグナルに基づいて注文
            for signal in signals:
                self._process_signal(signal, current_data.iloc[-1])
            
            # ポジション評価
            self._update_positions(current_data.iloc[-1])
            
            # 資産価値記録
            total_equity = self._calculate_total_equity(current_data.iloc[-1])
            self.equity_history.append(total_equity)
        
        return self._generate_result()
    
    def _process_signal(self, signal: Dict[str, Any], current_bar: pd.Series):
        """シグナル処理"""
        symbol = signal['symbol']
        action = signal['action']  # 'BUY' or 'SELL'
        quantity = signal.get('quantity', 100)
        
        if action == 'BUY':
            self._execute_buy_order(symbol, quantity, current_bar['close'])
        elif action == 'SELL':
            self._execute_sell_order(symbol, quantity, current_bar['close'])
    
    def _execute_buy_order(self, symbol: str, quantity: int, price: float):
        """買い注文実行"""
        # スリッページとコミッションを考慮
        execution_price = price * (1 + self.slippage_rate)
        total_cost = execution_price * quantity * (1 + self.commission_rate)
        
        if self.current_capital >= total_cost:
            # 注文実行
            trade = Trade(
                symbol=symbol,
                order_type=OrderType.BUY,
                quantity=quantity,
                price=execution_price,
                timestamp=datetime.now(),
                status=OrderStatus.EXECUTED,
                trade_id=f"T{len(self.trades)+1:06d}"
            )
            self.trades.append(trade)
            
            # ポジション更新
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_quantity = pos.quantity + quantity
                avg_price = ((pos.avg_price * pos.quantity) + 
                           (execution_price * quantity)) / total_quantity
                pos.quantity = total_quantity
                pos.avg_price = avg_price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=execution_price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                )
            
            # 資金更新
            self.current_capital -= total_cost
    
    def _execute_sell_order(self, symbol: str, quantity: int, price: float):
        """売り注文実行"""
        if symbol not in self.positions:
            return  # ポジションなし
        
        position = self.positions[symbol]
        sell_quantity = min(quantity, position.quantity)
        
        if sell_quantity <= 0:
            return
        
        # スリッページとコミッションを考慮
        execution_price = price * (1 - self.slippage_rate)
        total_proceeds = execution_price * sell_quantity * (1 - self.commission_rate)
        
        # 注文実行
        trade = Trade(
            symbol=symbol,
            order_type=OrderType.SELL,
            quantity=sell_quantity,
            price=execution_price,
            timestamp=datetime.now(),
            status=OrderStatus.EXECUTED,
            trade_id=f"T{len(self.trades)+1:06d}"
        )
        self.trades.append(trade)
        
        # 実現損益計算
        realized_pnl = (execution_price - position.avg_price) * sell_quantity
        position.realized_pnl += realized_pnl
        
        # ポジション更新
        position.quantity -= sell_quantity
        if position.quantity == 0:
            del self.positions[symbol]
        
        # 資金更新
        self.current_capital += total_proceeds
    
    def _update_positions(self, current_bar: pd.Series):
        """ポジション更新"""
        for symbol, position in self.positions.items():
            position.current_price = current_bar['close']
            position.unrealized_pnl = ((position.current_price - position.avg_price) * 
                                     position.quantity)
    
    def _calculate_total_equity(self, current_bar: pd.Series) -> float:
        """総資産価値計算"""
        total_equity = self.current_capital
        
        for symbol, position in self.positions.items():
            market_value = position.current_price * position.quantity
            total_equity += market_value
        
        return total_equity
    
    def _generate_result(self) -> BacktestResult:
        """結果生成"""
        final_capital = self.equity_history[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 最大ドローダウン計算
        equity_series = pd.Series(self.equity_history)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # シャープレシオ計算（簡易版）
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 年率化
        else:
            sharpe_ratio = 0.0
        
        # 勝率計算
        winning_trades = 0
        losing_trades = 0
        
        for i in range(1, len(self.trades)):
            if (i > 0 and 
                self.trades[i].order_type == OrderType.SELL and
                self.trades[i-1].order_type == OrderType.BUY and
                self.trades[i].symbol == self.trades[i-1].symbol):
                
                profit = ((self.trades[i].price - self.trades[i-1].price) * 
                         self.trades[i].quantity)
                
                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        total_completed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_completed_trades if total_completed_trades > 0 else 0.0
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            trades=self.trades.copy(),
            equity_curve=equity_series,
            performance_metrics={
                'volatility': returns.std() if len(returns) > 0 else 0.0,
                'best_trade': max([t.price for t in self.trades]) if self.trades else 0.0,
                'worst_trade': min([t.price for t in self.trades]) if self.trades else 0.0,
                'average_trade': np.mean([t.price for t in self.trades]) if self.trades else 0.0
            }
        )


class MockTradingStrategy:
    """取引戦略モック"""
    
    def __init__(self, name: str = "Test Strategy"):
        self.name = name
        self.parameters = {
            'sma_short': 20,
            'sma_long': 50,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """シグナル生成"""
        if len(data) < self.parameters['sma_long']:
            return []
        
        signals = []
        close = data['close']
        
        # 単純移動平均クロス戦略
        sma_short = close.rolling(window=self.parameters['sma_short']).mean()
        sma_long = close.rolling(window=self.parameters['sma_long']).mean()
        
        # RSI計算（簡易版）
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 最新の値
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        prev_sma_short = sma_short.iloc[-2]
        prev_sma_long = sma_long.iloc[-2]
        current_rsi = rsi.iloc[-1]
        
        # ゴールデンクロス（買いシグナル）
        if (current_sma_short > current_sma_long and 
            prev_sma_short <= prev_sma_long and
            current_rsi < 70):  # RSIが買われすぎでない
            signals.append({
                'symbol': 'TEST',
                'action': 'BUY',
                'quantity': 100,
                'reason': 'Golden Cross'
            })
        
        # デッドクロス（売りシグナル）
        elif (current_sma_short < current_sma_long and 
              prev_sma_short >= prev_sma_long and
              current_rsi > 30):  # RSIが売られすぎでない
            signals.append({
                'symbol': 'TEST',
                'action': 'SELL',
                'quantity': 100,
                'reason': 'Dead Cross'
            })
        
        return signals


def create_sample_market_data(days: int = 252) -> pd.DataFrame:
    """サンプル市場データ作成"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # トレンドのあるランダムウォーク
    trend = 0.0002  # 日次0.02%の上昇トレンド
    volatility = 0.02  # 日次2%のボラティリティ
    
    returns = np.random.normal(trend, volatility, days)
    prices = 1000 * np.exp(np.cumsum(returns))  # 幾何ブラウン運動
    
    # OHLC生成
    opens = prices * (1 + np.random.normal(0, 0.005, days))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    volumes = np.random.randint(100000, 1000000, days)
    
    return pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })


class TestMockBacktestEngine:
    """バックテストエンジンテストクラス"""
    
    @pytest.fixture
    def backtest_engine(self):
        """バックテストエンジンフィクスチャ"""
        return MockBacktestEngine(initial_capital=1000000)
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータフィクスチャ"""
        return create_sample_market_data()
    
    @pytest.fixture
    def trading_strategy(self):
        """取引戦略フィクスチャ"""
        return MockTradingStrategy()
    
    def test_engine_initialization(self, backtest_engine):
        """エンジン初期化テスト"""
        assert backtest_engine.initial_capital == 1000000
        assert backtest_engine.current_capital == 1000000
        assert len(backtest_engine.positions) == 0
        assert len(backtest_engine.trades) == 0
        assert backtest_engine.equity_history == [1000000]
    
    def test_strategy_setup(self, backtest_engine, trading_strategy):
        """戦略設定テスト"""
        backtest_engine.add_strategy(trading_strategy)
        assert hasattr(backtest_engine, 'strategy')
        assert backtest_engine.strategy.name == "Test Strategy"
    
    def test_data_setup(self, backtest_engine, sample_data):
        """データ設定テスト"""
        backtest_engine.set_data(sample_data)
        assert hasattr(backtest_engine, 'data')
        assert len(backtest_engine.data) == len(sample_data)
        assert backtest_engine.current_bar == 0
    
    def test_buy_order_execution(self, backtest_engine):
        """買い注文実行テスト"""
        symbol = "TEST"
        quantity = 100
        price = 1000.0
        
        initial_capital = backtest_engine.current_capital
        backtest_engine._execute_buy_order(symbol, quantity, price)
        
        # 資金減少チェック
        assert backtest_engine.current_capital < initial_capital
        
        # ポジション作成チェック
        assert symbol in backtest_engine.positions
        assert backtest_engine.positions[symbol].quantity == quantity
        
        # 取引記録チェック
        assert len(backtest_engine.trades) == 1
        assert backtest_engine.trades[0].symbol == symbol
        assert backtest_engine.trades[0].order_type == OrderType.BUY
    
    def test_sell_order_execution(self, backtest_engine):
        """売り注文実行テスト"""
        symbol = "TEST"
        quantity = 100
        buy_price = 1000.0
        sell_price = 1100.0
        
        # まず買い注文
        backtest_engine._execute_buy_order(symbol, quantity, buy_price)
        capital_after_buy = backtest_engine.current_capital
        
        # 売り注文
        backtest_engine._execute_sell_order(symbol, quantity, sell_price)
        
        # 資金増加チェック（利益分）
        assert backtest_engine.current_capital > capital_after_buy
        
        # ポジションクローズチェック
        assert symbol not in backtest_engine.positions
        
        # 取引記録チェック
        assert len(backtest_engine.trades) == 2
        assert backtest_engine.trades[1].order_type == OrderType.SELL
    
    def test_insufficient_funds(self, backtest_engine):
        """資金不足テスト"""
        # 資金を超える大きな注文
        symbol = "TEST"
        quantity = 100000  # 大量
        price = 1000.0
        
        initial_positions = len(backtest_engine.positions)
        initial_trades = len(backtest_engine.trades)
        
        backtest_engine._execute_buy_order(symbol, quantity, price)
        
        # 注文が実行されないことを確認
        assert len(backtest_engine.positions) == initial_positions
        assert len(backtest_engine.trades) == initial_trades
    
    def test_sell_without_position(self, backtest_engine):
        """ポジションなしでの売り注文テスト"""
        symbol = "TEST"
        quantity = 100
        price = 1000.0
        
        initial_capital = backtest_engine.current_capital
        initial_trades = len(backtest_engine.trades)
        
        backtest_engine._execute_sell_order(symbol, quantity, price)
        
        # 何も変化しないことを確認
        assert backtest_engine.current_capital == initial_capital
        assert len(backtest_engine.trades) == initial_trades
    
    def test_full_backtest_run(self, backtest_engine, sample_data, trading_strategy):
        """完全バックテスト実行テスト"""
        backtest_engine.add_strategy(trading_strategy)
        backtest_engine.set_data(sample_data)
        
        result = backtest_engine.run_backtest()
        
        # 結果の基本チェック
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 1000000
        assert result.final_capital > 0
        assert isinstance(result.total_return, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.win_rate, float)
        assert result.total_trades >= 0
        assert len(result.trades) == result.total_trades
        assert len(result.equity_curve) > 0
        assert isinstance(result.performance_metrics, dict)
    
    def test_performance_metrics_calculation(self, backtest_engine, sample_data, trading_strategy):
        """パフォーマンス指標計算テスト"""
        backtest_engine.add_strategy(trading_strategy)
        backtest_engine.set_data(sample_data)
        
        result = backtest_engine.run_backtest()
        
        # リターン計算
        expected_return = (result.final_capital - result.initial_capital) / result.initial_capital
        assert abs(result.total_return - expected_return) < 0.0001
        
        # 最大ドローダウンは負の値またはゼロ
        assert result.max_drawdown <= 0
        
        # 勝率は0-1の範囲
        assert 0 <= result.win_rate <= 1
        
        # 取引数の整合性
        assert result.winning_trades + result.losing_trades <= result.total_trades
    
    def test_equity_curve_generation(self, backtest_engine, sample_data, trading_strategy):
        """エクイティカーブ生成テスト"""
        backtest_engine.add_strategy(trading_strategy)
        backtest_engine.set_data(sample_data)
        
        result = backtest_engine.run_backtest()
        
        # エクイティカーブの基本チェック
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == result.initial_capital
        assert result.equity_curve.iloc[-1] == result.final_capital
        
        # 値がすべて正
        assert (result.equity_curve > 0).all()


class TestMockTradingStrategy:
    """取引戦略テストクラス"""
    
    @pytest.fixture
    def strategy(self):
        """戦略フィクスチャ"""
        return MockTradingStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータフィクスチャ"""
        return create_sample_market_data()
    
    def test_strategy_initialization(self, strategy):
        """戦略初期化テスト"""
        assert strategy.name == "Test Strategy"
        assert 'sma_short' in strategy.parameters
        assert 'sma_long' in strategy.parameters
        assert strategy.parameters['sma_short'] < strategy.parameters['sma_long']
    
    def test_signal_generation_insufficient_data(self, strategy):
        """データ不足時のシグナル生成テスト"""
        # 不十分なデータ
        short_data = create_sample_market_data(20)  # 20日分のみ
        signals = strategy.generate_signals(short_data)
        
        # シグナルなし
        assert len(signals) == 0
    
    def test_signal_generation_sufficient_data(self, strategy, sample_data):
        """十分なデータでのシグナル生成テスト"""
        signals = strategy.generate_signals(sample_data)
        
        # シグナル構造チェック
        for signal in signals:
            assert 'symbol' in signal
            assert 'action' in signal
            assert 'quantity' in signal
            assert 'reason' in signal
            assert signal['action'] in ['BUY', 'SELL']
            assert signal['quantity'] > 0
    
    def test_parameter_modification(self, strategy):
        """パラメータ変更テスト"""
        original_sma_short = strategy.parameters['sma_short']
        
        # パラメータ変更
        strategy.parameters['sma_short'] = 10
        
        assert strategy.parameters['sma_short'] != original_sma_short
        assert strategy.parameters['sma_short'] == 10


class TestBacktestIntegration:
    """バックテスト統合テスト"""
    
    def test_complete_backtest_workflow(self):
        """完全バックテストワークフローテスト"""
        # コンポーネント作成
        engine = MockBacktestEngine(initial_capital=500000)
        strategy = MockTradingStrategy()
        data = create_sample_market_data(100)
        
        # バックテスト実行
        engine.add_strategy(strategy)
        engine.set_data(data)
        result = engine.run_backtest()
        
        # 結果の妥当性チェック
        assert result.initial_capital == 500000
        assert result.final_capital > 0
        assert len(result.equity_curve) > 0
    
    def test_multiple_strategies_comparison(self):
        """複数戦略比較テスト"""
        data = create_sample_market_data(200)
        
        # 戦略1: 短期SMA
        strategy1 = MockTradingStrategy("Short SMA")
        strategy1.parameters['sma_short'] = 10
        strategy1.parameters['sma_long'] = 30
        
        # 戦略2: 長期SMA
        strategy2 = MockTradingStrategy("Long SMA")
        strategy2.parameters['sma_short'] = 30
        strategy2.parameters['sma_long'] = 60
        
        # 両戦略でバックテスト
        results = []
        for strategy in [strategy1, strategy2]:
            engine = MockBacktestEngine(initial_capital=1000000)
            engine.add_strategy(strategy)
            engine.set_data(data)
            result = engine.run_backtest()
            results.append(result)
        
        # 両方とも有効な結果
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.final_capital > 0
    
    def test_different_market_conditions(self):
        """異なる市場条件テスト"""
        strategy = MockTradingStrategy()
        
        # 上昇市場
        bull_data = create_sample_market_data(100)
        bull_data['close'] = bull_data['close'] * np.exp(np.linspace(0, 0.2, 100))
        
        # 下降市場
        bear_data = create_sample_market_data(100)
        bear_data['close'] = bear_data['close'] * np.exp(np.linspace(0, -0.2, 100))
        
        # 横ばい市場
        sideways_data = create_sample_market_data(100)
        sideways_data['close'] = sideways_data['close'].iloc[0] + np.random.normal(0, 10, 100)
        
        markets = [('bull', bull_data), ('bear', bear_data), ('sideways', sideways_data)]
        
        for market_name, market_data in markets:
            engine = MockBacktestEngine(initial_capital=1000000)
            engine.add_strategy(strategy)
            engine.set_data(market_data)
            result = engine.run_backtest()
            
            # 各市場で有効な結果
            assert isinstance(result, BacktestResult)
            assert result.final_capital > 0
    
    def test_commission_and_slippage_impact(self):
        """コミッションとスリッページの影響テスト"""
        strategy = MockTradingStrategy()
        data = create_sample_market_data(100)
        
        # コミッション・スリッページなし
        engine1 = MockBacktestEngine(initial_capital=1000000)
        engine1.commission_rate = 0.0
        engine1.slippage_rate = 0.0
        engine1.add_strategy(strategy)
        engine1.set_data(data)
        result1 = engine1.run_backtest()
        
        # コミッション・スリッページあり
        engine2 = MockBacktestEngine(initial_capital=1000000)
        engine2.commission_rate = 0.002  # 0.2%
        engine2.slippage_rate = 0.001    # 0.1%
        engine2.add_strategy(strategy)
        engine2.set_data(data)
        result2 = engine2.run_backtest()
        
        # コミッション・スリッページありの方が成績が悪いはず
        if result1.total_trades > 0:
            assert result2.final_capital <= result1.final_capital


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
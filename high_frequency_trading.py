#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High Frequency Trading Engine - 高頻度取引最適化システム
Issue #942対応: マイクロ秒レベル実行 + アルゴリズムトレーディング
"""

import asyncio
import numpy as np
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit, njit
import cython

# 統合モジュール
try:
    from quantum_ai_engine import quantum_ai_engine
    HAS_QUANTUM_AI = True
except ImportError:
    HAS_QUANTUM_AI = False

try:
    from blockchain_trading import trading_blockchain
    HAS_BLOCKCHAIN = True
except ImportError:
    HAS_BLOCKCHAIN = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False


@dataclass
class MarketData:
    """市場データ"""
    symbol: str
    timestamp: float
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    last_price: float
    volume: float
    sequence_id: int = 0


@dataclass
class Order:
    """注文"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    timestamp: float
    client_id: str
    status: str = 'NEW'  # NEW, FILLED, CANCELLED, REJECTED
    filled_quantity: float = 0.0
    average_price: float = 0.0


@dataclass
class Trade:
    """約定"""
    trade_id: str
    symbol: str
    price: float
    quantity: float
    timestamp: float
    buyer_order_id: str
    seller_order_id: str
    aggressor_side: str


@dataclass
class StrategyParams:
    """戦略パラメータ"""
    strategy_name: str
    symbol: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True


class MarketDataProcessor:
    """市場データ処理エンジン"""
    
    def __init__(self, max_depth: int = 1000):
        self.max_depth = max_depth
        self.order_books: Dict[str, Dict] = {}
        self.latest_trades: Dict[str, deque] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 統計計算用
        self.statistics: Dict[str, Dict] = defaultdict(dict)
        
        # 高速計算用バッファ
        self.fast_arrays = {}
    
    def update_order_book(self, symbol: str, bid_data: List[Tuple[float, float]], 
                         ask_data: List[Tuple[float, float]], timestamp: float):
        """板情報更新（最適化済み）"""
        if symbol not in self.order_books:
            self.order_books[symbol] = {
                'bids': {},
                'asks': {},
                'timestamp': timestamp,
                'sequence_id': 0
            }
        
        book = self.order_books[symbol]
        book['timestamp'] = timestamp
        book['sequence_id'] += 1
        
        # 高速更新
        book['bids'] = dict(bid_data[:self.max_depth])
        book['asks'] = dict(ask_data[:self.max_depth])
        
        # 統計更新
        self._update_statistics(symbol, book)
    
    @njit
    def _calculate_vwap_fast(prices: np.ndarray, volumes: np.ndarray) -> float:
        """VWAP高速計算"""
        if len(volumes) == 0 or np.sum(volumes) == 0:
            return 0.0
        return np.sum(prices * volumes) / np.sum(volumes)
    
    @njit
    def _calculate_volatility_fast(prices: np.ndarray, window: int = 20) -> float:
        """ボラティリティ高速計算"""
        if len(prices) < window:
            return 0.0
        
        returns = np.diff(np.log(prices[-window:]))
        return np.std(returns) * np.sqrt(252 * 24 * 60)  # 年率化
    
    def _update_statistics(self, symbol: str, book: Dict):
        """統計更新"""
        try:
            bids = book['bids']
            asks = book['asks']
            
            if not bids or not asks:
                return
            
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2.0
            
            # 価格履歴更新
            self.price_history[symbol].append(mid_price)
            
            stats = self.statistics[symbol]
            stats.update({
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_bps': (spread / mid_price) * 10000,
                'mid_price': mid_price,
                'timestamp': book['timestamp']
            })
            
            # VWAP計算（高速バージョン）
            if len(self.price_history[symbol]) >= 10:
                prices = np.array(list(self.price_history[symbol])[-100:])
                volumes = np.ones_like(prices)  # 簡略化
                
                stats['vwap'] = self._calculate_vwap_fast(prices, volumes)
                stats['volatility'] = self._calculate_volatility_fast(prices)
            
        except Exception as e:
            logging.error(f"Statistics update error for {symbol}: {e}")
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """市場データ取得"""
        if symbol not in self.order_books:
            return None
        
        book = self.order_books[symbol]
        stats = self.statistics.get(symbol, {})
        
        return MarketData(
            symbol=symbol,
            timestamp=book['timestamp'],
            bid_price=stats.get('best_bid', 0.0),
            ask_price=stats.get('best_ask', 0.0),
            bid_volume=0.0,  # 簡略化
            ask_volume=0.0,  # 簡略化
            last_price=stats.get('mid_price', 0.0),
            volume=0.0,
            sequence_id=book['sequence_id']
        )
    
    def get_statistics(self, symbol: str) -> Dict[str, Any]:
        """統計取得"""
        return self.statistics.get(symbol, {})


class OrderMatchingEngine:
    """注文照合エンジン"""
    
    def __init__(self):
        self.buy_orders: Dict[str, List[Order]] = defaultdict(list)
        self.sell_orders: Dict[str, List[Order]] = defaultdict(list)
        self.order_index: Dict[str, Order] = {}
        
        self.trade_id_counter = 0
        self.executed_trades: List[Trade] = []
        
        # パフォーマンス統計
        self.matching_stats = {
            'total_matches': 0,
            'average_latency_us': 0.0,
            'peak_throughput': 0,
            'total_volume': 0.0
        }
    
    def add_order(self, order: Order) -> List[Trade]:
        """注文追加と照合"""
        start_time = time.perf_counter()
        
        trades = []
        self.order_index[order.order_id] = order
        
        if order.side == 'BUY':
            trades = self._match_buy_order(order)
            if order.status == 'NEW':
                heapq.heappush(self.buy_orders[order.symbol], 
                             (-order.price, order.timestamp, order))
        else:
            trades = self._match_sell_order(order)
            if order.status == 'NEW':
                heapq.heappush(self.sell_orders[order.symbol], 
                             (order.price, order.timestamp, order))
        
        # パフォーマンス測定
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        self._update_matching_stats(latency_us, len(trades))
        
        return trades
    
    def _match_buy_order(self, buy_order: Order) -> List[Trade]:
        """買い注文照合"""
        trades = []
        remaining_qty = buy_order.quantity - buy_order.filled_quantity
        
        sell_queue = self.sell_orders[buy_order.symbol]
        
        while sell_queue and remaining_qty > 0:
            sell_price, sell_time, sell_order = sell_queue[0]
            
            # 価格条件チェック
            if buy_order.order_type == 'MARKET' or buy_order.price >= sell_price:
                heapq.heappop(sell_queue)
                
                # 約定数量計算
                available_qty = sell_order.quantity - sell_order.filled_quantity
                trade_qty = min(remaining_qty, available_qty)
                
                if trade_qty > 0:
                    # 約定生成
                    trade = self._create_trade(buy_order, sell_order, trade_qty, sell_price)
                    trades.append(trade)
                    
                    # 注文更新
                    buy_order.filled_quantity += trade_qty
                    sell_order.filled_quantity += trade_qty
                    remaining_qty -= trade_qty
                    
                    # ステータス更新
                    if buy_order.filled_quantity >= buy_order.quantity:
                        buy_order.status = 'FILLED'
                    if sell_order.filled_quantity >= sell_order.quantity:
                        sell_order.status = 'FILLED'
                    else:
                        # 部分約定の売り注文を再度キューに追加
                        heapq.heappush(sell_queue, (sell_price, sell_time, sell_order))
            else:
                break
        
        return trades
    
    def _match_sell_order(self, sell_order: Order) -> List[Trade]:
        """売り注文照合"""
        trades = []
        remaining_qty = sell_order.quantity - sell_order.filled_quantity
        
        buy_queue = self.buy_orders[sell_order.symbol]
        
        while buy_queue and remaining_qty > 0:
            neg_buy_price, buy_time, buy_order = buy_queue[0]
            buy_price = -neg_buy_price
            
            # 価格条件チェック
            if sell_order.order_type == 'MARKET' or sell_order.price <= buy_price:
                heapq.heappop(buy_queue)
                
                # 約定数量計算
                available_qty = buy_order.quantity - buy_order.filled_quantity
                trade_qty = min(remaining_qty, available_qty)
                
                if trade_qty > 0:
                    # 約定生成
                    trade = self._create_trade(buy_order, sell_order, trade_qty, buy_price)
                    trades.append(trade)
                    
                    # 注文更新
                    sell_order.filled_quantity += trade_qty
                    buy_order.filled_quantity += trade_qty
                    remaining_qty -= trade_qty
                    
                    # ステータス更新
                    if sell_order.filled_quantity >= sell_order.quantity:
                        sell_order.status = 'FILLED'
                    if buy_order.filled_quantity >= buy_order.quantity:
                        buy_order.status = 'FILLED'
                    else:
                        # 部分約定の買い注文を再度キューに追加
                        heapq.heappush(buy_queue, (neg_buy_price, buy_time, buy_order))
            else:
                break
        
        return trades
    
    def _create_trade(self, buy_order: Order, sell_order: Order, 
                     quantity: float, price: float) -> Trade:
        """約定生成"""
        self.trade_id_counter += 1
        
        trade = Trade(
            trade_id=f"T{self.trade_id_counter:08d}",
            symbol=buy_order.symbol,
            price=price,
            quantity=quantity,
            timestamp=time.time(),
            buyer_order_id=buy_order.order_id,
            seller_order_id=sell_order.order_id,
            aggressor_side=buy_order.side if buy_order.timestamp > sell_order.timestamp else sell_order.side
        )
        
        self.executed_trades.append(trade)
        return trade
    
    def _update_matching_stats(self, latency_us: float, trade_count: int):
        """照合統計更新"""
        stats = self.matching_stats
        
        # 移動平均での待ち時間更新
        alpha = 0.1
        stats['average_latency_us'] = (
            alpha * latency_us + (1 - alpha) * stats['average_latency_us']
        )
        
        stats['total_matches'] += trade_count
        
        # スループット更新（トレード/秒）
        if trade_count > 0:
            throughput = 1_000_000 / latency_us * trade_count  # trades per second
            stats['peak_throughput'] = max(stats['peak_throughput'], throughput)
    
    def get_order_book_snapshot(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """板スナップショット取得"""
        buy_queue = self.buy_orders.get(symbol, [])
        sell_queue = self.sell_orders.get(symbol, [])
        
        # 上位depth件を取得
        top_buys = sorted([(-price, time_stamp, order) 
                          for price, time_stamp, order in buy_queue 
                          if order.status == 'NEW'], reverse=True)[:depth]
        
        top_sells = sorted([(price, time_stamp, order) 
                           for price, time_stamp, order in sell_queue 
                           if order.status == 'NEW'])[:depth]
        
        return {
            'symbol': symbol,
            'bids': [(float(-price), order.quantity - order.filled_quantity) 
                    for price, _, order in top_buys],
            'asks': [(price, order.quantity - order.filled_quantity) 
                    for price, _, order in top_sells],
            'timestamp': time.time()
        }


class HighFrequencyStrategy:
    """高频度取引戦略基底クラス"""
    
    def __init__(self, name: str, symbols: List[str], params: StrategyParams):
        self.name = name
        self.symbols = symbols
        self.params = params
        
        self.positions: Dict[str, float] = defaultdict(float)
        self.pnl = 0.0
        self.trades_today = 0
        
        # リスク管理
        self.max_position = params.risk_limits.get('max_position', 1000.0)
        self.max_trades_per_second = params.risk_limits.get('max_trades_per_second', 10)
        self.max_daily_loss = params.risk_limits.get('max_daily_loss', -10000.0)
        
        self.last_trade_time = 0.0
        self.trade_count_1s = 0
    
    def should_trade(self, symbol: str, market_data: MarketData) -> bool:
        """取引可否判定"""
        # リスク制限チェック
        if abs(self.positions[symbol]) >= self.max_position:
            return False
        
        if self.pnl <= self.max_daily_loss:
            return False
        
        # 取引頻度制限
        current_time = time.time()
        if current_time - self.last_trade_time < 1.0:
            if self.trade_count_1s >= self.max_trades_per_second:
                return False
        else:
            self.trade_count_1s = 0
        
        return True
    
    def generate_signals(self, symbol: str, market_data: MarketData) -> List[Order]:
        """シグナル生成（派生クラスで実装）"""
        raise NotImplementedError
    
    def update_position(self, trade: Trade):
        """ポジション更新"""
        if trade.buyer_order_id.startswith(self.name):
            self.positions[trade.symbol] += trade.quantity
            self.pnl -= trade.quantity * trade.price
        elif trade.seller_order_id.startswith(self.name):
            self.positions[trade.symbol] -= trade.quantity
            self.pnl += trade.quantity * trade.price
        
        self.trades_today += 1


class MarketMakingStrategy(HighFrequencyStrategy):
    """マーケットメイキング戦略"""
    
    def generate_signals(self, symbol: str, market_data: MarketData) -> List[Order]:
        """マーケットメイキングシグナル生成"""
        if not self.should_trade(symbol, market_data):
            return []
        
        orders = []
        
        # スプレッドとボラティリティベースの価格計算
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.bid_price + market_data.ask_price) / 2.0
        
        # 動的スプレッド調整
        min_spread = self.params.parameters.get('min_spread_bps', 10) / 10000 * mid_price
        target_spread = max(spread * 0.3, min_spread)
        
        quantity = self.params.parameters.get('base_quantity', 100.0)
        
        # 買い注文
        buy_price = mid_price - target_spread / 2
        buy_order = Order(
            order_id=f"{self.name}_BUY_{symbol}_{int(time.time() * 1000000)}",
            symbol=symbol,
            side='BUY',
            price=buy_price,
            quantity=quantity,
            order_type='LIMIT',
            timestamp=time.time(),
            client_id=self.name
        )
        orders.append(buy_order)
        
        # 売り注文
        sell_price = mid_price + target_spread / 2
        sell_order = Order(
            order_id=f"{self.name}_SELL_{symbol}_{int(time.time() * 1000000)}",
            symbol=symbol,
            side='SELL',
            price=sell_price,
            quantity=quantity,
            order_type='LIMIT',
            timestamp=time.time(),
            client_id=self.name
        )
        orders.append(sell_order)
        
        return orders


class ArbitrageStrategy(HighFrequencyStrategy):
    """裁定取引戦略"""
    
    def __init__(self, name: str, symbols: List[str], params: StrategyParams):
        super().__init__(name, symbols, params)
        self.price_pairs = {}
        self.correlation_threshold = params.parameters.get('correlation_threshold', 0.95)
    
    def generate_signals(self, symbol: str, market_data: MarketData) -> List[Order]:
        """裁定機会検出"""
        if not self.should_trade(symbol, market_data):
            return []
        
        orders = []
        
        # ペア取引検出（簡略化）
        for other_symbol in self.symbols:
            if other_symbol != symbol and other_symbol in self.price_pairs:
                arb_signal = self._detect_arbitrage(symbol, other_symbol, market_data)
                
                if arb_signal:
                    orders.extend(arb_signal)
        
        self.price_pairs[symbol] = market_data.last_price
        return orders
    
    def _detect_arbitrage(self, symbol1: str, symbol2: str, current_data: MarketData) -> List[Order]:
        """裁定機会検出"""
        # 簡略化された裁定ロジック
        price1 = current_data.last_price
        price2 = self.price_pairs[symbol2]
        
        price_ratio = price1 / price2
        historical_ratio = self.params.parameters.get(f'ratio_{symbol1}_{symbol2}', 1.0)
        
        deviation = abs(price_ratio - historical_ratio) / historical_ratio
        
        if deviation > self.params.parameters.get('arb_threshold', 0.02):
            quantity = self.params.parameters.get('arb_quantity', 50.0)
            
            if price_ratio > historical_ratio:
                # symbol1が割高 -> 売り、symbol2が割安 -> 買い
                return [
                    Order(
                        order_id=f"{self.name}_ARB_SELL_{symbol1}_{int(time.time() * 1000000)}",
                        symbol=symbol1,
                        side='SELL',
                        price=price1,
                        quantity=quantity,
                        order_type='MARKET',
                        timestamp=time.time(),
                        client_id=self.name
                    )
                ]
        
        return []


class MomentumStrategy(HighFrequencyStrategy):
    """モメンタム戦略"""
    
    def __init__(self, name: str, symbols: List[str], params: StrategyParams):
        super().__init__(name, symbols, params)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.momentum_scores: Dict[str, float] = {}
    
    def generate_signals(self, symbol: str, market_data: MarketData) -> List[Order]:
        """モメンタムシグナル生成"""
        if not self.should_trade(symbol, market_data):
            return []
        
        self.price_history[symbol].append(market_data.last_price)
        
        if len(self.price_history[symbol]) < 20:
            return []
        
        # モメンタム計算
        momentum = self._calculate_momentum(symbol)
        self.momentum_scores[symbol] = momentum
        
        threshold = self.params.parameters.get('momentum_threshold', 0.02)
        quantity = self.params.parameters.get('momentum_quantity', 100.0)
        
        orders = []
        
        if momentum > threshold:
            # 強い上昇モメンタム -> 買い
            order = Order(
                order_id=f"{self.name}_MOM_BUY_{symbol}_{int(time.time() * 1000000)}",
                symbol=symbol,
                side='BUY',
                price=market_data.ask_price,
                quantity=quantity,
                order_type='MARKET',
                timestamp=time.time(),
                client_id=self.name
            )
            orders.append(order)
            
        elif momentum < -threshold:
            # 強い下降モメンタム -> 売り
            order = Order(
                order_id=f"{self.name}_MOM_SELL_{symbol}_{int(time.time() * 1000000)}",
                symbol=symbol,
                side='SELL',
                price=market_data.bid_price,
                quantity=quantity,
                order_type='MARKET',
                timestamp=time.time(),
                client_id=self.name
            )
            orders.append(order)
        
        return orders
    
    @njit
    def _calculate_momentum_fast(prices: np.ndarray, short_window: int, long_window: int) -> float:
        """高速モメンタム計算"""
        if len(prices) < long_window:
            return 0.0
        
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        
        return (short_ma - long_ma) / long_ma
    
    def _calculate_momentum(self, symbol: str) -> float:
        """モメンタム計算"""
        prices = np.array(list(self.price_history[symbol]))
        short_window = self.params.parameters.get('short_window', 5)
        long_window = self.params.parameters.get('long_window', 20)
        
        return self._calculate_momentum_fast(prices, short_window, long_window)


class HighFrequencyTradingEngine:
    """高頻度取引エンジン統合システム"""
    
    def __init__(self):
        self.market_processor = MarketDataProcessor()
        self.matching_engine = OrderMatchingEngine()
        
        # 戦略管理
        self.strategies: List[HighFrequencyStrategy] = []
        self.strategy_performance: Dict[str, Dict] = {}
        
        # 実行統計
        self.execution_stats = {
            'orders_per_second': 0.0,
            'trades_per_second': 0.0,
            'average_fill_ratio': 0.0,
            'total_volume': 0.0,
            'total_pnl': 0.0
        }
        
        # 非同期実行制御
        self.running = False
        self.order_queue = asyncio.Queue()
        self.market_data_queue = asyncio.Queue()
        
    def add_strategy(self, strategy: HighFrequencyStrategy):
        """戦略追加"""
        self.strategies.append(strategy)
        self.strategy_performance[strategy.name] = {
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    async def start(self):
        """エンジン開始"""
        self.running = True
        
        # 非同期タスク開始
        tasks = [
            self._market_data_processor(),
            self._order_processor(),
            self._strategy_executor(),
            self._performance_monitor()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """エンジン停止"""
        self.running = False
    
    async def _market_data_processor(self):
        """市場データ処理ループ"""
        while self.running:
            try:
                # 模擬市場データ生成
                await self._generate_market_data()
                await asyncio.sleep(0.001)  # 1ms間隔
                
            except Exception as e:
                logging.error(f"Market data processing error: {e}")
    
    async def _generate_market_data(self):
        """市場データ生成（シミュレーション）"""
        symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        
        for symbol in symbols:
            # ランダムウォークで価格生成
            base_price = 1000.0 + hash(symbol) % 500
            noise = np.random.normal(0, 1.0)
            
            bid_price = base_price + noise - 0.5
            ask_price = bid_price + np.random.uniform(0.1, 2.0)
            
            # 板情報生成
            bid_levels = [(bid_price - i * 0.1, np.random.uniform(100, 1000)) 
                         for i in range(5)]
            ask_levels = [(ask_price + i * 0.1, np.random.uniform(100, 1000)) 
                         for i in range(5)]
            
            # 板更新
            self.market_processor.update_order_book(
                symbol, bid_levels, ask_levels, time.time()
            )
            
            # 市場データキューに追加
            market_data = self.market_processor.get_market_data(symbol)
            if market_data:
                await self.market_data_queue.put(market_data)
    
    async def _order_processor(self):
        """注文処理ループ"""
        order_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # 注文処理（タイムアウト付き）
                order = await asyncio.wait_for(self.order_queue.get(), timeout=0.001)
                
                # 注文照合
                trades = self.matching_engine.add_order(order)
                
                # 約定処理
                for trade in trades:
                    await self._process_trade(trade)
                
                order_count += 1
                
                # 統計更新
                if order_count % 100 == 0:
                    elapsed = time.time() - start_time
                    self.execution_stats['orders_per_second'] = order_count / elapsed
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Order processing error: {e}")
    
    async def _strategy_executor(self):
        """戦略実行ループ"""
        while self.running:
            try:
                # 市場データ取得
                market_data = await asyncio.wait_for(
                    self.market_data_queue.get(), timeout=0.001
                )
                
                # 全戦略でシグナル生成
                for strategy in self.strategies:
                    if strategy.params.enabled and market_data.symbol in strategy.symbols:
                        orders = strategy.generate_signals(market_data.symbol, market_data)
                        
                        # 注文をキューに追加
                        for order in orders:
                            await self.order_queue.put(order)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Strategy execution error: {e}")
    
    async def _process_trade(self, trade: Trade):
        """約定処理"""
        # 戦略ポジション更新
        for strategy in self.strategies:
            if (trade.buyer_order_id.startswith(strategy.name) or 
                trade.seller_order_id.startswith(strategy.name)):
                strategy.update_position(trade)
                
                # 戦略パフォーマンス更新
                perf = self.strategy_performance[strategy.name]
                perf['total_trades'] += 1
                perf['total_pnl'] = strategy.pnl
        
        # 統計更新
        self.execution_stats['total_volume'] += trade.quantity * trade.price
        
        # ブロックチェーン記録（利用可能な場合）
        if HAS_BLOCKCHAIN:
            try:
                await self._record_trade_on_blockchain(trade)
            except Exception as e:
                logging.error(f"Blockchain recording error: {e}")
    
    async def _record_trade_on_blockchain(self, trade: Trade):
        """取引のブロックチェーン記録"""
        # 非同期でブロックチェーンに記録
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            None,
            trading_blockchain.execute_automated_trade,
            trade.symbol,
            'BUY' if trade.aggressor_side == 'BUY' else 'SELL',
            trade.quantity,
            trade.price
        )
    
    async def _performance_monitor(self):
        """パフォーマンス監視ループ"""
        while self.running:
            try:
                # システムパフォーマンス更新
                self._update_system_performance()
                
                # 1秒間隔でパフォーマンス監視
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
    
    def _update_system_performance(self):
        """システムパフォーマンス更新"""
        # 戦略パフォーマンス計算
        for strategy in self.strategies:
            perf = self.strategy_performance[strategy.name]
            
            # シャープレシオ計算（簡略化）
            if strategy.trades_today > 0:
                daily_return = strategy.pnl / strategy.trades_today
                perf['sharpe_ratio'] = daily_return / (abs(daily_return) * 0.1 + 1e-6)
        
        # 全体統計更新
        total_pnl = sum(s.pnl for s in self.strategies)
        self.execution_stats['total_pnl'] = total_pnl
        
        # パフォーマンス監視統合
        if HAS_PERFORMANCE_MONITOR:
            performance_monitor.track_analysis_time(
                'HFT_ENGINE', 0.001, 'high_frequency_trading'
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート取得"""
        return {
            'execution_statistics': self.execution_stats.copy(),
            'strategy_performance': self.strategy_performance.copy(),
            'matching_engine_stats': self.matching_engine.matching_stats.copy(),
            'market_data_stats': {
                symbol: self.market_processor.get_statistics(symbol)
                for symbol in ['STOCK_A', 'STOCK_B', 'STOCK_C']
            },
            'system_status': {
                'running': self.running,
                'active_strategies': len([s for s in self.strategies if s.params.enabled]),
                'total_strategies': len(self.strategies),
                'queue_sizes': {
                    'orders': self.order_queue.qsize(),
                    'market_data': self.market_data_queue.qsize()
                }
            }
        }


# グローバルインスタンス
hft_engine = HighFrequencyTradingEngine()


async def setup_and_test_hft():
    """HFTエンジンセットアップとテスト"""
    print("=== High Frequency Trading Engine Test ===")
    
    # 戦略作成とセットアップ
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
    # マーケットメイキング戦略
    mm_params = StrategyParams(
        strategy_name='MarketMaking',
        symbol='STOCK_A',
        parameters={
            'min_spread_bps': 5,
            'base_quantity': 100.0,
        },
        risk_limits={
            'max_position': 1000.0,
            'max_trades_per_second': 5,
            'max_daily_loss': -5000.0
        }
    )
    
    mm_strategy = MarketMakingStrategy('MM_STRATEGY', symbols, mm_params)
    hft_engine.add_strategy(mm_strategy)
    
    # モメンタム戦略
    momentum_params = StrategyParams(
        strategy_name='Momentum',
        symbol='STOCK_B',
        parameters={
            'momentum_threshold': 0.015,
            'momentum_quantity': 50.0,
            'short_window': 5,
            'long_window': 20
        },
        risk_limits={
            'max_position': 500.0,
            'max_trades_per_second': 3
        }
    )
    
    momentum_strategy = MomentumStrategy('MOMENTUM_STRATEGY', symbols, momentum_params)
    hft_engine.add_strategy(momentum_strategy)
    
    print(f"HFT Engine initialized with {len(hft_engine.strategies)} strategies")
    print("Strategies:", [s.name for s in hft_engine.strategies])
    
    # 短時間テスト実行
    print("\nRunning HFT simulation for 5 seconds...")
    
    # エンジン開始（5秒間）
    try:
        await asyncio.wait_for(hft_engine.start(), timeout=5.0)
    except asyncio.TimeoutError:
        await hft_engine.stop()
    
    # 結果表示
    print("\n=== Performance Report ===")
    report = hft_engine.get_performance_report()
    
    print(f"Orders per second: {report['execution_statistics']['orders_per_second']:.2f}")
    print(f"Total volume: ${report['execution_statistics']['total_volume']:,.2f}")
    print(f"Total PnL: ${report['execution_statistics']['total_pnl']:,.2f}")
    print(f"Matching engine latency: {report['matching_engine_stats']['average_latency_us']:.1f}μs")
    print(f"Peak throughput: {report['matching_engine_stats']['peak_throughput']:.0f} trades/sec")
    
    print("\nStrategy Performance:")
    for strategy_name, perf in report['strategy_performance'].items():
        print(f"  {strategy_name}:")
        print(f"    Trades: {perf['total_trades']}")
        print(f"    PnL: ${perf['total_pnl']:,.2f}")
        print(f"    Sharpe Ratio: {perf['sharpe_ratio']:.3f}")


if __name__ == "__main__":
    # HFTエンジンテスト実行
    asyncio.run(setup_and_test_hft())
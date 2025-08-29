#!/usr/bin/env python3
"""
Reactive Streams Implementation
リアクティブストリーミング実装
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import rx
from rx import operators as ops
from rx.core import Observer, Observable
from rx.subject import Subject, BehaviorSubject, ReplaySubject

T = TypeVar('T')


@dataclass
class MarketData:
    """市場データ"""
    symbol: str
    price: Decimal
    volume: int
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    
    @property
    def mid_price(self) -> Decimal:
        """中値計算"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.price


@dataclass
class TradeEvent:
    """取引イベント"""
    trade_id: UUID
    symbol: str
    side: str
    quantity: int
    price: Decimal
    timestamp: datetime
    portfolio_id: UUID
    execution_id: Optional[UUID] = None


@dataclass
class TechnicalSignal:
    """テクニカルシグナル"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 - 1.0
    indicator: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class ReactiveStream(ABC, Generic[T]):
    """リアクティブストリーム基底クラス"""
    
    def __init__(self):
        self._subject = Subject()
        self._subscribers: List[Observer] = []
        self._is_active = False
    
    @abstractmethod
    async def start(self) -> None:
        """ストリーム開始"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """ストリーム停止"""
        pass
    
    def subscribe(self, observer: Observer) -> None:
        """オブザーバー登録"""
        self._subscribers.append(observer)
        self._subject.subscribe(observer)
    
    def emit(self, value: T) -> None:
        """値の発行"""
        self._subject.on_next(value)
    
    def error(self, error: Exception) -> None:
        """エラー発行"""
        self._subject.on_error(error)
    
    def complete(self) -> None:
        """完了通知"""
        self._subject.on_completed()
    
    @property
    def observable(self) -> Observable:
        """Observable取得"""
        return self._subject.as_observable()


class MarketDataStream(ReactiveStream[MarketData]):
    """市場データストリーム"""
    
    def __init__(self, symbols: List[str], update_interval: float = 1.0):
        super().__init__()
        self.symbols = symbols
        self.update_interval = update_interval
        self._update_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """市場データストリーム開始"""
        self._is_active = True
        self._update_task = asyncio.create_task(self._data_update_loop())
    
    async def stop(self) -> None:
        """市場データストリーム停止"""
        self._is_active = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
    
    async def _data_update_loop(self) -> None:
        """データ更新ループ"""
        while self._is_active:
            try:
                for symbol in self.symbols:
                    market_data = await self._fetch_market_data(symbol)
                    self.emit(market_data)
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error(e)
                await asyncio.sleep(self.update_interval)
    
    async def _fetch_market_data(self, symbol: str) -> MarketData:
        """市場データ取得"""
        # 実際の実装では外部APIやWebSocket接続
        import random
        base_price = 1000
        price_change = (random.random() - 0.5) * 20
        
        return MarketData(
            symbol=symbol,
            price=Decimal(str(base_price + price_change)),
            volume=random.randint(1000, 10000),
            timestamp=datetime.utcnow(),
            bid=Decimal(str(base_price + price_change - 0.5)),
            ask=Decimal(str(base_price + price_change + 0.5))
        )


class TradeStream(ReactiveStream[TradeEvent]):
    """取引イベントストリーム"""
    
    def __init__(self):
        super().__init__()
        self._trade_queue = asyncio.Queue()
    
    async def start(self) -> None:
        """取引ストリーム開始"""
        self._is_active = True
        asyncio.create_task(self._trade_processing_loop())
    
    async def stop(self) -> None:
        """取引ストリーム停止"""
        self._is_active = False
    
    async def add_trade(self, trade: TradeEvent) -> None:
        """取引追加"""
        await self._trade_queue.put(trade)
    
    async def _trade_processing_loop(self) -> None:
        """取引処理ループ"""
        while self._is_active:
            try:
                trade = await asyncio.wait_for(
                    self._trade_queue.get(),
                    timeout=1.0
                )
                self.emit(trade)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error(e)


class EventStream(ReactiveStream[Dict[str, Any]]):
    """汎用イベントストリーム"""
    
    def __init__(self, event_types: List[str]):
        super().__init__()
        self.event_types = event_types
        self._event_buffer = ReplaySubject(buffer_size=1000)
    
    async def start(self) -> None:
        """イベントストリーム開始"""
        self._is_active = True
        # イベント監視開始
        asyncio.create_task(self._event_monitor())
    
    async def stop(self) -> None:
        """イベントストリーム停止"""
        self._is_active = False
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """イベント発行"""
        if event_type in self.event_types:
            event = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'id': str(time.time_ns())
            }
            
            self.emit(event)
            self._event_buffer.on_next(event)
    
    async def _event_monitor(self) -> None:
        """イベント監視"""
        # 実際の実装ではメッセージブローカーからのイベント受信
        pass


class TechnicalAnalysisStream:
    """テクニカル分析ストリーム"""
    
    def __init__(self, market_stream: MarketDataStream):
        self.market_stream = market_stream
        self.signal_subject = Subject()
        self._setup_analysis_pipeline()
    
    def _setup_analysis_pipeline(self) -> None:
        """分析パイプライン設定"""
        # 移動平均クロス戦略
        sma_pipeline = (
            self.market_stream.observable
            .pipe(
                ops.buffer_with_time_or_count(20.0, 20),  # 20秒または20データポイント
                ops.filter(lambda buffer: len(buffer) >= 20),
                ops.map(self._calculate_sma_signals),
                ops.filter(lambda signals: signals is not None)
            )
        )
        
        # RSI戦略
        rsi_pipeline = (
            self.market_stream.observable
            .pipe(
                ops.buffer_with_time_or_count(30.0, 30),
                ops.filter(lambda buffer: len(buffer) >= 14),
                ops.map(self._calculate_rsi_signals),
                ops.filter(lambda signals: signals is not None)
            )
        )
        
        # ボリンジャーバンド戦略
        bb_pipeline = (
            self.market_stream.observable
            .pipe(
                ops.buffer_with_time_or_count(25.0, 25),
                ops.filter(lambda buffer: len(buffer) >= 20),
                ops.map(self._calculate_bollinger_signals),
                ops.filter(lambda signals: signals is not None)
            )
        )
        
        # シグナル統合
        rx.merge(sma_pipeline, rsi_pipeline, bb_pipeline).subscribe(
            on_next=lambda signal: self.signal_subject.on_next(signal),
            on_error=lambda e: print(f"Analysis error: {e}")
        )
    
    def _calculate_sma_signals(self, data_buffer: List[MarketData]) -> Optional[TechnicalSignal]:
        """移動平均シグナル計算"""
        if len(data_buffer) < 20:
            return None
        
        try:
            prices = [float(d.price) for d in data_buffer]
            sma_short = sum(prices[-10:]) / 10  # 短期移動平均
            sma_long = sum(prices[-20:]) / 20   # 長期移動平均
            
            latest_data = data_buffer[-1]
            
            if sma_short > sma_long * 1.01:  # 1%以上上回る
                return TechnicalSignal(
                    symbol=latest_data.symbol,
                    signal_type='BUY',
                    strength=min((sma_short / sma_long - 1) * 10, 1.0),
                    indicator='SMA_CROSS',
                    value=sma_short - sma_long,
                    timestamp=latest_data.timestamp,
                    metadata={'sma_short': sma_short, 'sma_long': sma_long}
                )
            elif sma_short < sma_long * 0.99:  # 1%以上下回る
                return TechnicalSignal(
                    symbol=latest_data.symbol,
                    signal_type='SELL',
                    strength=min((1 - sma_short / sma_long) * 10, 1.0),
                    indicator='SMA_CROSS',
                    value=sma_short - sma_long,
                    timestamp=latest_data.timestamp,
                    metadata={'sma_short': sma_short, 'sma_long': sma_long}
                )
            
            return None
            
        except Exception as e:
            print(f"SMA calculation error: {e}")
            return None
    
    def _calculate_rsi_signals(self, data_buffer: List[MarketData]) -> Optional[TechnicalSignal]:
        """RSIシグナル計算"""
        if len(data_buffer) < 14:
            return None
        
        try:
            prices = [float(d.price) for d in data_buffer[-14:]]
            
            # RSI計算
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            latest_data = data_buffer[-1]
            
            if rsi < 30:  # 売られ過ぎ
                return TechnicalSignal(
                    symbol=latest_data.symbol,
                    signal_type='BUY',
                    strength=(30 - rsi) / 30,
                    indicator='RSI',
                    value=rsi,
                    timestamp=latest_data.timestamp,
                    metadata={'rsi': rsi, 'threshold': 30}
                )
            elif rsi > 70:  # 買われ過ぎ
                return TechnicalSignal(
                    symbol=latest_data.symbol,
                    signal_type='SELL',
                    strength=(rsi - 70) / 30,
                    indicator='RSI',
                    value=rsi,
                    timestamp=latest_data.timestamp,
                    metadata={'rsi': rsi, 'threshold': 70}
                )
            
            return None
            
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return None
    
    def _calculate_bollinger_signals(self, data_buffer: List[MarketData]) -> Optional[TechnicalSignal]:
        """ボリンジャーバンドシグナル計算"""
        if len(data_buffer) < 20:
            return None
        
        try:
            prices = [float(d.price) for d in data_buffer[-20:]]
            
            # ボリンジャーバンド計算
            sma = sum(prices) / len(prices)
            variance = sum((p - sma) ** 2 for p in prices) / len(prices)
            std_dev = variance ** 0.5
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            
            latest_data = data_buffer[-1]
            current_price = float(latest_data.price)
            
            if current_price < lower_band:  # 下限ブレイク
                return TechnicalSignal(
                    symbol=latest_data.symbol,
                    signal_type='BUY',
                    strength=min((lower_band - current_price) / std_dev, 1.0),
                    indicator='BOLLINGER',
                    value=current_price - lower_band,
                    timestamp=latest_data.timestamp,
                    metadata={
                        'upper_band': upper_band,
                        'lower_band': lower_band,
                        'sma': sma,
                        'std_dev': std_dev
                    }
                )
            elif current_price > upper_band:  # 上限ブレイク
                return TechnicalSignal(
                    symbol=latest_data.symbol,
                    signal_type='SELL',
                    strength=min((current_price - upper_band) / std_dev, 1.0),
                    indicator='BOLLINGER',
                    value=current_price - upper_band,
                    timestamp=latest_data.timestamp,
                    metadata={
                        'upper_band': upper_band,
                        'lower_band': lower_band,
                        'sma': sma,
                        'std_dev': std_dev
                    }
                )
            
            return None
            
        except Exception as e:
            print(f"Bollinger calculation error: {e}")
            return None
    
    @property
    def signals(self) -> Observable:
        """シグナルObservable取得"""
        return self.signal_subject.as_observable()


class RiskMonitoringStream:
    """リスク監視ストリーム"""
    
    def __init__(self, trade_stream: TradeStream, market_stream: MarketDataStream):
        self.trade_stream = trade_stream
        self.market_stream = market_stream
        self.risk_subject = Subject()
        self._setup_risk_pipeline()
    
    def _setup_risk_pipeline(self) -> None:
        """リスク監視パイプライン設定"""
        
        # ポジションリスク監視
        position_risk = (
            self.trade_stream.observable
            .pipe(
                ops.scan(self._accumulate_positions, {}),
                ops.map(self._calculate_position_risk),
                ops.filter(lambda risk: risk is not None)
            )
        )
        
        # 市場リスク監視
        market_risk = (
            self.market_stream.observable
            .pipe(
                ops.buffer_with_time(10.0),  # 10秒バッファ
                ops.map(self._calculate_market_risk),
                ops.filter(lambda risk: risk is not None)
            )
        )
        
        # リスクアラート統合
        rx.merge(position_risk, market_risk).subscribe(
            on_next=lambda alert: self.risk_subject.on_next(alert),
            on_error=lambda e: print(f"Risk monitoring error: {e}")
        )
    
    def _accumulate_positions(self, positions: Dict[str, int], trade: TradeEvent) -> Dict[str, int]:
        """ポジション集計"""
        current_pos = positions.get(trade.symbol, 0)
        
        if trade.side == 'BUY':
            positions[trade.symbol] = current_pos + trade.quantity
        elif trade.side == 'SELL':
            positions[trade.symbol] = current_pos - trade.quantity
        
        return positions
    
    def _calculate_position_risk(self, positions: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """ポジションリスク計算"""
        # 単一銘柄集中リスク
        total_position = sum(abs(pos) for pos in positions.values())
        if total_position == 0:
            return None
        
        for symbol, position in positions.items():
            concentration = abs(position) / total_position
            
            if concentration > 0.3:  # 30%以上集中
                return {
                    'type': 'POSITION_CONCENTRATION',
                    'symbol': symbol,
                    'concentration': concentration,
                    'position': position,
                    'severity': 'HIGH' if concentration > 0.5 else 'MEDIUM',
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return None
    
    def _calculate_market_risk(self, market_data: List[MarketData]) -> Optional[Dict[str, Any]]:
        """市場リスク計算"""
        if not market_data:
            return None
        
        # ボラティリティリスク
        for symbol in set(d.symbol for d in market_data):
            symbol_data = [d for d in market_data if d.symbol == symbol]
            
            if len(symbol_data) < 2:
                continue
            
            prices = [float(d.price) for d in symbol_data]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            if returns:
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                
                if volatility > 0.05:  # 5%以上のボラティリティ
                    return {
                        'type': 'HIGH_VOLATILITY',
                        'symbol': symbol,
                        'volatility': volatility,
                        'severity': 'HIGH' if volatility > 0.1 else 'MEDIUM',
                        'timestamp': datetime.utcnow().isoformat()
                    }
        
        return None
    
    @property
    def alerts(self) -> Observable:
        """リスクアラートObservable取得"""
        return self.risk_subject.as_observable()


# ストリーム統合管理
class StreamManager:
    """ストリーム統合管理"""
    
    def __init__(self):
        self.streams: Dict[str, ReactiveStream] = {}
        self.is_running = False
    
    def register_stream(self, name: str, stream: ReactiveStream) -> None:
        """ストリーム登録"""
        self.streams[name] = stream
    
    async def start_all(self) -> None:
        """全ストリーム開始"""
        self.is_running = True
        
        for name, stream in self.streams.items():
            try:
                await stream.start()
                print(f"Started stream: {name}")
            except Exception as e:
                print(f"Failed to start stream {name}: {e}")
    
    async def stop_all(self) -> None:
        """全ストリーム停止"""
        self.is_running = False
        
        for name, stream in self.streams.items():
            try:
                await stream.stop()
                print(f"Stopped stream: {name}")
            except Exception as e:
                print(f"Failed to stop stream {name}: {e}")
    
    def get_stream(self, name: str) -> Optional[ReactiveStream]:
        """ストリーム取得"""
        return self.streams.get(name)
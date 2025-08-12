#!/usr/bin/env python3
"""
超高速市場データ処理システム
Issue #366: 高頻度取引最適化エンジン - 低遅延データ処理

<100μs市場データ処理、リアルタイムOrder Book構築、
高速シグナル生成を実現するマイクロ秒レベルデータパイプライン
"""

import asyncio
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

# プロジェクトモジュール
try:
    from ..cache.advanced_cache_system import AdvancedCacheSystem
    from ..distributed.distributed_computing_manager import (
        DistributedComputingManager,
        DistributedTask,
        TaskType,
    )
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # モッククラス
    class DistributedComputingManager:
        def __init__(self):
            pass

        async def execute_distributed_task(self, task):
            return type("MockResult", (), {"success": True, "result": None})()

    class AdvancedCacheSystem:
        def __init__(self):
            pass


logger = get_context_logger(__name__)


class MessageType(IntEnum):
    """市場データメッセージタイプ"""

    TRADE = 1
    QUOTE = 2
    DEPTH_UPDATE = 3
    TRADE_SUMMARY = 4
    MARKET_STATUS = 5
    HEARTBEAT = 6


class OrderBookSide(IntEnum):
    """Order Book サイド"""

    BID = 1
    ASK = -1


@dataclass
class MarketUpdate:
    """市場データ更新（高速アクセス最適化）"""

    # Core identification (8 bytes)
    symbol_id: int = 0
    message_type: MessageType = MessageType.TRADE

    # Price/Size (16 bytes, fixed-point)
    price: int = 0  # Price * 10000
    size: int = 0  # Size * 10000

    # Timing (16 bytes)
    exchange_timestamp_ns: int = 0
    receive_timestamp_ns: int = field(default_factory=time.perf_counter_ns)

    # Optional fields (24 bytes)
    side: OrderBookSide = OrderBookSide.BID
    sequence_number: int = 0
    trade_id: int = 0

    # Computed fields (8 bytes)
    processing_latency_ns: int = 0

    def __post_init__(self):
        """後処理"""
        if self.receive_timestamp_ns == 0:
            self.receive_timestamp_ns = time.perf_counter_ns()

        if self.exchange_timestamp_ns > 0:
            self.processing_latency_ns = self.receive_timestamp_ns - self.exchange_timestamp_ns

    def get_price_float(self) -> float:
        """価格をfloat取得"""
        return self.price / 10000.0

    def get_size_float(self) -> float:
        """サイズをfloat取得"""
        return self.size / 10000.0

    def to_bytes(self) -> bytes:
        """固定サイズバイナリ化 (64 bytes)"""
        return struct.pack(
            "QIiiii QQ iiQ Q 8x",  # 8x = 8 byte padding
            self.symbol_id,
            self.message_type,
            self.price,
            self.size,
            self.exchange_timestamp_ns,
            self.receive_timestamp_ns,
            self.side,
            self.sequence_number,
            self.trade_id,
            self.processing_latency_ns,
        )


@dataclass
class OrderBookLevel:
    """Order Book レベル"""

    price: int = 0  # Fixed-point
    size: int = 0  # Fixed-point
    order_count: int = 0

    def get_price_float(self) -> float:
        return self.price / 10000.0

    def get_size_float(self) -> float:
        return self.size / 10000.0


class OrderBook:
    """
    高速Order Book

    マイクロ秒レベル更新、メモリ効率最適化
    """

    def __init__(self, symbol_id: int, max_levels: int = 20):
        self.symbol_id = symbol_id
        self.max_levels = max_levels

        # Order book levels (sorted arrays for speed)
        self.bids: List[OrderBookLevel] = []  # Descending price order
        self.asks: List[OrderBookLevel] = []  # Ascending price order

        # Fast lookup caches
        self.bid_price_to_index = {}
        self.ask_price_to_index = {}

        # Timing
        self.last_update_ns = 0
        self.sequence_number = 0

        # Statistics
        self.update_count = 0
        self.microsecond_updates = 0

    def update_level(self, side: OrderBookSide, price: int, size: int) -> bool:
        """
        レベル更新 (<10μs target)

        Returns:
            bool: 更新が発生したか
        """
        start_time = time.perf_counter_ns()

        try:
            if side == OrderBookSide.BID:
                updated = self._update_bid_level(price, size)
            else:
                updated = self._update_ask_level(price, size)

            if updated:
                self.last_update_ns = time.perf_counter_ns()
                self.sequence_number += 1
                self.update_count += 1

            return updated

        finally:
            processing_time_ns = time.perf_counter_ns() - start_time
            if processing_time_ns < 10000:  # <10μs
                self.microsecond_updates += 1

    def _update_bid_level(self, price: int, size: int) -> bool:
        """Bid レベル更新"""
        if size == 0:
            # Remove level
            return self._remove_bid_level(price)

        # Find insertion point (binary search for large books)
        insert_index = 0
        for i, bid in enumerate(self.bids):
            if bid.price == price:
                # Update existing level
                bid.size = size
                return True
            elif bid.price < price:
                insert_index = i
                break
            insert_index = i + 1

        # Insert new level
        new_level = OrderBookLevel(price=price, size=size, order_count=1)
        self.bids.insert(insert_index, new_level)

        # Maintain max levels
        if len(self.bids) > self.max_levels:
            removed = self.bids.pop()
            self.bid_price_to_index.pop(removed.price, None)

        # Update price index
        self._rebuild_bid_index()

        return True

    def _update_ask_level(self, price: int, size: int) -> bool:
        """Ask レベル更新"""
        if size == 0:
            # Remove level
            return self._remove_ask_level(price)

        # Find insertion point
        insert_index = 0
        for i, ask in enumerate(self.asks):
            if ask.price == price:
                # Update existing level
                ask.size = size
                return True
            elif ask.price > price:
                insert_index = i
                break
            insert_index = i + 1

        # Insert new level
        new_level = OrderBookLevel(price=price, size=size, order_count=1)
        self.asks.insert(insert_index, new_level)

        # Maintain max levels
        if len(self.asks) > self.max_levels:
            removed = self.asks.pop()
            self.ask_price_to_index.pop(removed.price, None)

        # Update price index
        self._rebuild_ask_index()

        return True

    def _remove_bid_level(self, price: int) -> bool:
        """Bid レベル削除"""
        for i, bid in enumerate(self.bids):
            if bid.price == price:
                self.bids.pop(i)
                self._rebuild_bid_index()
                return True
        return False

    def _remove_ask_level(self, price: int) -> bool:
        """Ask レベル削除"""
        for i, ask in enumerate(self.asks):
            if ask.price == price:
                self.asks.pop(i)
                self._rebuild_ask_index()
                return True
        return False

    def _rebuild_bid_index(self):
        """Bid価格インデックス再構築"""
        self.bid_price_to_index = {bid.price: i for i, bid in enumerate(self.bids)}

    def _rebuild_ask_index(self):
        """Ask価格インデックス再構築"""
        self.ask_price_to_index = {ask.price: i for i, ask in enumerate(self.asks)}

    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """最良Bid取得"""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """最良Ask取得"""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> Optional[int]:
        """スプレッド取得（固定小数点）"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    def get_mid_price(self) -> Optional[int]:
        """中間価格取得（固定小数点）"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) // 2
        return None

    def get_book_snapshot(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Order Bookスナップショット取得"""
        return {
            "bids": [(bid.get_price_float(), bid.get_size_float()) for bid in self.bids[:levels]],
            "asks": [(ask.get_price_float(), ask.get_size_float()) for ask in self.asks[:levels]],
            "timestamp_ns": self.last_update_ns,
            "sequence": self.sequence_number,
        }


class ZeroCopyUDPReceiver:
    """Zero-copy UDPレシーバー（高速市場データ受信）"""

    def __init__(
        self,
        multicast_group: str = "224.1.1.1",
        port: int = 12345,
        buffer_size: int = 64 * 1024 * 1024,  # 64MB
        enable_so_reuseport: bool = True,
    ):
        self.multicast_group = multicast_group
        self.port = port
        self.buffer_size = buffer_size
        self.enable_so_reuseport = enable_so_reuseport

        # Socket configuration
        self.socket = None
        self.running = False

        # Performance statistics
        self.packets_received = 0
        self.bytes_received = 0
        self.processing_latency_sum_ns = 0

        # Message parsing callback
        self.message_callback: Optional[Callable[[bytes], None]] = None

    def initialize_socket(self):
        """UDPソケット初期化"""
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Socket options for high performance
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.enable_so_reuseport:
                try:
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    logger.warning("SO_REUSEPORT not supported on this platform")

            # Buffer size optimization
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)

            # Bind to port
            self.socket.bind(("", self.port))

            # Join multicast group
            mreq = struct.pack("4sL", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            logger.info(f"UDP受信初期化完了: {self.multicast_group}:{self.port}")

        except Exception as e:
            logger.error(f"UDPソケット初期化失敗: {e}")
            raise

    def set_message_callback(self, callback: Callable[[bytes], None]):
        """メッセージ処理コールバック設定"""
        self.message_callback = callback

    def start_receiving(self):
        """受信開始"""
        if not self.socket:
            self.initialize_socket()

        self.running = True

        # High-performance receiving loop
        while self.running:
            try:
                # Receive packet
                data, addr = self.socket.recvfrom(65536)  # Max UDP packet size
                receive_time_ns = time.perf_counter_ns()

                # Update statistics
                self.packets_received += 1
                self.bytes_received += len(data)

                # Process message if callback set
                if self.message_callback:
                    process_start = time.perf_counter_ns()
                    self.message_callback(data)
                    process_end = time.perf_counter_ns()

                    self.processing_latency_sum_ns += process_end - process_start

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"UDP受信エラー: {e}")
                break

    def stop_receiving(self):
        """受信停止"""
        self.running = False
        if self.socket:
            self.socket.close()

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        avg_processing_latency_ns = 0
        if self.packets_received > 0:
            avg_processing_latency_ns = self.processing_latency_sum_ns / self.packets_received

        return {
            "packets_received": self.packets_received,
            "bytes_received": self.bytes_received,
            "avg_processing_latency_us": avg_processing_latency_ns / 1000,
            "throughput_mbps": (self.bytes_received * 8) / (1024 * 1024),  # Approximate
        }


class MarketDataParser:
    """高速市場データパーサー"""

    def __init__(self):
        # Message type parsers
        self.parsers = {
            MessageType.TRADE: self._parse_trade,
            MessageType.QUOTE: self._parse_quote,
            MessageType.DEPTH_UPDATE: self._parse_depth_update,
        }

        # Performance counters
        self.parsed_messages = 0
        self.parse_errors = 0
        self.total_parse_time_ns = 0

    def parse_message(self, raw_data: bytes) -> Optional[MarketUpdate]:
        """
        市場データメッセージ解析 (<50μs target)

        Args:
            raw_data: 生バイトデータ

        Returns:
            MarketUpdate or None
        """
        start_time = time.perf_counter_ns()

        try:
            if len(raw_data) < 8:  # Minimum header size
                return None

            # Extract message header (simplified)
            header = struct.unpack("II", raw_data[:8])
            symbol_id = header[0]
            message_type = MessageType(header[1])

            # Route to specific parser
            parser = self.parsers.get(message_type)
            if not parser:
                return None

            market_update = parser(symbol_id, raw_data[8:])

            if market_update:
                self.parsed_messages += 1

            return market_update

        except Exception as e:
            self.parse_errors += 1
            logger.debug(f"パース エラー: {e}")
            return None

        finally:
            end_time = time.perf_counter_ns()
            self.total_parse_time_ns += end_time - start_time

    def _parse_trade(self, symbol_id: int, data: bytes) -> Optional[MarketUpdate]:
        """取引データ解析"""
        if len(data) < 24:  # price(8) + size(8) + timestamp(8)
            return None

        try:
            price, size, timestamp = struct.unpack("QQQ", data[:24])

            return MarketUpdate(
                symbol_id=symbol_id,
                message_type=MessageType.TRADE,
                price=price,
                size=size,
                exchange_timestamp_ns=timestamp,
            )
        except:
            return None

    def _parse_quote(self, symbol_id: int, data: bytes) -> Optional[MarketUpdate]:
        """気配データ解析"""
        if len(data) < 32:  # bid_price(8) + bid_size(8) + ask_price(8) + ask_size(8)
            return None

        try:
            bid_price, bid_size, ask_price, ask_size = struct.unpack("QQQQ", data[:32])

            # Return as bid update (could return both bid and ask)
            return MarketUpdate(
                symbol_id=symbol_id,
                message_type=MessageType.QUOTE,
                price=bid_price,
                size=bid_size,
                side=OrderBookSide.BID,
            )
        except:
            return None

    def _parse_depth_update(self, symbol_id: int, data: bytes) -> Optional[MarketUpdate]:
        """板更新データ解析"""
        if len(data) < 20:  # side(4) + price(8) + size(8)
            return None

        try:
            side_raw, price, size = struct.unpack("IQQ", data[:20])
            side = OrderBookSide.BID if side_raw == 1 else OrderBookSide.ASK

            return MarketUpdate(
                symbol_id=symbol_id,
                message_type=MessageType.DEPTH_UPDATE,
                price=price,
                size=size,
                side=side,
            )
        except:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """パーサー統計"""
        avg_parse_time_ns = 0
        if self.parsed_messages > 0:
            avg_parse_time_ns = self.total_parse_time_ns / self.parsed_messages

        return {
            "parsed_messages": self.parsed_messages,
            "parse_errors": self.parse_errors,
            "avg_parse_time_us": avg_parse_time_ns / 1000,
            "success_rate": self.parsed_messages / max(self.parsed_messages + self.parse_errors, 1),
        }


class UltraFastMarketDataProcessor:
    """
    超高速市場データプロセッサー

    目標: <100μs end-to-end処理レイテンシー
    機能: UDP受信 → パース → OrderBook更新 → シグナル生成
    """

    def __init__(
        self,
        distributed_manager: Optional[DistributedComputingManager] = None,
        cache_system: Optional[AdvancedCacheSystem] = None,
        max_symbols: int = 1000,
        enable_udp_receiver: bool = False,  # テスト環境では無効
    ):
        """
        初期化

        Args:
            distributed_manager: 分散処理マネージャー
            cache_system: キャッシュシステム
            max_symbols: 最大銘柄数
            enable_udp_receiver: UDP受信有効化
        """
        self.distributed_manager = distributed_manager or DistributedComputingManager()
        self.cache_system = cache_system or AdvancedCacheSystem()
        self.max_symbols = max_symbols

        # Core components
        self.parser = MarketDataParser()

        # Order Books (symbol_id -> OrderBook)
        self.order_books: Dict[int, OrderBook] = {}

        # UDP Receiver (optional)
        self.udp_receiver = None
        if enable_udp_receiver:
            self.udp_receiver = ZeroCopyUDPReceiver()
            self.udp_receiver.set_message_callback(self._process_raw_message)

        # Signal generation callbacks
        self.signal_callbacks: List[Callable[[MarketUpdate], None]] = []

        # Performance statistics
        self.stats = {
            "messages_processed": 0,
            "order_book_updates": 0,
            "signals_generated": 0,
            "processing_latency_sum_us": 0.0,
            "max_processing_latency_us": 0.0,
        }

        # Threading
        self.processing_thread = None
        self.running = False

        logger.info("UltraFastMarketDataProcessor初期化完了")

    def add_signal_callback(self, callback: Callable[[MarketUpdate], None]):
        """シグナル生成コールバック追加"""
        self.signal_callbacks.append(callback)

    def _process_raw_message(self, raw_data: bytes):
        """生メッセージ処理 (UDP callback)"""
        start_time = time.perf_counter_ns()

        try:
            # Parse message
            market_update = self.parser.parse_message(raw_data)
            if not market_update:
                return

            # Process update
            self._process_market_update(market_update)

        finally:
            end_time = time.perf_counter_ns()
            processing_time_us = (end_time - start_time) / 1000

            # Update statistics
            self.stats["messages_processed"] += 1
            self.stats["processing_latency_sum_us"] += processing_time_us
            self.stats["max_processing_latency_us"] = max(
                self.stats["max_processing_latency_us"], processing_time_us
            )

    def _process_market_update(self, update: MarketUpdate):
        """
        市場データ更新処理 (<50μs target)

        Args:
            update: 市場データ更新
        """
        # Get or create order book
        order_book = self.order_books.get(update.symbol_id)
        if not order_book:
            if len(self.order_books) >= self.max_symbols:
                return  # Symbol limit reached
            order_book = OrderBook(update.symbol_id)
            self.order_books[update.symbol_id] = order_book

        # Update order book based on message type
        updated = False

        if update.message_type == MessageType.DEPTH_UPDATE:
            updated = order_book.update_level(update.side, update.price, update.size)
        elif update.message_type == MessageType.QUOTE:
            updated = order_book.update_level(update.side, update.price, update.size)

        if updated:
            self.stats["order_book_updates"] += 1

            # Generate signals
            self._generate_signals(update, order_book)

    def _generate_signals(self, update: MarketUpdate, order_book: OrderBook):
        """
        シグナル生成 (<25μs target)

        Args:
            update: 市場データ更新
            order_book: Order Book
        """
        try:
            # Notify signal callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(update)
                    self.stats["signals_generated"] += 1
                except Exception as e:
                    logger.debug(f"シグナルコールバックエラー: {e}")
        except Exception as e:
            logger.debug(f"シグナル生成エラー: {e}")

    async def process_market_update_async(self, update: MarketUpdate):
        """
        非同期市場データ更新処理

        Args:
            update: 市場データ更新
        """
        # Delegate to sync method for speed
        self._process_market_update(update)

    def process_market_updates_batch(self, updates: List[MarketUpdate]) -> int:
        """
        バッチ市場データ更新処理

        Args:
            updates: 市場データ更新リスト

        Returns:
            処理された更新数
        """
        processed = 0

        for update in updates:
            try:
                self._process_market_update(update)
                processed += 1
            except Exception as e:
                logger.debug(f"バッチ更新処理エラー: {e}")

        return processed

    def get_order_book(self, symbol_id: int) -> Optional[OrderBook]:
        """Order Book取得"""
        return self.order_books.get(symbol_id)

    def get_order_book_snapshot(self, symbol_id: int, levels: int = 5) -> Optional[Dict[str, Any]]:
        """Order Bookスナップショット取得"""
        order_book = self.get_order_book(symbol_id)
        if not order_book:
            return None

        return order_book.get_book_snapshot(levels)

    def get_market_summary(self, symbol_id: int) -> Optional[Dict[str, Any]]:
        """市場サマリー取得"""
        order_book = self.get_order_book(symbol_id)
        if not order_book:
            return None

        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()

        return {
            "symbol_id": symbol_id,
            "best_bid_price": best_bid.get_price_float() if best_bid else None,
            "best_bid_size": best_bid.get_size_float() if best_bid else None,
            "best_ask_price": best_ask.get_price_float() if best_ask else None,
            "best_ask_size": best_ask.get_size_float() if best_ask else None,
            "spread": spread / 10000.0 if spread else None,
            "mid_price": mid_price / 10000.0 if mid_price else None,
            "last_update_ns": order_book.last_update_ns,
            "sequence_number": order_book.sequence_number,
        }

    def start_udp_processing(self):
        """UDP処理開始"""
        if not self.udp_receiver:
            logger.warning("UDP受信が有効化されていません")
            return

        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(
            target=self.udp_receiver.start_receiving,
            name="MarketDataUDPProcessor",
            daemon=True,
        )
        self.processing_thread.start()
        logger.info("UDP市場データ処理開始")

    def stop_udp_processing(self):
        """UDP処理停止"""
        if not self.running:
            return

        self.running = False
        if self.udp_receiver:
            self.udp_receiver.stop_receiving()

        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        logger.info("UDP市場データ処理停止")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報取得"""
        avg_processing_latency_us = 0
        if self.stats["messages_processed"] > 0:
            avg_processing_latency_us = (
                self.stats["processing_latency_sum_us"] / self.stats["messages_processed"]
            )

        stats = self.stats.copy()
        stats.update(
            {
                "avg_processing_latency_us": avg_processing_latency_us,
                "active_order_books": len(self.order_books),
                "parser_stats": self.parser.get_stats(),
            }
        )

        if self.udp_receiver:
            stats["udp_receiver_stats"] = self.udp_receiver.get_stats()

        return stats

    async def cleanup(self):
        """クリーンアップ"""
        logger.info("UltraFastMarketDataProcessor クリーンアップ開始")

        self.stop_udp_processing()

        # Clear order books
        self.order_books.clear()
        self.signal_callbacks.clear()

        logger.info("クリーンアップ完了")


# Factory function
def create_market_data_processor(
    distributed_manager: Optional[DistributedComputingManager] = None,
    cache_system: Optional[AdvancedCacheSystem] = None,
    **config,
) -> UltraFastMarketDataProcessor:
    """UltraFastMarketDataProcessorファクトリ関数"""
    return UltraFastMarketDataProcessor(
        distributed_manager=distributed_manager, cache_system=cache_system, **config
    )


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #366 超高速市場データ処理テスト ===")

        processor = None
        try:
            # プロセッサー初期化
            processor = create_market_data_processor(
                max_symbols=100,
                enable_udp_receiver=False,  # テスト環境
            )

            # シグナルコールバック設定
            def signal_handler(update: MarketUpdate):
                print(f"シグナル: {update.symbol_id}, price={update.get_price_float()}")

            processor.add_signal_callback(signal_handler)

            # テストデータ生成
            test_updates = []

            # 1. 初期Order Book構築
            for i in range(5):
                # Bid levels
                bid_update = MarketUpdate(
                    symbol_id=1001,
                    message_type=MessageType.DEPTH_UPDATE,
                    side=OrderBookSide.BID,
                    price=(1000 - i) * 10000,  # $100.00, $99.90, etc.
                    size=100 * 10000,  # 100 shares
                )
                test_updates.append(bid_update)

                # Ask levels
                ask_update = MarketUpdate(
                    symbol_id=1001,
                    message_type=MessageType.DEPTH_UPDATE,
                    side=OrderBookSide.ASK,
                    price=(1010 + i) * 10000,  # $101.00, $101.10, etc.
                    size=100 * 10000,
                )
                test_updates.append(ask_update)

            print("\n1. バッチ更新処理テスト")
            start_time = time.perf_counter()
            processed_count = processor.process_market_updates_batch(test_updates)
            end_time = time.perf_counter()

            processing_time_ms = (end_time - start_time) * 1000
            print(
                f"バッチ処理: {processed_count}/{len(test_updates)} 更新, {processing_time_ms:.3f}ms"
            )
            print(f"平均処理時間: {(processing_time_ms * 1000) / processed_count:.1f}μs per update")

            print("\n2. Order Book状態確認")
            snapshot = processor.get_order_book_snapshot(1001, levels=3)
            if snapshot:
                print("Order Book:")
                print(f"  Bids: {snapshot['bids']}")
                print(f"  Asks: {snapshot['asks']}")
                print(f"  Sequence: {snapshot['sequence']}")

            print("\n3. 市場サマリー")
            summary = processor.get_market_summary(1001)
            if summary:
                print(f"  Best Bid: ${summary['best_bid_price']:.2f} @ {summary['best_bid_size']}")
                print(f"  Best Ask: ${summary['best_ask_price']:.2f} @ {summary['best_ask_size']}")
                print(f"  Spread: ${summary['spread']:.2f}")
                print(f"  Mid Price: ${summary['mid_price']:.2f}")

            print("\n4. 高速更新テスト")
            rapid_updates = []
            for i in range(100):
                update = MarketUpdate(
                    symbol_id=1001,
                    message_type=MessageType.DEPTH_UPDATE,
                    side=OrderBookSide.BID,
                    price=10000 * 10000,  # $1000.00
                    size=(100 + i) * 10000,
                )
                rapid_updates.append(update)

            start_time = time.perf_counter()
            rapid_processed = processor.process_market_updates_batch(rapid_updates)
            end_time = time.perf_counter()

            rapid_time_us = (end_time - start_time) * 1_000_000
            print(f"高速更新: {rapid_processed} updates in {rapid_time_us:.1f}μs")
            print(f"平均: {rapid_time_us / rapid_processed:.1f}μs per update")

            print("\n5. パフォーマンス統計")
            stats = processor.get_comprehensive_stats()
            for key, value in stats.items():
                if key == "parser_stats":
                    print("  Parser統計:")
                    for pk, pv in value.items():
                        print(f"    {pk}: {pv}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        finally:
            if processor:
                await processor.cleanup()

        print("\n=== 超高速市場データ処理テスト完了 ===")

    asyncio.run(main())

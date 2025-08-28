#!/usr/bin/env python3
"""
Distributed CQRS Implementation
分散CQRS実装
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union
from enum import Enum
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class MessagePriority(Enum):
    """メッセージ優先度"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass(frozen=True)
class Command:
    """コマンド基底クラス"""
    command_id: str
    aggregate_id: str
    user_id: str
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    
    def __post_init__(self):
        if not hasattr(self, 'command_id') or not self.command_id:
            object.__setattr__(self, 'command_id', str(uuid.uuid4()))
        if not hasattr(self, 'timestamp') or not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.now(timezone.utc))

@dataclass(frozen=True)
class Query:
    """クエリ基底クラス"""
    query_id: str
    user_id: str
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    
    def __post_init__(self):
        if not hasattr(self, 'query_id') or not self.query_id:
            object.__setattr__(self, 'query_id', str(uuid.uuid4()))
        if not hasattr(self, 'timestamp') or not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.now(timezone.utc))

# Trading Commands
@dataclass(frozen=True)
class ExecuteTradeCommand(Command):
    """取引実行コマンド"""
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    price: Optional[float] = None
    order_type: str = "MARKET"

@dataclass(frozen=True)
class CancelOrderCommand(Command):
    """注文取消コマンド"""
    order_id: str

@dataclass(frozen=True)
class UpdatePortfolioCommand(Command):
    """ポートフォリオ更新コマンド"""
    trades: List[Dict[str, Any]]

# Trading Queries
@dataclass(frozen=True)
class GetPortfolioQuery(Query):
    """ポートフォリオ取得クエリ"""
    portfolio_id: str
    as_of_date: Optional[datetime] = None

@dataclass(frozen=True)
class GetTradeHistoryQuery(Query):
    """取引履歴取得クエリ"""
    symbol: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100

@dataclass(frozen=True)
class GetMarketDataQuery(Query):
    """マーケットデータ取得クエリ"""
    symbols: List[str]
    timeframe: str = "1D"


class CommandHandler(ABC, Generic[T, R]):
    """コマンドハンドラー基底クラス"""
    
    @abstractmethod
    async def handle(self, command: T) -> TradingResult[R]:
        """コマンド処理"""
        pass
    
    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """処理可能確認"""
        pass


class QueryHandler(ABC, Generic[T, R]):
    """クエリハンドラー基底クラス"""
    
    @abstractmethod
    async def handle(self, query: T) -> TradingResult[R]:
        """クエリ処理"""
        pass
    
    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """処理可能確認"""
        pass


class MessageBus(ABC):
    """メッセージバス抽象基底クラス"""
    
    @abstractmethod
    async def send_command(self, command: Command) -> TradingResult[Any]:
        """コマンド送信"""
        pass
    
    @abstractmethod
    async def send_query(self, query: Query) -> TradingResult[Any]:
        """クエリ送信"""
        pass


class DistributedCommandBus(MessageBus):
    """分散コマンドバス"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self._command_handlers: Dict[Type, CommandHandler] = {}
        self._command_queue: List[Command] = []
        self._processing = False
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    def register_handler(self, command_type: Type[Command], handler: CommandHandler) -> None:
        """ハンドラー登録"""
        self._command_handlers[command_type] = handler
        logger.info(f"Registered command handler for {command_type.__name__}")
    
    async def send_command(self, command: Command) -> TradingResult[Any]:
        """コマンド送信"""
        try:
            # 適切なノード選択
            target_node = await self._select_node_for_command(command)
            
            # ローカル処理
            if target_node == "local":
                return await self._process_command_locally(command)
            
            # リモート送信
            return await self._send_command_to_node(command, target_node)
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return TradingResult.failure('COMMAND_ERROR', str(e))
    
    async def send_query(self, query: Query) -> TradingResult[Any]:
        """クエリ送信（コマンドバスでは使用しない）"""
        return TradingResult.failure('UNSUPPORTED', 'Use QueryBus for queries')
    
    async def start_processing(self) -> None:
        """処理開始"""
        if self._processing:
            return
        
        self._processing = True
        logger.info("Starting command processing")
        asyncio.create_task(self._process_command_queue())
    
    async def stop_processing(self) -> None:
        """処理停止"""
        self._processing = False
        logger.info("Stopped command processing")
    
    async def _process_command_queue(self) -> None:
        """コマンドキュー処理"""
        while self._processing:
            try:
                if not self._command_queue:
                    await asyncio.sleep(0.01)
                    continue
                
                # 優先度でソート
                self._command_queue.sort(key=lambda c: c.priority.value, reverse=True)
                
                # バッチ処理
                batch_size = min(10, len(self._command_queue))
                commands = self._command_queue[:batch_size]
                self._command_queue = self._command_queue[batch_size:]
                
                # 並列処理
                tasks = [self._process_command_locally(cmd) for cmd in commands]
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Command queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _select_node_for_command(self, command: Command) -> str:
        """コマンド用ノード選択"""
        # 簡単なハッシュベース分散
        hash_value = hash(command.aggregate_id) % len(self.nodes)
        return self.nodes[hash_value] if self.nodes else "local"
    
    async def _process_command_locally(self, command: Command) -> TradingResult[Any]:
        """ローカルコマンド処理"""
        handler = self._find_handler_for_command(command)
        if not handler:
            return TradingResult.failure('NO_HANDLER', f'No handler found for {type(command).__name__}')
        
        try:
            return await handler.handle(command)
        except Exception as e:
            logger.error(f"Command processing failed: {e}")
            return TradingResult.failure('PROCESSING_ERROR', str(e))
    
    async def _send_command_to_node(self, command: Command, node: str) -> TradingResult[Any]:
        """ノードへコマンド送信"""
        try:
            # 実際の実装では HTTP/gRPC を使用
            await asyncio.sleep(0.001)  # ネットワーク遅延シミュレーション
            logger.debug(f"Sent command {command.command_id} to node {node}")
            return TradingResult.success({"status": "sent", "node": node})
        except Exception as e:
            return TradingResult.failure('NETWORK_ERROR', str(e))
    
    def _find_handler_for_command(self, command: Command) -> Optional[CommandHandler]:
        """コマンドハンドラー検索"""
        command_type = type(command)
        return self._command_handlers.get(command_type)


class DistributedQueryBus(MessageBus):
    """分散クエリバス"""
    
    def __init__(self, read_replicas: List[str]):
        self.read_replicas = read_replicas
        self._query_handlers: Dict[Type, QueryHandler] = {}
        self._load_balancer_index = 0
        
    def register_handler(self, query_type: Type[Query], handler: QueryHandler) -> None:
        """ハンドラー登録"""
        self._query_handlers[query_type] = handler
        logger.info(f"Registered query handler for {query_type.__name__}")
    
    async def send_command(self, command: Command) -> TradingResult[Any]:
        """コマンド送信（クエリバスでは使用しない）"""
        return TradingResult.failure('UNSUPPORTED', 'Use CommandBus for commands')
    
    async def send_query(self, query: Query) -> TradingResult[Any]:
        """クエリ送信"""
        try:
            # 読み取り専用レプリカ選択
            replica = await self._select_read_replica()
            
            # ローカル処理
            if replica == "local":
                return await self._process_query_locally(query)
            
            # リモート送信
            return await self._send_query_to_replica(query, replica)
            
        except Exception as e:
            logger.error(f"Failed to send query: {e}")
            return TradingResult.failure('QUERY_ERROR', str(e))
    
    async def _select_read_replica(self) -> str:
        """読み取りレプリカ選択（ラウンドロビン）"""
        if not self.read_replicas:
            return "local"
        
        replica = self.read_replicas[self._load_balancer_index]
        self._load_balancer_index = (self._load_balancer_index + 1) % len(self.read_replicas)
        return replica
    
    async def _process_query_locally(self, query: Query) -> TradingResult[Any]:
        """ローカルクエリ処理"""
        handler = self._find_handler_for_query(query)
        if not handler:
            return TradingResult.failure('NO_HANDLER', f'No handler found for {type(query).__name__}')
        
        try:
            return await handler.handle(query)
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return TradingResult.failure('PROCESSING_ERROR', str(e))
    
    async def _send_query_to_replica(self, query: Query, replica: str) -> TradingResult[Any]:
        """レプリカへクエリ送信"""
        try:
            # 実際の実装では HTTP/gRPC を使用
            await asyncio.sleep(0.001)  # ネットワーク遅延シミュレーション
            logger.debug(f"Sent query {query.query_id} to replica {replica}")
            return TradingResult.success({"status": "queried", "replica": replica})
        except Exception as e:
            return TradingResult.failure('NETWORK_ERROR', str(e))
    
    def _find_handler_for_query(self, query: Query) -> Optional[QueryHandler]:
        """クエリハンドラー検索"""
        query_type = type(query)
        return self._query_handlers.get(query_type)


# 具象ハンドラー実装例
class ExecuteTradeCommandHandler(CommandHandler[ExecuteTradeCommand, Dict[str, Any]]):
    """取引実行コマンドハンドラー"""
    
    async def handle(self, command: ExecuteTradeCommand) -> TradingResult[Dict[str, Any]]:
        """取引実行"""
        try:
            # 取引実行ロジック
            trade_id = str(uuid.uuid4())
            
            result = {
                'trade_id': trade_id,
                'symbol': command.symbol,
                'action': command.action,
                'quantity': command.quantity,
                'price': command.price,
                'status': 'EXECUTED',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Trade executed: {trade_id}")
            return TradingResult.success(result)
            
        except Exception as e:
            return TradingResult.failure('EXECUTION_ERROR', str(e))
    
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, ExecuteTradeCommand)


class GetPortfolioQueryHandler(QueryHandler[GetPortfolioQuery, Dict[str, Any]]):
    """ポートフォリオ取得クエリハンドラー"""
    
    async def handle(self, query: GetPortfolioQuery) -> TradingResult[Dict[str, Any]]:
        """ポートフォリオ取得"""
        try:
            # ポートフォリオ取得ロジック
            portfolio = {
                'portfolio_id': query.portfolio_id,
                'cash': 1000000.0,
                'positions': [
                    {'symbol': 'AAPL', 'quantity': 100, 'market_value': 15000.0},
                    {'symbol': 'GOOGL', 'quantity': 50, 'market_value': 12500.0}
                ],
                'total_value': 1027500.0,
                'as_of': (query.as_of_date or datetime.now(timezone.utc)).isoformat()
            }
            
            return TradingResult.success(portfolio)
            
        except Exception as e:
            return TradingResult.failure('QUERY_ERROR', str(e))
    
    def can_handle(self, query: Query) -> bool:
        return isinstance(query, GetPortfolioQuery)


class EventualConsistency:
    """結果整合性管理"""
    
    def __init__(self, max_delay_seconds: int = 5):
        self.max_delay_seconds = max_delay_seconds
        self._pending_updates: Dict[str, List[Dict[str, Any]]] = {}
        self._consistency_checkers: List[Callable] = []
    
    def register_consistency_checker(self, checker: Callable) -> None:
        """整合性チェッカー登録"""
        self._consistency_checkers.append(checker)
    
    async def track_update(self, aggregate_id: str, update_info: Dict[str, Any]) -> None:
        """更新追跡"""
        if aggregate_id not in self._pending_updates:
            self._pending_updates[aggregate_id] = []
        
        update_info['timestamp'] = datetime.now(timezone.utc)
        self._pending_updates[aggregate_id].append(update_info)
        
        # 遅延チェック
        asyncio.create_task(self._check_consistency_after_delay(aggregate_id))
    
    async def _check_consistency_after_delay(self, aggregate_id: str) -> None:
        """遅延後整合性チェック"""
        await asyncio.sleep(self.max_delay_seconds)
        
        # 整合性チェック実行
        for checker in self._consistency_checkers:
            try:
                await checker(aggregate_id)
            except Exception as e:
                logger.error(f"Consistency check failed for {aggregate_id}: {e}")
        
        # 完了した更新を削除
        if aggregate_id in self._pending_updates:
            current_time = datetime.now(timezone.utc)
            self._pending_updates[aggregate_id] = [
                update for update in self._pending_updates[aggregate_id]
                if (current_time - update['timestamp']).seconds < self.max_delay_seconds
            ]
    
    async def get_pending_updates(self, aggregate_id: str) -> List[Dict[str, Any]]:
        """保留中更新取得"""
        return self._pending_updates.get(aggregate_id, [])
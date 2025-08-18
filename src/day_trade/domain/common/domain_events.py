"""
ドメインイベント

DDD（Domain-Driven Design）におけるドメインイベントの実装
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Callable
from uuid import UUID, uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class DomainEvent(ABC):
    """ドメインイベント基底クラス"""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.now)
    event_version: int = 1

    @property
    @abstractmethod
    def event_type(self) -> str:
        """イベントタイプ"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'event_id': str(self.event_id),
            'event_type': self.event_type,
            'occurred_at': self.occurred_at.isoformat(),
            'event_version': self.event_version,
            'payload': self._get_payload()
        }

    @abstractmethod
    def _get_payload(self) -> Dict[str, Any]:
        """イベント固有のペイロード取得"""
        pass


class DomainEventHandler(ABC):
    """ドメインイベントハンドラー基底クラス"""

    @abstractmethod
    def handle(self, event: DomainEvent) -> None:
        """イベント処理"""
        pass


class DomainEventDispatcher:
    """ドメインイベントディスパッチャー"""

    def __init__(self):
        self._handlers: Dict[Type[DomainEvent], List[DomainEventHandler]] = {}
        self._async_handlers: Dict[Type[DomainEvent], List[Callable]] = {}
        self._event_store: List[DomainEvent] = []
        self._failed_events: List[Dict[str, Any]] = []

    def register_handler(
        self,
        event_type: Type[DomainEvent],
        handler: DomainEventHandler
    ) -> None:
        """イベントハンドラー登録"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        logger.info(f"Handler registered for {event_type.__name__}: {handler.__class__.__name__}")

    def register_async_handler(
        self,
        event_type: Type[DomainEvent],
        handler: Callable
    ) -> None:
        """非同期イベントハンドラー登録"""
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []

        self._async_handlers[event_type].append(handler)
        logger.info(f"Async handler registered for {event_type.__name__}")

    def dispatch(self, event: DomainEvent) -> None:
        """イベントディスパッチ"""
        # イベントストアに保存
        self._event_store.append(event)

        event_type = type(event)

        # 同期ハンドラー実行
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler.handle(event)
                    logger.debug(f"Event {event.event_type} handled by {handler.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Error handling event {event.event_type}: {e}")
                    self._failed_events.append({
                        'event': event.to_dict(),
                        'handler': handler.__class__.__name__,
                        'error': str(e),
                        'failed_at': datetime.now().isoformat()
                    })

        # 非同期ハンドラー実行（ここでは仮実装）
        if event_type in self._async_handlers:
            for handler in self._async_handlers[event_type]:
                try:
                    # 実際の実装では async/await やタスクキューを使用
                    handler(event)
                    logger.debug(f"Event {event.event_type} handled asynchronously")
                except Exception as e:
                    logger.error(f"Error in async handler for event {event.event_type}: {e}")

    def dispatch_all(self, events: List[DomainEvent]) -> None:
        """複数イベントの一括ディスパッチ"""
        for event in events:
            self.dispatch(event)

    def get_events(
        self,
        event_type: Optional[Type[DomainEvent]] = None,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """イベント取得"""
        events = self._event_store

        if event_type:
            events = [e for e in events if isinstance(e, event_type)]

        if limit:
            events = events[-limit:]

        return events.copy()

    def get_failed_events(self) -> List[Dict[str, Any]]:
        """失敗したイベント取得"""
        return self._failed_events.copy()

    def clear_events(self) -> None:
        """イベントストアクリア"""
        cleared_count = len(self._event_store)
        self._event_store.clear()
        logger.info(f"Cleared {cleared_count} events from store")

    def replay_failed_events(self) -> int:
        """失敗したイベントの再実行"""
        retry_count = 0
        failed_events_copy = self._failed_events.copy()
        self._failed_events.clear()

        for failed_event_info in failed_events_copy:
            try:
                # 実際の実装では、イベントを再構築して再実行
                logger.info(f"Retrying failed event: {failed_event_info['event']['event_id']}")
                retry_count += 1
            except Exception as e:
                logger.error(f"Failed to retry event: {e}")
                self._failed_events.append(failed_event_info)

        return retry_count


# グローバルディスパッチャー
domain_event_dispatcher = DomainEventDispatcher()


# 具体的なドメインイベント実装例

@dataclass
class TradeExecutedEvent(DomainEvent):
    """取引実行イベント"""
    trade_id: UUID = None
    symbol: str = ""
    quantity: int = 0
    price: str = "0"  # Decimal は JSON シリアライズのため文字列
    direction: str = ""
    commission: str = "0"

    @property
    def event_type(self) -> str:
        return "trade_executed"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            'trade_id': str(self.trade_id),
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'direction': self.direction,
            'commission': self.commission
        }


@dataclass
class PositionOpenedEvent(DomainEvent):
    """ポジション開始イベント"""
    position_id: UUID = None
    symbol: str = ""
    initial_quantity: int = 0

    @property
    def event_type(self) -> str:
        return "position_opened"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            'position_id': str(self.position_id),
            'symbol': self.symbol,
            'initial_quantity': self.initial_quantity
        }


@dataclass
class PositionClosedEvent(DomainEvent):
    """ポジション終了イベント"""
    position_id: UUID = None
    symbol: str = ""
    realized_pnl: str = "0"
    total_trades: int = 0

    @property
    def event_type(self) -> str:
        return "position_closed"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            'position_id': str(self.position_id),
            'symbol': self.symbol,
            'realized_pnl': self.realized_pnl,
            'total_trades': self.total_trades
        }


@dataclass
class RiskThresholdExceededEvent(DomainEvent):
    """リスク閾値超過イベント"""
    portfolio_id: UUID = None
    risk_type: str = ""
    current_value: str = "0"
    threshold_value: str = "0"
    severity: str = ""

    @property
    def event_type(self) -> str:
        return "risk_threshold_exceeded"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            'portfolio_id': str(self.portfolio_id),
            'risk_type': self.risk_type,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'severity': self.severity
        }


@dataclass
class MarketDataUpdatedEvent(DomainEvent):
    """市場データ更新イベント"""
    symbol: str = ""
    price: str = "0"
    volume: int = 0
    change_percent: str = "0"

    @property
    def event_type(self) -> str:
        return "market_data_updated"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'change_percent': self.change_percent
        }


# イベントハンドラー実装例

class TradeEventLogger(DomainEventHandler):
    """取引イベントロガー"""

    def handle(self, event: DomainEvent) -> None:
        if isinstance(event, TradeExecutedEvent):
            logger.info(
                f"Trade executed: {event.symbol} {event.quantity}株 "
                f"@{event.price} ({event.direction})"
            )


class RiskMonitoringHandler(DomainEventHandler):
    """リスクモニタリングハンドラー"""

    def handle(self, event: DomainEvent) -> None:
        if isinstance(event, RiskThresholdExceededEvent):
            logger.warning(
                f"Risk threshold exceeded: {event.risk_type} "
                f"current={event.current_value} threshold={event.threshold_value}"
            )

            # 実際の実装では、アラート送信やポジション調整などを行う
            self._send_risk_alert(event)

    def _send_risk_alert(self, event: RiskThresholdExceededEvent) -> None:
        """リスクアラート送信（仮実装）"""
        logger.critical(f"RISK ALERT: {event.risk_type} exceeded for portfolio {event.portfolio_id}")


class PortfolioUpdater(DomainEventHandler):
    """ポートフォリオ更新ハンドラー"""

    def handle(self, event: DomainEvent) -> None:
        if isinstance(event, (TradeExecutedEvent, PositionClosedEvent)):
            logger.info(f"Updating portfolio based on {event.event_type}")
            # 実際の実装では、ポートフォリオの状態を更新


# デフォルトハンドラー登録
domain_event_dispatcher.register_handler(TradeExecutedEvent, TradeEventLogger())
domain_event_dispatcher.register_handler(RiskThresholdExceededEvent, RiskMonitoringHandler())
domain_event_dispatcher.register_handler(TradeExecutedEvent, PortfolioUpdater())
domain_event_dispatcher.register_handler(PositionClosedEvent, PortfolioUpdater())
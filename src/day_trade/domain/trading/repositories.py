"""
取引ドメイン リポジトリインターフェース

ヘキサゴナルアーキテクチャにおけるポートの定義
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from .entities import Trade, Position, Portfolio, TradeId, PositionId
from ..common.value_objects import Symbol


class TradeRepository(ABC):
    """取引リポジトリインターフェース"""

    @abstractmethod
    def save(self, trade: Trade) -> None:
        """取引保存"""
        pass

    @abstractmethod
    def find_by_id(self, trade_id: TradeId) -> Optional[Trade]:
        """ID による取引検索"""
        pass

    @abstractmethod
    def find_by_symbol(self, symbol: Symbol) -> List[Trade]:
        """銘柄による取引検索"""
        pass

    @abstractmethod
    def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Trade]:
        """日付範囲による取引検索"""
        pass

    @abstractmethod
    def find_by_direction(self, direction: str) -> List[Trade]:
        """取引方向による検索"""
        pass

    @abstractmethod
    def count_by_symbol(self, symbol: Symbol) -> int:
        """銘柄別取引件数"""
        pass

    @abstractmethod
    def get_latest_trades(self, limit: int = 10) -> List[Trade]:
        """最新取引取得"""
        pass


class PositionRepository(ABC):
    """ポジションリポジトリインターフェース"""

    @abstractmethod
    def save(self, position: Position) -> None:
        """ポジション保存"""
        pass

    @abstractmethod
    def find_by_id(self, position_id: PositionId) -> Optional[Position]:
        """ID によるポジション検索"""
        pass

    @abstractmethod
    def find_by_symbol(self, symbol: Symbol) -> Optional[Position]:
        """銘柄によるポジション検索"""
        pass

    @abstractmethod
    def find_open_positions(self) -> List[Position]:
        """オープンポジション取得"""
        pass

    @abstractmethod
    def find_closed_positions(self) -> List[Position]:
        """クローズポジション取得"""
        pass

    @abstractmethod
    def remove(self, position_id: PositionId) -> bool:
        """ポジション削除"""
        pass


class PortfolioRepository(ABC):
    """ポートフォリオリポジトリインターフェース"""

    @abstractmethod
    def save(self, portfolio: Portfolio) -> None:
        """ポートフォリオ保存"""
        pass

    @abstractmethod
    def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """ID によるポートフォリオ検索"""
        pass

    @abstractmethod
    def get_default_portfolio(self) -> Portfolio:
        """デフォルトポートフォリオ取得"""
        pass


class MarketDataRepository(ABC):
    """市場データリポジトリインターフェース"""

    @abstractmethod
    def get_current_price(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """現在価格取得"""
        pass

    @abstractmethod
    def get_historical_prices(
        self,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """履歴価格取得"""
        pass

    @abstractmethod
    def get_market_data(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """市場データ取得"""
        pass

    @abstractmethod
    def subscribe_to_price_updates(
        self,
        symbol: Symbol,
        callback: callable
    ) -> None:
        """価格更新購読"""
        pass


class TradingSignalRepository(ABC):
    """取引シグナルリポジトリインターフェース"""

    @abstractmethod
    def save_signal(self, signal: Dict[str, Any]) -> None:
        """シグナル保存"""
        pass

    @abstractmethod
    def get_signals_by_symbol(
        self,
        symbol: Symbol,
        signal_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """銘柄別シグナル取得"""
        pass

    @abstractmethod
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """アクティブシグナル取得"""
        pass

    @abstractmethod
    def mark_signal_processed(self, signal_id: str) -> None:
        """シグナル処理済みマーク"""
        pass


class RiskMetricsRepository(ABC):
    """リスクメトリクスリポジトリインターフェース"""

    @abstractmethod
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """メトリクス保存"""
        pass

    @abstractmethod
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """最新メトリクス取得"""
        pass

    @abstractmethod
    def get_metrics_history(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """メトリクス履歴取得"""
        pass


class PerformanceRepository(ABC):
    """パフォーマンスリポジトリインターフェース"""

    @abstractmethod
    def save_performance_record(self, record: Dict[str, Any]) -> None:
        """パフォーマンスレコード保存"""
        pass

    @abstractmethod
    def get_daily_performance(self, date: datetime) -> Optional[Dict[str, Any]]:
        """日次パフォーマンス取得"""
        pass

    @abstractmethod
    def get_performance_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        pass

    @abstractmethod
    def get_benchmark_comparison(
        self,
        benchmark: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """ベンチマーク比較取得"""
        pass


class ConfigurationRepository(ABC):
    """設定リポジトリインターフェース"""

    @abstractmethod
    def get_trading_config(self) -> Dict[str, Any]:
        """取引設定取得"""
        pass

    @abstractmethod
    def save_trading_config(self, config: Dict[str, Any]) -> None:
        """取引設定保存"""
        pass

    @abstractmethod
    def get_risk_config(self) -> Dict[str, Any]:
        """リスク設定取得"""
        pass

    @abstractmethod
    def save_risk_config(self, config: Dict[str, Any]) -> None:
        """リスク設定保存"""
        pass

    @abstractmethod
    def get_watchlist(self) -> List[Symbol]:
        """ウォッチリスト取得"""
        pass

    @abstractmethod
    def update_watchlist(self, symbols: List[Symbol]) -> None:
        """ウォッチリスト更新"""
        pass


class AuditLogRepository(ABC):
    """監査ログリポジトリインターフェース"""

    @abstractmethod
    def log_action(
        self,
        action: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """アクション記録"""
        pass

    @abstractmethod
    def get_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        action_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """ログ取得"""
        pass

    @abstractmethod
    def get_user_activity(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """ユーザー活動取得"""
        pass


class UnitOfWork(ABC):
    """作業単位パターン"""

    @abstractmethod
    def __enter__(self):
        """コンテキスト開始"""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキスト終了"""
        pass

    @abstractmethod
    def commit(self) -> None:
        """変更コミット"""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """変更ロールバック"""
        pass

    @property
    @abstractmethod
    def trades(self) -> TradeRepository:
        """取引リポジトリ"""
        pass

    @property
    @abstractmethod
    def positions(self) -> PositionRepository:
        """ポジションリポジトリ"""
        pass

    @property
    @abstractmethod
    def portfolios(self) -> PortfolioRepository:
        """ポートフォリオリポジトリ"""
        pass

    @property
    @abstractmethod
    def market_data(self) -> MarketDataRepository:
        """市場データリポジトリ"""
        pass

    @property
    @abstractmethod
    def signals(self) -> TradingSignalRepository:
        """シグナルリポジトリ"""
        pass

    @property
    @abstractmethod
    def risk_metrics(self) -> RiskMetricsRepository:
        """リスクメトリクスリポジトリ"""
        pass

    @property
    @abstractmethod
    def performance(self) -> PerformanceRepository:
        """パフォーマンスリポジトリ"""
        pass

    @property
    @abstractmethod
    def configuration(self) -> ConfigurationRepository:
        """設定リポジトリ"""
        pass

    @property
    @abstractmethod
    def audit_log(self) -> AuditLogRepository:
        """監査ログリポジトリ"""
        pass
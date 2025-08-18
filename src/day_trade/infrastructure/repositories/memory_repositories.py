"""
メモリベースリポジトリ実装

テスト用および軽量実装のためのインメモリリポジトリ
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from copy import deepcopy

from ...domain.trading.entities import Trade, Position, Portfolio, TradeId, PositionId
from ...domain.trading.repositories import (
    TradeRepository, PositionRepository, PortfolioRepository,
    MarketDataRepository, TradingSignalRepository, RiskMetricsRepository,
    PerformanceRepository, ConfigurationRepository, AuditLogRepository,
    UnitOfWork
)
from ...domain.common.value_objects import Symbol


class MemoryTradeRepository(TradeRepository):
    """メモリベース取引リポジトリ"""

    def __init__(self):
        self._trades: Dict[UUID, Trade] = {}

    def save(self, trade: Trade) -> None:
        """取引保存"""
        self._trades[trade.id.value] = deepcopy(trade)

    def find_by_id(self, trade_id: TradeId) -> Optional[Trade]:
        """ID による取引検索"""
        return deepcopy(self._trades.get(trade_id.value))

    def find_by_symbol(self, symbol: Symbol) -> List[Trade]:
        """銘柄による取引検索"""
        return [
            deepcopy(trade) for trade in self._trades.values()
            if trade.symbol == symbol
        ]

    def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Trade]:
        """日付範囲による取引検索"""
        return [
            deepcopy(trade) for trade in self._trades.values()
            if start_date <= trade.executed_at <= end_date
        ]

    def find_by_direction(self, direction: str) -> List[Trade]:
        """取引方向による検索"""
        return [
            deepcopy(trade) for trade in self._trades.values()
            if trade.direction == direction
        ]

    def count_by_symbol(self, symbol: Symbol) -> int:
        """銘柄別取引件数"""
        return len(self.find_by_symbol(symbol))

    def get_latest_trades(self, limit: int = 10) -> List[Trade]:
        """最新取引取得"""
        sorted_trades = sorted(
            self._trades.values(),
            key=lambda t: t.executed_at,
            reverse=True
        )
        return [deepcopy(trade) for trade in sorted_trades[:limit]]


class MemoryPositionRepository(PositionRepository):
    """メモリベースポジションリポジトリ"""

    def __init__(self):
        self._positions: Dict[UUID, Position] = {}

    def save(self, position: Position) -> None:
        """ポジション保存"""
        self._positions[position.id.value] = deepcopy(position)

    def find_by_id(self, position_id: PositionId) -> Optional[Position]:
        """ID によるポジション検索"""
        return deepcopy(self._positions.get(position_id.value))

    def find_by_symbol(self, symbol: Symbol) -> Optional[Position]:
        """銘柄によるポジション検索"""
        for position in self._positions.values():
            if position.symbol == symbol:
                return deepcopy(position)
        return None

    def find_open_positions(self) -> List[Position]:
        """オープンポジション取得"""
        return [
            deepcopy(position) for position in self._positions.values()
            if not position.is_closed
        ]

    def find_closed_positions(self) -> List[Position]:
        """クローズポジション取得"""
        return [
            deepcopy(position) for position in self._positions.values()
            if position.is_closed
        ]

    def remove(self, position_id: PositionId) -> bool:
        """ポジション削除"""
        if position_id.value in self._positions:
            del self._positions[position_id.value]
            return True
        return False


class MemoryPortfolioRepository(PortfolioRepository):
    """メモリベースポートフォリオリポジトリ"""

    def __init__(self):
        self._portfolios: Dict[UUID, Portfolio] = {}
        self._default_portfolio_id = uuid4()
        # デフォルトポートフォリオ作成
        self._portfolios[self._default_portfolio_id] = Portfolio(self._default_portfolio_id)

    def save(self, portfolio: Portfolio) -> None:
        """ポートフォリオ保存"""
        self._portfolios[portfolio.id] = deepcopy(portfolio)

    def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """ID によるポートフォリオ検索"""
        return deepcopy(self._portfolios.get(portfolio_id))

    def get_default_portfolio(self) -> Portfolio:
        """デフォルトポートフォリオ取得"""
        return deepcopy(self._portfolios[self._default_portfolio_id])


class MemoryMarketDataRepository(MarketDataRepository):
    """メモリベース市場データリポジトリ"""

    def __init__(self):
        self._price_data: Dict[Symbol, Dict[str, Any]] = {}
        self._historical_data: Dict[Symbol, List[Dict[str, Any]]] = {}
        self._subscribers: Dict[Symbol, List[callable]] = {}

    def get_current_price(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """現在価格取得"""
        return self._price_data.get(symbol)

    def get_historical_prices(
        self,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """履歴価格取得"""
        historical = self._historical_data.get(symbol, [])
        return [
            data for data in historical
            if start_date <= data['timestamp'] <= end_date
        ]

    def get_market_data(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """市場データ取得"""
        return self.get_current_price(symbol)

    def subscribe_to_price_updates(
        self,
        symbol: Symbol,
        callback: callable
    ) -> None:
        """価格更新購読"""
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []
        self._subscribers[symbol].append(callback)

    def update_price(self, symbol: Symbol, price_data: Dict[str, Any]) -> None:
        """価格更新（テスト用）"""
        self._price_data[symbol] = price_data

        # 購読者に通知
        if symbol in self._subscribers:
            for callback in self._subscribers[symbol]:
                try:
                    callback(symbol, price_data)
                except Exception:
                    pass  # エラーは無視


class MemoryTradingSignalRepository(TradingSignalRepository):
    """メモリベース取引シグナルリポジトリ"""

    def __init__(self):
        self._signals: List[Dict[str, Any]] = []
        self._processed_signals: set = set()

    def save_signal(self, signal: Dict[str, Any]) -> None:
        """シグナル保存"""
        signal['id'] = str(uuid4())
        signal['created_at'] = datetime.now()
        self._signals.append(signal)

    def get_signals_by_symbol(
        self,
        symbol: Symbol,
        signal_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """銘柄別シグナル取得"""
        result = [
            signal for signal in self._signals
            if signal.get('symbol') == symbol.code
        ]

        if signal_type:
            result = [
                signal for signal in result
                if signal.get('signal_type') == signal_type
            ]

        return result

    def get_active_signals(self) -> List[Dict[str, Any]]:
        """アクティブシグナル取得"""
        return [
            signal for signal in self._signals
            if signal['id'] not in self._processed_signals
        ]

    def mark_signal_processed(self, signal_id: str) -> None:
        """シグナル処理済みマーク"""
        self._processed_signals.add(signal_id)


class MemoryRiskMetricsRepository(RiskMetricsRepository):
    """メモリベースリスクメトリクスリポジトリ"""

    def __init__(self):
        self._metrics: List[Dict[str, Any]] = []

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """メトリクス保存"""
        metrics['timestamp'] = datetime.now()
        self._metrics.append(metrics)

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """最新メトリクス取得"""
        if not self._metrics:
            return None
        return max(self._metrics, key=lambda m: m['timestamp'])

    def get_metrics_history(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """メトリクス履歴取得"""
        return [
            metrics for metrics in self._metrics
            if start_date <= metrics['timestamp'] <= end_date
        ]


class MemoryPerformanceRepository(PerformanceRepository):
    """メモリベースパフォーマンスリポジトリ"""

    def __init__(self):
        self._performance_records: List[Dict[str, Any]] = []

    def save_performance_record(self, record: Dict[str, Any]) -> None:
        """パフォーマンスレコード保存"""
        record['timestamp'] = datetime.now()
        self._performance_records.append(record)

    def get_daily_performance(self, date: datetime) -> Optional[Dict[str, Any]]:
        """日次パフォーマンス取得"""
        for record in self._performance_records:
            if record['timestamp'].date() == date.date():
                return record
        return None

    def get_performance_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        records = [
            record for record in self._performance_records
            if start_date <= record['timestamp'] <= end_date
        ]

        if not records:
            return {'total_return': 0, 'trade_count': 0}

        total_return = sum(record.get('daily_return', 0) for record in records)

        return {
            'total_return': total_return,
            'trade_count': len(records),
            'avg_daily_return': total_return / len(records) if records else 0
        }

    def get_benchmark_comparison(
        self,
        benchmark: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """ベンチマーク比較取得"""
        # 簡易実装
        summary = self.get_performance_summary(start_date, end_date)
        return {
            'portfolio_return': summary.get('total_return', 0),
            'benchmark_return': 0.05,  # 固定値
            'alpha': summary.get('total_return', 0) - 0.05,
            'beta': 1.0
        }


class MemoryConfigurationRepository(ConfigurationRepository):
    """メモリベース設定リポジトリ"""

    def __init__(self):
        self._trading_config = {
            'max_positions': 10,
            'max_position_size': 1000,
            'min_volume': 1000,
            'volatility_threshold': 0.02
        }
        self._risk_config = {
            'max_daily_loss': 10000,
            'max_position_risk': 0.05,
            'risk_tolerance': 0.1
        }
        self._watchlist = [
            Symbol('7203'), Symbol('9984'), Symbol('6758')
        ]

    def get_trading_config(self) -> Dict[str, Any]:
        """取引設定取得"""
        return deepcopy(self._trading_config)

    def save_trading_config(self, config: Dict[str, Any]) -> None:
        """取引設定保存"""
        self._trading_config.update(config)

    def get_risk_config(self) -> Dict[str, Any]:
        """リスク設定取得"""
        return deepcopy(self._risk_config)

    def save_risk_config(self, config: Dict[str, Any]) -> None:
        """リスク設定保存"""
        self._risk_config.update(config)

    def get_watchlist(self) -> List[Symbol]:
        """ウォッチリスト取得"""
        return deepcopy(self._watchlist)

    def update_watchlist(self, symbols: List[Symbol]) -> None:
        """ウォッチリスト更新"""
        self._watchlist = deepcopy(symbols)


class MemoryAuditLogRepository(AuditLogRepository):
    """メモリベース監査ログリポジトリ"""

    def __init__(self):
        self._logs: List[Dict[str, Any]] = []

    def log_action(
        self,
        action: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """アクション記録"""
        log_entry = {
            'id': str(uuid4()),
            'action': action,
            'details': deepcopy(details),
            'user_id': user_id,
            'timestamp': datetime.now()
        }
        self._logs.append(log_entry)

    def get_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        action_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """ログ取得"""
        result = [
            log for log in self._logs
            if start_date <= log['timestamp'] <= end_date
        ]

        if action_type:
            result = [
                log for log in result
                if log['action'] == action_type
            ]

        return result

    def get_user_activity(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """ユーザー活動取得"""
        return [
            log for log in self._logs
            if (log['user_id'] == user_id and
                start_date <= log['timestamp'] <= end_date)
        ]


class MemoryUnitOfWork(UnitOfWork):
    """メモリベース作業単位"""

    def __init__(self):
        self._trades = MemoryTradeRepository()
        self._positions = MemoryPositionRepository()
        self._portfolios = MemoryPortfolioRepository()
        self._market_data = MemoryMarketDataRepository()
        self._signals = MemoryTradingSignalRepository()
        self._risk_metrics = MemoryRiskMetricsRepository()
        self._performance = MemoryPerformanceRepository()
        self._configuration = MemoryConfigurationRepository()
        self._audit_log = MemoryAuditLogRepository()
        self._committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()

    def commit(self) -> None:
        """変更コミット"""
        self._committed = True

    def rollback(self) -> None:
        """変更ロールバック"""
        # メモリ実装では実際のロールバックは困難
        # 実際の実装ではトランザクション管理が必要
        pass

    @property
    def trades(self) -> MemoryTradeRepository:
        return self._trades

    @property
    def positions(self) -> MemoryPositionRepository:
        return self._positions

    @property
    def portfolios(self) -> MemoryPortfolioRepository:
        return self._portfolios

    @property
    def market_data(self) -> MemoryMarketDataRepository:
        return self._market_data

    @property
    def signals(self) -> MemoryTradingSignalRepository:
        return self._signals

    @property
    def risk_metrics(self) -> MemoryRiskMetricsRepository:
        return self._risk_metrics

    @property
    def performance(self) -> MemoryPerformanceRepository:
        return self._performance

    @property
    def configuration(self) -> MemoryConfigurationRepository:
        return self._configuration

    @property
    def audit_log(self) -> MemoryAuditLogRepository:
        return self._audit_log
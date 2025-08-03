"""
トランザクション管理強化ユーティリティ
Issue #187: データベース操作のトランザクション管理の拡張機能
"""

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from ..models.database import db_manager
from ..utils.logging_config import get_context_logger, log_business_event, log_error_with_context, log_performance_metric

logger = get_context_logger(__name__, component="transaction_manager")


class TransactionIsolationLevel(Enum):
    """トランザクション分離レベル"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionStatus(Enum):
    """トランザクション状態"""
    STARTED = "started"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionMetrics:
    """トランザクションメトリクス"""
    transaction_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TransactionStatus = TransactionStatus.STARTED
    isolation_level: Optional[TransactionIsolationLevel] = None
    retry_count: int = 0
    operations_count: int = 0
    affected_tables: Set[str] = field(default_factory=set)
    error_message: Optional[str] = None
    session_id: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """トランザクション実行時間"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def duration_ms(self) -> Optional[float]:
        """トランザクション実行時間（ミリ秒）"""
        if self.duration:
            return self.duration.total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "transaction_id": self.transaction_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "isolation_level": self.isolation_level.value if self.isolation_level else None,
            "retry_count": self.retry_count,
            "operations_count": self.operations_count,
            "affected_tables": list(self.affected_tables),
            "error_message": self.error_message,
            "session_id": self.session_id,
            "duration_ms": self.duration_ms
        }


class TransactionMonitor:
    """トランザクション監視クラス"""

    def __init__(self):
        self.active_transactions: Dict[str, TransactionMetrics] = {}
        self.completed_transactions: List[TransactionMetrics] = []
        self.lock = threading.Lock()
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """SQLAlchemyイベントリスナーを設定"""
        @event.listens_for(db_manager.engine, "before_execute")
        def receive_before_execute(conn, clauseelement, multiparams, params, execution_options):
            """SQL実行前のイベント"""
            # トランザクション内のSQL実行をカウント
            transaction_id = getattr(conn.info, 'transaction_id', None)
            if transaction_id and transaction_id in self.active_transactions:
                with self.lock:
                    self.active_transactions[transaction_id].operations_count += 1

    def start_transaction(
        self,
        transaction_id: Optional[str] = None,
        isolation_level: Optional[TransactionIsolationLevel] = None,
        session_id: Optional[str] = None
    ) -> str:
        """トランザクション開始の記録"""
        if not transaction_id:
            transaction_id = str(uuid4())

        metrics = TransactionMetrics(
            transaction_id=transaction_id,
            start_time=datetime.now(),
            isolation_level=isolation_level,
            session_id=session_id
        )

        with self.lock:
            self.active_transactions[transaction_id] = metrics

        logger.info("トランザクション開始",
                   transaction_id=transaction_id,
                   isolation_level=isolation_level.value if isolation_level else None,
                   session_id=session_id)

        return transaction_id

    def update_transaction(
        self,
        transaction_id: str,
        status: Optional[TransactionStatus] = None,
        affected_table: Optional[str] = None,
        retry_count: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """トランザクション状態の更新"""
        with self.lock:
            if transaction_id in self.active_transactions:
                metrics = self.active_transactions[transaction_id]

                if status:
                    metrics.status = status
                if affected_table:
                    metrics.affected_tables.add(affected_table)
                if retry_count is not None:
                    metrics.retry_count = retry_count
                if error_message:
                    metrics.error_message = error_message

    def complete_transaction(
        self,
        transaction_id: str,
        status: TransactionStatus,
        error_message: Optional[str] = None
    ):
        """トランザクション完了の記録"""
        with self.lock:
            if transaction_id in self.active_transactions:
                metrics = self.active_transactions[transaction_id]
                metrics.end_time = datetime.now()
                metrics.status = status
                if error_message:
                    metrics.error_message = error_message

                # 完了済みリストに移動
                self.completed_transactions.append(metrics)
                del self.active_transactions[transaction_id]

                # 古い記録を清理（直近1000件まで保持）
                if len(self.completed_transactions) > 1000:
                    self.completed_transactions = self.completed_transactions[-1000:]

        # メトリクスログ
        log_performance_metric(
            "transaction_duration",
            metrics.duration_ms or 0,
            "ms",
            transaction_id=transaction_id,
            status=status.value,
            retry_count=metrics.retry_count,
            operations_count=metrics.operations_count,
            affected_tables_count=len(metrics.affected_tables)
        )

        logger.info("トランザクション完了",
                   transaction_id=transaction_id,
                   status=status.value,
                   duration_ms=metrics.duration_ms,
                   operations_count=metrics.operations_count,
                   retry_count=metrics.retry_count)

    def get_active_transactions(self) -> List[TransactionMetrics]:
        """実行中のトランザクション一覧"""
        with self.lock:
            return list(self.active_transactions.values())

    def get_completed_transactions(
        self,
        limit: int = 100,
        status_filter: Optional[TransactionStatus] = None
    ) -> List[TransactionMetrics]:
        """完了済みトランザクション一覧"""
        with self.lock:
            transactions = self.completed_transactions.copy()

        if status_filter:
            transactions = [t for t in transactions if t.status == status_filter]

        return sorted(transactions, key=lambda x: x.start_time, reverse=True)[:limit]

    def get_transaction_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """トランザクション統計情報"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            recent_transactions = [
                t for t in self.completed_transactions
                if t.start_time >= cutoff_time
            ]

        if not recent_transactions:
            return {
                "period_hours": hours,
                "total_transactions": 0,
                "success_rate": 0.0,
                "average_duration_ms": 0.0,
                "total_operations": 0,
                "retry_rate": 0.0
            }

        successful = [t for t in recent_transactions if t.status == TransactionStatus.COMMITTED]
        failed = [t for t in recent_transactions if t.status in [TransactionStatus.FAILED, TransactionStatus.ROLLED_BACK]]
        retried = [t for t in recent_transactions if t.retry_count > 0]

        durations = [t.duration_ms for t in recent_transactions if t.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "period_hours": hours,
            "total_transactions": len(recent_transactions),
            "successful_transactions": len(successful),
            "failed_transactions": len(failed),
            "retried_transactions": len(retried),
            "success_rate": len(successful) / len(recent_transactions) * 100,
            "retry_rate": len(retried) / len(recent_transactions) * 100,
            "average_duration_ms": avg_duration,
            "total_operations": sum(t.operations_count for t in recent_transactions),
            "affected_tables": len(set().union(*[t.affected_tables for t in recent_transactions]))
        }


# グローバルトランザクション監視インスタンス
transaction_monitor = TransactionMonitor()


class EnhancedTransactionManager:
    """強化されたトランザクション管理クラス"""

    def __init__(self, monitor: Optional[TransactionMonitor] = None):
        self.monitor = monitor or transaction_monitor

    @contextmanager
    def enhanced_transaction(
        self,
        isolation_level: Optional[TransactionIsolationLevel] = None,
        retry_count: int = 3,
        retry_delay: float = 0.1,
        timeout_seconds: Optional[float] = None,
        readonly: bool = False,
        transaction_id: Optional[str] = None
    ):
        """
        強化されたトランザクション管理

        Args:
            isolation_level: 分離レベル
            retry_count: 再試行回数
            retry_delay: 再試行間隔
            timeout_seconds: タイムアウト（秒）
            readonly: 読み取り専用モード
            transaction_id: カスタムトランザクションID

        Usage:
            with enhanced_transaction_manager.enhanced_transaction(
                isolation_level=TransactionIsolationLevel.REPEATABLE_READ,
                timeout_seconds=30.0
            ) as session:
                # トランザクション処理
                pass
        """
        transaction_id = transaction_id or str(uuid4())

        # トランザクション開始を記録
        self.monitor.start_transaction(
            transaction_id=transaction_id,
            isolation_level=isolation_level
        )

        start_time = time.time()

        for attempt in range(retry_count + 1):
            session = None
            try:
                session = db_manager.get_session()

                # トランザクション設定
                if isolation_level:
                    session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.value}"))

                if readonly:
                    session.execute(text("SET TRANSACTION READ ONLY"))

                # セッション情報にトランザクションIDを設定
                session.info['transaction_id'] = transaction_id

                # タイムアウト監視
                session.execute(text("BEGIN"))

                self.monitor.update_transaction(transaction_id, status=TransactionStatus.ACTIVE)

                # タイムアウトチェック用の開始時間
                operation_start = time.time()

                try:
                    yield session

                    # タイムアウトチェック
                    if timeout_seconds and (time.time() - operation_start) > timeout_seconds:
                        raise OperationalError("Transaction timeout", None, None)

                    session.commit()

                    # 成功を記録
                    self.monitor.complete_transaction(
                        transaction_id,
                        TransactionStatus.COMMITTED
                    )

                    # ビジネスイベントログ
                    log_business_event(
                        "transaction_completed",
                        transaction_id=transaction_id,
                        duration_ms=(time.time() - start_time) * 1000,
                        attempt=attempt + 1,
                        isolation_level=isolation_level.value if isolation_level else None,
                        readonly=readonly
                    )

                    return

                except Exception as inner_error:
                    session.rollback()
                    raise inner_error

            except (OperationalError, IntegrityError) as e:
                self.monitor.update_transaction(
                    transaction_id,
                    retry_count=attempt,
                    error_message=str(e)
                )

                if attempt < retry_count and db_manager._is_retriable_error(e):
                    logger.warning("トランザクション再試行",
                                 transaction_id=transaction_id,
                                 attempt=attempt + 1,
                                 max_attempts=retry_count + 1,
                                 error=str(e))
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                    continue
                else:
                    # 失敗を記録
                    self.monitor.complete_transaction(
                        transaction_id,
                        TransactionStatus.FAILED,
                        str(e)
                    )

                    log_error_with_context(e, {
                        "transaction_id": transaction_id,
                        "attempts": attempt + 1,
                        "isolation_level": isolation_level.value if isolation_level else None,
                        "readonly": readonly
                    })
                    raise

            except Exception as e:
                # その他のエラー
                self.monitor.complete_transaction(
                    transaction_id,
                    TransactionStatus.FAILED,
                    str(e)
                )

                log_error_with_context(e, {
                    "transaction_id": transaction_id,
                    "error_type": "unexpected",
                    "isolation_level": isolation_level.value if isolation_level else None
                })
                raise

            finally:
                if session:
                    session.close()

    @contextmanager
    def distributed_transaction(self, transaction_contexts: List[Callable]):
        """
        分散トランザクション管理（2フェーズコミット風）

        Args:
            transaction_contexts: トランザクション処理のリスト

        Note:
            本格的な分散トランザクションではなく、
            複数のトランザクション処理を調整する簡易版
        """
        transaction_id = str(uuid4())
        logger.info("分散トランザクション開始", transaction_id=transaction_id)

        sessions = []
        prepared_transactions = []

        try:
            # Phase 1: Prepare (各トランザクションを準備)
            for i, context_func in enumerate(transaction_contexts):
                session = db_manager.get_session()
                sessions.append(session)
                session.begin()

                try:
                    # トランザクション処理を実行
                    result = context_func(session)
                    prepared_transactions.append((session, result))

                    # コミット前の整合性チェック
                    session.flush()

                except Exception as e:
                    logger.error(f"分散トランザクションPhase1失敗 (context {i})",
                               transaction_id=transaction_id,
                               error=str(e))
                    raise

            # Phase 2: Commit (すべて成功した場合のみコミット)
            for session, result in prepared_transactions:
                session.commit()

            log_business_event(
                "distributed_transaction_completed",
                transaction_id=transaction_id,
                contexts_count=len(transaction_contexts),
                status="success"
            )

            logger.info("分散トランザクション完了", transaction_id=transaction_id)
            yield  # コンテキストマネージャーとして使用するためのyield

        except Exception as e:
            # ロールバック処理
            for session in sessions:
                try:
                    session.rollback()
                except Exception as rollback_error:
                    logger.error("分散トランザクションロールバック失敗",
                               transaction_id=transaction_id,
                               rollback_error=str(rollback_error))

            log_business_event(
                "distributed_transaction_failed",
                transaction_id=transaction_id,
                contexts_count=len(transaction_contexts),
                error=str(e)
            )

            logger.error("分散トランザクション失敗", transaction_id=transaction_id, error=str(e))
            raise

        finally:
            # セッションクリーンアップ
            for session in sessions:
                try:
                    session.close()
                except Exception:
                    pass

    def execute_with_savepoint(
        self,
        operations: List[Callable],
        savepoint_name: str = "sp1"
    ):
        """
        セーブポイントを使用した部分ロールバック対応処理

        Args:
            operations: 実行する操作のリスト
            savepoint_name: セーブポイント名
        """
        transaction_id = str(uuid4())

        with self.enhanced_transaction() as session:
            try:
                # セーブポイント作成
                session.execute(text(f"SAVEPOINT {savepoint_name}"))

                for i, operation in enumerate(operations):
                    try:
                        operation(session)
                        # 各操作後に中間状態をflush
                        session.flush()

                    except Exception as operation_error:
                        logger.warning(f"操作 {i} でエラー、セーブポイントにロールバック",
                                     transaction_id=transaction_id,
                                     operation_index=i,
                                     error=str(operation_error))

                        # セーブポイントにロールバック
                        session.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint_name}"))

                        # エラーを再発生させるか、継続するかは呼び出し元が決定
                        raise

                # セーブポイント削除
                session.execute(text(f"RELEASE SAVEPOINT {savepoint_name}"))

            except Exception as e:
                logger.error("セーブポイント処理失敗", transaction_id=transaction_id, error=str(e))
                raise


# グローバルインスタンス
enhanced_transaction_manager = EnhancedTransactionManager()


def with_transaction_monitoring(func: Callable) -> Callable:
    """
    トランザクション監視付きデコレーター

    Usage:
        @with_transaction_monitoring
        def my_database_operation():
            # データベース処理
            pass
    """
    def wrapper(*args, **kwargs):
        transaction_id = str(uuid4())

        start_time = time.time()
        transaction_monitor.start_transaction(transaction_id)

        try:
            result = func(*args, **kwargs)
            transaction_monitor.complete_transaction(
                transaction_id,
                TransactionStatus.COMMITTED
            )
            return result

        except Exception as e:
            transaction_monitor.complete_transaction(
                transaction_id,
                TransactionStatus.FAILED,
                str(e)
            )
            raise

    return wrapper


def get_transaction_health_report() -> Dict[str, Any]:
    """
    トランザクションヘルスレポート生成

    Returns:
        システムのトランザクション健全性レポート
    """
    stats_24h = transaction_monitor.get_transaction_statistics(24)
    stats_1h = transaction_monitor.get_transaction_statistics(1)
    active_transactions = transaction_monitor.get_active_transactions()

    # 長時間実行中のトランザクションを検出
    now = datetime.now()
    long_running_threshold = timedelta(minutes=5)
    long_running = [
        t for t in active_transactions
        if (now - t.start_time) > long_running_threshold
    ]

    return {
        "report_time": datetime.now().isoformat(),
        "active_transactions_count": len(active_transactions),
        "long_running_transactions_count": len(long_running),
        "stats_24h": stats_24h,
        "stats_1h": stats_1h,
        "health_status": "healthy" if stats_24h["success_rate"] > 95 else "warning",
        "recommendations": _generate_health_recommendations(stats_24h, long_running)
    }


def _generate_health_recommendations(stats: Dict[str, Any], long_running: List[TransactionMetrics]) -> List[str]:
    """健全性改善提案を生成"""
    recommendations = []

    if stats["success_rate"] < 95:
        recommendations.append("トランザクション成功率が低下しています。エラーログを確認してください。")

    if stats["retry_rate"] > 10:
        recommendations.append("再試行率が高いです。データベース負荷やロック競合を確認してください。")

    if stats["average_duration_ms"] > 1000:
        recommendations.append("トランザクション実行時間が長いです。クエリ最適化を検討してください。")

    if long_running:
        recommendations.append(f"{len(long_running)}個の長時間実行トランザクションが検出されました。")

    if not recommendations:
        recommendations.append("トランザクション処理は健全に動作しています。")

    return recommendations

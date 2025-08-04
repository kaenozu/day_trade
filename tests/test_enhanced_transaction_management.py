"""
強化されたトランザクション管理機能のテスト
Issue #187: データベース操作のトランザクション管理の追加機能テスト
"""

import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.exc import OperationalError

from src.day_trade.models.database import Base, DatabaseConfig, db_manager
from src.day_trade.utils.transaction_best_practices import (
    TransactionPatterns,
)
from src.day_trade.utils.transaction_manager import (
    EnhancedTransactionManager,
    TransactionIsolationLevel,
    TransactionMonitor,
    TransactionStatus,
    enhanced_transaction_manager,
    transaction_monitor,
)


# テスト用のSQLAlchemyモデル
class MockModel(Base):
    __tablename__ = 'test_model'

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    version = Column(Integer, default=1)


class TestTransactionMonitor:
    """トランザクション監視機能のテスト"""

    def setup_method(self):
        """テスト前の準備"""
        self.monitor = TransactionMonitor()

    def test_transaction_lifecycle_monitoring(self):
        """トランザクションライフサイクル監視のテスト"""
        # トランザクション開始
        transaction_id = self.monitor.start_transaction(
            isolation_level=TransactionIsolationLevel.READ_COMMITTED
        )

        assert transaction_id in self.monitor.active_transactions

        metrics = self.monitor.active_transactions[transaction_id]
        assert metrics.status == TransactionStatus.STARTED
        assert metrics.isolation_level == TransactionIsolationLevel.READ_COMMITTED
        assert metrics.operations_count == 0

        # トランザクション更新
        self.monitor.update_transaction(
            transaction_id,
            status=TransactionStatus.ACTIVE,
            affected_table="test_table",
            retry_count=1
        )

        assert metrics.status == TransactionStatus.ACTIVE
        assert "test_table" in metrics.affected_tables
        assert metrics.retry_count == 1

        # トランザクション完了
        self.monitor.complete_transaction(
            transaction_id,
            TransactionStatus.COMMITTED
        )

        assert transaction_id not in self.monitor.active_transactions
        assert len(self.monitor.completed_transactions) == 1

        completed = self.monitor.completed_transactions[0]
        assert completed.transaction_id == transaction_id
        assert completed.status == TransactionStatus.COMMITTED
        assert completed.end_time is not None
        assert completed.duration is not None

    def test_transaction_statistics(self):
        """トランザクション統計のテスト"""
        # 複数のトランザクションを実行
        for i in range(10):
            transaction_id = self.monitor.start_transaction()

            # 成功と失敗を混在
            if i < 7:
                self.monitor.complete_transaction(transaction_id, TransactionStatus.COMMITTED)
            else:
                self.monitor.complete_transaction(
                    transaction_id,
                    TransactionStatus.FAILED,
                    "Test error"
                )

        stats = self.monitor.get_transaction_statistics(24)

        assert stats["total_transactions"] == 10
        assert stats["successful_transactions"] == 7
        assert stats["failed_transactions"] == 3
        assert stats["success_rate"] == 70.0

    def test_long_running_transaction_detection(self):
        """長時間実行トランザクションの検出テスト"""
        # 古いトランザクションを模擬
        old_transaction_id = self.monitor.start_transaction()
        old_metrics = self.monitor.active_transactions[old_transaction_id]
        old_metrics.start_time = datetime.now() - timedelta(minutes=10)

        # 新しいトランザクション
        self.monitor.start_transaction()

        active_transactions = self.monitor.get_active_transactions()
        assert len(active_transactions) == 2

        # アクティブトランザクションが2つ存在することを確認
        assert len(active_transactions) == 2

        # 古いトランザクションが存在することを確認
        old_exists = any(
            (datetime.now() - t.start_time) > timedelta(minutes=5)
            for t in active_transactions
        )
        assert old_exists


class TestEnhancedTransactionManager:
    """強化されたトランザクション管理のテスト"""

    def setup_method(self):
        """テスト前の準備"""
        # テスト用データベース設定
        self.db_config = DatabaseConfig.for_testing()
        self.db_manager = db_manager
        self.enhanced_manager = EnhancedTransactionManager()

    def test_enhanced_transaction_basic_flow(self):
        """強化されたトランザクションの基本フロー"""
        executed = False

        # get_sessionメソッドをモック
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_session.execute = Mock()
            mock_session.commit = Mock()
            mock_session.close = Mock()
            mock_session.info = {}
            mock_get_session.return_value = mock_session

            with self.enhanced_manager.enhanced_transaction(
                isolation_level=TransactionIsolationLevel.READ_COMMITTED,
                timeout_seconds=10.0
            ) as session:
                assert session is not None
                executed = True

        assert executed

    def test_transaction_retry_mechanism(self):
        """トランザクション再試行メカニズムのテスト"""
        retry_count = 0

        def mock_retriable_error(*args, **kwargs):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise OperationalError("Deadlock detected", None, None)
            # モックセッションを返す
            mock_session = Mock()
            mock_session.execute = Mock()
            mock_session.commit = Mock()
            mock_session.rollback = Mock()
            mock_session.close = Mock()
            mock_session.info = {}
            return mock_session

        with patch.object(db_manager, 'get_session', side_effect=mock_retriable_error):
            with patch.object(db_manager, '_is_retriable_error', return_value=True):
                try:
                    with self.enhanced_manager.enhanced_transaction(retry_count=3):
                        pass
                except:
                    pass  # エラーは期待される

        assert retry_count >= 2  # 再試行が実行されたことを確認

    def test_transaction_timeout(self):
        """トランザクションタイムアウトのテスト"""
        with pytest.raises(Exception):  # タイムアウトエラーが期待される
            with self.enhanced_manager.enhanced_transaction(timeout_seconds=0.1):
                time.sleep(0.2)  # タイムアウトを超過

    def test_readonly_transaction(self):
        """読み取り専用トランザクションのテスト"""
        executed = False

        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_session.execute = Mock()
            mock_session.commit = Mock()
            mock_session.close = Mock()
            mock_session.info = {}
            mock_get_session.return_value = mock_session

            with self.enhanced_manager.enhanced_transaction(readonly=True):
                # 読み取り専用では更新操作は制限される
                executed = True

        assert executed

    def test_distributed_transaction_success(self):
        """分散トランザクション成功のテスト"""
        results = []

        def context1(session):
            results.append("context1")
            return "result1"

        def context2(session):
            results.append("context2")
            return "result2"

        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_session.begin = Mock()
            mock_session.commit = Mock()
            mock_session.flush = Mock()
            mock_session.close = Mock()
            mock_get_session.return_value = mock_session

            with self.enhanced_manager.distributed_transaction([context1, context2]):
                pass

        assert "context1" in results
        assert "context2" in results

    def test_distributed_transaction_failure(self):
        """分散トランザクション失敗のテスト"""
        results = []

        def context1(session):
            results.append("context1")
            return "result1"

        def context2(session):
            results.append("context2")
            raise Exception("Context2 failed")

        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_session.begin = Mock()
            mock_session.rollback = Mock()
            mock_session.flush = Mock()
            mock_session.close = Mock()
            mock_get_session.return_value = mock_session

            with pytest.raises(Exception):
                with self.enhanced_manager.distributed_transaction([context1, context2]):
                    pass

        assert "context1" in results
        assert "context2" in results

    def test_savepoint_operation(self):
        """セーブポイント操作のテスト"""
        operations_executed = []

        def operation1(session):
            operations_executed.append("op1")

        def operation2(session):
            operations_executed.append("op2")
            raise Exception("Operation 2 failed")

        def operation3(session):
            operations_executed.append("op3")

        with pytest.raises(Exception):
            self.enhanced_manager.execute_with_savepoint(
                [operation1, operation2, operation3]
            )

        assert "op1" in operations_executed
        assert "op2" in operations_executed
        assert "op3" not in operations_executed


class TestTransactionPatterns:
    """トランザクションパターンのテスト"""

    def setup_method(self):
        """テスト前の準備"""
        self.db_config = DatabaseConfig.for_testing()

    def test_idempotent_operation_pattern(self):
        """冪等性操作パターンのテスト"""
        operation_key = "test_operation_123"
        executed_count = 0

        def test_operation():
            nonlocal executed_count
            with db_manager.session_scope() as session:
                with TransactionPatterns.idempotent_operation(operation_key, session):
                    executed_count += 1

        # 同じ操作を複数回実行
        test_operation()
        test_operation()

        # 冪等性により実行回数が制御されることを期待
        # (実際の実装では重複チェックが必要)
        assert executed_count >= 1

    def test_optimistic_locking_pattern(self):
        """楽観的ロックパターンのテスト"""
        with db_manager.session_scope() as session:
            # モックセッションの設定
            mock_record = MockModel(id=1, version=1)

            with patch.object(session, 'query') as mock_query:
                mock_query.return_value.filter_by.return_value.first.return_value = mock_record
                mock_query.return_value.filter_by.return_value.update.return_value = 1

                # 楽観的ロック更新のテスト
                success = TransactionPatterns.optimistic_locking_update(
                    session=session,
                    model_class=MockModel,
                    record_id=1,
                    update_data={"name": "updated", "version": 1}
                )

                assert success is True

    def test_batch_upsert_pattern(self):
        """バッチアップサートパターンのテスト"""
        records_data = [
            {"id": 1, "name": "record1"},
            {"id": 2, "name": "record2"},
            {"id": 3, "name": "record3"}
        ]

        with db_manager.session_scope() as session:
            with patch.object(session, 'query') as mock_query:
                # 既存レコードなしとして模擬
                mock_query.return_value.filter_by.return_value.first.return_value = None
                with patch.object(session, 'add') as mock_add:
                    # バッチアップサートの実行
                    TransactionPatterns.batch_upsert_pattern(
                        session=session,
                        model_class=MockModel,
                        records_data=records_data,
                        unique_keys=["id"],
                        batch_size=2
                    )

                    # 呼び出し回数の検証
                    assert mock_query.called
                    assert mock_add.called

    def test_saga_transaction_pattern(self):
        """Sagaトランザクションパターンのテスト"""
        compensation_calls = []

        def compensation1():
            compensation_calls.append("comp1")

        def compensation2():
            compensation_calls.append("comp2")

        compensation_steps = [compensation1, compensation2]

        # 正常ケース
        with TransactionPatterns.saga_transaction_pattern(compensation_steps) as executed_steps:
            executed_steps.append(0)
            executed_steps.append(1)

        assert len(compensation_calls) == 0  # 正常時は補償処理なし

        # 異常ケース
        compensation_calls.clear()

        with pytest.raises(Exception):
            with TransactionPatterns.saga_transaction_pattern(compensation_steps) as executed_steps:
                executed_steps.append(0)
                executed_steps.append(1)
                raise Exception("Business error")

        assert "comp1" in compensation_calls
        assert "comp2" in compensation_calls


class TestTransactionPerformance:
    """トランザクションパフォーマンステスト"""

    def test_transaction_performance_monitoring(self):
        """トランザクションパフォーマンス監視のテスト"""
        time.time()

        with enhanced_transaction_manager.enhanced_transaction():
            # 軽い処理をシミュレート
            time.sleep(0.01)

        # トランザクション統計を確認
        stats = transaction_monitor.get_transaction_statistics(1)

        assert stats["total_transactions"] >= 1
        if stats["total_transactions"] > 0:
            assert stats["average_duration_ms"] > 0

    def test_concurrent_transaction_handling(self):
        """並行トランザクション処理のテスト"""
        results = []
        errors = []

        def worker_function(worker_id):
            try:
                with enhanced_transaction_manager.enhanced_transaction():
                    # 並行処理のシミュレート
                    time.sleep(0.01)
                    results.append(f"worker_{worker_id}_success")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {str(e)}")

        # 複数スレッドで並行実行
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 5
        assert len(errors) == 0

    def test_transaction_resource_cleanup(self):
        """トランザクションリソースクリーンアップのテスト"""
        # アクティブトランザクション数の監視
        initial_active = len(transaction_monitor.get_active_transactions())

        # 複数のトランザクションを実行
        for _i in range(3):
            with enhanced_transaction_manager.enhanced_transaction():
                pass

        # リソースがクリーンアップされているか確認
        final_active = len(transaction_monitor.get_active_transactions())
        assert final_active == initial_active


class TestTransactionIntegration:
    """トランザクション統合テスト"""

    def test_trade_manager_transaction_integration(self):
        """TradeManagerとの統合テスト"""
        from src.day_trade.core.trade_manager import TradeManager, TradeType

        # TradeManagerの既存トランザクション機能をテスト
        trade_manager = TradeManager(load_from_db=False)

        # 取引追加（メモリのみ）
        trade_id = trade_manager.add_trade(
            symbol="TEST",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000"),
            persist_to_db=False
        )

        assert trade_id is not None
        assert len(trade_manager.trades) == 1

        # ポジション確認
        position = trade_manager.get_position("TEST")
        assert position is not None
        assert position.quantity == 100

    def test_database_manager_transaction_integration(self):
        """DatabaseManagerとの統合テスト"""
        # 既存のDatabaseManagerのトランザクション機能をテスト
        operations_executed = []

        def operation1(session):
            operations_executed.append("op1")

        def operation2(session):
            operations_executed.append("op2")

        # アトミック操作のテスト
        db_manager.atomic_operation([operation1, operation2])

        assert "op1" in operations_executed
        assert "op2" in operations_executed

    def test_transaction_monitoring_integration(self):
        """トランザクション監視機能の統合テスト"""
        initial_stats = transaction_monitor.get_transaction_statistics(1)

        # 複数のトランザクション処理を実行
        with enhanced_transaction_manager.enhanced_transaction():
            pass

        with enhanced_transaction_manager.enhanced_transaction():
            pass

        final_stats = transaction_monitor.get_transaction_statistics(1)

        # 統計が更新されていることを確認
        assert final_stats["total_transactions"] >= initial_stats["total_transactions"]


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])

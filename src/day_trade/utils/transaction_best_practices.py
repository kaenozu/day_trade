"""
トランザクション管理ベストプラクティス集
Issue #187: データベース操作のトランザクション管理の知見とパターン集
"""

from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, OperationalError

from ..models.database import db_manager
from ..utils.logging_config import get_context_logger, log_business_event, log_error_with_context
from .transaction_manager import enhanced_transaction_manager, TransactionIsolationLevel

logger = get_context_logger(__name__, component="transaction_best_practices")


class TransactionPatterns:
    """よく使用されるトランザクションパターンの実装集"""

    @staticmethod
    @contextmanager
    def idempotent_operation(operation_key: str, session: Session):
        """
        冪等性保証パターン

        同じ操作を複数回実行しても安全になるようにする

        Args:
            operation_key: 操作の一意キー
            session: データベースセッション
        """
        # 操作ログテーブルで重複実行を防ぐ
        # (実際の実装では専用のoperation_logテーブルを使用)

        operation_logger = logger.bind(
            pattern="idempotent_operation",
            operation_key=operation_key
        )

        try:
            # 操作が既に実行済みかチェック
            # existing_operation = session.query(OperationLog).filter_by(
            #     operation_key=operation_key
            # ).first()
            #
            # if existing_operation:
            #     operation_logger.info("操作は既に実行済みです")
            #     return

            operation_logger.info("冪等操作開始")
            yield session

            # 操作完了を記録
            # operation_log = OperationLog(
            #     operation_key=operation_key,
            #     executed_at=datetime.now(),
            #     status="completed"
            # )
            # session.add(operation_log)

            operation_logger.info("冪等操作完了")

        except Exception as e:
            operation_logger.error("冪等操作失敗", error=str(e))
            raise

    @staticmethod
    def optimistic_locking_update(
        session: Session,
        model_class,
        record_id: Any,
        update_data: Dict[str, Any],
        version_field: str = "version"
    ) -> bool:
        """
        楽観的ロックによる更新パターン

        Args:
            session: データベースセッション
            model_class: 更新対象のモデルクラス
            record_id: レコードID
            update_data: 更新データ
            version_field: バージョンフィールド名

        Returns:
            更新成功フラグ
        """
        update_logger = logger.bind(
            pattern="optimistic_locking_update",
            model_class=model_class.__name__,
            record_id=record_id
        )

        try:
            # 現在のレコードを取得
            record = session.query(model_class).filter_by(id=record_id).first()
            if not record:
                update_logger.warning("更新対象レコードが存在しません")
                return False

            current_version = getattr(record, version_field)
            expected_version = update_data.pop(version_field, current_version)

            # バージョンチェック
            if current_version != expected_version:
                update_logger.warning("楽観的ロック競合検出",
                                    current_version=current_version,
                                    expected_version=expected_version)
                return False

            # 更新実行（バージョンをインクリメント）
            update_data[version_field] = current_version + 1

            rows_updated = session.query(model_class).filter_by(
                id=record_id,
                **{version_field: current_version}
            ).update(update_data)

            if rows_updated == 0:
                update_logger.warning("他のプロセスにより更新されました")
                return False

            session.flush()
            update_logger.info("楽観的ロック更新成功",
                             new_version=current_version + 1)
            return True

        except Exception as e:
            update_logger.error("楽観的ロック更新失敗", error=str(e))
            raise

    @staticmethod
    def batch_upsert_pattern(
        session: Session,
        model_class,
        records_data: List[Dict[str, Any]],
        unique_keys: List[str],
        batch_size: int = 1000
    ):
        """
        バッチアップサート（存在すれば更新、なければ挿入）パターン

        Args:
            session: データベースセッション
            model_class: 対象モデルクラス
            records_data: レコードデータのリスト
            unique_keys: 一意キーのフィールド名リスト
            batch_size: バッチサイズ
        """
        batch_logger = logger.bind(
            pattern="batch_upsert",
            model_class=model_class.__name__,
            total_records=len(records_data),
            batch_size=batch_size
        )

        batch_logger.info("バッチアップサート開始")

        inserted_count = 0
        updated_count = 0

        try:
            for i in range(0, len(records_data), batch_size):
                batch = records_data[i:i + batch_size]

                for record_data in batch:
                    # 一意キーで既存レコードを検索
                    filter_conditions = {
                        key: record_data[key] for key in unique_keys
                        if key in record_data
                    }

                    existing_record = session.query(model_class).filter_by(
                        **filter_conditions
                    ).first()

                    if existing_record:
                        # 更新
                        for key, value in record_data.items():
                            if hasattr(existing_record, key):
                                setattr(existing_record, key, value)
                        updated_count += 1
                    else:
                        # 挿入
                        new_record = model_class(**record_data)
                        session.add(new_record)
                        inserted_count += 1

                # バッチごとにflush
                session.flush()

                batch_logger.debug("バッチ処理完了",
                                 batch_number=i // batch_size + 1,
                                 processed=min(i + batch_size, len(records_data)))

            batch_logger.info("バッチアップサート完了",
                            inserted_count=inserted_count,
                            updated_count=updated_count)

        except Exception as e:
            batch_logger.error("バッチアップサート失敗", error=str(e))
            raise

    @staticmethod
    @contextmanager
    def saga_transaction_pattern(compensation_steps: List[Callable]):
        """
        Sagaパターン（分散トランザクション代替）

        Args:
            compensation_steps: 補償処理のリスト（逆順で実行される）
        """
        saga_logger = logger.bind(pattern="saga_transaction")
        executed_steps = []

        try:
            saga_logger.info("Sagaトランザクション開始", steps_count=len(compensation_steps))

            # 前進フェーズは呼び出し元で実行
            yield executed_steps

            saga_logger.info("Sagaトランザクション正常完了")

        except Exception as e:
            saga_logger.error("Sagaトランザクション失敗、補償処理開始", error=str(e))

            # 補償フェーズ（実行済みステップを逆順で取り消し）
            for step_index in reversed(executed_steps):
                try:
                    if step_index < len(compensation_steps):
                        compensation_steps[step_index]()
                        saga_logger.info("補償処理完了", step=step_index)
                except Exception as compensation_error:
                    saga_logger.error("補償処理失敗",
                                    step=step_index,
                                    error=str(compensation_error))

            saga_logger.info("補償処理完了")
            raise


class TransactionOptimizationTips:
    """トランザクション最適化のヒント集"""

    @staticmethod
    def minimize_transaction_scope_example():
        """トランザクションスコープを最小化する例"""

        # ❌ 悪い例：長時間のトランザクション
        def bad_example():
            with db_manager.session_scope() as session:
                # 外部API呼び出し（時間がかかる）
                # external_data = fetch_external_api()

                # データベース処理
                # record = Model(data=external_data)
                # session.add(record)
                pass

        # ✅ 良い例：短時間のトランザクション
        def good_example():
            # 外部API呼び出し（トランザクション外で実行）
            # external_data = fetch_external_api()

            # データベース処理のみトランザクション内で実行
            with db_manager.session_scope() as session:
                # record = Model(data=external_data)
                # session.add(record)
                pass

    @staticmethod
    def proper_isolation_level_usage():
        """適切な分離レベルの使用例"""

        # レポート生成（ダーティリード許容）
        def generate_report():
            with enhanced_transaction_manager.enhanced_transaction(
                isolation_level=TransactionIsolationLevel.READ_UNCOMMITTED,
                readonly=True
            ) as session:
                # 高速なレポート生成
                # result = session.query(...).all()
                pass

        # 金融取引（厳密な一貫性要求）
        def financial_transaction():
            with enhanced_transaction_manager.enhanced_transaction(
                isolation_level=TransactionIsolationLevel.SERIALIZABLE
            ) as session:
                # 厳密な整合性が必要な処理
                # account = session.query(Account).filter_by(id=account_id).first()
                # account.balance -= amount
                pass

    @staticmethod
    def bulk_operation_optimization():
        """バルク操作の最適化例"""

        def optimized_bulk_insert(data_list: List[Dict]):
            """最適化されたバルク挿入"""

            # 大量データを小さなバッチに分割
            BATCH_SIZE = 1000

            for i in range(0, len(data_list), BATCH_SIZE):
                batch = data_list[i:i + BATCH_SIZE]

                with db_manager.session_scope() as session:
                    # バルク挿入使用
                    # session.bulk_insert_mappings(Model, batch)

                    # 定期的にメモリをクリア
                    session.expunge_all()

    @staticmethod
    def deadlock_prevention_strategies():
        """デッドロック防止戦略"""

        def ordered_resource_access():
            """リソースアクセスの順序統一"""

            def update_accounts(account_id1: int, account_id2: int, amount: Decimal):
                # アカウントIDを昇順でソート（デッドロック防止）
                ordered_ids = sorted([account_id1, account_id2])

                with db_manager.session_scope() as session:
                    # 常に同じ順序でロック取得
                    for account_id in ordered_ids:
                        # account = session.query(Account).filter_by(id=account_id).with_for_update().first()
                        pass

                    # 実際の更新処理
                    # update_account_balance(session, account_id1, -amount)
                    # update_account_balance(session, account_id2, +amount)

    @staticmethod
    def connection_pool_optimization():
        """コネクションプール最適化"""

        def monitor_connection_pool():
            """コネクションプール状態の監視"""
            pool = db_manager.engine.pool

            pool_status = {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalidated()
            }

            logger.info("コネクションプール状態", **pool_status)
            return pool_status


class TransactionTesting:
    """トランザクション処理のテストパターン"""

    @staticmethod
    def test_transaction_rollback():
        """トランザクションロールバックのテスト"""

        def test_function():
            # テスト用のトランザクション
            with db_manager.session_scope() as session:
                # テストデータ挿入
                # test_record = TestModel(data="test")
                # session.add(test_record)
                # session.flush()

                # 意図的にエラーを発生
                raise Exception("Test rollback")

        # ロールバック後にデータが残っていないことを確認
        try:
            test_function()
        except Exception:
            pass

        # データが存在しないことを確認
        with db_manager.session_scope() as session:
            # count = session.query(TestModel).filter_by(data="test").count()
            # assert count == 0, "Rollback failed"
            pass

    @staticmethod
    def test_concurrent_access():
        """並行アクセスのテスト"""
        import threading
        import time

        def concurrent_update_test():
            results = []

            def update_worker(worker_id: int):
                try:
                    with db_manager.session_scope() as session:
                        # 楽観的ロックテスト
                        # record = session.query(TestModel).filter_by(id=1).first()
                        # time.sleep(0.1)  # 競合状態を作る
                        # record.value += 1
                        results.append(f"Worker {worker_id}: Success")
                except Exception as e:
                    results.append(f"Worker {worker_id}: {str(e)}")

            # 複数スレッドで同時実行
            threads = []
            for i in range(5):
                thread = threading.Thread(target=update_worker, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            return results


def create_transaction_performance_benchmark():
    """トランザクションパフォーマンステスト作成"""

    import time
    from typing import NamedTuple

    class BenchmarkResult(NamedTuple):
        operation_name: str
        total_time: float
        operations_count: int
        ops_per_second: float
        average_time_per_op: float

    def benchmark_transaction_patterns() -> List[BenchmarkResult]:
        """各種トランザクションパターンのパフォーマンステスト"""

        results = []

        # 1. 単一トランザクション vs バッチトランザクション
        def single_transactions(count: int):
            start_time = time.time()
            for i in range(count):
                with db_manager.session_scope() as session:
                    # record = TestModel(value=i)
                    # session.add(record)
                    pass
            return time.time() - start_time

        def batch_transaction(count: int):
            start_time = time.time()
            with db_manager.session_scope() as session:
                for i in range(count):
                    # record = TestModel(value=i)
                    # session.add(record)
                    pass
            return time.time() - start_time

        # ベンチマーク実行
        operation_count = 1000

        single_time = single_transactions(operation_count)
        results.append(BenchmarkResult(
            "single_transactions",
            single_time,
            operation_count,
            operation_count / single_time,
            single_time / operation_count
        ))

        batch_time = batch_transaction(operation_count)
        results.append(BenchmarkResult(
            "batch_transaction",
            batch_time,
            operation_count,
            operation_count / batch_time,
            batch_time / operation_count
        ))

        return results


# 使用例とベストプラクティスの実演
if __name__ == "__main__":

    # 1. 冪等性保証の例
    with db_manager.session_scope() as session:
        with TransactionPatterns.idempotent_operation("user_registration_123", session):
            # ユーザー登録処理（重複実行防止）
            pass

    # 2. 楽観的ロック更新の例
    with db_manager.session_scope() as session:
        success = TransactionPatterns.optimistic_locking_update(
            session=session,
            model_class=None,  # 実際のモデルクラスを指定
            record_id=1,
            update_data={"name": "updated_name", "version": 1}
        )
        logger.info("楽観的ロック更新結果", success=success)

    # 3. Sagaパターンの例
    def compensation_step_1():
        logger.info("補償処理1実行")

    def compensation_step_2():
        logger.info("補償処理2実行")

    try:
        with TransactionPatterns.saga_transaction_pattern([compensation_step_1, compensation_step_2]) as executed_steps:
            # ステップ1実行
            executed_steps.append(0)
            logger.info("ステップ1完了")

            # ステップ2実行
            executed_steps.append(1)
            logger.info("ステップ2完了")

            # エラーを意図的に発生（補償処理テスト）
            # raise Exception("Business logic error")

    except Exception as e:
        logger.info("Sagaパターンテスト完了", error=str(e))

    logger.info("トランザクションベストプラクティス例実行完了")

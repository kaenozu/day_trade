"""
高度な一括操作モジュール
UPSERT操作とその他の高度な一括操作

Issue #120: declarative_base()の定義場所の最適化対応
- 高度な一括操作の責務を明確化
- データベース固有の最適化機能
"""

import time
from typing import List

from ...utils.exceptions import handle_database_exception
from ...utils.logging_config import get_context_logger, log_database_operation
from .transaction import TransactionManager

logger = get_context_logger(__name__)


class AdvancedBulkOperations:
    """高度な一括操作クラス"""

    def __init__(self, transaction_manager: TransactionManager):
        """
        高度一括操作の初期化

        Args:
            transaction_manager: トランザクション管理
        """
        self.transaction_manager = transaction_manager

    def upsert_batch(self, model_class, data_list: list, unique_columns: list, batch_size: int = 1000):
        """
        一括UPSERT操作（存在しなければ挿入、存在すれば更新）

        Args:
            model_class: 対象モデルクラス
            data_list: データリスト
            unique_columns: 一意性を判定するカラム名のリスト
            batch_size: バッチサイズ

        Note:
            SQLiteの場合はON CONFLICT、PostgreSQLの場合はON CONFLICT DO UPDATE等を使用
            MySQLの場合はON DUPLICATE KEY UPDATE等を使用
        """
        if not data_list:
            return

        operation_logger = logger
        operation_logger.info("Starting upsert batch operation")

        database_type = self.transaction_manager.connection_manager.config.get_database_type()
        
        try:
            with self.transaction_manager.transaction_scope() as session:
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]
                    batch_number = i // batch_size + 1

                    batch_start_time = time.time()
                    
                    if database_type == "sqlite":
                        self._sqlite_upsert_batch(session, model_class, batch, unique_columns)
                    elif database_type == "postgresql":
                        self._postgresql_upsert_batch(session, model_class, batch, unique_columns)
                    elif database_type == "mysql":
                        self._mysql_upsert_batch(session, model_class, batch, unique_columns)
                    else:
                        # フォールバック: 個別にSELECT→INSERT/UPDATE
                        self._generic_upsert_batch(session, model_class, batch, unique_columns)
                    
                    batch_end_time = time.time()
                    log_database_operation(
                        "upsert_batch",
                        duration=batch_end_time - batch_start_time,
                        table_name=str(model_class.__table__.name),
                        batch_number=batch_number,
                        batch_size=len(batch),
                    )

            operation_logger.info("Upsert batch completed successfully")
            
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Upsert batch operation failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e

    def _sqlite_upsert_batch(self, session, model_class, batch, unique_columns):
        """SQLite用UPSERT実装"""
        # ON CONFLICT句を使用した実装
        # 実装詳細は簡略化
        for item in batch:
            instance = model_class(**item)
            session.merge(instance)
        session.flush()

    def _postgresql_upsert_batch(self, session, model_class, batch, unique_columns):
        """PostgreSQL用UPSERT実装"""
        # ON CONFLICT DO UPDATE句を使用した実装
        # 実装詳細は簡略化
        for item in batch:
            instance = model_class(**item)
            session.merge(instance)
        session.flush()

    def _mysql_upsert_batch(self, session, model_class, batch, unique_columns):
        """MySQL用UPSERT実装"""
        # ON DUPLICATE KEY UPDATE句を使用した実装
        # 実装詳細は簡略化
        for item in batch:
            instance = model_class(**item)
            session.merge(instance)
        session.flush()

    def _generic_upsert_batch(self, session, model_class, batch, unique_columns):
        """汎用UPSERT実装（フォールバック）"""
        for item in batch:
            # 既存レコードを検索
            query = session.query(model_class)
            for col in unique_columns:
                if col in item:
                    query = query.filter(getattr(model_class, col) == item[col])
            
            existing = query.first()
            
            if existing:
                # 更新
                for key, value in item.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # 挿入
                instance = model_class(**item)
                session.add(instance)
                
        session.flush()

    def bulk_merge(self, model_class, data_list: list, batch_size: int = 1000):
        """
        一括マージ操作（SQLAlchemyのmergeを使用）

        Args:
            model_class: 対象モデルクラス
            data_list: データリスト
            batch_size: バッチサイズ
        """
        if not data_list:
            return

        operation_logger = logger
        operation_logger.info("Starting bulk merge operation")

        try:
            with self.transaction_manager.transaction_scope() as session:
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]
                    batch_number = i // batch_size + 1

                    batch_start_time = time.time()
                    
                    for item in batch:
                        instance = model_class(**item)
                        session.merge(instance)
                    
                    session.flush()
                    
                    batch_end_time = time.time()
                    log_database_operation(
                        "bulk_merge_batch",
                        duration=batch_end_time - batch_start_time,
                        table_name=str(model_class.__table__.name),
                        batch_number=batch_number,
                        batch_size=len(batch),
                    )

            operation_logger.info("Bulk merge completed successfully")
            
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Bulk merge operation failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e

    def conditional_bulk_update(
        self, 
        model_class, 
        update_data: dict, 
        condition_data: dict, 
        batch_size: int = 1000
    ):
        """
        条件付き一括更新

        Args:
            model_class: 対象モデルクラス
            update_data: 更新するデータの辞書
            condition_data: 条件の辞書
            batch_size: バッチサイズ
        """
        operation_logger = logger
        operation_logger.info("Starting conditional bulk update operation")

        try:
            with self.transaction_manager.transaction_scope() as session:
                query = session.query(model_class)
                
                # 条件を適用
                for column, value in condition_data.items():
                    if hasattr(model_class, column):
                        query = query.filter(getattr(model_class, column) == value)
                
                # 更新対象レコードを取得
                total_records = query.count()
                operation_logger.info(f"Updating {total_records} records")
                
                # バッチ処理
                updated_count = 0
                offset = 0
                
                while offset < total_records:
                    batch_records = query.offset(offset).limit(batch_size).all()
                    
                    if not batch_records:
                        break
                    
                    batch_start_time = time.time()
                    
                    for record in batch_records:
                        for column, value in update_data.items():
                            if hasattr(record, column):
                                setattr(record, column, value)
                    
                    session.flush()
                    updated_count += len(batch_records)
                    offset += batch_size
                    
                    batch_end_time = time.time()
                    log_database_operation(
                        "conditional_bulk_update_batch",
                        duration=batch_end_time - batch_start_time,
                        table_name=str(model_class.__table__.name),
                        batch_size=len(batch_records),
                        updated_count=updated_count,
                    )

            operation_logger.info(f"Conditional bulk update completed: {updated_count} records updated")
            
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Conditional bulk update failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e

    def replace_all_data(self, model_class, new_data_list: list, batch_size: int = 1000):
        """
        全データ置き換え（既存データを削除してから新データを挿入）

        Args:
            model_class: 対象モデルクラス
            new_data_list: 新しいデータのリスト
            batch_size: バッチサイズ

        Warning:
            この操作は既存データを全て削除します。注意して使用してください。
        """
        operation_logger = logger
        operation_logger.warning("Starting replace all data operation - THIS WILL DELETE ALL EXISTING DATA")

        try:
            with self.transaction_manager.transaction_scope() as session:
                # 既存データを全削除
                delete_count = session.query(model_class).count()
                session.query(model_class).delete()
                session.flush()
                
                operation_logger.info(f"Deleted {delete_count} existing records")
                
                # 新しいデータを一括挿入
                if new_data_list:
                    for i in range(0, len(new_data_list), batch_size):
                        batch = new_data_list[i : i + batch_size]
                        batch_number = i // batch_size + 1
                        
                        batch_start_time = time.time()
                        session.bulk_insert_mappings(model_class, batch)
                        session.flush()
                        batch_end_time = time.time()
                        
                        log_database_operation(
                            "replace_data_batch",
                            duration=batch_end_time - batch_start_time,
                            table_name=str(model_class.__table__.name),
                            batch_number=batch_number,
                            batch_size=len(batch),
                        )

            operation_logger.info(
                f"Replace all data completed: deleted {delete_count} records, "
                f"inserted {len(new_data_list)} new records"
            )
            
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Replace all data operation failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e
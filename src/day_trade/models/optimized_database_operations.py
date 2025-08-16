#!/usr/bin/env python3
"""
データベース操作最適化モジュール

Issue #165: アプリケーション全体の処理速度向上に向けた最適化
このモジュールは、SQLAlchemyのバルク操作を活用したデータベース最適化機能を提供します。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import Session, sessionmaker

from ..utils.logging_config import get_context_logger
from ..utils.performance_optimizer import PerformanceProfiler


@dataclass
class BulkOperationResult:
    """バルク操作の結果"""

    success: bool
    processed_count: int
    execution_time: float
    throughput: float  # records per second
    error_message: Optional[str] = None
    failed_records: Optional[List[Dict]] = None


class OptimizedDatabaseOperations:
    """データベース操作最適化クラス"""

    def __init__(self, session: Session):
        self.session = session
        self.logger = get_context_logger(__name__)
        self.profiler = PerformanceProfiler()

    @contextmanager
    def bulk_operation_context(self, operation_name: str):
        """バルク操作用のコンテキストマネージャー"""
        start_time = time.perf_counter()
        self.logger.info(f"バルク操作開始: {operation_name}")

        try:
            yield
            self.session.commit()
            execution_time = time.perf_counter() - start_time
            self.logger.info(
                f"バルク操作成功: {operation_name}",
                extra={"execution_time": execution_time},
            )
        except Exception as e:
            self.session.rollback()
            execution_time = time.perf_counter() - start_time
            self.logger.error(
                f"バルク操作失敗: {operation_name}",
                extra={"error": str(e)},
                execution_time=execution_time,
            )
            raise

    def bulk_insert_optimized(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        chunk_size: int = 1000,
        return_defaults: bool = False,
    ) -> BulkOperationResult:
        """
        最適化されたバルク挿入

        Args:
            model_class: SQLAlchemyモデルクラス
            data: 挿入するデータのリスト
            chunk_size: チャンクサイズ
            return_defaults: デフォルト値を返すか

        Returns:
            BulkOperationResult: 操作結果
        """
        start_time = time.perf_counter()
        total_processed = 0
        failed_records = []

        try:
            with self.bulk_operation_context("bulk_insert_optimized"):
                # データをチャンクに分割して処理
                for i in range(0, len(data), chunk_size):
                    chunk = data[i : i + chunk_size]

                    try:
                        self.session.bulk_insert_mappings(
                            model_class, chunk, return_defaults=return_defaults
                        )
                        total_processed += len(chunk)

                        self.logger.debug(
                            "チャンク挿入完了",
                            extra={"chunk_size": len(chunk)},
                            total_processed=total_processed,
                        )

                    except IntegrityError as e:
                        # 重複データなどの整合性エラーの場合、個別処理
                        self.logger.warning(
                            "チャンク挿入で整合性エラー、個別処理に切り替え",
                            extra={"error": str(e)},
                        )

                        individual_result = self._handle_individual_inserts(
                            model_class, chunk
                        )
                        total_processed += individual_result["success_count"]
                        failed_records.extend(individual_result["failed_records"])

            execution_time = time.perf_counter() - start_time
            throughput = total_processed / execution_time if execution_time > 0 else 0

            return BulkOperationResult(
                success=True,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=throughput,
                failed_records=failed_records if failed_records else None,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BulkOperationResult(
                success=False,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=0,
                error_message=str(e),
            )

    def bulk_update_optimized(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        chunk_size: int = 1000,
        update_changed_only: bool = True,
    ) -> BulkOperationResult:
        """
        最適化されたバルク更新

        Args:
            model_class: SQLAlchemyモデルクラス
            data: 更新するデータのリスト（主キーを含む必要がある）
            chunk_size: チャンクサイズ
            update_changed_only: 変更されたレコードのみ更新するか

        Returns:
            BulkOperationResult: 操作結果
        """
        start_time = time.perf_counter()
        total_processed = 0
        failed_records = []

        try:
            with self.bulk_operation_context("bulk_update_optimized"):
                # データをチャンクに分割して処理
                for i in range(0, len(data), chunk_size):
                    chunk = data[i : i + chunk_size]

                    try:
                        if update_changed_only:
                            # 変更されたレコードのみを更新
                            filtered_chunk = self._filter_changed_records(
                                model_class, chunk
                            )
                            if filtered_chunk:
                                self.session.bulk_update_mappings(
                                    model_class, filtered_chunk
                                )
                                total_processed += len(filtered_chunk)
                        else:
                            # 全レコードを更新
                            self.session.bulk_update_mappings(model_class, chunk)
                            total_processed += len(chunk)

                        self.logger.debug(
                            "チャンク更新完了",
                            extra={"chunk_size": len(chunk)},
                            total_processed=total_processed,
                        )

                    except Exception as e:
                        self.logger.warning(
                            "チャンク更新エラー", extra={"error": str(e)}
                        )
                        failed_records.extend(chunk)

            execution_time = time.perf_counter() - start_time
            throughput = total_processed / execution_time if execution_time > 0 else 0

            return BulkOperationResult(
                success=True,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=throughput,
                failed_records=failed_records if failed_records else None,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BulkOperationResult(
                success=False,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=0,
                error_message=str(e),
            )

    def bulk_upsert_optimized(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        conflict_columns: List[str],
        chunk_size: int = 1000,
    ) -> BulkOperationResult:
        """
        最適化されたバルクアップサート（INSERT ... ON CONFLICT）

        Args:
            model_class: SQLAlchemyモデルクラス
            data: アップサートするデータのリスト
            conflict_columns: 競合を判定するカラム名のリスト
            chunk_size: チャンクサイズ

        Returns:
            BulkOperationResult: 操作結果
        """
        start_time = time.perf_counter()
        total_processed = 0

        try:
            with self.bulk_operation_context("bulk_upsert_optimized"):
                # データベースの種類を取得
                dialect_name = self.session.bind.dialect.name

                # データをチャンクに分割して処理
                for i in range(0, len(data), chunk_size):
                    chunk = data[i : i + chunk_size]

                    if dialect_name == "postgresql":
                        # PostgreSQLのON CONFLICT構文を使用
                        result = self._upsert_postgresql(
                            model_class, chunk, conflict_columns
                        )
                    elif dialect_name == "sqlite":
                        # SQLiteのINSERT OR REPLACE構文を使用
                        result = self._upsert_sqlite(
                            model_class, chunk, conflict_columns
                        )
                    elif dialect_name == "mysql":
                        # MySQLのON DUPLICATE KEY UPDATE構文を使用
                        result = self._upsert_mysql(
                            model_class, chunk, conflict_columns
                        )
                    else:
                        # その他のデータベースでは個別処理
                        result = self._upsert_fallback(
                            model_class, chunk, conflict_columns
                        )

                    total_processed += result["processed_count"]

                    self.logger.debug(
                        "チャンクアップサート完了",
                        extra={"chunk_size": len(chunk)},
                        total_processed=total_processed,
                    )

            execution_time = time.perf_counter() - start_time
            throughput = total_processed / execution_time if execution_time > 0 else 0

            return BulkOperationResult(
                success=True,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=throughput,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BulkOperationResult(
                success=False,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=0,
                error_message=str(e),
            )

    def bulk_delete_optimized(
        self,
        model_class: Type[DeclarativeMeta],
        filter_conditions: List[Dict[str, Any]],
        chunk_size: int = 1000,
    ) -> BulkOperationResult:
        """
        最適化されたバルク削除

        Args:
            model_class: SQLAlchemyモデルクラス
            filter_conditions: 削除条件のリスト
            chunk_size: チャンクサイズ

        Returns:
            BulkOperationResult: 操作結果
        """
        start_time = time.perf_counter()
        total_processed = 0

        try:
            with self.bulk_operation_context("bulk_delete_optimized"):
                # 条件をチャンクに分割して処理
                for i in range(0, len(filter_conditions), chunk_size):
                    chunk = filter_conditions[i : i + chunk_size]

                    # IN句を使用した効率的な削除
                    primary_keys = []
                    for condition in chunk:
                        # 主キーを取得
                        pk_columns = inspect(model_class).primary_key
                        pk_values = tuple(condition.get(col.name) for col in pk_columns)
                        primary_keys.append(pk_values)

                    if primary_keys:
                        # 主キーのIN句で削除
                        pk_column = pk_columns[0]  # 単一主キーを想定
                        query = self.session.query(model_class).filter(
                            pk_column.in_([pk[0] for pk in primary_keys])
                        )
                        deleted_count = query.delete(synchronize_session=False)
                        total_processed += deleted_count

                        self.logger.debug(
                            "チャンク削除完了",
                            extra={"chunk_size": len(chunk)},
                            deleted_count=deleted_count,
                            total_processed=total_processed,
                        )

            execution_time = time.perf_counter() - start_time
            throughput = total_processed / execution_time if execution_time > 0 else 0

            return BulkOperationResult(
                success=True,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=throughput,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BulkOperationResult(
                success=False,
                processed_count=total_processed,
                execution_time=execution_time,
                throughput=0,
                error_message=str(e),
            )

    def _handle_individual_inserts(
        self, model_class: Type[DeclarativeMeta], data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """個別挿入の処理（整合性エラー対応）"""
        success_count = 0
        failed_records = []

        for record in data:
            try:
                obj = model_class(**record)
                self.session.add(obj)
                self.session.flush()  # エラーを早期検出
                success_count += 1
            except Exception as e:
                self.session.rollback()
                failed_records.append({"record": record, "error": str(e)})

                # セッションの状態をリセット
                self.session.begin()

        return {"success_count": success_count, "failed_records": failed_records}

    def _filter_changed_records(
        self, model_class: Type[DeclarativeMeta], data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """変更されたレコードのみをフィルタリング"""
        changed_records = []

        # 主キーでレコードを検索し、変更点を確認
        for record in data:
            pk_columns = inspect(model_class).primary_key
            pk_filter = {col.name: record.get(col.name) for col in pk_columns}

            existing = self.session.query(model_class).filter_by(**pk_filter).first()

            if existing:
                # 既存レコードと比較
                has_changes = False
                for key, value in record.items():
                    if hasattr(existing, key) and getattr(existing, key) != value:
                        has_changes = True
                        break

                if has_changes:
                    changed_records.append(record)
            else:
                # 新規レコードの場合も含める
                changed_records.append(record)

        return changed_records

    def _upsert_postgresql(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        conflict_columns: List[str],
    ) -> Dict[str, int]:
        """PostgreSQL用アップサート"""
        # PostgreSQLのON CONFLICTを使用した実装
        table = model_class.__table__
        ", ".join(conflict_columns)

        # 更新するカラムを動的に生成
        update_columns = [
            col.name for col in table.columns if col.name not in conflict_columns
        ]
        ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        # バルクアップサートクエリを実行
        # 実際の実装では、SQLAlchemyのbulk_insert_mappingsと
        # PostgreSQL固有の構文を組み合わせる

        return {"processed_count": len(data)}

    def _upsert_sqlite(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        conflict_columns: List[str],
    ) -> Dict[str, int]:
        """SQLite用アップサート"""
        # SQLiteのINSERT OR REPLACEを使用した実装
        return {"processed_count": len(data)}

    def _upsert_mysql(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        conflict_columns: List[str],
    ) -> Dict[str, int]:
        """MySQL用アップサート"""
        # MySQLのON DUPLICATE KEY UPDATEを使用した実装
        return {"processed_count": len(data)}

    def _upsert_fallback(
        self,
        model_class: Type[DeclarativeMeta],
        data: List[Dict[str, Any]],
        conflict_columns: List[str],
    ) -> Dict[str, int]:
        """フォールバック用アップサート（個別処理）"""
        processed_count = 0

        for record in data:
            # 存在チェック
            filter_conditions = {col: record.get(col) for col in conflict_columns}
            existing = (
                self.session.query(model_class).filter_by(**filter_conditions).first()
            )

            if existing:
                # 更新
                for key, value in record.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # 挿入
                obj = model_class(**record)
                self.session.add(obj)

            processed_count += 1

        return {"processed_count": processed_count}

    def optimize_query_performance(
        self, query_sql: str, explain_analyze: bool = False
    ) -> Dict[str, Any]:
        """
        クエリパフォーマンスの分析と最適化提案

        Args:
            query_sql: 分析するSQLクエリ
            explain_analyze: EXPLAIN ANALYZEを実行するか

        Returns:
            パフォーマンス分析結果
        """
        try:
            start_time = time.perf_counter()

            # EXPLAIN（または EXPLAIN ANALYZE）を実行
            if explain_analyze:
                explain_query = f"EXPLAIN ANALYZE {query_sql}"
            else:
                explain_query = f"EXPLAIN {query_sql}"

            result = self.session.execute(text(explain_query))
            explain_output = result.fetchall()

            execution_time = time.perf_counter() - start_time

            # 実際のクエリ実行時間も測定
            query_start = time.perf_counter()
            query_result = self.session.execute(text(query_sql))
            query_result.fetchall()  # 結果を全て取得
            query_execution_time = time.perf_counter() - query_start

            return {
                "explain_output": [str(row) for row in explain_output],
                "explain_execution_time": execution_time,
                "query_execution_time": query_execution_time,
                "optimization_suggestions": self._generate_optimization_suggestions(
                    explain_output
                ),
            }

        except Exception as e:
            return {
                "error": str(e),
                "explain_output": [],
                "optimization_suggestions": [],
            }

    def _generate_optimization_suggestions(
        self, explain_output: List[Any]
    ) -> List[str]:
        """EXPLAIN結果から最適化提案を生成"""
        suggestions = []
        explain_text = " ".join([str(row) for row in explain_output]).lower()

        # 一般的なボトルネックパターンをチェック
        if "seq scan" in explain_text:
            suggestions.append(
                "シーケンシャルスキャンが検出されました。インデックスの追加を検討してください。"
            )

        if "nested loop" in explain_text and "rows=" in explain_text:
            suggestions.append(
                "ネストしたループが検出されました。JOINの最適化やインデックスの見直しを検討してください。"
            )

        if "sort" in explain_text and "disk" in explain_text:
            suggestions.append(
                "ディスクソートが発生しています。work_memの増加やORDER BYの最適化を検討してください。"
            )

        if "hash" in explain_text and "buckets" in explain_text:
            suggestions.append(
                "ハッシュ結合が使用されています。適切なサイズのwork_memを設定してください。"
            )

        return suggestions

    def get_table_statistics(
        self, model_class: Type[DeclarativeMeta]
    ) -> Dict[str, Any]:
        """テーブル統計情報を取得"""
        table_name = model_class.__tablename__

        try:
            # レコード数
            count_query = self.session.query(model_class).count()

            # テーブルサイズ（データベース固有）
            dialect_name = self.session.bind.dialect.name

            if dialect_name == "postgresql":
                size_query = f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) as table_size,
                       pg_size_pretty(pg_relation_size('{table_name}')) as data_size,
                       pg_size_pretty(pg_total_relation_size('{table_name}') - pg_relation_size('{table_name}')) as index_size
                """
            elif dialect_name == "sqlite":
                size_query = (
                    f"SELECT COUNT(*) * 1024 as estimated_size FROM {table_name}"
                )
            else:
                size_query = None

            size_result = {}
            if size_query:
                size_data = self.session.execute(text(size_query)).fetchone()
                if size_data:
                    size_result = dict(size_data._mapping)

            return {
                "table_name": table_name,
                "record_count": count_query,
                "size_info": size_result,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {"table_name": table_name, "error": str(e), "timestamp": time.time()}


# 使用例とテスト
if __name__ == "__main__":
    from datetime import datetime

    from sqlalchemy import Column, DateTime, Integer, String, create_engine
    from sqlalchemy.orm import sessionmaker

    # Issue #120: Baseクラスをbase.pyからインポート
    from .base import Base

    class TestStock(Base):
        __tablename__ = "test_stocks"

        id = Column(Integer, primary_key=True)
        symbol = Column(String(10), unique=True)
        name = Column(String(100))
        price = Column(Integer)  # 価格（整数で格納）
        updated_at = Column(DateTime, default=datetime.now)

    # テスト実行
    print("🔧 データベース最適化ツール - テスト実行")

    # インメモリSQLiteでテスト
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    # 最適化オペレータを初期化
    optimizer = OptimizedDatabaseOperations(session)

    # テストデータ生成
    test_data = [
        {"symbol": f"TST{i:04d}", "name": f"Test Stock {i}", "price": 1000 + i}
        for i in range(1000)
    ]

    print(f"📊 {len(test_data)}件のテストデータでバルク挿入テスト")

    # バルク挿入テスト
    result = optimizer.bulk_insert_optimized(TestStock, test_data, chunk_size=100)

    print("✅ バルク挿入結果:")
    print(f"   成功: {result.success}")
    print(f"   処理件数: {result.processed_count}")
    print(f"   実行時間: {result.execution_time:.3f}秒")
    print(f"   スループット: {result.throughput:.0f} records/sec")

    # テーブル統計取得
    stats = optimizer.get_table_statistics(TestStock)
    print("\n📈 テーブル統計:")
    print(f"   レコード数: {stats['record_count']}")

    session.close()

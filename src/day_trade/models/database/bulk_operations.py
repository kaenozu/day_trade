"""
一括操作管理モジュール（統合版）
大量データの一括挿入・更新、チャンク単位での読み込み

Issue #120: declarative_base()の定義場所の最適化対応
- 一括操作の責務を明確化、モジュール分割で300行以下に最適化
- パフォーマンス最適化
"""

from typing import Any, Generator, List, Optional

from .basic_operations import BasicBulkOperations
from .read_operations import ReadOperations
from .advanced_operations import AdvancedBulkOperations
from .transaction import TransactionManager

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class BulkOperationsManager:
    """一括操作統合管理クラス（300行以下に最適化）"""

    def __init__(self, transaction_manager: TransactionManager):
        """
        一括操作管理の初期化

        Args:
            transaction_manager: トランザクション管理
        """
        self.transaction_manager = transaction_manager
        self.basic_ops = BasicBulkOperations(transaction_manager)
        self.read_ops = ReadOperations(transaction_manager)
        self.advanced_ops = AdvancedBulkOperations(transaction_manager)

    # 基本操作のデリゲート
    def bulk_insert(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括挿入（堅牢性向上版）

        Args:
            model_class: 挿入するモデルクラス
            data_list: 挿入するデータのリスト（辞書形式）
            batch_size: バッチサイズ

        Raises:
            DatabaseError: 一括挿入に失敗した場合
        """
        return self.basic_ops.bulk_insert(model_class, data_list, batch_size)

    def bulk_update(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括更新（堅牢性向上版）

        Args:
            model_class: 更新するモデルクラス
            data_list: 更新するデータのリスト（辞書形式、idが必要）
            batch_size: バッチサイズ

        Raises:
            DatabaseError: 一括更新に失敗した場合
        """
        return self.basic_ops.bulk_update(model_class, data_list, batch_size)

    def bulk_delete(self, model_class, filter_conditions: list, batch_size: int = 1000):
        """
        大量データの一括削除

        Args:
            model_class: 削除するモデルクラス
            filter_conditions: 削除条件のリスト
            batch_size: バッチサイズ

        Raises:
            DatabaseError: 一括削除に失敗した場合
        """
        return self.basic_ops.bulk_delete(model_class, filter_conditions, batch_size)

    # 読み込み操作のデリゲート
    def read_in_chunks(
        self,
        model_class,
        chunk_size: int = 1000,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
    ) -> Generator[List[Any], None, None]:
        """
        指定されたモデルからデータをチャンク単位で読み込むジェネレータ

        Args:
            model_class: 読み込むモデルクラス
            chunk_size: 1チャンクあたりのレコード数
            filters: 読み込みに適用するフィルターのリスト (SQLAlchemy filter句)
            order_by: 読み込み順序 (SQLAlchemy order_by句)

        Yields:
            List[Any]: チャンクごとのモデルインスタンスのリスト
        """
        return self.read_ops.read_in_chunks(model_class, chunk_size, filters, order_by)

    def streaming_read(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
        buffer_size: int = 10000,
    ) -> Generator[Any, None, None]:
        """
        メモリ効率的なストリーミング読み込み

        Args:
            model_class: 読み込むモデルクラス
            filters: フィルター条件
            order_by: ソート順
            buffer_size: バッファサイズ

        Yields:
            個々のモデルインスタンス
        """
        return self.read_ops.streaming_read(model_class, filters, order_by, buffer_size)

    def count_records(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
    ) -> int:
        """レコード数をカウント"""
        return self.read_ops.count_records(model_class, filters)

    def exists(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
    ) -> bool:
        """指定条件のレコードが存在するかチェック"""
        return self.read_ops.exists(model_class, filters)

    def find_first(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
    ) -> Optional[Any]:
        """指定条件の最初のレコードを取得"""
        return self.read_ops.find_first(model_class, filters, order_by)

    def paginate(
        self,
        model_class,
        page: int = 1,
        per_page: int = 100,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
    ) -> dict:
        """ページネーション機能"""
        return self.read_ops.paginate(model_class, page, per_page, filters, order_by)

    # 高度な操作のデリゲート
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
        return self.advanced_ops.upsert_batch(model_class, data_list, unique_columns, batch_size)

    def bulk_merge(self, model_class, data_list: list, batch_size: int = 1000):
        """一括マージ操作（SQLAlchemyのmergeを使用）"""
        return self.advanced_ops.bulk_merge(model_class, data_list, batch_size)

    def conditional_bulk_update(
        self, 
        model_class, 
        update_data: dict, 
        condition_data: dict, 
        batch_size: int = 1000
    ):
        """条件付き一括更新"""
        return self.advanced_ops.conditional_bulk_update(
            model_class, update_data, condition_data, batch_size
        )

    def replace_all_data(self, model_class, new_data_list: list, batch_size: int = 1000):
        """
        全データ置き換え（既存データを削除してから新データを挿入）

        Warning:
            この操作は既存データを全て削除します。注意して使用してください。
        """
        return self.advanced_ops.replace_all_data(model_class, new_data_list, batch_size)

    # 統合便利メソッド
    def get_operation_stats(self) -> dict:
        """一括操作の統計情報を取得"""
        return {
            "basic_operations": {
                "available": ["bulk_insert", "bulk_update", "bulk_delete"]
            },
            "read_operations": {
                "available": ["read_in_chunks", "streaming_read", "count_records", 
                             "exists", "find_first", "paginate"]
            },
            "advanced_operations": {
                "available": ["upsert_batch", "bulk_merge", "conditional_bulk_update", 
                             "replace_all_data"]
            }
        }
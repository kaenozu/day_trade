"""
読み込み操作モジュール
チャンク単位・ストリーミング読み込み機能

Issue #120: declarative_base()の定義場所の最適化対応
- 読み込み操作の責務を明確化
- メモリ効率的な読み込み処理
"""

from typing import Any, Generator, List, Optional

from ...utils.logging_config import get_context_logger
from .transaction import TransactionManager

logger = get_context_logger(__name__)


class ReadOperations:
    """読み込み操作クラス"""

    def __init__(self, transaction_manager: TransactionManager):
        """
        読み込み操作の初期化

        Args:
            transaction_manager: トランザクション管理
        """
        self.transaction_manager = transaction_manager

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
        operation_logger = logger.bind(
            operation="read_in_chunks",
            model_class=model_class.__name__,
            chunk_size=chunk_size,
        )
        operation_logger.info("Starting chunked read operation")

        offset = 0
        total_read = 0
        while True:
            with self.transaction_manager.read_only_session() as session:
                query = session.query(model_class)
                if filters:
                    for f in filters:
                        query = query.filter(f)
                if order_by is not None:
                    query = query.order_by(order_by)

                # チャンク読み込み
                chunk = query.offset(offset).limit(chunk_size).all()

                if not chunk:
                    break  # データがなくなったら終了

                total_read += len(chunk)
                operation_logger.debug(
                    f"Read chunk: offset={offset}, size={len(chunk)}, total_read={total_read}"
                )
                yield chunk
                offset += chunk_size

            if len(chunk) < chunk_size:
                break  # 最後のチャンクがchunk_size未満なら全件読み込み完了

        operation_logger.info(
            "Chunked read operation completed", extra={"total_records_read": total_read}
        )

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
        operation_logger = logger.bind(
            operation="streaming_read",
            model_class=model_class.__name__,
            buffer_size=buffer_size,
        )
        operation_logger.info("Starting streaming read operation")

        processed_count = 0
        with self.transaction_manager.read_only_session() as session:
            query = session.query(model_class)
            
            if filters:
                for f in filters:
                    query = query.filter(f)
            if order_by is not None:
                query = query.order_by(order_by)

            # SQLAlchemyの結果をストリーミング処理
            for record in query.yield_per(buffer_size):
                processed_count += 1
                if processed_count % buffer_size == 0:
                    operation_logger.debug(f"Processed {processed_count} records")
                yield record

        operation_logger.info(
            "Streaming read completed", extra={"total_records_processed": processed_count}
        )

    def count_records(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
    ) -> int:
        """
        レコード数をカウント

        Args:
            model_class: 対象モデルクラス
            filters: フィルター条件

        Returns:
            int: レコード数
        """
        operation_logger = logger.bind(
            operation="count_records",
            model_class=model_class.__name__,
        )
        
        with self.transaction_manager.read_only_session() as session:
            query = session.query(model_class)
            
            if filters:
                for f in filters:
                    query = query.filter(f)
            
            count = query.count()
            operation_logger.info(f"Record count: {count}")
            return count

    def exists(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
    ) -> bool:
        """
        指定条件のレコードが存在するかチェック

        Args:
            model_class: 対象モデルクラス
            filters: フィルター条件

        Returns:
            bool: 存在する場合True
        """
        operation_logger = logger.bind(
            operation="exists_check",
            model_class=model_class.__name__,
        )
        
        with self.transaction_manager.read_only_session() as session:
            query = session.query(model_class)
            
            if filters:
                for f in filters:
                    query = query.filter(f)
            
            exists = session.query(query.exists()).scalar()
            operation_logger.debug(f"Exists result: {exists}")
            return exists

    def find_first(
        self,
        model_class,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        指定条件の最初のレコードを取得

        Args:
            model_class: 対象モデルクラス
            filters: フィルター条件
            order_by: ソート順

        Returns:
            モデルインスタンスまたはNone
        """
        operation_logger = logger.bind(
            operation="find_first",
            model_class=model_class.__name__,
        )
        
        with self.transaction_manager.read_only_session() as session:
            query = session.query(model_class)
            
            if filters:
                for f in filters:
                    query = query.filter(f)
            if order_by is not None:
                query = query.order_by(order_by)
            
            result = query.first()
            operation_logger.debug(f"Found first record: {result is not None}")
            return result

    def paginate(
        self,
        model_class,
        page: int = 1,
        per_page: int = 100,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
    ) -> dict:
        """
        ページネーション機能

        Args:
            model_class: 対象モデルクラス
            page: ページ番号（1から始まる）
            per_page: ページあたりのレコード数
            filters: フィルター条件
            order_by: ソート順

        Returns:
            dict: ページネーション情報とデータ
        """
        operation_logger = logger.bind(
            operation="paginate",
            model_class=model_class.__name__,
            page=page,
            per_page=per_page,
        )
        
        with self.transaction_manager.read_only_session() as session:
            query = session.query(model_class)
            
            if filters:
                for f in filters:
                    query = query.filter(f)
            if order_by is not None:
                query = query.order_by(order_by)
            
            # 総レコード数を取得
            total_count = query.count()
            
            # オフセット計算
            offset = (page - 1) * per_page
            
            # ページデータを取得
            items = query.offset(offset).limit(per_page).all()
            
            # ページネーション情報の計算
            has_next = (offset + per_page) < total_count
            has_prev = page > 1
            total_pages = (total_count + per_page - 1) // per_page  # 切り上げ計算
            
            result = {
                "items": items,
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            }
            
            operation_logger.info(
                f"Pagination completed: page {page}/{total_pages}, {len(items)} items"
            )
            return result
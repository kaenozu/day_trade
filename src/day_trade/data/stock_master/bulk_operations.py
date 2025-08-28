"""
銘柄マスタの一括処理操作モジュール

このモジュールは大量の銘柄データを効率的に処理する機能を提供します。
一括追加、一括更新、一括削除、upsert機能等を実装しています。
"""

from typing import Dict, List

from ...models.bulk_operations import AdvancedBulkOperations
from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockBulkOperations:
    """銘柄一括処理操作クラス"""

    def __init__(self, db_manager):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
        """
        self.db_manager = db_manager
        self.bulk_operations = AdvancedBulkOperations(db_manager)

    def bulk_add_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        銘柄の一括追加（AdvancedBulkOperations使用・パフォーマンス最適化版）

        Args:
            stocks_data: 銘柄データのリスト
                例: [{'code': '1000', 'name': '株式会社A', 'market': '東証プライム', ...}, ...]
            batch_size: バッチサイズ

        Returns:
            追加結果の統計情報
        """
        if not stocks_data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        try:
            # データの検証と準備
            validated_data = []
            for stock_data in stocks_data:
                if not stock_data.get("code") or not stock_data.get("name"):
                    logger.warning(f"無効な銘柄データをスキップ: {stock_data}")
                    continue
                validated_data.append(
                    {
                        "code": stock_data["code"],
                        "name": stock_data["name"],
                        "market": stock_data.get("market"),
                        "sector": stock_data.get("sector"),
                        "industry": stock_data.get("industry"),
                    }
                )

            # AdvancedBulkOperationsを使用して一括挿入
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="ignore",  # 重複は無視
                chunk_size=batch_size,
                unique_columns=["code"],
            )

            logger.info(f"銘柄一括追加完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括追加エラー: {e}")
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "errors": len(stocks_data),
            }

    def bulk_update_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        銘柄の一括更新（AdvancedBulkOperations使用・パフォーマンス最適化版）

        Args:
            stocks_data: 更新する銘柄データのリスト
                例: [{'code': '1000', 'name': '新社名', 'sector': '新セクター', ...}, ...]
            batch_size: バッチサイズ

        Returns:
            更新結果の統計情報
        """
        if not stocks_data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        try:
            # データの検証と準備
            validated_data = []
            for stock_data in stocks_data:
                code = stock_data.get("code")
                if not code:
                    logger.warning(f"銘柄コードが無効です: {stock_data}")
                    continue
                validated_data.append(
                    {
                        "code": code,
                        "name": stock_data.get("name"),
                        "market": stock_data.get("market"),
                        "sector": stock_data.get("sector"),
                        "industry": stock_data.get("industry"),
                    }
                )

            # AdvancedBulkOperationsを使用してupsert（挿入または更新）
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="update",  # 重複時は更新
                chunk_size=batch_size,
                unique_columns=["code"],
            )

            logger.info(f"銘柄一括更新完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括更新エラー: {e}")
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "errors": len(stocks_data),
            }

    def bulk_upsert_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        銘柄の一括upsert（AdvancedBulkOperations使用・存在すれば更新、なければ追加）

        Args:
            stocks_data: 銘柄データのリスト
            batch_size: バッチサイズ

        Returns:
            実行結果の統計情報
        """
        if not stocks_data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        try:
            # データの検証と準備
            validated_data = []
            for stock_data in stocks_data:
                code = stock_data.get("code")
                if not code:
                    logger.warning(f"銘柄コードが無効です: {stock_data}")
                    continue
                validated_data.append(
                    {
                        "code": code,
                        "name": stock_data.get("name"),
                        "market": stock_data.get("market"),
                        "sector": stock_data.get("sector"),
                        "industry": stock_data.get("industry"),
                    }
                )

            # AdvancedBulkOperationsを使用してupsert
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="update",  # 重複時は更新
                chunk_size=batch_size,
                unique_columns=["code"],
            )

            logger.info(f"銘柄一括upsert完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括upsertエラー: {e}")
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "errors": len(stocks_data),
            }

    def bulk_delete_stocks(self, codes: List[str], batch_size: int = 1000) -> Dict[str, int]:
        """
        銘柄の一括削除

        Args:
            codes: 削除する銘柄コードのリスト
            batch_size: バッチサイズ

        Returns:
            削除結果の統計情報
        """
        if not codes:
            return {"deleted": 0, "failed": 0, "not_found": 0, "total": 0}

        logger.info(f"銘柄一括削除開始: {len(codes)}件")

        deleted_count = 0
        failed_count = 0
        not_found_count = 0

        try:
            # バッチごとに処理
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i : i + batch_size]

                with self.db_manager.session_scope() as session:
                    try:
                        # 削除対象の銘柄を取得
                        stocks_to_delete = (
                            session.query(Stock)
                            .filter(Stock.code.in_(batch_codes))
                            .all()
                        )

                        found_codes = {stock.code for stock in stocks_to_delete}
                        not_found_codes = set(batch_codes) - found_codes

                        # 削除実行
                        for stock in stocks_to_delete:
                            session.delete(stock)
                            deleted_count += 1

                        not_found_count += len(not_found_codes)

                        logger.debug(
                            f"バッチ削除完了: 削除={len(stocks_to_delete)}, "
                            f"見つからず={len(not_found_codes)}"
                        )

                    except Exception as e:
                        logger.error(f"バッチ削除エラー: {e}")
                        failed_count += len(batch_codes)

        except Exception as e:
            logger.error(f"銘柄一括削除エラー: {e}")
            failed_count = len(codes)

        result = {
            "deleted": deleted_count,
            "failed": failed_count,
            "not_found": not_found_count,
            "total": len(codes),
        }

        logger.info(f"銘柄一括削除完了: {result}")
        return result

    def bulk_validate_stocks(self, stocks_data: List[dict]) -> Dict[str, List[dict]]:
        """
        銘柄データの一括バリデーション

        Args:
            stocks_data: バリデーション対象の銘柄データリスト

        Returns:
            バリデーション結果（valid/invalid データの分離）
        """
        valid_data = []
        invalid_data = []

        for stock_data in stocks_data:
            validation_errors = self._validate_single_stock(stock_data)

            if validation_errors:
                invalid_data.append({
                    **stock_data,
                    "validation_errors": validation_errors
                })
            else:
                valid_data.append(stock_data)

        logger.info(
            f"バリデーション完了: 有効={len(valid_data)}, 無効={len(invalid_data)}"
        )

        return {
            "valid": valid_data,
            "invalid": invalid_data,
            "summary": {
                "total": len(stocks_data),
                "valid_count": len(valid_data),
                "invalid_count": len(invalid_data),
            }
        }

    def _validate_single_stock(self, stock_data: dict) -> List[str]:
        """単一銘柄データのバリデーション"""
        validation_errors = []

        # 必須フィールドチェック
        if not stock_data.get("code"):
            validation_errors.append("code is required")

        if not stock_data.get("name"):
            validation_errors.append("name is required")

        # データ型チェック
        code = stock_data.get("code")
        if code and not isinstance(code, str):
            validation_errors.append("code must be string")

        # コード形式チェック（数値のみ）
        if code and not code.isdigit():
            validation_errors.append("code must be numeric")

        # 名前の長さチェック
        name = stock_data.get("name")
        if name and len(name) > 100:
            validation_errors.append("name too long (max 100 chars)")

        return validation_errors
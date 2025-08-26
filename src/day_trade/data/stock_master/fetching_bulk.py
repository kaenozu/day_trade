"""
銘柄マスタの一括データ取得・更新機能モジュール

このモジュールは複数銘柄の情報を一括で取得・更新する機能を提供します。
セクター情報の一括更新機能も含みます。
"""

import time
from typing import Dict, List

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockDataFetcherBulk:
    """銘柄データ一括取得・更新クラス"""

    def __init__(self, db_manager, stock_fetcher, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            stock_fetcher: StockFetcherインスタンス
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.stock_fetcher = stock_fetcher
        self.config = config or {}

    def bulk_fetch_and_update_companies(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, int]:
        """
        複数銘柄の企業情報を一括取得・更新（StockFetcher経由）

        Args:
            codes: 銘柄コードのリスト
            batch_size: バッチサイズ（APIレートリミット対応）
            delay: バッチ間の遅延（秒）

        Returns:
            更新結果の統計情報
        """
        if not codes:
            return {"success": 0, "failed": 0, "skipped": 0, "total": 0}

        logger.info(f"企業情報一括取得開始: {len(codes)}銘柄")

        success_count = 0
        failed_count = 0
        skipped_count = 0
        start_time = time.time()

        try:
            # 新しい一括取得機能を使用
            bulk_company_data = self.stock_fetcher.bulk_get_company_info(
                codes=codes, batch_size=batch_size, delay=delay
            )

            # 取得結果を処理
            success_count, failed_count, skipped_count = self._process_bulk_company_data(
                bulk_company_data
            )

        except Exception as e:
            logger.error(f"一括企業情報取得エラー: {e}")
            # フォールバック処理
            success_count, failed_count, skipped_count = self._fallback_individual_processing(
                codes, batch_size, delay
            )

        return self._create_bulk_result(codes, success_count, failed_count, skipped_count, start_time)

    def update_sector_information_bulk(
        self, codes: List[str], batch_size: int = 20, delay: float = 0.1
    ) -> Dict[str, int]:
        """
        複数銘柄のセクター情報を一括更新（Issue #133対応）

        Args:
            codes: 銘柄コードのリスト
            batch_size: バッチサイズ（APIレートリミット対応）
            delay: バッチ間の遅延（秒）

        Returns:
            更新結果の統計情報
        """
        if not codes:
            return {"updated": 0, "failed": 0, "skipped": 0, "total": 0}

        logger.info(f"セクター情報一括更新開始: {len(codes)}銘柄")

        updated_count = 0
        failed_count = 0
        skipped_count = 0

        # バッチ処理でAPIレートリミットを回避
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            logger.info(
                f"セクター情報バッチ処理: {i // batch_size + 1}/{(len(codes) + batch_size - 1) // batch_size}"
            )

            with self.db_manager.session_scope() as session:
                for code in batch_codes:
                    try:
                        # 現在の銘柄情報を取得
                        stock = session.query(Stock).filter(Stock.code == code).first()
                        if not stock:
                            logger.warning(f"銘柄が見つかりません: {code}")
                            skipped_count += 1
                            continue

                        # セクター情報が既に存在する場合はスキップ（オプション）
                        if stock.sector and stock.industry:
                            logger.debug(
                                f"セクター情報が既に存在: {code} - {stock.sector}"
                            )
                            skipped_count += 1
                            continue

                        # StockFetcherから企業情報を取得
                        company_info = self.stock_fetcher.get_company_info(code)
                        if not company_info:
                            logger.warning(f"企業情報を取得できません: {code}")
                            failed_count += 1
                            continue

                        # セクター情報を更新
                        updated = False
                        if (
                            company_info.get("sector")
                            and company_info["sector"] != stock.sector
                        ):
                            stock.sector = company_info["sector"]
                            updated = True

                        if (
                            company_info.get("industry")
                            and company_info["industry"] != stock.industry
                        ):
                            stock.industry = company_info["industry"]
                            updated = True

                        if updated:
                            session.flush()
                            logger.info(
                                f"セクター情報を更新: {code} - {stock.sector}/{stock.industry}"
                            )
                            updated_count += 1
                        else:
                            skipped_count += 1

                    except Exception as e:
                        logger.error(f"セクター情報更新エラー ({code}): {e}")
                        failed_count += 1

            # バッチ間の遅延
            if i + batch_size < len(codes) and delay > 0:
                time.sleep(delay)

        result = {
            "updated": updated_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": len(codes),
        }

        logger.info(f"セクター情報一括更新完了: {result}")
        return result

    def auto_update_missing_sector_info(self, max_stocks: int = 100) -> Dict[str, int]:
        """
        セクター情報が空の銘柄を自動的に更新（ユーティリティ）

        Args:
            max_stocks: 一度に処理する銘柄の上限数

        Returns:
            更新結果の統計情報
        """
        logger.info(f"セクター情報の自動更新を開始: 上限{max_stocks}銘柄")

        # セクター情報が空の銘柄を取得
        from .search import StockSearcher

        searcher = StockSearcher(self.db_manager, self.config)
        codes_to_update = searcher.get_stocks_without_sector_info(limit=max_stocks)

        if not codes_to_update:
            logger.info("セクター情報の更新が必要な銘柄はありません")
            return {"updated": 0, "failed": 0, "skipped": 0, "total": 0}

        # 一括更新を実行
        return self.update_sector_information_bulk(
            codes_to_update,
            batch_size=self.config.get("fetch_batch_size", 20),
            delay=self.config.get("fetch_delay_seconds", 0.1),
        )

    def _process_bulk_company_data(self, bulk_company_data: Dict) -> tuple:
        """一括取得データの処理"""
        success_count = 0
        failed_count = 0 
        skipped_count = 0

        for code, company_info in bulk_company_data.items():
            try:
                if company_info:
                    with self.db_manager.session_scope() as session:
                        stock = session.query(Stock).filter(Stock.code == code).first()

                        if stock:
                            stock.name = company_info.get("name", stock.name)
                            stock.sector = company_info.get("sector", stock.sector)
                            stock.industry = company_info.get("industry", stock.industry)
                        else:
                            stock = Stock(
                                code=code,
                                name=company_info.get("name", ""),
                                market="東証プライム",
                                sector=company_info.get("sector", ""),
                                industry=company_info.get("industry", ""),
                            )
                            session.add(stock)

                        session.commit()
                        success_count += 1
                else:
                    skipped_count += 1
                    logger.warning(f"企業情報が取得できませんでした: {code}")

            except Exception as e:
                failed_count += 1
                logger.error(f"銘柄情報処理エラー ({code}): {e}")

        return success_count, failed_count, skipped_count

    def _fallback_individual_processing(self, codes: List[str], batch_size: int, delay: float) -> tuple:
        """フォールバック個別処理"""
        from .fetching_core import StockDataFetcherCore
        
        logger.warning("個別処理にフォールバック")
        success_count = 0
        failed_count = 0
        skipped_count = 0

        # コア機能のインスタンスを作成
        core_fetcher = StockDataFetcherCore(self.db_manager, self.stock_fetcher, self.config)

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            
            for code in batch_codes:
                try:
                    stock_info = core_fetcher.fetch_and_update_stock_info_as_dict(code)
                    if stock_info:
                        success_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"銘柄情報取得失敗 ({code}): {e}")
                    failed_count += 1

            if i + batch_size < len(codes) and delay > 0:
                time.sleep(delay)

        return success_count, failed_count, skipped_count

    def _create_bulk_result(self, codes: List[str], success_count: int, failed_count: int, 
                          skipped_count: int, start_time: float) -> Dict[str, int]:
        """一括処理結果の作成"""
        total_elapsed = time.time() - start_time
        avg_time_per_stock = total_elapsed / len(codes) if codes else 0

        result = {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": len(codes),
            "elapsed_seconds": total_elapsed,
            "avg_time_per_stock": avg_time_per_stock,
        }

        logger.info(
            f"企業情報一括取得完了: 成功={success_count}, 失敗={failed_count}, "
            f"スキップ={skipped_count}, 合計={len(codes)} "
            f"({total_elapsed:.2f}秒, 平均{avg_time_per_stock:.3f}秒/銘柄)"
        )
        return result
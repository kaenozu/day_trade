"""
銘柄マスタのユーティリティ機能モジュール

このモジュールは銘柄データの検証、統計情報取得、分析機能等を提供します。
"""

from typing import Dict, List, Optional

from sqlalchemy import func

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockMasterUtils:
    """銘柄マスタユーティリティクラス"""

    def __init__(self, db_manager, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.config = config or {}

    def validate_stock_data(self, code: str, name: str = None) -> bool:
        """
        銘柄データのバリデーション

        Args:
            code: 証券コード
            name: 銘柄名

        Returns:
            バリデーション結果
        """
        if self.config.get("require_code", True) and not code:
            return False
        if self.config.get("require_name", True) and not name:
            return False
        if self.config.get("validate_code_format", True) and not code.isdigit():
            return False
        if name and len(name) > self.config.get("max_name_length", 100):
            return False
        return True

    def get_sector_distribution() -> Dict[str, int]:
        """
        セクター別銘柄数の分布を取得

        Returns:
            セクター名: 銘柄数 の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # セクター別の銘柄数を集計
                results = (
                    session.query(Stock.sector, func.count(Stock.id))
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.sector != "")
                    .group_by(Stock.sector)
                    .order_by(func.count(Stock.id).desc())
                    .all()
                )

                distribution = {sector: count for sector, count in results}

                # セクター情報が空の銘柄数を追加
                missing_count = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.sector.is_(None)) | (Stock.sector == ""))
                    .scalar()
                )

                if missing_count > 0:
                    distribution["(セクター情報なし)"] = missing_count

                return distribution

        except Exception as e:
            logger.error(f"セクター分布取得エラー: {e}")
            return {}

    def get_industry_distribution(self) -> Dict[str, int]:
        """
        業種別銘柄数の分布を取得

        Returns:
            業種名: 銘柄数 の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # 業種別の銘柄数を集計
                results = (
                    session.query(Stock.industry, func.count(Stock.id))
                    .filter(Stock.industry.isnot(None))
                    .filter(Stock.industry != "")
                    .group_by(Stock.industry)
                    .order_by(func.count(Stock.id).desc())
                    .all()
                )

                distribution = {industry: count for industry, count in results}

                # 業種情報が空の銘柄数を追加
                missing_count = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.industry.is_(None)) | (Stock.industry == ""))
                    .scalar()
                )

                if missing_count > 0:
                    distribution["(業種情報なし)"] = missing_count

                return distribution

        except Exception as e:
            logger.error(f"業種分布取得エラー: {e}")
            return {}

    def get_market_distribution(self) -> Dict[str, int]:
        """
        市場区分別銘柄数の分布を取得

        Returns:
            市場区分: 銘柄数 の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # 市場区分別の銘柄数を集計
                results = (
                    session.query(Stock.market, func.count(Stock.id))
                    .filter(Stock.market.isnot(None))
                    .filter(Stock.market != "")
                    .group_by(Stock.market)
                    .order_by(func.count(Stock.id).desc())
                    .all()
                )

                distribution = {market: count for market, count in results}

                # 市場情報が空の銘柄数を追加
                missing_count = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.market.is_(None)) | (Stock.market == ""))
                    .scalar()
                )

                if missing_count > 0:
                    distribution["(市場情報なし)"] = missing_count

                return distribution

        except Exception as e:
            logger.error(f"市場分布取得エラー: {e}")
            return {}

    def get_master_statistics(self) -> Dict[str, int]:
        """
        銘柄マスタの統計情報を取得

        Returns:
            統計情報の辞書
        """
        stats = {}

        try:
            with self.db_manager.session_scope() as session:
                # 総銘柄数
                stats["total_stocks"] = session.query(Stock).count()

                # セクター情報あり/なし
                stats["stocks_with_sector"] = (
                    session.query(Stock)
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.sector != "")
                    .count()
                )
                stats["stocks_without_sector"] = stats["total_stocks"] - stats["stocks_with_sector"]

                # 業種情報あり/なし
                stats["stocks_with_industry"] = (
                    session.query(Stock)
                    .filter(Stock.industry.isnot(None))
                    .filter(Stock.industry != "")
                    .count()
                )
                stats["stocks_without_industry"] = stats["total_stocks"] - stats["stocks_with_industry"]

                # 市場情報あり/なし
                stats["stocks_with_market"] = (
                    session.query(Stock)
                    .filter(Stock.market.isnot(None))
                    .filter(Stock.market != "")
                    .count()
                )
                stats["stocks_without_market"] = stats["total_stocks"] - stats["stocks_with_market"]

                # セクター数
                stats["unique_sectors"] = (
                    session.query(Stock.sector)
                    .distinct()
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.sector != "")
                    .count()
                )

                # 業種数
                stats["unique_industries"] = (
                    session.query(Stock.industry)
                    .distinct()
                    .filter(Stock.industry.isnot(None))
                    .filter(Stock.industry != "")
                    .count()
                )

                # 市場区分数
                stats["unique_markets"] = (
                    session.query(Stock.market)
                    .distinct()
                    .filter(Stock.market.isnot(None))
                    .filter(Stock.market != "")
                    .count()
                )

                logger.info(f"銘柄マスタ統計情報を取得: {stats}")

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")

        return stats

    def apply_stock_limit(self, requested_limit: int) -> int:
        """
        設定に基づいて銘柄数制限を適用

        Args:
            requested_limit: 要求された制限数

        Returns:
            適用する実際の制限数
        """
        # 設定から最大銘柄数制限を取得
        limits_config = self.config.get("limits", {})
        max_stock_count = limits_config.get("max_stock_count")
        max_search_limit = limits_config.get("max_search_limit", 1000)

        # テストモード判定（環境変数またはconfig）
        import os

        is_test_mode = (
            os.getenv("TEST_MODE", "false").lower() == "true"
            or os.getenv("PYTEST_CURRENT_TEST") is not None
        )

        if is_test_mode:
            test_limit = limits_config.get("test_mode_limit", 100)
            effective_limit = min(requested_limit, test_limit)
            logger.info(
                f"テストモード: 銘柄数制限を適用 {requested_limit} -> {effective_limit}"
            )
            return effective_limit

        # 通常モードでの制限適用
        effective_limit = min(requested_limit, max_search_limit)

        if max_stock_count is not None:
            effective_limit = min(effective_limit, max_stock_count)
            if effective_limit < requested_limit:
                logger.info(
                    f"銘柄数制限を適用: {requested_limit} -> {effective_limit} "
                    f"(max_stock_count: {max_stock_count})"
                )

        return effective_limit

    def set_stock_limit(self, limit: Optional[int]) -> None:
        """
        動的に銘柄数制限を設定

        Args:
            limit: 設定する制限数（Noneで制限解除）
        """
        if "limits" not in self.config:
            self.config["limits"] = {}

        self.config["limits"]["max_stock_count"] = limit

        if limit is None:
            logger.info("銘柄数制限を解除しました")
        else:
            logger.info(f"銘柄数制限を設定しました: {limit}")

    def get_stock_limit(self) -> Optional[int]:
        """
        現在の銘柄数制限を取得

        Returns:
            現在の制限数（Noneは制限なし）
        """
        return self.config.get("limits", {}).get("max_stock_count")

    def apply_session_management(self, stock: Stock, session, detached: bool):
        """
        セッション管理の適用

        Args:
            stock: 対象のStockオブジェクト
            session: データベースセッション
            detached: セッションから切り離すかどうか
        """
        # eager loadingで属性を事前読み込み
        if self.config.get("use_eager_loading", True):
            _ = (
                stock.id,
                stock.code,
                stock.name,
                stock.market,
                stock.sector,
                stock.industry,
            )

        # detachedが指定されている場合はセッションから切り離す
        if detached and self.config.get("auto_expunge", False):
            session.expunge(stock)

    def analyze_data_quality(self) -> Dict[str, float]:
        """
        データ品質分析

        Returns:
            データ品質指標の辞書
        """
        quality_metrics = {}

        try:
            stats = self.get_master_statistics()

            if stats.get("total_stocks", 0) > 0:
                total = stats["total_stocks"]

                # 完全性指標（セクター情報の充足率）
                quality_metrics["sector_completeness"] = (
                    stats.get("stocks_with_sector", 0) / total
                ) * 100

                # 完全性指標（業種情報の充足率）
                quality_metrics["industry_completeness"] = (
                    stats.get("stocks_with_industry", 0) / total
                ) * 100

                # 完全性指標（市場情報の充足率）
                quality_metrics["market_completeness"] = (
                    stats.get("stocks_with_market", 0) / total
                ) * 100

                # 全体的なデータ完全性スコア
                quality_metrics["overall_completeness"] = (
                    quality_metrics["sector_completeness"] +
                    quality_metrics["industry_completeness"] +
                    quality_metrics["market_completeness"]
                ) / 3

                logger.info(f"データ品質分析結果: {quality_metrics}")

        except Exception as e:
            logger.error(f"データ品質分析エラー: {e}")

        return quality_metrics

    def get_missing_data_report(self) -> Dict[str, List[str]]:
        """
        欠損データのレポートを生成

        Returns:
            欠損データの詳細レポート
        """
        report = {
            "missing_sector": [],
            "missing_industry": [],
            "missing_market": [],
            "missing_all": []
        }

        try:
            with self.db_manager.session_scope() as session:
                # セクター情報が空の銘柄
                stocks_missing_sector = (
                    session.query(Stock.code)
                    .filter((Stock.sector.is_(None)) | (Stock.sector == ""))
                    .limit(100)  # 上限を設定
                    .all()
                )
                report["missing_sector"] = [s.code for s in stocks_missing_sector]

                # 業種情報が空の銘柄
                stocks_missing_industry = (
                    session.query(Stock.code)
                    .filter((Stock.industry.is_(None)) | (Stock.industry == ""))
                    .limit(100)
                    .all()
                )
                report["missing_industry"] = [s.code for s in stocks_missing_industry]

                # 市場情報が空の銘柄
                stocks_missing_market = (
                    session.query(Stock.code)
                    .filter((Stock.market.is_(None)) | (Stock.market == ""))
                    .limit(100)
                    .all()
                )
                report["missing_market"] = [s.code for s in stocks_missing_market]

                # 全ての情報が空の銘柄
                stocks_missing_all = (
                    session.query(Stock.code)
                    .filter(
                        ((Stock.sector.is_(None)) | (Stock.sector == "")) &
                        ((Stock.industry.is_(None)) | (Stock.industry == "")) &
                        ((Stock.market.is_(None)) | (Stock.market == ""))
                    )
                    .limit(100)
                    .all()
                )
                report["missing_all"] = [s.code for s in stocks_missing_all]

                logger.info(f"欠損データレポート生成完了: {len(report)}")

        except Exception as e:
            logger.error(f"欠損データレポート生成エラー: {e}")

        return report


# Issue #133: セクター情報永続化のユーティリティ関数
def update_all_sector_information(
    stock_master_manager, batch_size: int = 20, max_stocks: int = 1000
) -> Dict[str, int]:
    """
    全銘柄のセクター情報を更新するユーティリティ関数

    Args:
        stock_master_manager: StockMasterManagerインスタンス
        batch_size: バッチサイズ
        max_stocks: 一度に処理する銘柄の上限数

    Returns:
        更新結果の統計情報
    """
    return stock_master_manager.auto_update_missing_sector_info(max_stocks=max_stocks)


def get_sector_distribution(db_manager) -> Dict[str, int]:
    """
    セクター別銘柄数の分布を取得（スタンドアロン関数）

    Args:
        db_manager: データベースマネージャー

    Returns:
        セクター名: 銘柄数 の辞書
    """
    utils = StockMasterUtils(db_manager)
    return utils.get_sector_distribution()


def analyze_stock_code_patterns(db_manager) -> Dict[str, int]:
    """証券コードのパターン分析"""
    patterns = {f"{i}000-{i}999": 0 for i in range(1, 10)}
    patterns["others"] = 0

    try:
        with db_manager.session_scope() as session:
            stocks = session.query(Stock.code).all()
            
            for stock in stocks:
                try:
                    code_num = int(stock.code)
                    category = f"{code_num // 1000}000-{code_num // 1000}999"
                    if category in patterns:
                        patterns[category] += 1
                    else:
                        patterns["others"] += 1
                except ValueError:
                    patterns["others"] += 1
    except Exception as e:
        logger.error(f"コードパターン分析エラー: {e}")

    return patterns
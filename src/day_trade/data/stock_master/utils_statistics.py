"""
銘柄マスタの統計・分析機能モジュール

このモジュールは銘柄データの統計情報取得、分析機能を提供します。
"""

from typing import Dict

from sqlalchemy import func

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockStatisticsUtils:
    """銘柄データ統計・分析クラス"""

    def __init__(self, db_manager, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.config = config or {}

    def get_sector_distribution(self) -> Dict[str, int]:
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
        市場別銘柄数の分布を取得

        Returns:
            市場名: 銘柄数 の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # 市場別の銘柄数を集計
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
        マスタの総合統計情報を取得

        Returns:
            統計情報の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # 基本統計
                total_stocks = session.query(func.count(Stock.id)).scalar()
                
                # 完全な情報を持つ銘柄数
                complete_stocks = (
                    session.query(func.count(Stock.id))
                    .filter(Stock.code.isnot(None))
                    .filter(Stock.name.isnot(None))
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.industry.isnot(None))
                    .filter(Stock.market.isnot(None))
                    .filter(Stock.code != "")
                    .filter(Stock.name != "")
                    .filter(Stock.sector != "")
                    .filter(Stock.industry != "")
                    .filter(Stock.market != "")
                    .scalar()
                )

                # 各フィールドの欠損数
                missing_name = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.name.is_(None)) | (Stock.name == ""))
                    .scalar()
                )

                missing_sector = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.sector.is_(None)) | (Stock.sector == ""))
                    .scalar()
                )

                missing_industry = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.industry.is_(None)) | (Stock.industry == ""))
                    .scalar()
                )

                missing_market = (
                    session.query(func.count(Stock.id))
                    .filter((Stock.market.is_(None)) | (Stock.market == ""))
                    .scalar()
                )

                # ユニークな値の数
                unique_sectors = (
                    session.query(Stock.sector)
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.sector != "")
                    .distinct()
                    .count()
                )

                unique_industries = (
                    session.query(Stock.industry)
                    .filter(Stock.industry.isnot(None))
                    .filter(Stock.industry != "")
                    .distinct()
                    .count()
                )

                unique_markets = (
                    session.query(Stock.market)
                    .filter(Stock.market.isnot(None))
                    .filter(Stock.market != "")
                    .distinct()
                    .count()
                )

                return {
                    "total_stocks": total_stocks,
                    "complete_stocks": complete_stocks,
                    "completion_rate": round((complete_stocks / total_stocks * 100), 2) if total_stocks > 0 else 0,
                    "missing_name": missing_name,
                    "missing_sector": missing_sector,
                    "missing_industry": missing_industry,
                    "missing_market": missing_market,
                    "unique_sectors": unique_sectors,
                    "unique_industries": unique_industries,
                    "unique_markets": unique_markets,
                }

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {}

    def get_code_range_distribution(self) -> Dict[str, int]:
        """
        コード範囲別の銘柄分布を取得

        Returns:
            コード範囲: 銘柄数 の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # コード範囲別の集計
                ranges = {
                    "1000-1999": (1000, 1999),
                    "2000-2999": (2000, 2999),
                    "3000-3999": (3000, 3999),
                    "4000-4999": (4000, 4999),
                    "5000-5999": (5000, 5999),
                    "6000-6999": (6000, 6999),
                    "7000-7999": (7000, 7999),
                    "8000-8999": (8000, 8999),
                    "9000-9999": (9000, 9999),
                }

                distribution = {}
                
                for range_name, (start, end) in ranges.items():
                    count = (
                        session.query(func.count(Stock.id))
                        .filter(Stock.code.between(str(start), str(end)))
                        .scalar()
                    )
                    if count > 0:
                        distribution[range_name] = count

                return distribution

        except Exception as e:
            logger.error(f"コード範囲分布取得エラー: {e}")
            return {}

    def get_top_sectors(self, limit: int = 10) -> Dict[str, int]:
        """
        上位セクターの取得

        Args:
            limit: 取得する上位数

        Returns:
            上位セクターと銘柄数
        """
        try:
            with self.db_manager.session_scope() as session:
                results = (
                    session.query(Stock.sector, func.count(Stock.id))
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.sector != "")
                    .group_by(Stock.sector)
                    .order_by(func.count(Stock.id).desc())
                    .limit(limit)
                    .all()
                )

                return {sector: count for sector, count in results}

        except Exception as e:
            logger.error(f"上位セクター取得エラー: {e}")
            return {}

    def get_top_industries(self, limit: int = 10) -> Dict[str, int]:
        """
        上位業種の取得

        Args:
            limit: 取得する上位数

        Returns:
            上位業種と銘柄数
        """
        try:
            with self.db_manager.session_scope() as session:
                results = (
                    session.query(Stock.industry, func.count(Stock.id))
                    .filter(Stock.industry.isnot(None))
                    .filter(Stock.industry != "")
                    .group_by(Stock.industry)
                    .order_by(func.count(Stock.id).desc())
                    .limit(limit)
                    .all()
                )

                return {industry: count for industry, count in results}

        except Exception as e:
            logger.error(f"上位業種取得エラー: {e}")
            return {}
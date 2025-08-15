#!/usr/bin/env python3
"""
動的銘柄選択システム
デイトレード向けの適切な銘柄を動的に選択する
"""

import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from .topix500_master import TOPIX500MasterManager
    from ..utils.logging_config import get_context_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

    def get_context_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_context_logger(__name__)


@dataclass
class SymbolSelectionCriteria:
    """銘柄選択基準"""
    min_market_cap: Optional[float] = 100_000_000_000  # 1000億円以上
    max_market_cap: Optional[float] = None
    min_liquidity_score: Optional[float] = 0.5  # 流動性スコア
    excluded_sectors: List[str] = None  # 除外セクター
    preferred_sectors: List[str] = None  # 優先セクター
    max_symbols: int = 50  # 最大銘柄数

    def __post_init__(self):
        if self.excluded_sectors is None:
            self.excluded_sectors = []
        if self.preferred_sectors is None:
            self.preferred_sectors = []


class DynamicSymbolSelector:
    """動的銘柄選択システム"""

    def __init__(self, db_path: str = "data/topix500_master.db"):
        """
        初期化

        Args:
            db_path: TOPIX500データベースパス
        """
        self.topix_master = TOPIX500MasterManager(db_path)
        logger.info("動的銘柄選択システム初期化完了")

    def select_daytrading_symbols(self, criteria: SymbolSelectionCriteria = None) -> List[str]:
        """
        デイトレード向け銘柄選択

        Args:
            criteria: 選択基準

        Returns:
            選択された銘柄コードリスト
        """
        if criteria is None:
            criteria = SymbolSelectionCriteria()

        try:
            with sqlite3.connect(self.topix_master.db_path) as conn:
                cursor = conn.cursor()

                # ベースクエリ
                query = """
                    SELECT code, name, market_cap, topix_weight, sector_code, sector_name
                    FROM topix500_master
                    WHERE is_active = TRUE
                """
                params = []

                # 時価総額フィルター
                if criteria.min_market_cap:
                    query += " AND market_cap >= ?"
                    params.append(criteria.min_market_cap)

                if criteria.max_market_cap:
                    query += " AND market_cap <= ?"
                    params.append(criteria.max_market_cap)

                # セクターフィルター
                if criteria.excluded_sectors:
                    placeholders = ",".join(["?" for _ in criteria.excluded_sectors])
                    query += f" AND sector_code NOT IN ({placeholders})"
                    params.extend(criteria.excluded_sectors)

                if criteria.preferred_sectors:
                    placeholders = ",".join(["?" for _ in criteria.preferred_sectors])
                    query += f" AND sector_code IN ({placeholders})"
                    params.extend(criteria.preferred_sectors)

                # 並び順：時価総額 × TOPIX組入比重
                query += " ORDER BY (market_cap * COALESCE(topix_weight, 0.001)) DESC"

                # 件数制限
                if criteria.max_symbols:
                    query += f" LIMIT {criteria.max_symbols}"

                cursor.execute(query, params)
                results = cursor.fetchall()

                symbols = [row[0] for row in results]

                logger.info(f"デイトレード向け銘柄選択完了: {len(symbols)}銘柄")
                if symbols:
                    logger.info(f"選択銘柄例: {', '.join(symbols[:5])}")

                return symbols

        except Exception as e:
            logger.error(f"銘柄選択エラー: {e}")
            raise RuntimeError(f"動的銘柄選択に失敗しました: {e}") from e

    def get_liquid_symbols(self, limit: int = 20) -> List[str]:
        """
        高流動性銘柄取得（デイトレード特化）

        Args:
            limit: 取得件数

        Returns:
            高流動性銘柄リスト
        """
        criteria = SymbolSelectionCriteria(
            min_market_cap=300_000_000_000,  # 3000億円以上（高流動性）
            max_symbols=limit,
            excluded_sectors=["REIT", "ETF"]  # デイトレード向きでないセクターを除外
        )

        return self.select_daytrading_symbols(criteria)

    def get_volatile_symbols(self, limit: int = 10) -> List[str]:
        """
        高ボラティリティ銘柄取得（デイトレード機会重視）

        Args:
            limit: 取得件数

        Returns:
            高ボラティリティ銘柄リスト
        """
        criteria = SymbolSelectionCriteria(
            min_market_cap=50_000_000_000,  # 500億円以上
            max_market_cap=1_000_000_000_000,  # 1兆円以下（大型すぎる銘柄は除外）
            preferred_sectors=["情報・通信業", "サービス業", "電気機器"],  # ボラが高いセクター
            max_symbols=limit
        )

        return self.select_daytrading_symbols(criteria)

    def get_balanced_portfolio(self, limit: int = 15) -> List[str]:
        """
        バランス型ポートフォリオ銘柄（安定性重視）

        Args:
            limit: 取得件数

        Returns:
            バランス型銘柄リスト
        """
        criteria = SymbolSelectionCriteria(
            min_market_cap=200_000_000_000,  # 2000億円以上
            max_symbols=limit,
            excluded_sectors=["REIT", "ETF", "その他金融業"]
        )

        return self.select_daytrading_symbols(criteria)


    def get_sector_diversified_symbols(self, limit: int = 20) -> List[str]:
        """
        セクター分散された銘柄選択

        Args:
            limit: 取得件数

        Returns:
            セクター分散銘柄リスト
        """
        try:
            with sqlite3.connect(self.topix_master.db_path) as conn:
                cursor = conn.cursor()

                # セクター別上位銘柄を取得
                cursor.execute("""
                    WITH ranked_symbols AS (
                        SELECT
                            code,
                            sector_code,
                            ROW_NUMBER() OVER (
                                PARTITION BY sector_code
                                ORDER BY (market_cap * COALESCE(topix_weight, 0.001)) DESC
                            ) as rn
                        FROM topix500_master
                        WHERE is_active = TRUE
                          AND market_cap >= 100000000000
                    )
                    SELECT code
                    FROM ranked_symbols
                    WHERE rn <= 2  -- 各セクターから上位2銘柄
                    ORDER BY sector_code, rn
                    LIMIT ?
                """, (limit,))

                symbols = [row[0] for row in cursor.fetchall()]
                logger.info(f"セクター分散銘柄選択: {len(symbols)}銘柄")

                if not symbols:
                    raise RuntimeError("セクター分散銘柄が見つかりません")
                return symbols

        except Exception as e:
            logger.error(f"セクター分散銘柄選択エラー: {e}")
            raise RuntimeError(f"セクター分散銘柄選択に失敗しました: {e}") from e


# テスト用
if __name__ == "__main__":
    selector = DynamicSymbolSelector()

    print("=== 高流動性銘柄（デイトレード向け）===")
    liquid_symbols = selector.get_liquid_symbols(10)
    for symbol in liquid_symbols:
        print(f"  {symbol}")

    print("\n=== 高ボラティリティ銘柄 ===")
    volatile_symbols = selector.get_volatile_symbols(5)
    for symbol in volatile_symbols:
        print(f"  {symbol}")

    print("\n=== セクター分散銘柄 ===")
    diversified_symbols = selector.get_sector_diversified_symbols(15)
    for symbol in diversified_symbols:
        print(f"  {symbol}")
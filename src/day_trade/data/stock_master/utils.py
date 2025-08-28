"""
銘柄マスタのユーティリティ機能モジュール（統合版）

このモジュールは銘柄データの検証、統計情報取得、分析機能等を統合して提供します。
"""

from typing import Dict, List, Optional

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger
from .utils_management import StockManagementUtils
from .utils_statistics import StockStatisticsUtils
from .utils_validation import StockValidationUtils

logger = get_context_logger(__name__)


class StockMasterUtils:
    """銘柄マスタユーティリティクラス（統合版）"""

    def __init__(self, db_manager, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.config = config or {}
        
        # 各機能モジュールを初期化
        self.validation = StockValidationUtils(db_manager, config)
        self.statistics = StockStatisticsUtils(db_manager, config)
        self.management = StockManagementUtils(db_manager, config)

    # ==========================================================================
    # バリデーション機能（StockValidationUtilsに委譲）
    # ==========================================================================

    def validate_stock_data(self, code: str, name: str = None) -> bool:
        """
        銘柄データのバリデーション

        Args:
            code: 証券コード
            name: 銘柄名

        Returns:
            バリデーション結果
        """
        return self.validation.validate_stock_data(code, name)

    def analyze_data_quality(self) -> Dict[str, float]:
        """
        データ品質の分析

        Returns:
            品質指標の辞書
        """
        return self.validation.analyze_data_quality()

    def get_missing_data_report(self) -> Dict[str, List[str]]:
        """
        欠損データのレポートを生成

        Returns:
            欠損項目別の銘柄コードリスト
        """
        return self.validation.get_missing_data_report()

    # ==========================================================================
    # 統計機能（StockStatisticsUtilsに委譲）
    # ==========================================================================

    def get_sector_distribution(self) -> Dict[str, int]:
        """
        セクター別銘柄数の分布を取得

        Returns:
            セクター名: 銘柄数 の辞書
        """
        return self.statistics.get_sector_distribution()

    def get_industry_distribution(self) -> Dict[str, int]:
        """
        業種別銘柄数の分布を取得

        Returns:
            業種名: 銘柄数 の辞書
        """
        return self.statistics.get_industry_distribution()

    def get_market_distribution(self) -> Dict[str, int]:
        """
        市場別銘柄数の分布を取得

        Returns:
            市場名: 銘柄数 の辞書
        """
        return self.statistics.get_market_distribution()

    def get_master_statistics(self) -> Dict[str, int]:
        """
        マスタの総合統計情報を取得

        Returns:
            統計情報の辞書
        """
        return self.statistics.get_master_statistics()

    # ==========================================================================
    # 管理機能（StockManagementUtilsに委譲）
    # ==========================================================================

    def apply_stock_limit(self, requested_limit: int) -> int:
        """
        設定に基づいて銘柄数制限を適用

        Args:
            requested_limit: 要求された制限数

        Returns:
            適用する実際の制限数
        """
        return self.management.apply_stock_limit(requested_limit)

    def set_stock_limit(self, limit: Optional[int]) -> None:
        """
        動的に銘柄数制限を設定

        Args:
            limit: 設定する制限数（Noneで制限解除）
        """
        return self.management.set_stock_limit(limit)

    def get_stock_limit(self) -> Optional[int]:
        """
        現在の銘柄数制限を取得

        Returns:
            現在の制限数（Noneは制限なし）
        """
        return self.management.get_stock_limit()

    def apply_session_management(self, stock: Stock, session, detached: bool) -> None:
        """
        セッション管理の適用

        Args:
            stock: 対象のStockオブジェクト
            session: データベースセッション
            detached: セッションから切り離すかどうか
        """
        return self.management.apply_session_management(stock, session, detached)


# ==========================================================================
# パッケージレベルのユーティリティ関数（後方互換性のため）
# ==========================================================================

def update_all_sector_information(
    batch_size: int = 20, max_stocks: int = 1000
) -> Dict[str, int]:
    """
    全銘柄のセクター情報を更新するユーティリティ関数

    Args:
        batch_size: バッチサイズ
        max_stocks: 一度に処理する銘柄の上限数

    Returns:
        更新結果の統計情報
    """
    from . import stock_master
    return stock_master.auto_update_missing_sector_info(max_stocks=max_stocks)


def get_sector_distribution(db_manager) -> Dict[str, int]:
    """
    セクター別銘柄数の分布を取得（パッケージレベル関数）

    Args:
        db_manager: データベースマネージャー

    Returns:
        セクター名: 銘柄数 の辞書
    """
    utils = StockStatisticsUtils(db_manager)
    return utils.get_sector_distribution()


def analyze_stock_code_patterns(db_manager) -> Dict[str, int]:
    """
    証券コードのパターン分析

    Args:
        db_manager: データベースマネージャー

    Returns:
        コード範囲別の分布
    """
    utils = StockStatisticsUtils(db_manager)
    return utils.get_code_range_distribution()
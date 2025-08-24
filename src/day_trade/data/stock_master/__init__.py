"""
銘柄マスタパッケージ

このパッケージは元のstock_master.pyの機能を複数のモジュールに分割したものです。
バックワード互換性を提供し、既存のコードが変更なしで動作することを保証します。
"""

# メインマネージャークラスをインポート
from .manager import StockMasterManager, create_stock_master_manager

# 各機能モジュールをインポート
from .bulk_operations import StockBulkOperations
from .fetching import StockDataFetcher
from .operations import StockOperations
from .search import StockSearcher
from .utils import StockMasterUtils

# ユーティリティ関数をインポート
from .utils import (
    analyze_stock_code_patterns,
    get_sector_distribution,
    update_all_sector_information,
)

# 後方互換性のためのグローバルインスタンス
# 元のstock_master.pyで提供されていたグローバル変数を再現
stock_master = StockMasterManager()

# パブリックAPI
__all__ = [
    # メインクラス
    "StockMasterManager",
    "create_stock_master_manager",
    
    # 機能別クラス
    "StockOperations",
    "StockSearcher", 
    "StockDataFetcher",
    "StockBulkOperations",
    "StockMasterUtils",
    
    # ユーティリティ関数
    "update_all_sector_information",
    "get_sector_distribution",
    "analyze_stock_code_patterns",
    
    # グローバルインスタンス（後方互換性）
    "stock_master",
]

# バージョン情報
__version__ = "2.0.0"

# パッケージレベルでのドキュメント
def get_module_info():
    """
    パッケージの情報を取得
    
    Returns:
        パッケージの構成情報
    """
    return {
        "package_name": "stock_master",
        "version": __version__,
        "description": "銘柄マスタ管理システム（モジュール分割版）",
        "modules": {
            "manager": "メインマネージャークラス（統合インターフェース）",
            "operations": "基本CRUD操作（追加・更新・削除・取得）",
            "search": "検索機能（名前・セクター・業種・複合条件）",
            "fetching": "外部API連携によるデータ取得・更新",
            "bulk_operations": "大量データの一括処理",
            "utils": "ユーティリティ機能・統計情報・データ品質分析"
        },
        "compatibility": {
            "original_file": "stock_master.py (1487 lines)",
            "split_date": "2024",
            "backward_compatible": True,
            "breaking_changes": None
        }
    }
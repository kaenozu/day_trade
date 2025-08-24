#!/usr/bin/env python3
"""
マスターデータ管理（MDM）システム
統合されたマスターデータ管理・ガバナンス・品質管理システム

バックワード互換性のため、元のMasterDataManagerクラスを
統合されたコンポーネントから再構築して提供します。
"""

# メインマネージャーをエクスポート
from .manager import MasterDataManager, create_master_data_manager

# 型定義をエクスポート
from .types import (
    DataClassification,
    DataDomain,
    DataElement,
    DataGovernancePolicy,
    DataLineage,
    DataSteward,
    DataStewardshipRole,
    MasterDataEntity,
    MasterDataStatus,
)

# コンポーネントをエクスポート（必要に応じて直接アクセス可能）
from .catalog_dashboard import CatalogDashboard
from .database_manager import DatabaseManager
from .default_setup import DefaultSetup
from .governance_manager import GovernanceManager
from .integration_rules import (
    CompanyDataIntegrationRule,
    CurrencyDataIntegrationRule,
    DataIntegrationRule,
    StockDataIntegrationRule,
    get_default_integration_rules,
)
from .quality_assessor import QualityAssessor

# バージョン情報
__version__ = "2.0.0"
__author__ = "MDM System"

# すべてのエクスポート項目
__all__ = [
    # メインクラス
    "MasterDataManager",
    "create_master_data_manager",
    
    # 型定義
    "DataClassification",
    "DataDomain",
    "DataElement",
    "DataGovernancePolicy", 
    "DataLineage",
    "DataSteward",
    "DataStewardshipRole",
    "MasterDataEntity",
    "MasterDataStatus",
    
    # コンポーネント
    "CatalogDashboard",
    "DatabaseManager",
    "DefaultSetup",
    "GovernanceManager",
    "QualityAssessor",
    
    # データ統合ルール
    "DataIntegrationRule",
    "StockDataIntegrationRule",
    "CompanyDataIntegrationRule",
    "CurrencyDataIntegrationRule",
    "get_default_integration_rules",
]

# 元のmaster_data_manager.pyとの互換性を保つため
# 直接インポートされる可能性のあるクラスをモジュールレベルで利用可能に
MasterDataManager = MasterDataManager
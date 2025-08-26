#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム

このモジュールは、分割された各コンポーネントを統合し、
元のenterprise_master_data_management.pyとのバックワード互換性を提供します。
"""

# 分割されたモジュールから必要なクラス・関数・定数をインポート
from .enums import (
    ApprovalStatus,
    AuditAction,
    ChangeType,
    DataGovernanceLevel,
    EntityStatus,
    HierarchyType,
    MasterDataType,
    QualityScoreLevel,
    RiskLevel,
    ValidationResult,
    get_quality_level,
    get_risk_level_from_score,
    QUALITY_THRESHOLDS,
)

from .models import (
    ChangeHistory,
    DataChangeRequest,
    DataGovernancePolicy,
    MasterDataEntity,
    MasterDataHierarchy,
    QualityHistory,
    QualityMetric,
)

from .quality_manager import QualityManager
from .governance_manager import GovernanceManager
from .entity_manager import EntityManager
from .change_manager import ChangeManager
from .hierarchy import HierarchyManager
from .database_operations import DatabaseOperations
from .catalog_manager import CatalogManager
from .main import EnterpriseMasterDataManagement, create_enterprise_mdm_system

# バックワード互換性のためのエイリアス定義
# 元のファイルで使用されていた名前に対応

# メインクラスのバックワード互換性
EnterpriseMasterDataManagement = EnterpriseMasterDataManagement

# ファクトリー関数のバックワード互換性
create_enterprise_mdm_system = create_enterprise_mdm_system

# 列挙型のバックワード互換性（元のファイルからそのまま利用可能）
MasterDataType = MasterDataType
DataGovernanceLevel = DataGovernanceLevel
ChangeType = ChangeType
ApprovalStatus = ApprovalStatus

# データクラスのバックワード互換性
MasterDataEntity = MasterDataEntity
DataChangeRequest = DataChangeRequest
DataGovernancePolicy = DataGovernancePolicy
MasterDataHierarchy = MasterDataHierarchy

# 新しい機能の公開
__all__ = [
    # メインクラス・関数
    "EnterpriseMasterDataManagement",
    "create_enterprise_mdm_system",
    
    # 列挙型
    "MasterDataType",
    "DataGovernanceLevel", 
    "ChangeType",
    "ApprovalStatus",
    "QualityScoreLevel",
    "RiskLevel",
    "ValidationResult",
    "EntityStatus",
    "HierarchyType",
    "AuditAction",
    
    # データモデル
    "MasterDataEntity",
    "DataChangeRequest", 
    "DataGovernancePolicy",
    "MasterDataHierarchy",
    "QualityMetric",
    "ChangeHistory",
    "QualityHistory",
    
    # 管理クラス
    "QualityManager",
    "GovernanceManager",
    "EntityManager",
    "ChangeManager",
    "HierarchyManager",
    "DatabaseOperations",
    "CatalogManager",
    
    # ユーティリティ関数
    "get_quality_level",
    "get_risk_level_from_score",
    
    # 定数
    "QUALITY_THRESHOLDS",
]

# バージョン情報
__version__ = "2.0.0"
__author__ = "Enterprise MDM Team"
__description__ = """
エンタープライズマスターデータ管理（MDM）システム

Issue #420: データ管理とデータ品質保証メカニズムの強化

このパッケージは、企業レベルのマスターデータ管理戦略を実装します：
- データ統合・統一化
- ゴールデンレコード管理  
- データガバナンス・ポリシー
- 階層・分類管理
- データカタログ・メタデータ管理
- データ品質・整合性保証
- アクセス制御・セキュリティ
- 変更追跡・監査証跡

Version 2.0では、以下の改善が行われました：
- モジュール分割による保守性向上（各ファイル300行以内）
- 機能別の責任分離
- より詳細な品質管理機能
- 改善されたガバナンス管理
- 包括的なデータカタログ機能
- 強化されたデータリネージュ追跡
"""

# 利用例とドキュメント
__doc_examples__ = """
## 基本的な使用方法

```python
import asyncio
from day_trade.data.enterprise_master import (
    create_enterprise_mdm_system,
    MasterDataType,
    ChangeType
)

async def main():
    # MDMシステム初期化
    mdm = create_enterprise_mdm_system("my_mdm.db")
    
    # エンティティ登録
    entity_id = await mdm.register_master_data_entity(
        entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
        primary_key="7203",
        attributes={
            "symbol": "7203",
            "name": "トヨタ自動車",
            "isin": "JP3633400001",
            "market": "TSE"
        },
        source_system="market_data",
        created_by="data_admin"
    )
    
    # データ変更リクエスト
    request_id = await mdm.request_data_change(
        entity_id=entity_id,
        change_type=ChangeType.UPDATE,
        proposed_changes={"sector": "自動車・輸送機器"},
        business_justification="業界分類の詳細化",
        requested_by="business_analyst"
    )
    
    # データカタログ取得
    catalog = await mdm.get_data_catalog()
    print(f"エンティティタイプ数: {catalog['total_entity_types']}")

asyncio.run(main())
```

## 高度な機能

```python
# カスタムガバナンスポリシー
from day_trade.data.enterprise_master import DataGovernancePolicy, DataGovernanceLevel

policy = DataGovernancePolicy(
    policy_id="custom_policy",
    policy_name="カスタム金融商品ポリシー",
    entity_types=[MasterDataType.FINANCIAL_INSTRUMENTS],
    rules=[
        {"field": "symbol", "required": True, "unique": True},
        {"field": "name", "required": True, "min_length": 2},
        {"field": "isin", "pattern": "^[A-Z]{2}[A-Z0-9]{10}$"}
    ],
    governance_level=DataGovernanceLevel.STRICT,
    quality_threshold=95.0
)

# データ階層作成
hierarchy_id = await mdm.create_data_hierarchy(
    name="業界分類階層",
    entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
    root_entity_id="industry_root",
    level_definitions={1: "大分類", 2: "中分類", 3: "小分類"}
)

# 品質評価実行
quality_report = await mdm.run_quality_assessment(
    entity_type=MasterDataType.FINANCIAL_INSTRUMENTS
)
```
"""

# 移行ガイド
__migration_guide__ = """
## Version 1.x から 2.0 への移行ガイド

### 変更点
1. ファイル構造の変更：単一ファイルから複数モジュールに分割
2. より詳細な機能分離
3. 新しい品質管理機能の追加
4. 改善されたエラーハンドリング

### 互換性
- 既存のAPIは完全に互換性を保持
- 既存のデータベース構造は変更なし
- 設定ファイルの形式は変更なし

### 推奨される移行手順
1. インポート文の確認（大部分は変更不要）
2. 新しい品質管理機能の活用を検討
3. 改善されたガバナンス機能の利用を検討
4. 詳細なデータカタログ機能の活用

### 新機能の利用
- QualityManager: 詳細なデータ品質評価
- CatalogManager: 包括的なデータカタログ生成
- 改善されたリネージュ追跡機能
"""
#!/usr/bin/env python3
"""
マスターデータ管理（MDM）システム
Issue #420: データ管理とデータ品質保証メカニズムの強化

企業レベルのマスターデータ管理戦略:
- データ統合・統一
- データガバナンス
- 階層管理
- データカタログ
- データ系譜管理
- 品質保証
- アクセス制御
- データ監査
"""

import asyncio
import json
import logging
import sqlite3
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from ..utils.data_quality_manager import DataQualityLevel, DataQualityMetrics
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"mdm_key_{hash(str(args))}"

    class DataQualityLevel(Enum):
        EXCELLENT = "excellent"
        GOOD = "good"
        FAIR = "fair"
        POOR = "poor"
        CRITICAL = "critical"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataDomain(Enum):
    """データドメイン"""

    FINANCIAL = "financial"  # 金融データ
    MARKET = "market"  # 市場データ
    SECURITY = "security"  # 証券データ
    REFERENCE = "reference"  # 参照データ
    CUSTOMER = "customer"  # 顧客データ
    PRODUCT = "product"  # 商品データ
    TRANSACTION = "transaction"  # 取引データ
    REGULATORY = "regulatory"  # 規制データ


class DataStewardshipRole(Enum):
    """データスチュワードシップ役割"""

    DATA_OWNER = "data_owner"  # データ所有者
    DATA_STEWARD = "data_steward"  # データスチュワード
    DATA_CUSTODIAN = "data_custodian"  # データ管理者
    DATA_USER = "data_user"  # データ利用者
    QUALITY_ANALYST = "quality_analyst"  # 品質分析者


class MasterDataStatus(Enum):
    """マスターデータ状態"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DataClassification(Enum):
    """データ分類"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class DataElement:
    """データ要素定義"""

    element_id: str
    name: str
    description: str
    data_type: str
    domain: DataDomain
    classification: DataClassification = DataClassification.INTERNAL
    is_pii: bool = False  # 個人識別情報
    business_rules: List[str] = dataclasses_field(default_factory=list)
    validation_rules: List[str] = dataclasses_field(default_factory=list)
    source_systems: List[str] = dataclasses_field(default_factory=list)
    created_at: datetime = dataclasses_field(default_factory=datetime.utcnow)
    updated_at: datetime = dataclasses_field(default_factory=datetime.utcnow)
    created_by: str = "system"
    metadata: Dict[str, Any] = dataclasses_field(default_factory=dict)


@dataclass
class MasterDataEntity:
    """マスターデータエンティティ"""

    entity_id: str
    entity_type: str  # "stock", "company", "currency", etc.
    primary_key: str
    attributes: Dict[str, Any]
    domain: DataDomain
    status: MasterDataStatus = MasterDataStatus.ACTIVE
    version: int = 1
    source_system: str = "mdm"
    data_quality_score: float = 1.0
    last_validated: datetime = dataclasses_field(default_factory=datetime.utcnow)
    created_at: datetime = dataclasses_field(default_factory=datetime.utcnow)
    updated_at: datetime = dataclasses_field(default_factory=datetime.utcnow)
    steward: Optional[str] = None
    lineage: List[str] = dataclasses_field(default_factory=list)  # データ系譜
    relationships: Dict[str, List[str]] = dataclasses_field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses_field(default_factory=dict)


@dataclass
class DataSteward:
    """データスチュワード"""

    steward_id: str
    name: str
    email: str
    role: DataStewardshipRole
    domains: List[DataDomain]
    responsibilities: List[str] = dataclasses_field(default_factory=list)
    active: bool = True
    created_at: datetime = dataclasses_field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = dataclasses_field(default_factory=dict)


@dataclass
class DataGovernancePolicy:
    """データガバナンスポリシー"""

    policy_id: str
    name: str
    description: str
    domain: DataDomain
    policy_type: str  # "retention", "access", "quality", "security"
    rules: List[Dict[str, Any]]
    enforcement_level: str  # "mandatory", "recommended", "optional"
    effective_date: datetime
    expiration_date: Optional[datetime] = None
    owner: str = "system"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = dataclasses_field(default_factory=dict)


@dataclass
class DataLineage:
    """データ系譜"""

    lineage_id: str
    source_entity: str
    target_entity: str
    transformation: str
    transformation_type: str  # "extract", "transform", "load", "calculation"
    confidence: float = 1.0  # 系譜の信頼度
    created_at: datetime = dataclasses_field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = dataclasses_field(default_factory=dict)


class DataIntegrationRule(ABC):
    """抽象データ統合ルール"""

    @abstractmethod
    async def integrate(
        self, source_data: Dict[str, Any], existing_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """データ統合実行"""
        pass

    @abstractmethod
    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報取得"""
        pass


class StockDataIntegrationRule(DataIntegrationRule):
    """株式データ統合ルール"""

    async def integrate(
        self, source_data: Dict[str, Any], existing_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """株式データ統合実行"""
        try:
            integrated = source_data.copy()

            if existing_data:
                # 既存データとの統合ロジック
                # 最新の価格情報を優先
                if "last_price" in source_data and "last_updated" in source_data:
                    existing_updated = existing_data.get("last_updated")
                    source_updated = source_data.get("last_updated")

                    if existing_updated and source_updated:
                        if pd.to_datetime(existing_updated) > pd.to_datetime(
                            source_updated
                        ):
                            integrated["last_price"] = existing_data["last_price"]
                            integrated["last_updated"] = existing_data["last_updated"]

                # 企業情報は手動変更を優先
                manual_fields = ["company_name", "industry", "sector"]
                for field in manual_fields:
                    if field in existing_data and existing_data.get(
                        f"{field}_manual", False
                    ):
                        integrated[field] = existing_data[field]
                        integrated[f"{field}_manual"] = True

                # バージョン管理
                integrated["version"] = existing_data.get("version", 1) + 1
                integrated["previous_version"] = existing_data.get("version", 1)

            # 品質チェック
            integrated = await self._apply_quality_rules(integrated)

            return integrated

        except Exception as e:
            logger.error(f"株式データ統合エラー: {e}")
            return source_data

    async def _apply_quality_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """品質ルール適用"""
        # 必須フィールドチェック
        required_fields = ["symbol", "company_name"]
        for field in required_fields:
            if field not in data or not data[field]:
                data[f"{field}_quality_issue"] = "missing_required_field"

        # データ型検証
        if "last_price" in data:
            try:
                data["last_price"] = float(data["last_price"])
            except (ValueError, TypeError):
                data["last_price_quality_issue"] = "invalid_price_format"

        return data

    def get_rule_info(self) -> Dict[str, Any]:
        return {
            "rule_type": "stock_data_integration",
            "version": "1.0",
            "priority_fields": ["last_price", "last_updated"],
            "manual_override_fields": ["company_name", "industry", "sector"],
        }


class MasterDataManager:
    """マスターデータ管理システム"""

    def __init__(
        self,
        storage_path: str = "data/mdm",
        enable_cache: bool = True,
        enable_audit: bool = True,
        data_retention_days: int = 2555,  # 7年（金融業界標準）
    ):
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.enable_audit = enable_audit
        self.data_retention_days = data_retention_days

        # ディレクトリ初期化
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # データベース初期化
        self.db_path = self.storage_path / "mdm.db"
        self._initialize_database()

        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=128,  # MDMは大量のマスターデータをキャッシュ
                    l2_memory_mb=512,
                    l3_disk_mb=2048,
                )
                logger.info("MDMキャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        # データ統合ルール
        self.integration_rules: Dict[str, DataIntegrationRule] = {
            "stock": StockDataIntegrationRule()
        }

        # インメモリキャッシュ
        self.data_elements: Dict[str, DataElement] = {}
        self.master_entities: Dict[str, MasterDataEntity] = {}
        self.data_stewards: Dict[str, DataSteward] = {}
        self.governance_policies: Dict[str, DataGovernancePolicy] = {}
        self.data_lineages: Dict[str, DataLineage] = {}

        # デフォルト設定
        self._setup_default_data_elements()
        self._setup_default_governance_policies()
        self._setup_default_stewards()

        logger.info("マスターデータ管理システム初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - キャッシュ: {'有効' if enable_cache else '無効'}")
        logger.info(f"  - 監査: {'有効' if enable_audit else '無効'}")
        logger.info(f"  - データ保持期間: {data_retention_days}日")

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # データ要素テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_elements (
                    element_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    data_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    classification TEXT,
                    is_pii INTEGER DEFAULT 0,
                    business_rules TEXT,
                    validation_rules TEXT,
                    source_systems TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT,
                    metadata TEXT
                )
            """
            )

            # マスターエンティティテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS master_entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    primary_key TEXT NOT NULL,
                    attributes TEXT,
                    domain TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    version INTEGER DEFAULT 1,
                    source_system TEXT,
                    data_quality_score REAL DEFAULT 1.0,
                    last_validated TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    steward TEXT,
                    lineage TEXT,
                    relationships TEXT,
                    metadata TEXT
                )
            """
            )

            # データスチュワードテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_stewards (
                    steward_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    domains TEXT,
                    responsibilities TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """
            )

            # ガバナンスポリシーテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS governance_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    domain TEXT NOT NULL,
                    policy_type TEXT NOT NULL,
                    rules TEXT,
                    enforcement_level TEXT,
                    effective_date TEXT NOT NULL,
                    expiration_date TEXT,
                    owner TEXT,
                    approved_by TEXT,
                    approved_at TEXT,
                    metadata TEXT
                )
            """
            )

            # データ系譜テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    source_entity TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    transformation TEXT,
                    transformation_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """
            )

            # 監査ログテーブル（監査有効時）
            if self.enable_audit:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        audit_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        old_values TEXT,
                        new_values TEXT,
                        user_id TEXT,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    )
                """
                )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON master_entities(entity_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_domain ON master_entities(domain)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_status ON master_entities(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_source ON data_lineage(source_entity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_target ON data_lineage(target_entity)"
            )

    def _setup_default_data_elements(self):
        """デフォルトデータ要素設定"""
        # 株式コード要素
        stock_symbol = DataElement(
            element_id="stock_symbol",
            name="株式コード",
            description="証券取引所における株式の識別コード",
            data_type="string",
            domain=DataDomain.SECURITY,
            classification=DataClassification.PUBLIC,
            business_rules=["4桁の数字", "日本株式市場"],
            validation_rules=["^[0-9]{4}$"],
            source_systems=["trading_system", "market_data"],
        )
        self.data_elements[stock_symbol.element_id] = stock_symbol

        # 企業名要素
        company_name = DataElement(
            element_id="company_name",
            name="企業名",
            description="上場企業の正式名称",
            data_type="string",
            domain=DataDomain.SECURITY,
            classification=DataClassification.PUBLIC,
            business_rules=["正式な企業名", "日本語表記"],
            validation_rules=["最大長255文字"],
            source_systems=["corporate_data", "regulatory_filing"],
        )
        self.data_elements[company_name.element_id] = company_name

        # 株価要素
        stock_price = DataElement(
            element_id="stock_price",
            name="株価",
            description="株式の現在価格または終値",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て価格"],
            validation_rules=[">0", "小数点以下2桁まで"],
            source_systems=["market_data", "trading_system"],
        )
        self.data_elements[stock_price.element_id] = stock_price

    def _setup_default_governance_policies(self):
        """デフォルトガバナンスポリシー設定"""
        # データ品質ポリシー
        quality_policy = DataGovernancePolicy(
            policy_id="data_quality_policy",
            name="データ品質管理ポリシー",
            description="すべてのマスターデータは定義された品質基準を満たす必要があります",
            domain=DataDomain.FINANCIAL,
            policy_type="quality",
            rules=[
                {
                    "rule": "completeness",
                    "threshold": 0.95,
                    "description": "必須フィールドの完全性は95%以上",
                },
                {
                    "rule": "accuracy",
                    "threshold": 0.98,
                    "description": "データの正確性は98%以上",
                },
                {
                    "rule": "timeliness",
                    "threshold": 60,
                    "description": "データの鮮度は60分以内",
                },
            ],
            enforcement_level="mandatory",
            effective_date=datetime.utcnow(),
            owner="data_governance_team",
        )
        self.governance_policies[quality_policy.policy_id] = quality_policy

        # データ保持ポリシー
        retention_policy = DataGovernancePolicy(
            policy_id="data_retention_policy",
            name="データ保持ポリシー",
            description="金融業界規制に準拠したデータ保持期間",
            domain=DataDomain.FINANCIAL,
            policy_type="retention",
            rules=[
                {
                    "rule": "financial_data",
                    "retention_years": 7,
                    "description": "金融データは7年間保持",
                },
                {
                    "rule": "audit_log",
                    "retention_years": 10,
                    "description": "監査ログは10年間保持",
                },
                {
                    "rule": "pii_data",
                    "retention_years": 3,
                    "description": "個人識別情報は3年間保持",
                },
            ],
            enforcement_level="mandatory",
            effective_date=datetime.utcnow(),
            owner="compliance_team",
        )
        self.governance_policies[retention_policy.policy_id] = retention_policy

    def _setup_default_stewards(self):
        """デフォルトデータスチュワード設定"""
        # マーケットデータスチュワード
        market_steward = DataSteward(
            steward_id="market_data_steward",
            name="マーケットデータスチュワード",
            email="market.steward@company.com",
            role=DataStewardshipRole.DATA_STEWARD,
            domains=[DataDomain.MARKET, DataDomain.SECURITY],
            responsibilities=[
                "市場データの品質監視",
                "価格データの整合性確認",
                "データソースの管理",
                "品質問題のエスカレーション",
            ],
        )
        self.data_stewards[market_steward.steward_id] = market_steward

        # リファレンスデータスチュワード
        reference_steward = DataSteward(
            steward_id="reference_data_steward",
            name="リファレンスデータスチュワード",
            email="reference.steward@company.com",
            role=DataStewardshipRole.DATA_STEWARD,
            domains=[DataDomain.REFERENCE, DataDomain.REGULATORY],
            responsibilities=[
                "参照データの維持管理",
                "マスターデータの標準化",
                "データ定義の管理",
                "変更管理プロセスの実行",
            ],
        )
        self.data_stewards[reference_steward.steward_id] = reference_steward

    async def register_master_entity(
        self,
        entity_type: str,
        primary_key: str,
        attributes: Dict[str, Any],
        domain: DataDomain,
        source_system: str = "mdm",
        steward_id: Optional[str] = None,
    ) -> str:
        """マスターエンティティ登録"""
        entity_id = f"{entity_type}_{primary_key}_{int(time.time())}"

        logger.info(f"マスターエンティティ登録: {entity_id} ({entity_type})")

        try:
            # 既存エンティティチェック
            existing_entity = await self._find_existing_entity(entity_type, primary_key)

            # データ統合実行
            if entity_type in self.integration_rules:
                rule = self.integration_rules[entity_type]
                integrated_attributes = await rule.integrate(
                    attributes, existing_entity.attributes if existing_entity else None
                )
            else:
                integrated_attributes = attributes

            # エンティティ作成
            entity = MasterDataEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                primary_key=primary_key,
                attributes=integrated_attributes,
                domain=domain,
                source_system=source_system,
                steward=steward_id,
                version=existing_entity.version + 1 if existing_entity else 1,
            )

            # データ品質評価
            entity.data_quality_score = await self._assess_entity_quality(entity)

            # ガバナンスポリシー適用
            await self._apply_governance_policies(entity)

            # データベース保存
            await self._save_entity_to_db(entity)

            # キャッシュ更新
            if self.cache_manager:
                cache_key = generate_unified_cache_key("mdm_entity", entity_id)
                self.cache_manager.put(cache_key, entity, priority=5.0)

            self.master_entities[entity_id] = entity

            # データ系譜記録
            if existing_entity:
                await self._record_data_lineage(
                    source_entity=existing_entity.entity_id,
                    target_entity=entity_id,
                    transformation="data_integration",
                    transformation_type="transform",
                )

            # 監査ログ記録
            if self.enable_audit:
                await self._record_audit_log(
                    entity_type="master_entity",
                    entity_id=entity_id,
                    operation="register",
                    new_values=entity.attributes,
                    user_id=source_system,
                )

            logger.info(f"マスターエンティティ登録完了: {entity_id}")
            return entity_id

        except Exception as e:
            logger.error(f"マスターエンティティ登録エラー: {e}")
            raise

    async def get_master_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """マスターエンティティ取得"""
        # キャッシュチェック
        if entity_id in self.master_entities:
            return self.master_entities[entity_id]

        if self.cache_manager:
            cache_key = generate_unified_cache_key("mdm_entity", entity_id)
            cached_entity = self.cache_manager.get(cache_key)
            if cached_entity:
                self.master_entities[entity_id] = cached_entity
                return cached_entity

        # データベース検索
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_id, entity_type, primary_key, attributes, domain, status,
                           version, source_system, data_quality_score, last_validated,
                           created_at, updated_at, steward, lineage, relationships, metadata
                    FROM master_entities WHERE entity_id = ?
                """,
                    (entity_id,),
                )

                row = cursor.fetchone()
                if row:
                    entity = MasterDataEntity(
                        entity_id=row[0],
                        entity_type=row[1],
                        primary_key=row[2],
                        attributes=json.loads(row[3]) if row[3] else {},
                        domain=DataDomain(row[4]),
                        status=MasterDataStatus(row[5]),
                        version=row[6],
                        source_system=row[7],
                        data_quality_score=row[8],
                        last_validated=datetime.fromisoformat(row[9]),
                        created_at=datetime.fromisoformat(row[10]),
                        updated_at=datetime.fromisoformat(row[11]),
                        steward=row[12],
                        lineage=json.loads(row[13]) if row[13] else [],
                        relationships=json.loads(row[14]) if row[14] else {},
                        metadata=json.loads(row[15]) if row[15] else {},
                    )

                    # キャッシュ更新
                    self.master_entities[entity_id] = entity
                    if self.cache_manager:
                        cache_key = generate_unified_cache_key("mdm_entity", entity_id)
                        self.cache_manager.put(cache_key, entity, priority=4.0)

                    return entity

                return None

        except Exception as e:
            logger.error(f"マスターエンティティ取得エラー: {e}")
            return None

    async def search_master_entities(
        self,
        entity_type: Optional[str] = None,
        domain: Optional[DataDomain] = None,
        status: Optional[MasterDataStatus] = None,
        primary_key_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> List[MasterDataEntity]:
        """マスターエンティティ検索"""
        try:
            query_conditions = []
            query_params = []

            if entity_type:
                query_conditions.append("entity_type = ?")
                query_params.append(entity_type)

            if domain:
                query_conditions.append("domain = ?")
                query_params.append(domain.value)

            if status:
                query_conditions.append("status = ?")
                query_params.append(status.value)

            if primary_key_pattern:
                query_conditions.append("primary_key LIKE ?")
                query_params.append(f"%{primary_key_pattern}%")

            where_clause = ""
            if query_conditions:
                where_clause = "WHERE " + " AND ".join(query_conditions)

            query = f"""
                SELECT entity_id, entity_type, primary_key, attributes, domain, status,
                       version, source_system, data_quality_score, last_validated,
                       created_at, updated_at, steward, lineage, relationships, metadata
                FROM master_entities
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT ?
            """
            query_params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, query_params)

                entities = []
                for row in cursor.fetchall():
                    entity = MasterDataEntity(
                        entity_id=row[0],
                        entity_type=row[1],
                        primary_key=row[2],
                        attributes=json.loads(row[3]) if row[3] else {},
                        domain=DataDomain(row[4]),
                        status=MasterDataStatus(row[5]),
                        version=row[6],
                        source_system=row[7],
                        data_quality_score=row[8],
                        last_validated=datetime.fromisoformat(row[9]),
                        created_at=datetime.fromisoformat(row[10]),
                        updated_at=datetime.fromisoformat(row[11]),
                        steward=row[12],
                        lineage=json.loads(row[13]) if row[13] else [],
                        relationships=json.loads(row[14]) if row[14] else {},
                        metadata=json.loads(row[15]) if row[15] else {},
                    )
                    entities.append(entity)

                return entities

        except Exception as e:
            logger.error(f"マスターエンティティ検索エラー: {e}")
            return []

    async def update_master_entity(
        self, entity_id: str, attributes: Dict[str, Any], user_id: str = "system"
    ) -> bool:
        """マスターエンティティ更新"""
        logger.info(f"マスターエンティティ更新: {entity_id}")

        try:
            # 既存エンティティ取得
            existing_entity = await self.get_master_entity(entity_id)
            if not existing_entity:
                logger.warning(f"更新対象エンティティが見つかりません: {entity_id}")
                return False

            # 監査ログ用の変更前データ保存
            old_attributes = existing_entity.attributes.copy()

            # データ統合実行
            if existing_entity.entity_type in self.integration_rules:
                rule = self.integration_rules[existing_entity.entity_type]
                integrated_attributes = await rule.integrate(
                    attributes, existing_entity.attributes
                )
            else:
                integrated_attributes = {**existing_entity.attributes, **attributes}

            # エンティティ更新
            existing_entity.attributes = integrated_attributes
            existing_entity.version += 1
            existing_entity.updated_at = datetime.utcnow()

            # データ品質再評価
            existing_entity.data_quality_score = await self._assess_entity_quality(
                existing_entity
            )

            # データベース更新
            await self._update_entity_in_db(existing_entity)

            # キャッシュ更新
            self.master_entities[entity_id] = existing_entity
            if self.cache_manager:
                cache_key = generate_unified_cache_key("mdm_entity", entity_id)
                self.cache_manager.put(cache_key, existing_entity, priority=5.0)

            # 監査ログ記録
            if self.enable_audit:
                await self._record_audit_log(
                    entity_type="master_entity",
                    entity_id=entity_id,
                    operation="update",
                    old_values=old_attributes,
                    new_values=integrated_attributes,
                    user_id=user_id,
                )

            logger.info(f"マスターエンティティ更新完了: {entity_id}")
            return True

        except Exception as e:
            logger.error(f"マスターエンティティ更新エラー: {e}")
            return False

    async def create_data_catalog(self) -> Dict[str, Any]:
        """データカタログ作成"""
        logger.info("データカタログ作成中...")

        try:
            catalog = {
                "generated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "data_elements": {},
                "master_entities": {},
                "domains": {},
                "stewards": {},
                "governance_policies": {},
                "quality_metrics": {},
                "lineage_summary": {},
            }

            # データ要素カタログ
            for element_id, element in self.data_elements.items():
                catalog["data_elements"][element_id] = {
                    "name": element.name,
                    "description": element.description,
                    "data_type": element.data_type,
                    "domain": element.domain.value,
                    "classification": element.classification.value,
                    "is_pii": element.is_pii,
                    "business_rules": element.business_rules,
                    "validation_rules": element.validation_rules,
                    "source_systems": element.source_systems,
                }

            # マスターエンティティサマリー
            entities_by_type = {}
            entities_by_domain = {}
            total_quality_score = 0
            entity_count = 0

            all_entities = await self.search_master_entities(limit=1000)
            for entity in all_entities:
                # エンティティ種別別統計
                if entity.entity_type not in entities_by_type:
                    entities_by_type[entity.entity_type] = 0
                entities_by_type[entity.entity_type] += 1

                # ドメイン別統計
                domain_key = entity.domain.value
                if domain_key not in entities_by_domain:
                    entities_by_domain[domain_key] = 0
                entities_by_domain[domain_key] += 1

                # 品質スコア集計
                total_quality_score += entity.data_quality_score
                entity_count += 1

            catalog["master_entities"] = {
                "total_count": entity_count,
                "by_type": entities_by_type,
                "by_domain": entities_by_domain,
                "average_quality_score": total_quality_score / entity_count
                if entity_count > 0
                else 0,
            }

            # ドメイン情報
            for domain in DataDomain:
                catalog["domains"][domain.value] = {
                    "name": domain.value,
                    "entity_count": entities_by_domain.get(domain.value, 0),
                    "stewards": [
                        steward.steward_id
                        for steward in self.data_stewards.values()
                        if domain in steward.domains
                    ],
                }

            # スチュワード情報
            for steward_id, steward in self.data_stewards.items():
                catalog["stewards"][steward_id] = {
                    "name": steward.name,
                    "role": steward.role.value,
                    "domains": [domain.value for domain in steward.domains],
                    "responsibilities": steward.responsibilities,
                    "active": steward.active,
                }

            # ガバナンスポリシー
            for policy_id, policy in self.governance_policies.items():
                catalog["governance_policies"][policy_id] = {
                    "name": policy.name,
                    "description": policy.description,
                    "domain": policy.domain.value,
                    "policy_type": policy.policy_type,
                    "enforcement_level": policy.enforcement_level,
                    "effective_date": policy.effective_date.isoformat(),
                    "owner": policy.owner,
                }

            # 品質メトリクス
            catalog["quality_metrics"] = await self._calculate_global_quality_metrics()

            # データ系譜サマリー
            catalog["lineage_summary"] = await self._get_lineage_summary()

            # カタログファイル保存
            catalog_file = self.storage_path / f"data_catalog_{int(time.time())}.json"
            with open(catalog_file, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)

            logger.info(f"データカタログ作成完了: {catalog_file}")
            return catalog

        except Exception as e:
            logger.error(f"データカタログ作成エラー: {e}")
            return {"error": str(e)}

    async def _find_existing_entity(
        self, entity_type: str, primary_key: str
    ) -> Optional[MasterDataEntity]:
        """既存エンティティ検索"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT entity_id FROM master_entities
                    WHERE entity_type = ? AND primary_key = ? AND status = 'active'
                    ORDER BY version DESC LIMIT 1
                """,
                    (entity_type, primary_key),
                )

                row = cursor.fetchone()
                if row:
                    return await self.get_master_entity(row[0])

                return None

        except Exception as e:
            logger.error(f"既存エンティティ検索エラー: {e}")
            return None

    async def _assess_entity_quality(self, entity: MasterDataEntity) -> float:
        """エンティティ品質評価"""
        try:
            quality_score = 1.0

            # 必須属性チェック
            required_attrs = self._get_required_attributes(entity.entity_type)
            missing_attrs = [
                attr for attr in required_attrs if attr not in entity.attributes
            ]
            if missing_attrs:
                quality_score -= len(missing_attrs) * 0.1

            # データ型チェック
            type_violations = self._validate_attribute_types(entity)
            quality_score -= len(type_violations) * 0.05

            # ビジネスルールチェック
            rule_violations = await self._validate_business_rules(entity)
            quality_score -= len(rule_violations) * 0.05

            # 鮮度チェック
            data_age_hours = (
                datetime.utcnow() - entity.updated_at
            ).total_seconds() / 3600
            if data_age_hours > 24:  # 24時間以上古い
                quality_score -= 0.1
            elif data_age_hours > 168:  # 1週間以上古い
                quality_score -= 0.2

            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error(f"品質評価エラー: {e}")
            return 0.5  # デフォルト値

    def _get_required_attributes(self, entity_type: str) -> List[str]:
        """エンティティ種別の必須属性取得"""
        required_attrs_map = {
            "stock": ["symbol", "company_name"],
            "company": ["name", "industry"],
            "currency": ["code", "name"],
            "exchange": ["code", "name", "country"],
        }
        return required_attrs_map.get(entity_type, [])

    def _validate_attribute_types(self, entity: MasterDataEntity) -> List[str]:
        """属性データ型検証"""
        violations = []

        try:
            for attr_name, attr_value in entity.attributes.items():
                # データ要素定義との照合
                for element in self.data_elements.values():
                    if element.name.lower().replace(" ", "_") == attr_name:
                        expected_type = element.data_type

                        if expected_type == "string" and not isinstance(
                            attr_value, str
                        ):
                            violations.append(f"{attr_name}: 文字列型が期待されます")
                        elif expected_type == "decimal" and not isinstance(
                            attr_value, (int, float)
                        ):
                            violations.append(f"{attr_name}: 数値型が期待されます")
                        elif expected_type == "integer" and not isinstance(
                            attr_value, int
                        ):
                            violations.append(f"{attr_name}: 整数型が期待されます")

                        break

            return violations

        except Exception as e:
            logger.error(f"属性データ型検証エラー: {e}")
            return []

    async def _validate_business_rules(self, entity: MasterDataEntity) -> List[str]:
        """ビジネスルール検証"""
        violations = []

        try:
            # 株式コードのビジネスルール
            if entity.entity_type == "stock" and "symbol" in entity.attributes:
                symbol = entity.attributes["symbol"]
                if (
                    not isinstance(symbol, str)
                    or not symbol.isdigit()
                    or len(symbol) != 4
                ):
                    violations.append("株式コードは4桁の数字である必要があります")

            # 価格のビジネスルール
            price_fields = ["last_price", "open_price", "high_price", "low_price"]
            for field in price_fields:
                if field in entity.attributes:
                    price = entity.attributes[field]
                    if isinstance(price, (int, float)) and price <= 0:
                        violations.append(f"{field}は正の数値である必要があります")

            return violations

        except Exception as e:
            logger.error(f"ビジネスルール検証エラー: {e}")
            return []

    async def _apply_governance_policies(self, entity: MasterDataEntity):
        """ガバナンスポリシー適用"""
        try:
            # ドメイン関連ポリシー適用
            domain_policies = [
                policy
                for policy in self.governance_policies.values()
                if policy.domain == entity.domain
                or policy.domain == DataDomain.FINANCIAL
            ]

            for policy in domain_policies:
                if policy.policy_type == "quality":
                    # 品質ポリシー適用
                    for policy_rule in policy.rules:
                        if policy_rule["rule"] == "completeness":
                            required_attrs = self._get_required_attributes(
                                entity.entity_type
                            )
                            completeness = (
                                len(
                                    [
                                        attr
                                        for attr in required_attrs
                                        if attr in entity.attributes
                                    ]
                                )
                                / len(required_attrs)
                                if required_attrs
                                else 1.0
                            )

                            if completeness < policy_rule["threshold"]:
                                entity.metadata[
                                    "quality_violations"
                                ] = entity.metadata.get("quality_violations", [])
                                entity.metadata["quality_violations"].append(
                                    f"完全性違反: {completeness:.2f} < {policy_rule['threshold']}"
                                )

                elif policy.policy_type == "retention":
                    # データ保持ポリシー適用（エンティティの最終更新日時をチェックし、古い場合は警告）
                    retention_threshold = datetime.utcnow() - timedelta(
                        days=policy.rules[0]["retention_years"] * 365
                    )
                    if entity.updated_at < retention_threshold:
                        entity.metadata["retention_violations"] = entity.metadata.get(
                            "retention_violations", []
                        )
                        entity.metadata["retention_violations"].append(
                            f"保持期間違反: 最終更新日時が {retention_threshold.isoformat()} より古い"
                        )

        except Exception as e:
            logger.error(f"ガバナンスポリシー適用エラー: {e}")

    async def _save_entity_to_db(self, entity: MasterDataEntity):
        """エンティティデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO master_entities
                (entity_id, entity_type, primary_key, attributes, domain, status,
                 version, source_system, data_quality_score, last_validated,
                 created_at, updated_at, steward, lineage, relationships, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entity.entity_id,
                    entity.entity_type,
                    entity.primary_key,
                    json.dumps(entity.attributes, default=str),
                    entity.domain.value,
                    entity.status.value,
                    entity.version,
                    entity.source_system,
                    entity.data_quality_score,
                    entity.last_validated.isoformat(),
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.steward,
                    json.dumps(entity.lineage),
                    json.dumps(entity.relationships),
                    json.dumps(entity.metadata, default=str),
                ),
            )

    async def _update_entity_in_db(self, entity: MasterDataEntity):
        """エンティティデータベース更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE master_entities SET
                    attributes = ?, status = ?, version = ?, data_quality_score = ?,
                    last_validated = ?, updated_at = ?, steward = ?, lineage = ?,
                    relationships = ?, metadata = ?
                WHERE entity_id = ?
            """,
                (
                    json.dumps(entity.attributes, default=str),
                    entity.status.value,
                    entity.version,
                    entity.data_quality_score,
                    entity.last_validated.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.steward,
                    json.dumps(entity.lineage),
                    json.dumps(entity.relationships),
                    json.dumps(entity.metadata, default=str),
                    entity.entity_id,
                ),
            )

    async def _record_data_lineage(
        self,
        source_entity: str,
        target_entity: str,
        transformation: str,
        transformation_type: str,
        confidence: float = 1.0,
    ):
        """データ系譜記録"""
        lineage_id = (
            f"lineage_{int(time.time())}_{hash(f'{source_entity}_{target_entity}')}"
        )

        lineage = DataLineage(
            lineage_id=lineage_id,
            source_entity=source_entity,
            target_entity=target_entity,
            transformation=transformation,
            transformation_type=transformation_type,
            confidence=confidence,
        )

        self.data_lineages[lineage_id] = lineage

        # データベース保存
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_lineage
                (lineage_id, source_entity, target_entity, transformation,
                 transformation_type, confidence, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    lineage.lineage_id,
                    lineage.source_entity,
                    lineage.target_entity,
                    lineage.transformation,
                    lineage.transformation_type,
                    lineage.confidence,
                    lineage.created_at.isoformat(),
                    json.dumps(lineage.metadata),
                ),
            )

    async def _record_audit_log(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        user_id: str = "system",
    ):
        """監査ログ記録"""
        if not self.enable_audit:
            return

        audit_id = (
            f"audit_{int(time.time())}_{hash(f'{entity_type}_{entity_id}_{operation}')}"
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_log
                (audit_id, entity_type, entity_id, operation, old_values,
                 new_values, user_id, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    audit_id,
                    entity_type,
                    entity_id,
                    operation,
                    json.dumps(old_values, default=str) if old_values else None,
                    json.dumps(new_values, default=str) if new_values else None,
                    user_id,
                    datetime.utcnow().isoformat(),
                    json.dumps({}),
                ),
            )

    async def _calculate_global_quality_metrics(self) -> Dict[str, Any]:
        """グローバル品質メトリクス計算"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 全体品質統計
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entities,
                        AVG(data_quality_score) as avg_quality_score,
                        MIN(data_quality_score) as min_quality_score,
                        MAX(data_quality_score) as max_quality_score
                    FROM master_entities WHERE status = 'active'
                """
                )

                stats = cursor.fetchone()

                # ドメイン別品質統計
                cursor = conn.execute(
                    """
                    SELECT
                        domain,
                        COUNT(*) as count,
                        AVG(data_quality_score) as avg_score
                    FROM master_entities WHERE status = 'active'
                    GROUP BY domain
                """
                )

                domain_stats = {}
                for row in cursor.fetchall():
                    domain_stats[row[0]] = {
                        "entity_count": row[1],
                        "average_quality_score": row[2],
                    }

                return {
                    "total_entities": stats[0] or 0,
                    "average_quality_score": stats[1] or 0,
                    "min_quality_score": stats[2] or 0,
                    "max_quality_score": stats[3] or 0,
                    "domain_statistics": domain_stats,
                    "calculated_at": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"グローバル品質メトリクス計算エラー: {e}")
            return {}

    async def _get_lineage_summary(self) -> Dict[str, Any]:
        """データ系譜サマリー取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 系譜統計
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_lineages,
                        COUNT(DISTINCT source_entity) as unique_sources,
                        COUNT(DISTINCT target_entity) as unique_targets,
                        AVG(confidence) as avg_confidence
                    FROM data_lineage
                """
                )

                stats = cursor.fetchone()

                # 変換タイプ別統計
                cursor = conn.execute(
                    """
                    SELECT transformation_type, COUNT(*) as count
                    FROM data_lineage
                    GROUP BY transformation_type
                """
                )

                transformation_stats = {}
                for row in cursor.fetchall():
                    transformation_stats[row[0]] = row[1]

                return {
                    "total_lineages": stats[0] or 0,
                    "unique_sources": stats[1] or 0,
                    "unique_targets": stats[2] or 0,
                    "average_confidence": stats[3] or 0,
                    "transformation_types": transformation_stats,
                }

        except Exception as e:
            logger.error(f"データ系譜サマリー取得エラー: {e}")
            return {}

    async def get_mdm_dashboard(self) -> Dict[str, Any]:
        """MDMダッシュボード情報取得"""
        try:
            # 基本統計
            total_entities = len(self.master_entities)
            active_stewards = sum(
                1 for steward in self.data_stewards.values() if steward.active
            )

            # 品質メトリクス
            quality_metrics = await self._calculate_global_quality_metrics()

            # 最近の活動
            recent_activities = await self._get_recent_activities()

            # ガバナンス状況
            governance_status = {
                "total_policies": len(self.governance_policies),
                "active_policies": len(
                    [
                        p
                        for p in self.governance_policies.values()
                        if p.expiration_date is None
                        or p.expiration_date > datetime.utcnow()
                    ]
                ),
            }

            return {
                "generated_at": datetime.utcnow().isoformat(),
                "system_status": "active",
                "statistics": {
                    "total_entities": total_entities,
                    "data_elements": len(self.data_elements),
                    "active_stewards": active_stewards,
                    "data_lineages": len(self.data_lineages),
                },
                "quality_metrics": quality_metrics,
                "governance_status": governance_status,
                "recent_activities": recent_activities,
                "domain_distribution": await self._get_domain_distribution(),
            }

        except Exception as e:
            logger.error(f"MDMダッシュボード情報取得エラー: {e}")
            return {"error": str(e)}

    async def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """最近の活動取得"""
        activities = []

        try:
            if self.enable_audit:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT entity_type, entity_id, operation, user_id, timestamp
                        FROM audit_log
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """
                    )

                    for row in cursor.fetchall():
                        activities.append(
                            {
                                "entity_type": row[0],
                                "entity_id": row[1],
                                "operation": row[2],
                                "user_id": row[3],
                                "timestamp": row[4],
                            }
                        )

            return activities

        except Exception as e:
            logger.error(f"最近の活動取得エラー: {e}")
            return []

    async def _get_domain_distribution(self) -> Dict[str, int]:
        """ドメイン分布取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT domain, COUNT(*) as count
                    FROM master_entities
                    WHERE status = 'active'
                    GROUP BY domain
                """
                )

                distribution = {}
                for row in cursor.fetchall():
                    distribution[row[0]] = row[1]

                return distribution

        except Exception as e:
            logger.error(f"ドメイン分布取得エラー: {e}")
            return {}

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("マスターデータ管理システム クリーンアップ開始")

        # インメモリキャッシュクリア
        self.data_elements.clear()
        self.master_entities.clear()
        self.data_stewards.clear()
        self.governance_policies.clear()
        self.data_lineages.clear()

        # 古いログデータクリーンアップ（保持期間超過）
        if self.enable_audit:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    retention_date = datetime.utcnow() - timedelta(
                        days=self.data_retention_days
                    )
                    conn.execute(
                        """
                        DELETE FROM audit_log
                        WHERE timestamp < ?
                    """,
                        (retention_date.isoformat(),),
                    )

                    deleted_count = conn.total_changes
                    if deleted_count > 0:
                        logger.info(f"古い監査ログクリーンアップ: {deleted_count}件")
            except Exception as e:
                logger.error(f"監査ログクリーンアップエラー: {e}")

        logger.info("マスターデータ管理システム クリーンアップ完了")


# Factory function
def create_master_data_manager(
    storage_path: str = "data/mdm",
    enable_cache: bool = True,
    enable_audit: bool = True,
    data_retention_days: int = 2555,
) -> MasterDataManager:
    """マスターデータ管理システム作成"""
    return MasterDataManager(
        storage_path=storage_path,
        enable_cache=enable_cache,
        enable_audit=enable_audit,
        data_retention_days=data_retention_days,
    )


if __name__ == "__main__":
    # テスト実行
    async def test_master_data_manager():
        print("=== Issue #420 マスターデータ管理（MDM）システムテスト ===")

        try:
            # MDMシステム初期化
            mdm = create_master_data_manager(
                storage_path="test_mdm",
                enable_cache=True,
                enable_audit=True,
                data_retention_days=30,
            )

            print("\n1. マスターデータ管理システム初期化完了")
            print(f"   ストレージパス: {mdm.storage_path}")
            print(f"   データ要素数: {len(mdm.data_elements)}")
            print(f"   ガバナンスポリシー数: {len(mdm.governance_policies)}")
            print(f"   データスチュワード数: {len(mdm.data_stewards)}")

            # マスターエンティティ登録テスト
            print("\n2. マスターエンティティ登録テスト...")

            # 株式エンティティ登録
            stock_attributes = {
                "symbol": "7203",
                "company_name": "トヨタ自動車株式会社",
                "industry": "輸送用機器",
                "sector": "製造業",
                "last_price": 2500.0,
                "last_updated": datetime.utcnow().isoformat(),
                "market_cap": 35000000000000,  # 35兆円
                "listing_date": "1949-05-16",
            }

            stock_entity_id = await mdm.register_master_entity(
                entity_type="stock",
                primary_key="7203",
                attributes=stock_attributes,
                domain=DataDomain.SECURITY,
                source_system="market_data",
                steward_id="market_data_steward",
            )

            print(f"   株式エンティティ登録: {stock_entity_id}")

            # 企業エンティティ登録
            company_attributes = {
                "name": "トヨタ自動車株式会社",
                "industry": "輸送用機器",
                "founded": "1937-08-28",
                "headquarters": "愛知県豊田市",
                "employees": 366283,
                "website": "https://toyota.jp",
            }

            company_entity_id = await mdm.register_master_entity(
                entity_type="company",
                primary_key="toyota",
                attributes=company_attributes,
                domain=DataDomain.REFERENCE,
                steward_id="reference_data_steward",
            )

            print(f"   企業エンティティ登録: {company_entity_id}")

            # エンティティ取得テスト
            print("\n3. エンティティ取得テスト...")
            retrieved_stock = await mdm.get_master_entity(stock_entity_id)
            if retrieved_stock:
                print(
                    f"   取得成功: {retrieved_stock.entity_type} - {retrieved_stock.primary_key}"
                )
                print(f"   品質スコア: {retrieved_stock.data_quality_score:.3f}")
                print(f"   バージョン: {retrieved_stock.version}")

            # エンティティ更新テスト
            print("\n4. エンティティ更新テスト...")
            updated_attributes = {
                "last_price": 2520.0,
                "last_updated": datetime.utcnow().isoformat(),
                "price_change": 20.0,
                "price_change_percent": 0.8,
            }

            update_success = await mdm.update_master_entity(
                stock_entity_id, updated_attributes, "test_user"
            )

            print(f"   更新結果: {'成功' if update_success else '失敗'}")

            # エンティティ検索テスト
            print("\n5. エンティティ検索テスト...")

            # 株式エンティティ検索
            stock_entities = await mdm.search_master_entities(
                entity_type="stock", domain=DataDomain.SECURITY, limit=10
            )
            print(f"   株式エンティティ検索結果: {len(stock_entities)}件")

            # 全エンティティ検索
            all_entities = await mdm.search_master_entities(limit=100)
            print(f"   全エンティティ検索結果: {len(all_entities)}件")

            # データカタログ作成テスト
            print("\n6. データカタログ作成テスト...")
            catalog = await mdm.create_data_catalog()

            print(f"   カタログ生成時刻: {catalog['generated_at']}")
            print(f"   データ要素数: {len(catalog['data_elements'])}")
            print("   マスターエンティティ:")
            print(f"     総数: {catalog['master_entities']['total_count']}")
            print(
                f"     平均品質スコア: {catalog['master_entities']['average_quality_score']:.3f}"
            )
            print(f"   ドメイン数: {len(catalog['domains'])}")
            print(f"   スチュワード数: {len(catalog['stewards'])}")
            print(f"   ガバナンスポリシー数: {len(catalog['governance_policies'])}")

            # MDMダッシュボード確認
            print("\n7. MDMダッシュボード確認...")
            dashboard = await mdm.get_mdm_dashboard()

            print(f"   システム状態: {dashboard['system_status']}")
            stats = dashboard["statistics"]
            print("   統計情報:")
            print(f"     総エンティティ数: {stats['total_entities']}")
            print(f"     データ要素数: {stats['data_elements']}")
            print(f"     アクティブスチュワード数: {stats['active_stewards']}")

            if "quality_metrics" in dashboard and dashboard["quality_metrics"]:
                quality = dashboard["quality_metrics"]
                print("   品質メトリクス:")
                print(f"     平均品質スコア: {quality['average_quality_score']:.3f}")
                print(f"     最小品質スコア: {quality['min_quality_score']:.3f}")
                print(f"     最大品質スコア: {quality['max_quality_score']:.3f}")

            if "governance_status" in dashboard:
                governance = dashboard["governance_status"]
                print("   ガバナンス状況:")
                print(f"     総ポリシー数: {governance['total_policies']}")
                print(f"     アクティブポリシー数: {governance['active_policies']}")

            # ドメイン分布
            if "domain_distribution" in dashboard:
                print("   ドメイン分布:")
                for domain, count in dashboard["domain_distribution"].items():
                    print(f"     {domain}: {count}件")

            # 最近の活動
            if "recent_activities" in dashboard and dashboard["recent_activities"]:
                print(f"   最近の活動: {len(dashboard['recent_activities'])}件")
                for activity in dashboard["recent_activities"][:3]:
                    print(
                        f"     {activity['operation']} - {activity['entity_type']} ({activity['timestamp']})"
                    )

            # クリーンアップ
            await mdm.cleanup()

            print("\n✅ Issue #420 マスターデータ管理システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_master_data_manager())

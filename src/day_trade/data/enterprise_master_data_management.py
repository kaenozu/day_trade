#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム
Issue #420: データ管理とデータ品質保証メカニズムの強化

企業レベルのマスターデータ管理戦略:
- データ統合・統一化
- ゴールデンレコード管理
- データガバナンス・ポリシー
- 階層・分類管理
- データカタログ・メタデータ管理
- データ品質・整合性保証
- アクセス制御・セキュリティ
- 変更追跡・監査証跡
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from ..monitoring.structured_logging_enhancement import (
        StructuredLoggingEnhancementSystem,
    )
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .comprehensive_data_quality_system import ComprehensiveDataQualitySystem
    from .enhanced_data_version_control import EnhancedDataVersionControl
    from .master_data_manager import MasterDataManager, MasterDataSet

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

    # Fallback definitions
    class MasterDataSet:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class MasterDataType(Enum):
    """マスターデータ種別"""

    FINANCIAL_INSTRUMENTS = "financial_instruments"
    MARKET_SEGMENTS = "market_segments"
    CURRENCY_CODES = "currency_codes"
    INDUSTRY_CODES = "industry_codes"
    EXCHANGE_CODES = "exchange_codes"
    COUNTRY_CODES = "country_codes"
    TRADING_HOLIDAYS = "trading_holidays"
    REFERENCE_RATES = "reference_rates"
    RISK_RATINGS = "risk_ratings"
    REGULATORY_DATA = "regulatory_data"


class DataGovernanceLevel(Enum):
    """データガバナンスレベル"""

    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    REGULATORY = "regulatory"


class ChangeType(Enum):
    """変更種別"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    SPLIT = "split"
    ARCHIVE = "archive"


class ApprovalStatus(Enum):
    """承認状態"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


@dataclass
class MasterDataEntity:
    """マスターデータエンティティ"""

    entity_id: str
    entity_type: MasterDataType
    primary_key: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    governance_level: DataGovernanceLevel = DataGovernanceLevel.STANDARD
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    updated_by: str = "system"
    is_active: bool = True
    is_golden_record: bool = False
    source_systems: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)


@dataclass
class DataChangeRequest:
    """データ変更リクエスト"""

    request_id: str
    entity_id: str
    change_type: ChangeType
    proposed_changes: Dict[str, Any]
    business_justification: str
    requested_by: str
    requested_at: datetime
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataGovernancePolicy:
    """データガバナンスポリシー"""

    policy_id: str
    policy_name: str
    entity_types: List[MasterDataType]
    rules: List[Dict[str, Any]]
    governance_level: DataGovernanceLevel
    requires_approval: bool = True
    approval_roles: List[str] = field(default_factory=list)
    retention_period: Optional[int] = None  # days
    audit_required: bool = True
    quality_threshold: float = 80.0
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MasterDataHierarchy:
    """マスターデータ階層"""

    hierarchy_id: str
    name: str
    entity_type: MasterDataType
    root_entity_id: str
    parent_child_mapping: Dict[str, List[str]] = field(default_factory=dict)
    level_definitions: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnterpriseMasterDataManagement:
    """エンタープライズマスターデータ管理システム"""

    def __init__(self, db_path: str = "enterprise_mdm.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path

        # データベース初期化
        self._initialize_database()

        # コンポーネント
        self.quality_system = None
        self.version_control = None
        self.cache_manager = UnifiedCacheManager() if DEPENDENCIES_AVAILABLE else None

        # ガバナンスポリシー
        self.governance_policies: Dict[str, DataGovernancePolicy] = {}

        # 階層管理
        self.hierarchies: Dict[str, MasterDataHierarchy] = {}

        # 統計
        self.stats = {
            "total_entities": 0,
            "golden_records": 0,
            "pending_changes": 0,
            "data_quality_checks": 0,
            "governance_violations": 0,
        }

        self._initialize_components()
        self._load_default_policies()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # マスターデータエンティティテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS master_data_entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    primary_key TEXT NOT NULL,
                    attributes TEXT NOT NULL,
                    metadata TEXT,
                    quality_score REAL DEFAULT 0.0,
                    governance_level TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_golden_record BOOLEAN DEFAULT FALSE,
                    source_systems TEXT,
                    related_entities TEXT
                )
            """
            )

            # 変更リクエストテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_change_requests (
                    request_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    proposed_changes TEXT NOT NULL,
                    business_justification TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    requested_at DATETIME NOT NULL,
                    approval_status TEXT NOT NULL,
                    approved_by TEXT,
                    approved_at DATETIME,
                    rejection_reason TEXT,
                    impact_assessment TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_id) REFERENCES master_data_entities(entity_id)
                )
            """
            )

            # ガバナンスポリシーテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS governance_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_name TEXT NOT NULL,
                    entity_types TEXT NOT NULL,
                    rules TEXT NOT NULL,
                    governance_level TEXT NOT NULL,
                    requires_approval BOOLEAN DEFAULT TRUE,
                    approval_roles TEXT,
                    retention_period INTEGER,
                    audit_required BOOLEAN DEFAULT TRUE,
                    quality_threshold REAL DEFAULT 80.0,
                    created_by TEXT NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """
            )

            # データ階層テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_hierarchies (
                    hierarchy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    root_entity_id TEXT NOT NULL,
                    parent_child_mapping TEXT NOT NULL,
                    level_definitions TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 変更履歴テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS change_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    old_version INTEGER,
                    new_version INTEGER,
                    changed_fields TEXT,
                    changed_by TEXT NOT NULL,
                    changed_at DATETIME NOT NULL,
                    business_context TEXT,
                    approval_reference TEXT,
                    metadata TEXT
                )
            """
            )

            # データ品質履歴テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_issues TEXT,
                    quality_metrics TEXT,
                    checked_at DATETIME NOT NULL,
                    checked_by TEXT,
                    remediation_actions TEXT,
                    metadata TEXT
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON master_data_entities(entity_type, is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_golden ON master_data_entities(is_golden_record, entity_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_status ON data_change_requests(approval_status, requested_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_entity ON change_history(entity_id, changed_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_entity ON quality_history(entity_id, checked_at)"
            )

            conn.commit()

    def _initialize_components(self):
        """コンポーネント初期化"""
        if DEPENDENCIES_AVAILABLE:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
                self.version_control = EnhancedDataVersionControl()
                self.logger.info("MDMシステムコンポーネント初期化完了")
            except Exception as e:
                self.logger.warning(f"コンポーネント初期化エラー: {e}")

    def _load_default_policies(self):
        """デフォルトガバナンスポリシー読み込み"""
        default_policies = [
            DataGovernancePolicy(
                policy_id="financial_instruments_policy",
                policy_name="金融商品マスタポリシー",
                entity_types=[MasterDataType.FINANCIAL_INSTRUMENTS],
                rules=[
                    {"field": "symbol", "required": True, "unique": True},
                    {"field": "name", "required": True, "min_length": 2},
                    {"field": "isin", "pattern": "^[A-Z]{2}[A-Z0-9]{10}$"},
                    {"field": "market", "required": True},
                ],
                governance_level=DataGovernanceLevel.STRICT,
                approval_roles=["data_steward", "compliance_officer"],
                quality_threshold=90.0,
            ),
            DataGovernancePolicy(
                policy_id="market_reference_policy",
                policy_name="市場参照データポリシー",
                entity_types=[
                    MasterDataType.MARKET_SEGMENTS,
                    MasterDataType.EXCHANGE_CODES,
                ],
                rules=[
                    {"field": "code", "required": True, "unique": True},
                    {"field": "description", "required": True},
                ],
                governance_level=DataGovernanceLevel.STANDARD,
                approval_roles=["data_steward"],
                quality_threshold=85.0,
            ),
            DataGovernancePolicy(
                policy_id="regulatory_data_policy",
                policy_name="規制データポリシー",
                entity_types=[MasterDataType.REGULATORY_DATA],
                rules=[
                    {"field": "regulation_id", "required": True, "unique": True},
                    {"field": "effective_date", "required": True, "type": "date"},
                    {"field": "authority", "required": True},
                ],
                governance_level=DataGovernanceLevel.REGULATORY,
                approval_roles=["compliance_officer", "legal_team"],
                audit_required=True,
                retention_period=2555,  # 7 years
                quality_threshold=95.0,
            ),
        ]

        for policy in default_policies:
            self.governance_policies[policy.policy_id] = policy
            self._save_governance_policy(policy)

    async def register_master_data_entity(
        self,
        entity_type: MasterDataType,
        primary_key: str,
        attributes: Dict[str, Any],
        source_system: str,
        created_by: str = "system",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """マスターデータエンティティ登録"""
        try:
            metadata = metadata or {}
            current_time = datetime.now(timezone.utc)

            # エンティティID生成
            entity_id = f"{entity_type.value}_{hashlib.md5(primary_key.encode()).hexdigest()[:8]}_{int(time.time())}"

            # ガバナンスポリシー適用
            policy = self._get_applicable_policy(entity_type)
            governance_level = policy.governance_level if policy else DataGovernanceLevel.STANDARD

            # データ品質チェック
            quality_score = await self._calculate_entity_quality(attributes, policy)

            # バリデーション実行
            validation_result = await self._validate_entity_attributes(attributes, policy)
            if not validation_result["is_valid"]:
                raise ValueError(
                    f"データバリデーションエラー: {', '.join(validation_result['errors'])}"
                )

            # エンティティ作成
            entity = MasterDataEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                primary_key=primary_key,
                attributes=attributes,
                metadata={**metadata, "source_system": source_system},
                quality_score=quality_score,
                governance_level=governance_level,
                created_by=created_by,
                updated_by=created_by,
                source_systems=[source_system],
            )

            # ゴールデンレコード判定
            entity.is_golden_record = await self._determine_golden_record_status(entity)

            # データベース保存
            await self._save_entity(entity)

            # 変更履歴記録
            await self._record_change_history(
                entity_id,
                ChangeType.CREATE,
                None,
                1,
                list(attributes.keys()),
                created_by,
                current_time,
                "Initial entity registration",
                metadata,
            )

            # 統計更新
            self.stats["total_entities"] += 1
            if entity.is_golden_record:
                self.stats["golden_records"] += 1

            self.logger.info(f"マスターデータエンティティ登録完了: {entity_id}")
            return entity_id

        except Exception as e:
            self.logger.error(f"エンティティ登録エラー: {e}")
            raise

    async def request_data_change(
        self,
        entity_id: str,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
        business_justification: str,
        requested_by: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """データ変更リクエスト作成"""
        try:
            metadata = metadata or {}
            current_time = datetime.now(timezone.utc)

            # リクエストID生成
            request_id = f"req_{entity_id}_{change_type.value}_{int(time.time())}"

            # エンティティ存在確認
            entity = await self._get_entity(entity_id)
            if not entity:
                raise ValueError(f"エンティティが見つかりません: {entity_id}")

            # ガバナンスポリシー確認
            policy = self._get_applicable_policy(entity.entity_type)
            requires_approval = policy.requires_approval if policy else True

            # 影響評価実行
            impact_assessment = await self._assess_change_impact(
                entity, change_type, proposed_changes
            )

            # 変更リクエスト作成
            change_request = DataChangeRequest(
                request_id=request_id,
                entity_id=entity_id,
                change_type=change_type,
                proposed_changes=proposed_changes,
                business_justification=business_justification,
                requested_by=requested_by,
                requested_at=current_time,
                approval_status=(
                    ApprovalStatus.PENDING if requires_approval else ApprovalStatus.APPROVED
                ),
                impact_assessment=impact_assessment,
                metadata=metadata,
            )

            # 自動承認の場合
            if not requires_approval:
                change_request.approved_by = "system"
                change_request.approved_at = current_time
                # 変更を即座に適用
                await self._apply_approved_changes(change_request)

            # データベース保存
            await self._save_change_request(change_request)

            # 統計更新
            if change_request.approval_status == ApprovalStatus.PENDING:
                self.stats["pending_changes"] += 1

            self.logger.info(f"データ変更リクエスト作成: {request_id}")
            return request_id

        except Exception as e:
            self.logger.error(f"変更リクエスト作成エラー: {e}")
            raise

    async def approve_change_request(
        self, request_id: str, approved_by: str, approval_notes: str = ""
    ) -> bool:
        """変更リクエスト承認"""
        try:
            # リクエスト取得
            change_request = await self._get_change_request(request_id)
            if not change_request:
                raise ValueError(f"変更リクエストが見つかりません: {request_id}")

            if change_request.approval_status != ApprovalStatus.PENDING:
                raise ValueError(
                    f"リクエストは承認待ち状態ではありません: {change_request.approval_status}"
                )

            current_time = datetime.now(timezone.utc)

            # 承認情報更新
            change_request.approval_status = ApprovalStatus.APPROVED
            change_request.approved_by = approved_by
            change_request.approved_at = current_time
            change_request.metadata["approval_notes"] = approval_notes

            # 変更適用
            await self._apply_approved_changes(change_request)

            # データベース更新
            await self._update_change_request(change_request)

            # 統計更新
            self.stats["pending_changes"] -= 1

            self.logger.info(f"変更リクエスト承認: {request_id}")
            return True

        except Exception as e:
            self.logger.error(f"変更リクエスト承認エラー: {e}")
            return False

    async def reject_change_request(
        self, request_id: str, rejected_by: str, rejection_reason: str
    ) -> bool:
        """変更リクエスト却下"""
        try:
            # リクエスト取得
            change_request = await self._get_change_request(request_id)
            if not change_request:
                raise ValueError(f"変更リクエストが見つかりません: {request_id}")

            # 却下情報更新
            change_request.approval_status = ApprovalStatus.REJECTED
            change_request.approved_by = rejected_by
            change_request.approved_at = datetime.now(timezone.utc)
            change_request.rejection_reason = rejection_reason

            # データベース更新
            await self._update_change_request(change_request)

            # 統計更新
            self.stats["pending_changes"] -= 1

            self.logger.info(f"変更リクエスト却下: {request_id}")
            return True

        except Exception as e:
            self.logger.error(f"変更リクエスト却下エラー: {e}")
            return False

    async def _apply_approved_changes(self, change_request: DataChangeRequest):
        """承認済み変更の適用"""
        try:
            entity = await self._get_entity(change_request.entity_id)
            if not entity:
                raise ValueError(f"エンティティが見つかりません: {change_request.entity_id}")

            old_version = entity.version
            changed_fields = []

            if change_request.change_type == ChangeType.UPDATE:
                # 属性更新
                for field, new_value in change_request.proposed_changes.items():
                    if field in entity.attributes:
                        if entity.attributes[field] != new_value:
                            entity.attributes[field] = new_value
                            changed_fields.append(field)
                    else:
                        entity.attributes[field] = new_value
                        changed_fields.append(field)

            elif change_request.change_type == ChangeType.DELETE:
                entity.is_active = False
                changed_fields.append("is_active")

            elif change_request.change_type == ChangeType.ARCHIVE:
                entity.is_active = False
                entity.metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
                changed_fields.extend(["is_active", "metadata"])

            # バージョン更新
            entity.version += 1
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = change_request.approved_by or change_request.requested_by

            # 品質スコア再計算
            policy = self._get_applicable_policy(entity.entity_type)
            entity.quality_score = await self._calculate_entity_quality(entity.attributes, policy)

            # ゴールデンレコード状態再評価
            entity.is_golden_record = await self._determine_golden_record_status(entity)

            # データベース更新
            await self._update_entity(entity)

            # 変更履歴記録
            await self._record_change_history(
                entity.entity_id,
                change_request.change_type,
                old_version,
                entity.version,
                changed_fields,
                entity.updated_by,
                entity.updated_at,
                change_request.business_justification,
                {"request_id": change_request.request_id},
            )

        except Exception as e:
            self.logger.error(f"変更適用エラー: {e}")
            raise

    async def _calculate_entity_quality(
        self, attributes: Dict[str, Any], policy: Optional[DataGovernancePolicy]
    ) -> float:
        """エンティティ品質スコア計算"""
        try:
            if not policy:
                return 75.0  # デフォルトスコア

            total_score = 0.0
            max_score = 0.0

            for rule in policy.rules:
                field_name = rule.get("field")
                if not field_name:
                    continue

                max_score += 100
                field_value = attributes.get(field_name)

                # 必須フィールドチェック
                if rule.get("required", False):
                    if field_value is None or field_value == "":
                        continue  # 0点
                    total_score += 30

                # 一意性チェック（簡略化）
                if rule.get("unique", False) and field_value:
                    total_score += 20

                # パターンマッチングチェック
                if rule.get("pattern") and isinstance(field_value, str):
                    import re

                    if re.match(rule["pattern"], field_value):
                        total_score += 25

                # 長さチェック
                if rule.get("min_length") and isinstance(field_value, str):
                    if len(field_value) >= rule["min_length"]:
                        total_score += 15

                # 型チェック
                if rule.get("type") and field_value is not None:
                    expected_type = rule["type"]
                    if expected_type == "date":
                        try:
                            datetime.fromisoformat(str(field_value))
                            total_score += 10
                        except:
                            pass
                    elif expected_type == "number":
                        if isinstance(field_value, (int, float)):
                            total_score += 10

            if max_score > 0:
                return min(100.0, (total_score / max_score) * 100)
            else:
                return 85.0  # デフォルト

        except Exception as e:
            self.logger.error(f"品質スコア計算エラー: {e}")
            return 50.0

    async def _validate_entity_attributes(
        self, attributes: Dict[str, Any], policy: Optional[DataGovernancePolicy]
    ) -> Dict[str, Any]:
        """エンティティ属性バリデーション"""
        errors = []
        warnings = []

        try:
            if not policy:
                return {"is_valid": True, "errors": errors, "warnings": warnings}

            for rule in policy.rules:
                field_name = rule.get("field")
                if not field_name:
                    continue

                field_value = attributes.get(field_name)

                # 必須フィールドチェック
                if rule.get("required", False):
                    if field_value is None or field_value == "":
                        errors.append(f"必須フィールド '{field_name}' が未設定です")
                        continue

                if field_value is None:
                    continue

                # パターンマッチングチェック
                if rule.get("pattern") and isinstance(field_value, str):
                    import re

                    if not re.match(rule["pattern"], field_value):
                        errors.append(
                            f"フィールド '{field_name}' がパターン '{rule['pattern']}' と一致しません"
                        )

                # 長さチェック
                if rule.get("min_length") and isinstance(field_value, str):
                    if len(field_value) < rule["min_length"]:
                        errors.append(
                            f"フィールド '{field_name}' の長さが最小値 {rule['min_length']} 未満です"
                        )

                if rule.get("max_length") and isinstance(field_value, str):
                    if len(field_value) > rule["max_length"]:
                        warnings.append(
                            f"フィールド '{field_name}' の長さが推奨最大値 {rule['max_length']} を超えています"
                        )

            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
            }

        except Exception as e:
            self.logger.error(f"バリデーションエラー: {e}")
            return {
                "is_valid": False,
                "errors": [f"バリデーション実行エラー: {str(e)}"],
                "warnings": [],
            }

    async def _determine_golden_record_status(self, entity: MasterDataEntity) -> bool:
        """ゴールデンレコード状態判定"""
        try:
            # 基本判定条件
            quality_threshold = 90.0  # ゴールデンレコード品質閾値

            # 品質スコアチェック
            if entity.quality_score < quality_threshold:
                return False

            # 複数ソースからのデータ統合確認
            if len(entity.source_systems) < 1:
                return False

            # エンティティタイプ別の特別条件
            if entity.entity_type == MasterDataType.FINANCIAL_INSTRUMENTS:
                # 金融商品の場合、ISINコードが必須
                if "isin" not in entity.attributes or not entity.attributes["isin"]:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"ゴールデンレコード判定エラー: {e}")
            return False

    async def _assess_change_impact(
        self,
        entity: MasterDataEntity,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """変更影響評価"""
        try:
            impact = {
                "risk_level": "low",
                "affected_systems": entity.source_systems.copy(),
                "related_entities": entity.related_entities.copy(),
                "business_impact": "minimal",
                "technical_impact": "minimal",
                "recommendations": [],
            }

            # 変更タイプ別の影響評価
            if change_type == ChangeType.DELETE:
                impact["risk_level"] = "high"
                impact["business_impact"] = "high"
                impact["recommendations"].append("削除前に関連データの確認が必要")

            elif change_type == ChangeType.MERGE:
                impact["risk_level"] = "medium"
                impact["technical_impact"] = "medium"
                impact["recommendations"].append("データ統合後の整合性確認が必要")

            # 重要フィールドの変更チェック
            critical_fields = ["primary_key", "symbol", "isin", "code"]
            for field in critical_fields:
                if field in proposed_changes and field in entity.attributes:
                    if entity.attributes[field] != proposed_changes[field]:
                        impact["risk_level"] = "high"
                        impact["business_impact"] = "high"
                        impact["recommendations"].append(
                            f"重要フィールド '{field}' の変更は慎重な確認が必要"
                        )

            # 品質スコア予測
            if change_type == ChangeType.UPDATE:
                temp_attributes = entity.attributes.copy()
                temp_attributes.update(proposed_changes)
                policy = self._get_applicable_policy(entity.entity_type)
                predicted_quality = await self._calculate_entity_quality(temp_attributes, policy)

                impact["predicted_quality_score"] = predicted_quality

                if predicted_quality < entity.quality_score - 10:
                    impact["risk_level"] = "medium"
                    impact["recommendations"].append("品質スコア低下の可能性があります")

            return impact

        except Exception as e:
            self.logger.error(f"影響評価エラー: {e}")
            return {
                "risk_level": "unknown",
                "error": str(e),
                "recommendations": ["影響評価を手動で実施してください"],
            }

    def _get_applicable_policy(self, entity_type: MasterDataType) -> Optional[DataGovernancePolicy]:
        """適用可能なガバナンスポリシー取得"""
        for policy in self.governance_policies.values():
            if entity_type in policy.entity_types:
                return policy
        return None

    async def create_data_hierarchy(
        self,
        name: str,
        entity_type: MasterDataType,
        root_entity_id: str,
        level_definitions: Dict[int, str],
        created_by: str = "system",
    ) -> str:
        """データ階層作成"""
        try:
            hierarchy_id = f"hierarchy_{entity_type.value}_{int(time.time())}"

            hierarchy = MasterDataHierarchy(
                hierarchy_id=hierarchy_id,
                name=name,
                entity_type=entity_type,
                root_entity_id=root_entity_id,
                level_definitions=level_definitions,
                metadata={"created_by": created_by},
            )

            self.hierarchies[hierarchy_id] = hierarchy

            # データベース保存
            await self._save_hierarchy(hierarchy)

            self.logger.info(f"データ階層作成: {hierarchy_id}")
            return hierarchy_id

        except Exception as e:
            self.logger.error(f"階層作成エラー: {e}")
            raise

    async def get_data_catalog(
        self, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """データカタログ取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if entity_type:
                    cursor = conn.execute(
                        """
                        SELECT entity_type, COUNT(*) as total_count,
                               SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_count,
                               SUM(CASE WHEN is_golden_record THEN 1 ELSE 0 END) as golden_count,
                               AVG(quality_score) as avg_quality,
                               MIN(created_at) as earliest_created,
                               MAX(updated_at) as latest_updated
                        FROM master_data_entities
                        WHERE entity_type = ?
                        GROUP BY entity_type
                    """,
                        (entity_type.value,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT entity_type, COUNT(*) as total_count,
                               SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_count,
                               SUM(CASE WHEN is_golden_record THEN 1 ELSE 0 END) as golden_count,
                               AVG(quality_score) as avg_quality,
                               MIN(created_at) as earliest_created,
                               MAX(updated_at) as latest_updated
                        FROM master_data_entities
                        GROUP BY entity_type
                    """
                    )

                catalog_data = []
                for row in cursor.fetchall():
                    catalog_data.append(
                        {
                            "entity_type": row[0],
                            "total_count": row[1],
                            "active_count": row[2],
                            "golden_records": row[3],
                            "average_quality": round(row[4], 2) if row[4] else 0,
                            "earliest_created": row[5],
                            "latest_updated": row[6],
                        }
                    )

                # ガバナンス情報追加
                for item in catalog_data:
                    entity_type_enum = MasterDataType(item["entity_type"])
                    policy = self._get_applicable_policy(entity_type_enum)
                    if policy:
                        item["governance_policy"] = policy.policy_name
                        item["governance_level"] = policy.governance_level.value
                        item["requires_approval"] = policy.requires_approval

                return {
                    "catalog": catalog_data,
                    "total_entity_types": len(catalog_data),
                    "system_stats": self.stats,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"データカタログ取得エラー: {e}")
            return {"error": str(e)}

    async def get_entity_lineage(self, entity_id: str) -> Dict[str, Any]:
        """エンティティデータ系譜取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 変更履歴取得
                cursor = conn.execute(
                    """
                    SELECT change_type, old_version, new_version, changed_fields,
                           changed_by, changed_at, business_context, metadata
                    FROM change_history
                    WHERE entity_id = ?
                    ORDER BY changed_at DESC
                """,
                    (entity_id,),
                )

                change_history = []
                for row in cursor.fetchall():
                    change_history.append(
                        {
                            "change_type": row[0],
                            "old_version": row[1],
                            "new_version": row[2],
                            "changed_fields": json.loads(row[3]) if row[3] else [],
                            "changed_by": row[4],
                            "changed_at": row[5],
                            "business_context": row[6],
                            "metadata": json.loads(row[7]) if row[7] else {},
                        }
                    )

                # 品質履歴取得
                cursor = conn.execute(
                    """
                    SELECT quality_score, quality_issues, checked_at, checked_by
                    FROM quality_history
                    WHERE entity_id = ?
                    ORDER BY checked_at DESC LIMIT 20
                """,
                    (entity_id,),
                )

                quality_history = []
                for row in cursor.fetchall():
                    quality_history.append(
                        {
                            "quality_score": row[0],
                            "quality_issues": json.loads(row[1]) if row[1] else [],
                            "checked_at": row[2],
                            "checked_by": row[3],
                        }
                    )

                return {
                    "entity_id": entity_id,
                    "change_history": change_history,
                    "quality_history": quality_history,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"データ系譜取得エラー: {e}")
            return {"entity_id": entity_id, "error": str(e)}

    # データベースヘルパーメソッド
    async def _save_entity(self, entity: MasterDataEntity):
        """エンティティ保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO master_data_entities
                (entity_id, entity_type, primary_key, attributes, metadata, quality_score,
                 governance_level, version, created_at, updated_at, created_by, updated_by,
                 is_active, is_golden_record, source_systems, related_entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entity.entity_id,
                    entity.entity_type.value,
                    entity.primary_key,
                    json.dumps(entity.attributes, ensure_ascii=False),
                    json.dumps(entity.metadata, ensure_ascii=False),
                    entity.quality_score,
                    entity.governance_level.value,
                    entity.version,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.created_by,
                    entity.updated_by,
                    entity.is_active,
                    entity.is_golden_record,
                    json.dumps(entity.source_systems),
                    json.dumps(entity.related_entities),
                ),
            )
            conn.commit()

    async def _update_entity(self, entity: MasterDataEntity):
        """エンティティ更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE master_data_entities
                SET attributes = ?, metadata = ?, quality_score = ?, version = ?,
                    updated_at = ?, updated_by = ?, is_active = ?, is_golden_record = ?,
                    source_systems = ?, related_entities = ?
                WHERE entity_id = ?
            """,
                (
                    json.dumps(entity.attributes, ensure_ascii=False),
                    json.dumps(entity.metadata, ensure_ascii=False),
                    entity.quality_score,
                    entity.version,
                    entity.updated_at.isoformat(),
                    entity.updated_by,
                    entity.is_active,
                    entity.is_golden_record,
                    json.dumps(entity.source_systems),
                    json.dumps(entity.related_entities),
                    entity.entity_id,
                ),
            )
            conn.commit()

    async def _get_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """エンティティ取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM master_data_entities WHERE entity_id = ?
            """,
                (entity_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return MasterDataEntity(
                entity_id=row[0],
                entity_type=MasterDataType(row[1]),
                primary_key=row[2],
                attributes=json.loads(row[3]),
                metadata=json.loads(row[4]) if row[4] else {},
                quality_score=row[5],
                governance_level=DataGovernanceLevel(row[6]),
                version=row[7],
                created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9]),
                created_by=row[10],
                updated_by=row[11],
                is_active=bool(row[12]),
                is_golden_record=bool(row[13]),
                source_systems=json.loads(row[14]) if row[14] else [],
                related_entities=json.loads(row[15]) if row[15] else [],
            )

    async def _save_change_request(self, request: DataChangeRequest):
        """変更リクエスト保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_change_requests
                (request_id, entity_id, change_type, proposed_changes, business_justification,
                 requested_by, requested_at, approval_status, approved_by, approved_at,
                 rejection_reason, impact_assessment, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    request.request_id,
                    request.entity_id,
                    request.change_type.value,
                    json.dumps(request.proposed_changes, ensure_ascii=False),
                    request.business_justification,
                    request.requested_by,
                    request.requested_at.isoformat(),
                    request.approval_status.value,
                    request.approved_by,
                    request.approved_at.isoformat() if request.approved_at else None,
                    request.rejection_reason,
                    json.dumps(request.impact_assessment, ensure_ascii=False),
                    json.dumps(request.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()

    async def _get_change_request(self, request_id: str) -> Optional[DataChangeRequest]:
        """変更リクエスト取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM data_change_requests WHERE request_id = ?
            """,
                (request_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return DataChangeRequest(
                request_id=row[0],
                entity_id=row[1],
                change_type=ChangeType(row[2]),
                proposed_changes=json.loads(row[3]),
                business_justification=row[4],
                requested_by=row[5],
                requested_at=datetime.fromisoformat(row[6]),
                approval_status=ApprovalStatus(row[7]),
                approved_by=row[8],
                approved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                rejection_reason=row[10],
                impact_assessment=json.loads(row[11]) if row[11] else {},
                metadata=json.loads(row[12]) if row[12] else {},
            )

    async def _update_change_request(self, request: DataChangeRequest):
        """変更リクエスト更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE data_change_requests
                SET approval_status = ?, approved_by = ?, approved_at = ?,
                    rejection_reason = ?, metadata = ?
                WHERE request_id = ?
            """,
                (
                    request.approval_status.value,
                    request.approved_by,
                    request.approved_at.isoformat() if request.approved_at else None,
                    request.rejection_reason,
                    json.dumps(request.metadata, ensure_ascii=False),
                    request.request_id,
                ),
            )
            conn.commit()

    async def _record_change_history(
        self,
        entity_id: str,
        change_type: ChangeType,
        old_version: Optional[int],
        new_version: int,
        changed_fields: List[str],
        changed_by: str,
        changed_at: datetime,
        business_context: str,
        metadata: Dict[str, Any],
    ):
        """変更履歴記録"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO change_history
                (entity_id, change_type, old_version, new_version, changed_fields,
                 changed_by, changed_at, business_context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entity_id,
                    change_type.value,
                    old_version,
                    new_version,
                    json.dumps(changed_fields),
                    changed_by,
                    changed_at.isoformat(),
                    business_context,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            conn.commit()

    def _save_governance_policy(self, policy: DataGovernancePolicy):
        """ガバナンスポリシー保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO governance_policies
                (policy_id, policy_name, entity_types, rules, governance_level,
                 requires_approval, approval_roles, retention_period, audit_required,
                 quality_threshold, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    policy.policy_id,
                    policy.policy_name,
                    json.dumps([et.value for et in policy.entity_types]),
                    json.dumps(policy.rules, ensure_ascii=False),
                    policy.governance_level.value,
                    policy.requires_approval,
                    json.dumps(policy.approval_roles),
                    policy.retention_period,
                    policy.audit_required,
                    policy.quality_threshold,
                    policy.created_by,
                    policy.created_at.isoformat(),
                ),
            )
            conn.commit()

    async def _save_hierarchy(self, hierarchy: MasterDataHierarchy):
        """階層保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_hierarchies
                (hierarchy_id, name, entity_type, root_entity_id, parent_child_mapping,
                 level_definitions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    hierarchy.hierarchy_id,
                    hierarchy.name,
                    hierarchy.entity_type.value,
                    hierarchy.root_entity_id,
                    json.dumps(hierarchy.parent_child_mapping, ensure_ascii=False),
                    json.dumps(hierarchy.level_definitions, ensure_ascii=False),
                    json.dumps(hierarchy.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()


# Factory function
def create_enterprise_mdm_system(
    db_path: str = "enterprise_mdm.db",
) -> EnterpriseMasterDataManagement:
    """エンタープライズMDMシステム作成"""
    return EnterpriseMasterDataManagement(db_path)


if __name__ == "__main__":
    # テスト実行
    async def test_enterprise_mdm_system():
        print("=== エンタープライズマスターデータ管理システムテスト ===")

        try:
            # システム初期化
            mdm_system = create_enterprise_mdm_system("test_enterprise_mdm.db")

            print("\n1. MDMシステム初期化完了")
            print(f"   ガバナンスポリシー数: {len(mdm_system.governance_policies)}")

            # 金融商品マスターデータ登録
            print("\n2. 金融商品マスターデータ登録...")

            stock_entities = []

            # TOPIX500銘柄サンプル
            sample_stocks = [
                {
                    "symbol": "7203",
                    "name": "トヨタ自動車",
                    "isin": "JP3633400001",
                    "market": "TSE",
                    "sector": "自動車",
                },
                {
                    "symbol": "9984",
                    "name": "ソフトバンクグループ",
                    "isin": "JP3436100006",
                    "market": "TSE",
                    "sector": "情報通信",
                },
                {
                    "symbol": "6758",
                    "name": "ソニーグループ",
                    "isin": "JP3435000009",
                    "market": "TSE",
                    "sector": "電気機器",
                },
                {
                    "symbol": "9432",
                    "name": "日本電信電話",
                    "isin": "JP3432600004",
                    "market": "TSE",
                    "sector": "情報通信",
                },
            ]

            for stock in sample_stocks:
                entity_id = await mdm_system.register_master_data_entity(
                    entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
                    primary_key=stock["symbol"],
                    attributes=stock,
                    source_system="topix_master",
                    created_by="data_admin",
                )
                stock_entities.append(entity_id)
                print(f"   登録完了: {stock['symbol']} - {entity_id}")

            # 市場コードマスター登録
            print("\n3. 市場コードマスター登録...")

            market_codes = [
                {"code": "TSE", "description": "東京証券取引所", "country": "JP"},
                {"code": "OSE", "description": "大阪証券取引所", "country": "JP"},
                {
                    "code": "NYSE",
                    "description": "ニューヨーク証券取引所",
                    "country": "US",
                },
                {"code": "NASDAQ", "description": "ナスダック", "country": "US"},
            ]

            for market in market_codes:
                entity_id = await mdm_system.register_master_data_entity(
                    entity_type=MasterDataType.EXCHANGE_CODES,
                    primary_key=market["code"],
                    attributes=market,
                    source_system="market_reference",
                    created_by="data_admin",
                )
                print(f"   市場コード登録: {market['code']} - {entity_id}")

            # データ変更リクエストテスト
            print("\n4. データ変更リクエストテスト...")

            change_request_id = await mdm_system.request_data_change(
                entity_id=stock_entities[0],
                change_type=ChangeType.UPDATE,
                proposed_changes={
                    "sector": "自動車・輸送機器",
                    "market_cap": 35000000000000,
                },
                business_justification="業界分類の詳細化とマーケットキャップ情報追加",
                requested_by="business_analyst",
            )

            print(f"   変更リクエスト作成: {change_request_id}")

            # 変更リクエスト承認
            approval_result = await mdm_system.approve_change_request(
                change_request_id, "data_steward", "業界分類詳細化は適切。承認します。"
            )

            print(f"   変更リクエスト承認: {'成功' if approval_result else '失敗'}")

            # データ階層作成テスト
            print("\n5. データ階層作成テスト...")

            hierarchy_id = await mdm_system.create_data_hierarchy(
                name="業界分類階層",
                entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
                root_entity_id="industry_root",
                level_definitions={1: "大分類", 2: "中分類", 3: "小分類"},
                created_by="data_architect",
            )

            print(f"   データ階層作成: {hierarchy_id}")

            # データカタログ取得
            print("\n6. データカタログ取得...")

            catalog = await mdm_system.get_data_catalog()
            print(f"   エンティティタイプ数: {catalog['total_entity_types']}")
            print("   システム統計:")

            for key, value in catalog["system_stats"].items():
                print(f"     {key}: {value}")

            print("   カタログ詳細:")
            for item in catalog["catalog"]:
                print(
                    f"     {item['entity_type']}: {item['total_count']}件 (ゴールデン: {item['golden_records']}件)"
                )
                print(
                    f"       平均品質: {item['average_quality']}, ガバナンス: {item.get('governance_level', 'N/A')}"
                )

            # エンティティ系譜取得
            print("\n7. エンティティデータ系譜取得...")

            lineage = await mdm_system.get_entity_lineage(stock_entities[0])
            print(f"   変更履歴: {len(lineage['change_history'])}件")
            print(f"   品質履歴: {len(lineage['quality_history'])}件")

            if lineage["change_history"]:
                latest_change = lineage["change_history"][0]
                print(
                    f"   最新変更: {latest_change['change_type']} by {latest_change['changed_by']}"
                )
                print(f"   変更フィールド: {latest_change['changed_fields']}")

            print("\n[成功] エンタープライズマスターデータ管理システムテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_enterprise_mdm_system())

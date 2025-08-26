#!/usr/bin/env python3
"""
マスターデータ管理（MDM）共通型定義
データクラスと列挙型定義
"""

import warnings
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
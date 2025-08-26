#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - データモデル定義

このモジュールは、MDMシステムで使用されるデータクラスとモデルを定義します。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .enums import (
    ApprovalStatus,
    ChangeType,
    DataGovernanceLevel,
    MasterDataType,
)


@dataclass
class MasterDataEntity:
    """マスターデータエンティティ
    
    MDMシステムで管理される基本的なデータエンティティを表現します。
    """
    
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

    def __post_init__(self):
        """初期化後の処理"""
        if not self.entity_id:
            raise ValueError("entity_id は必須です")
        if not self.primary_key:
            raise ValueError("primary_key は必須です")
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換
        
        Returns:
            Dict[str, Any]: エンティティの辞書表現
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "primary_key": self.primary_key,
            "attributes": self.attributes,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "governance_level": self.governance_level.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "is_active": self.is_active,
            "is_golden_record": self.is_golden_record,
            "source_systems": self.source_systems,
            "related_entities": self.related_entities,
        }


@dataclass
class DataChangeRequest:
    """データ変更リクエスト
    
    マスターデータの変更要求を管理するためのモデルです。
    """
    
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

    def __post_init__(self):
        """初期化後の処理"""
        if not self.request_id:
            raise ValueError("request_id は必須です")
        if not self.entity_id:
            raise ValueError("entity_id は必須です")
        if not self.business_justification:
            raise ValueError("business_justification は必須です")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換
        
        Returns:
            Dict[str, Any]: 変更リクエストの辞書表現
        """
        return {
            "request_id": self.request_id,
            "entity_id": self.entity_id,
            "change_type": self.change_type.value,
            "proposed_changes": self.proposed_changes,
            "business_justification": self.business_justification,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "approval_status": self.approval_status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "impact_assessment": self.impact_assessment,
            "metadata": self.metadata,
        }


@dataclass
class DataGovernancePolicy:
    """データガバナンスポリシー
    
    マスターデータのガバナンス規則と承認フローを定義します。
    """
    
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

    def __post_init__(self):
        """初期化後の処理"""
        if not self.policy_id:
            raise ValueError("policy_id は必須です")
        if not self.policy_name:
            raise ValueError("policy_name は必須です")
        if not self.entity_types:
            raise ValueError("entity_types は必須です")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換
        
        Returns:
            Dict[str, Any]: ポリシーの辞書表現
        """
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "entity_types": [et.value for et in self.entity_types],
            "rules": self.rules,
            "governance_level": self.governance_level.value,
            "requires_approval": self.requires_approval,
            "approval_roles": self.approval_roles,
            "retention_period": self.retention_period,
            "audit_required": self.audit_required,
            "quality_threshold": self.quality_threshold,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MasterDataHierarchy:
    """マスターデータ階層
    
    マスターデータ間の階層関係を管理します。
    """
    
    hierarchy_id: str
    name: str
    entity_type: MasterDataType
    root_entity_id: str
    parent_child_mapping: Dict[str, List[str]] = field(default_factory=dict)
    level_definitions: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後の処理"""
        if not self.hierarchy_id:
            raise ValueError("hierarchy_id は必須です")
        if not self.name:
            raise ValueError("name は必須です")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換
        
        Returns:
            Dict[str, Any]: 階層の辞書表現
        """
        return {
            "hierarchy_id": self.hierarchy_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "root_entity_id": self.root_entity_id,
            "parent_child_mapping": self.parent_child_mapping,
            "level_definitions": self.level_definitions,
            "metadata": self.metadata,
        }

    def get_children(self, parent_id: str) -> List[str]:
        """子エンティティIDリストを取得
        
        Args:
            parent_id: 親エンティティID
            
        Returns:
            List[str]: 子エンティティIDのリスト
        """
        return self.parent_child_mapping.get(parent_id, [])

    def add_child(self, parent_id: str, child_id: str):
        """子エンティティを追加
        
        Args:
            parent_id: 親エンティティID
            child_id: 子エンティティID
        """
        if parent_id not in self.parent_child_mapping:
            self.parent_child_mapping[parent_id] = []
        if child_id not in self.parent_child_mapping[parent_id]:
            self.parent_child_mapping[parent_id].append(child_id)


@dataclass
class QualityMetric:
    """品質メトリック
    
    データ品質の測定結果を表現します。
    """
    
    metric_name: str
    score: float
    weight: float = 1.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後の処理"""
        if not 0 <= self.score <= 100:
            raise ValueError("score は 0-100 の範囲である必要があります")
        if self.weight < 0:
            raise ValueError("weight は非負数である必要があります")


@dataclass
class ChangeHistory:
    """変更履歴
    
    マスターデータの変更履歴を記録します。
    """
    
    id: Optional[int]
    entity_id: str
    change_type: ChangeType
    old_version: Optional[int]
    new_version: int
    changed_fields: List[str]
    changed_by: str
    changed_at: datetime
    business_context: str
    approval_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換
        
        Returns:
            Dict[str, Any]: 変更履歴の辞書表現
        """
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "change_type": self.change_type.value,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "changed_fields": self.changed_fields,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at.isoformat(),
            "business_context": self.business_context,
            "approval_reference": self.approval_reference,
            "metadata": self.metadata,
        }


@dataclass
class QualityHistory:
    """品質履歴
    
    データ品質の履歴を記録します。
    """
    
    id: Optional[int]
    entity_id: str
    quality_score: float
    quality_issues: List[str]
    quality_metrics: Dict[str, QualityMetric]
    checked_at: datetime
    checked_by: Optional[str] = None
    remediation_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換
        
        Returns:
            Dict[str, Any]: 品質履歴の辞書表現
        """
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "quality_score": self.quality_score,
            "quality_issues": self.quality_issues,
            "quality_metrics": {
                name: {
                    "metric_name": metric.metric_name,
                    "score": metric.score,
                    "weight": metric.weight,
                    "issues": metric.issues,
                    "recommendations": metric.recommendations,
                    "metadata": metric.metadata,
                }
                for name, metric in self.quality_metrics.items()
            },
            "checked_at": self.checked_at.isoformat(),
            "checked_by": self.checked_by,
            "remediation_actions": self.remediation_actions,
            "metadata": self.metadata,
        }
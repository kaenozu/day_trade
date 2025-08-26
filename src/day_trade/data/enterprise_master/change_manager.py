#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - 変更管理

このモジュールは、データ変更リクエストの管理と承認フローを担当します。
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .enums import ApprovalStatus, ChangeType, RiskLevel
from .models import DataChangeRequest, MasterDataEntity
from .governance_manager import GovernanceManager
from .entity_manager import EntityManager


class ChangeManager:
    """変更管理クラス
    
    データ変更リクエストの作成、承認、適用を担当します。
    """
    
    def __init__(
        self,
        governance_manager: GovernanceManager,
        entity_manager: EntityManager
    ):
        self.logger = logging.getLogger(__name__)
        self.governance_manager = governance_manager
        self.entity_manager = entity_manager
        self._change_requests: Dict[str, DataChangeRequest] = {}

    async def create_change_request(
        self,
        entity_id: str,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
        business_justification: str,
        requested_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """データ変更リクエスト作成
        
        Args:
            entity_id: 対象エンティティID
            change_type: 変更タイプ
            proposed_changes: 提案する変更内容
            business_justification: ビジネス上の根拠
            requested_by: リクエスト者
            metadata: 追加メタデータ
            
        Returns:
            Tuple[bool, str, Optional[str]]: (成功フラグ, リクエストID, エラーメッセージ)
        """
        try:
            metadata = metadata or {}
            current_time = datetime.now(timezone.utc)

            # エンティティ存在確認
            entity = await self.entity_manager.get_entity(entity_id)
            if not entity:
                return False, "", f"エンティティが見つかりません: {entity_id}"

            # リクエストID生成
            request_id = f"req_{entity_id}_{change_type.value}_{int(time.time())}"

            # 承認要否確認
            requires_approval = self.governance_manager.check_approval_required(
                entity.entity_type, change_type.value
            )

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

            # 保存
            self._change_requests[request_id] = change_request

            self.logger.info(f"変更リクエスト作成: {request_id}")
            return True, request_id, None

        except Exception as e:
            self.logger.error(f"変更リクエスト作成エラー: {e}")
            return False, "", str(e)

    async def approve_change_request(
        self,
        request_id: str,
        approved_by: str,
        approval_notes: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """変更リクエスト承認
        
        Args:
            request_id: リクエストID
            approved_by: 承認者
            approval_notes: 承認メモ
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            change_request = self._change_requests.get(request_id)
            if not change_request:
                return False, f"変更リクエストが見つかりません: {request_id}"

            if change_request.approval_status != ApprovalStatus.PENDING:
                return False, f"リクエストは承認待ち状態ではありません: {change_request.approval_status.value}"

            current_time = datetime.now(timezone.utc)

            # 承認情報更新
            change_request.approval_status = ApprovalStatus.APPROVED
            change_request.approved_by = approved_by
            change_request.approved_at = current_time
            change_request.metadata["approval_notes"] = approval_notes

            # 変更適用
            success, error_msg = await self._apply_approved_changes(change_request)
            if not success:
                return False, f"変更適用エラー: {error_msg}"

            self.logger.info(f"変更リクエスト承認: {request_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"変更リクエスト承認エラー: {e}")
            return False, str(e)

    async def reject_change_request(
        self,
        request_id: str,
        rejected_by: str,
        rejection_reason: str
    ) -> Tuple[bool, Optional[str]]:
        """変更リクエスト却下
        
        Args:
            request_id: リクエストID
            rejected_by: 却下者
            rejection_reason: 却下理由
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            change_request = self._change_requests.get(request_id)
            if not change_request:
                return False, f"変更リクエストが見つかりません: {request_id}"

            if change_request.approval_status != ApprovalStatus.PENDING:
                return False, f"リクエストは承認待ち状態ではありません: {change_request.approval_status.value}"

            # 却下情報更新
            change_request.approval_status = ApprovalStatus.REJECTED
            change_request.approved_by = rejected_by
            change_request.approved_at = datetime.now(timezone.utc)
            change_request.rejection_reason = rejection_reason

            self.logger.info(f"変更リクエスト却下: {request_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"変更リクエスト却下エラー: {e}")
            return False, str(e)

    async def withdraw_change_request(
        self, request_id: str, withdrawn_by: str
    ) -> Tuple[bool, Optional[str]]:
        """変更リクエスト取り下げ
        
        Args:
            request_id: リクエストID
            withdrawn_by: 取り下げ実行者
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            change_request = self._change_requests.get(request_id)
            if not change_request:
                return False, f"変更リクエストが見つかりません: {request_id}"

            if change_request.approval_status != ApprovalStatus.PENDING:
                return False, f"承認待ち以外のリクエストは取り下げできません: {change_request.approval_status.value}"

            # 取り下げ情報更新
            change_request.approval_status = ApprovalStatus.WITHDRAWN
            change_request.metadata["withdrawn_by"] = withdrawn_by
            change_request.metadata["withdrawn_at"] = datetime.now(timezone.utc).isoformat()

            self.logger.info(f"変更リクエスト取り下げ: {request_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"変更リクエスト取り下げエラー: {e}")
            return False, str(e)

    async def get_change_request(self, request_id: str) -> Optional[DataChangeRequest]:
        """変更リクエスト取得
        
        Args:
            request_id: リクエストID
            
        Returns:
            Optional[DataChangeRequest]: 変更リクエスト
        """
        return self._change_requests.get(request_id)

    async def list_change_requests(
        self,
        approval_status: Optional[ApprovalStatus] = None,
        entity_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        limit: int = 100
    ) -> List[DataChangeRequest]:
        """変更リクエスト一覧取得
        
        Args:
            approval_status: 承認状態フィルタ
            entity_id: エンティティIDフィルタ
            requested_by: リクエスト者フィルタ
            limit: 取得件数制限
            
        Returns:
            List[DataChangeRequest]: 変更リクエスト一覧
        """
        try:
            requests = []
            
            for request in self._change_requests.values():
                # フィルタ適用
                if approval_status and request.approval_status != approval_status:
                    continue
                
                if entity_id and request.entity_id != entity_id:
                    continue
                
                if requested_by and request.requested_by != requested_by:
                    continue
                
                requests.append(request)
            
            # リクエスト日時の降順でソート
            requests.sort(key=lambda r: r.requested_at, reverse=True)
            
            return requests[:limit]

        except Exception as e:
            self.logger.error(f"変更リクエスト一覧取得エラー: {e}")
            return []

    async def _apply_approved_changes(
        self, change_request: DataChangeRequest
    ) -> Tuple[bool, Optional[str]]:
        """承認済み変更の適用
        
        Args:
            change_request: 変更リクエスト
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            entity = await self.entity_manager.get_entity(change_request.entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {change_request.entity_id}"

            if change_request.change_type == ChangeType.UPDATE:
                # 属性更新
                success, error_msg = await self.entity_manager.update_entity(
                    change_request.entity_id,
                    change_request.proposed_changes,
                    change_request.approved_by or change_request.requested_by,
                    {"change_request_id": change_request.request_id}
                )
                return success, error_msg

            elif change_request.change_type == ChangeType.DELETE:
                # エンティティ削除
                success, error_msg = await self.entity_manager.delete_entity(
                    change_request.entity_id,
                    change_request.approved_by or change_request.requested_by
                )
                return success, error_msg

            elif change_request.change_type == ChangeType.ARCHIVE:
                # エンティティアーカイブ
                success, error_msg = await self.entity_manager.archive_entity(
                    change_request.entity_id,
                    change_request.approved_by or change_request.requested_by
                )
                return success, error_msg

            else:
                return False, f"未対応の変更タイプ: {change_request.change_type.value}"

        except Exception as e:
            self.logger.error(f"変更適用エラー: {e}")
            return False, str(e)

    async def _assess_change_impact(
        self,
        entity: MasterDataEntity,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """変更影響評価
        
        Args:
            entity: 対象エンティティ
            change_type: 変更タイプ
            proposed_changes: 提案する変更
            
        Returns:
            Dict[str, Any]: 影響評価結果
        """
        try:
            impact = {
                "risk_level": RiskLevel.LOW.value,
                "affected_systems": entity.source_systems.copy(),
                "related_entities": entity.related_entities.copy(),
                "business_impact": "minimal",
                "technical_impact": "minimal",
                "recommendations": [],
                "estimated_effort": "low",
                "rollback_complexity": "simple",
            }

            # 変更タイプ別の影響評価
            if change_type == ChangeType.DELETE:
                impact["risk_level"] = RiskLevel.HIGH.value
                impact["business_impact"] = "high"
                impact["technical_impact"] = "high"
                impact["rollback_complexity"] = "complex"
                impact["recommendations"].append("削除前に関連データの確認が必要")
                impact["recommendations"].append("バックアップの作成を推奨")

            elif change_type == ChangeType.MERGE:
                impact["risk_level"] = RiskLevel.MEDIUM.value
                impact["technical_impact"] = "medium"
                impact["estimated_effort"] = "medium"
                impact["rollback_complexity"] = "complex"
                impact["recommendations"].append("データ統合後の整合性確認が必要")

            elif change_type == ChangeType.SPLIT:
                impact["risk_level"] = RiskLevel.HIGH.value
                impact["technical_impact"] = "high"
                impact["estimated_effort"] = "high"
                impact["rollback_complexity"] = "very_complex"
                impact["recommendations"].append("データ分割の設計を慎重に検討してください")

            # 重要フィールドの変更チェック
            critical_fields = self._get_critical_fields(entity.entity_type)
            for field in critical_fields:
                if field in proposed_changes and field in entity.attributes:
                    if entity.attributes[field] != proposed_changes[field]:
                        impact["risk_level"] = RiskLevel.HIGH.value
                        impact["business_impact"] = "high"
                        impact["recommendations"].append(
                            f"重要フィールド '{field}' の変更は慎重な確認が必要"
                        )

            # ゴールデンレコードの変更
            if entity.is_golden_record:
                impact["risk_level"] = RiskLevel.MEDIUM.value
                impact["business_impact"] = "medium"
                impact["recommendations"].append("ゴールデンレコードの変更は影響範囲が広いため注意が必要")

            # 品質スコア予測
            if change_type == ChangeType.UPDATE:
                predicted_impact = self._predict_quality_impact(entity, proposed_changes)
                impact.update(predicted_impact)

            return impact

        except Exception as e:
            self.logger.error(f"影響評価エラー: {e}")
            return {
                "risk_level": RiskLevel.UNKNOWN.value,
                "error": str(e),
                "recommendations": ["影響評価を手動で実施してください"],
            }

    def _get_critical_fields(self, entity_type) -> List[str]:
        """エンティティタイプの重要フィールド取得
        
        Args:
            entity_type: エンティティタイプ
            
        Returns:
            List[str]: 重要フィールドリスト
        """
        critical_field_map = {
            "financial_instruments": ["symbol", "isin", "name", "market"],
            "exchange_codes": ["code", "description"],
            "currency_codes": ["code", "name"],
            "industry_codes": ["code", "name"],
        }
        
        return critical_field_map.get(entity_type.value, ["primary_key", "name"])

    def _predict_quality_impact(
        self, entity: MasterDataEntity, proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """品質影響予測
        
        Args:
            entity: エンティティ
            proposed_changes: 提案する変更
            
        Returns:
            Dict[str, Any]: 品質影響予測結果
        """
        try:
            # 簡単な品質影響予測ロジック
            quality_impact = {
                "current_quality_score": entity.quality_score,
                "predicted_quality_change": 0,
                "quality_concerns": []
            }

            # 必須フィールドの削除チェック
            policy = self.governance_manager.get_applicable_policy(entity.entity_type)
            if policy:
                for rule in policy.rules:
                    field = rule.get("field")
                    if rule.get("required", False) and field in proposed_changes:
                        new_value = proposed_changes[field]
                        if new_value is None or new_value == "":
                            quality_impact["predicted_quality_change"] -= 20
                            quality_impact["quality_concerns"].append(
                                f"必須フィールド '{field}' が空になります"
                            )

            # データ型の不整合チェック
            for field, new_value in proposed_changes.items():
                current_value = entity.attributes.get(field)
                if current_value is not None and type(current_value) != type(new_value):
                    quality_impact["predicted_quality_change"] -= 5
                    quality_impact["quality_concerns"].append(
                        f"フィールド '{field}' のデータ型が変更されます"
                    )

            quality_impact["predicted_quality_score"] = max(
                0, entity.quality_score + quality_impact["predicted_quality_change"]
            )

            return quality_impact

        except Exception as e:
            self.logger.error(f"品質影響予測エラー: {e}")
            return {
                "current_quality_score": entity.quality_score,
                "predicted_quality_change": 0,
                "quality_concerns": ["品質影響の予測に失敗しました"]
            }

    async def get_change_statistics(self) -> Dict[str, Any]:
        """変更統計情報取得
        
        Returns:
            Dict[str, Any]: 変更統計情報
        """
        try:
            stats = {
                "total_requests": len(self._change_requests),
                "by_status": {status.value: 0 for status in ApprovalStatus},
                "by_change_type": {change_type.value: 0 for change_type in ChangeType},
                "by_risk_level": {},
                "average_approval_time_hours": 0.0,
                "pending_high_risk": 0,
            }

            approval_times = []
            
            for request in self._change_requests.values():
                # 状態別集計
                stats["by_status"][request.approval_status.value] += 1
                
                # 変更タイプ別集計
                stats["by_change_type"][request.change_type.value] += 1
                
                # リスクレベル別集計
                risk_level = request.impact_assessment.get("risk_level", "unknown")
                if risk_level not in stats["by_risk_level"]:
                    stats["by_risk_level"][risk_level] = 0
                stats["by_risk_level"][risk_level] += 1
                
                # 高リスクで承認待ちの件数
                if (request.approval_status == ApprovalStatus.PENDING and 
                    risk_level in ["high", "critical"]):
                    stats["pending_high_risk"] += 1
                
                # 承認時間計算
                if request.approved_at and request.approval_status == ApprovalStatus.APPROVED:
                    approval_time = (request.approved_at - request.requested_at).total_seconds() / 3600
                    approval_times.append(approval_time)
            
            # 平均承認時間
            if approval_times:
                stats["average_approval_time_hours"] = sum(approval_times) / len(approval_times)
            
            return stats

        except Exception as e:
            self.logger.error(f"変更統計情報取得エラー: {e}")
            return {"error": str(e)}
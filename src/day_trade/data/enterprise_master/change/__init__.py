#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - 変更管理モジュール

変更管理機能の統合インターフェース
"""

from .change_request import ChangeRequestManager
from .change_statistics import ChangeStatistics  
from .impact_assessment import ImpactAssessment

__all__ = [
    "ChangeRequestManager",
    "ChangeStatistics",
    "ImpactAssessment",
]


class ChangeManager:
    """統合変更管理クラス"""
    
    def __init__(self, governance_manager, entity_manager):
        # 各コンポーネント初期化
        self.impact_assessor = ImpactAssessment(governance_manager)
        self.request_manager = ChangeRequestManager(
            governance_manager, entity_manager, self.impact_assessor
        )
        self.statistics = ChangeStatistics(self.request_manager._change_requests)
    
    # 変更リクエスト管理の委譲
    async def create_change_request(self, *args, **kwargs):
        return await self.request_manager.create_change_request(*args, **kwargs)
    
    async def approve_change_request(self, *args, **kwargs):
        return await self.request_manager.approve_change_request(*args, **kwargs)
    
    async def reject_change_request(self, *args, **kwargs):
        return await self.request_manager.reject_change_request(*args, **kwargs)
    
    async def withdraw_change_request(self, *args, **kwargs):
        return await self.request_manager.withdraw_change_request(*args, **kwargs)
    
    async def get_change_request(self, request_id: str):
        return await self.request_manager.get_change_request(request_id)
    
    async def list_change_requests(self, *args, **kwargs):
        return await self.request_manager.list_change_requests(*args, **kwargs)
    
    async def get_pending_requests_by_priority(self):
        return await self.request_manager.get_pending_requests_by_priority()
    
    async def get_change_request_history(self, entity_id: str):
        return await self.request_manager.get_change_request_history(entity_id)
    
    async def bulk_approve_requests(self, *args, **kwargs):
        return await self.request_manager.bulk_approve_requests(*args, **kwargs)
    
    # 影響評価の委譲
    async def assess_change_impact(self, *args, **kwargs):
        return await self.impact_assessor.assess_change_impact(*args, **kwargs)
    
    async def assess_system_wide_impact(self, *args, **kwargs):
        return await self.impact_assessor.assess_system_wide_impact(*args, **kwargs)
    
    async def generate_risk_matrix(self, *args, **kwargs):
        return await self.impact_assessor.generate_risk_matrix(*args, **kwargs)
    
    # 統計情報の委譲
    async def get_change_statistics(self):
        return await self.statistics.get_change_statistics()
    
    async def get_trend_analysis(self, *args, **kwargs):
        return await self.statistics.get_trend_analysis(*args, **kwargs)
    
    async def get_approval_performance(self):
        return await self.statistics.get_approval_performance()
    
    async def get_rejection_analysis(self):
        return await self.statistics.get_rejection_analysis()
    
    async def get_workload_analysis(self):
        return await self.statistics.get_workload_analysis()

    # 変更適用機能
    async def apply_approved_changes(self, change_request):
        """承認済み変更の適用"""
        from ..enums import ChangeType
        
        try:
            entity = await self.request_manager.entity_manager.get_entity(
                change_request.entity_id
            )
            if not entity:
                return False, f"エンティティが見つかりません: {change_request.entity_id}"

            if change_request.change_type == ChangeType.UPDATE:
                # 属性更新
                success, error_msg = await self.request_manager.entity_manager.update_entity(
                    change_request.entity_id,
                    change_request.proposed_changes,
                    change_request.approved_by or change_request.requested_by,
                    {"change_request_id": change_request.request_id}
                )
                return success, error_msg

            elif change_request.change_type == ChangeType.DELETE:
                # エンティティ削除
                success, error_msg = await self.request_manager.entity_manager.delete_entity(
                    change_request.entity_id,
                    change_request.approved_by or change_request.requested_by
                )
                return success, error_msg

            elif change_request.change_type == ChangeType.ARCHIVE:
                # エンティティアーカイブ
                success, error_msg = await self.request_manager.entity_manager.archive_entity(
                    change_request.entity_id,
                    change_request.approved_by or change_request.requested_by
                )
                return success, error_msg

            else:
                return False, f"未対応の変更タイプ: {change_request.change_type.value}"

        except Exception as e:
            return False, str(e)
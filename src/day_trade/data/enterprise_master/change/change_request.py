#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - 変更リクエスト管理

変更リクエストの作成、承認、却下機能を提供
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..enums import ApprovalStatus, ChangeType
from ..models import DataChangeRequest


class ChangeRequestManager:
    """変更リクエスト管理クラス"""

    def __init__(self, governance_manager, entity_manager, impact_assessor):
        self.logger = logging.getLogger(__name__)
        self.governance_manager = governance_manager
        self.entity_manager = entity_manager
        self.impact_assessor = impact_assessor
        self._change_requests: Dict[str, DataChangeRequest] = {}

    async def create_change_request(
        self,
        entity_id: str,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
        business_justification: str,
        requested_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """データ変更リクエスト作成"""
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
            impact_assessment = await self.impact_assessor.assess_change_impact(
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

            # 保存
            self._change_requests[request_id] = change_request

            self.logger.info(f"変更リクエスト作成: {request_id}")
            return True, request_id, None

        except Exception as e:
            self.logger.error(f"変更リクエスト作成エラー: {e}")
            return False, "", str(e)

    async def approve_change_request(
        self, request_id: str, approved_by: str, approval_notes: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """変更リクエスト承認"""
        try:
            change_request = self._change_requests.get(request_id)
            if not change_request:
                return False, f"変更リクエストが見つかりません: {request_id}"

            if change_request.approval_status != ApprovalStatus.PENDING:
                return (
                    False,
                    f"リクエストは承認待ち状態ではありません: {change_request.approval_status.value}",
                )

            current_time = datetime.now(timezone.utc)

            # 承認情報更新
            change_request.approval_status = ApprovalStatus.APPROVED
            change_request.approved_by = approved_by
            change_request.approved_at = current_time
            change_request.metadata["approval_notes"] = approval_notes

            self.logger.info(f"変更リクエスト承認: {request_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"変更リクエスト承認エラー: {e}")
            return False, str(e)

    async def reject_change_request(
        self, request_id: str, rejected_by: str, rejection_reason: str
    ) -> Tuple[bool, Optional[str]]:
        """変更リクエスト却下"""
        try:
            change_request = self._change_requests.get(request_id)
            if not change_request:
                return False, f"変更リクエストが見つかりません: {request_id}"

            if change_request.approval_status != ApprovalStatus.PENDING:
                return (
                    False,
                    f"リクエストは承認待ち状態ではありません: {change_request.approval_status.value}",
                )

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
        """変更リクエスト取り下げ"""
        try:
            change_request = self._change_requests.get(request_id)
            if not change_request:
                return False, f"変更リクエストが見つかりません: {request_id}"

            if change_request.approval_status != ApprovalStatus.PENDING:
                return (
                    False,
                    f"承認待ち以外のリクエストは取り下げできません: {change_request.approval_status.value}",
                )

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
        """変更リクエスト取得"""
        return self._change_requests.get(request_id)

    async def list_change_requests(
        self,
        approval_status: Optional[ApprovalStatus] = None,
        entity_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        limit: int = 100,
    ) -> List[DataChangeRequest]:
        """変更リクエスト一覧取得"""
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

    async def get_pending_requests_by_priority(self) -> List[DataChangeRequest]:
        """優先度順の承認待ちリクエスト取得"""
        try:
            pending_requests = [
                req
                for req in self._change_requests.values()
                if req.approval_status == ApprovalStatus.PENDING
            ]

            # 優先度順でソート（高リスク、古い順）
            def priority_key(req):
                risk_level_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                risk_level = req.impact_assessment.get("risk_level", "low")
                return (risk_level_priority.get(risk_level, 1), -req.requested_at.timestamp())

            pending_requests.sort(key=priority_key, reverse=True)
            return pending_requests

        except Exception as e:
            self.logger.error(f"優先度順承認待ちリクエスト取得エラー: {e}")
            return []

    async def get_change_request_history(self, entity_id: str) -> List[DataChangeRequest]:
        """エンティティの変更リクエスト履歴取得"""
        try:
            entity_requests = [
                req
                for req in self._change_requests.values()
                if req.entity_id == entity_id
            ]

            # リクエスト日時の降順でソート
            entity_requests.sort(key=lambda r: r.requested_at, reverse=True)
            return entity_requests

        except Exception as e:
            self.logger.error(f"変更リクエスト履歴取得エラー: {e}")
            return []

    async def bulk_approve_requests(
        self, request_ids: List[str], approved_by: str, approval_notes: str = ""
    ) -> Dict[str, Any]:
        """一括承認"""
        try:
            results = {"successful": 0, "failed": 0, "errors": []}

            for request_id in request_ids:
                success, error_msg = await self.approve_change_request(
                    request_id, approved_by, approval_notes
                )

                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({"request_id": request_id, "error": error_msg})

            return results

        except Exception as e:
            self.logger.error(f"一括承認エラー: {e}")
            return {"successful": 0, "failed": len(request_ids), "errors": [{"error": str(e)}]}
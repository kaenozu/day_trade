#!/usr/bin/env python3
"""
マスターデータガバナンス管理
データガバナンスポリシーの適用・管理
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .types import (
    DataDomain,
    DataGovernancePolicy,
    DataSteward,
    DataStewardshipRole,
    MasterDataEntity,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class GovernanceManager:
    """ガバナンス管理クラス"""

    def __init__(self):
        """初期化"""
        self.governance_policies: Dict[str, DataGovernancePolicy] = {}
        self.data_stewards: Dict[str, DataSteward] = {}
        self._setup_default_policies()
        self._setup_default_stewards()

    def _setup_default_policies(self):
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

        # データアクセス制御ポリシー
        access_policy = DataGovernancePolicy(
            policy_id="data_access_control_policy",
            name="データアクセス制御ポリシー",
            description="データ分類に基づくアクセス制御",
            domain=DataDomain.FINANCIAL,
            policy_type="access",
            rules=[
                {
                    "rule": "confidential_data_access",
                    "roles": ["data_owner", "data_steward"],
                    "description": "機密データは所有者とスチュワードのみアクセス可能",
                },
                {
                    "rule": "restricted_data_access",
                    "roles": ["data_owner"],
                    "description": "制限データは所有者のみアクセス可能",
                },
                {
                    "rule": "audit_requirement",
                    "enabled": True,
                    "description": "すべてのアクセスを監査ログに記録",
                },
            ],
            enforcement_level="mandatory",
            effective_date=datetime.utcnow(),
            owner="security_team",
        )
        self.governance_policies[access_policy.policy_id] = access_policy

        # データ更新承認ポリシー
        approval_policy = DataGovernancePolicy(
            policy_id="data_update_approval_policy",
            name="データ更新承認ポリシー",
            description="重要データの更新には承認が必要",
            domain=DataDomain.SECURITY,
            policy_type="approval",
            rules=[
                {
                    "rule": "master_data_approval",
                    "threshold": "high_impact",
                    "approvers": ["data_steward", "data_owner"],
                    "description": "高影響度データは承認が必要",
                },
                {
                    "rule": "bulk_update_approval",
                    "threshold": 100,
                    "description": "100件以上の一括更新は承認が必要",
                },
            ],
            enforcement_level="mandatory",
            effective_date=datetime.utcnow(),
            owner="data_governance_team",
        )
        self.governance_policies[approval_policy.policy_id] = approval_policy

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

        # データ品質分析者
        quality_analyst = DataSteward(
            steward_id="data_quality_analyst",
            name="データ品質分析者",
            email="quality.analyst@company.com",
            role=DataStewardshipRole.QUALITY_ANALYST,
            domains=[DataDomain.FINANCIAL, DataDomain.MARKET, DataDomain.SECURITY],
            responsibilities=[
                "データ品質メトリクス監視",
                "品質問題の分析・報告",
                "品質改善施策の提案",
                "品質ダッシュボードの維持",
            ],
        )
        self.data_stewards[quality_analyst.steward_id] = quality_analyst

    async def apply_governance_policies(self, entity: MasterDataEntity):
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
                await self._apply_single_policy(entity, policy)

        except Exception as e:
            logger.error(f"ガバナンスポリシー適用エラー: {e}")

    async def _apply_single_policy(self, entity: MasterDataEntity, policy: DataGovernancePolicy):
        """単一ポリシー適用"""
        if policy.policy_type == "quality":
            await self._apply_quality_policy(entity, policy)
        elif policy.policy_type == "retention":
            await self._apply_retention_policy(entity, policy)
        elif policy.policy_type == "access":
            await self._apply_access_policy(entity, policy)
        elif policy.policy_type == "approval":
            await self._apply_approval_policy(entity, policy)

    async def _apply_quality_policy(self, entity: MasterDataEntity, policy: DataGovernancePolicy):
        """品質ポリシー適用"""
        try:
            for policy_rule in policy.rules:
                if policy_rule["rule"] == "completeness":
                    completeness = self._calculate_completeness(entity)
                    threshold = policy_rule["threshold"]

                    if completeness < threshold:
                        violation = {
                            "policy_id": policy.policy_id,
                            "rule": policy_rule["rule"],
                            "threshold": threshold,
                            "actual": completeness,
                            "severity": "high" if policy.enforcement_level == "mandatory" else "medium",
                            "description": f"完全性違反: {completeness:.2f} < {threshold}",
                        }
                        self._add_policy_violation(entity, violation)

                elif policy_rule["rule"] == "accuracy":
                    # 正確性チェック（品質スコアベース）
                    accuracy = entity.data_quality_score
                    threshold = policy_rule["threshold"]

                    if accuracy < threshold:
                        violation = {
                            "policy_id": policy.policy_id,
                            "rule": policy_rule["rule"],
                            "threshold": threshold,
                            "actual": accuracy,
                            "severity": "high",
                            "description": f"正確性違反: {accuracy:.2f} < {threshold}",
                        }
                        self._add_policy_violation(entity, violation)

                elif policy_rule["rule"] == "timeliness":
                    # 適時性チェック（更新からの経過時間）
                    age_minutes = (datetime.utcnow() - entity.updated_at).total_seconds() / 60
                    threshold = policy_rule["threshold"]

                    if age_minutes > threshold:
                        violation = {
                            "policy_id": policy.policy_id,
                            "rule": policy_rule["rule"],
                            "threshold": threshold,
                            "actual": age_minutes,
                            "severity": "medium",
                            "description": f"適時性違反: {age_minutes:.1f}分 > {threshold}分",
                        }
                        self._add_policy_violation(entity, violation)

        except Exception as e:
            logger.error(f"品質ポリシー適用エラー: {e}")

    async def _apply_retention_policy(self, entity: MasterDataEntity, policy: DataGovernancePolicy):
        """保持ポリシー適用"""
        try:
            for policy_rule in policy.rules:
                if policy_rule["rule"] == "financial_data":
                    retention_years = policy_rule["retention_years"]
                    retention_threshold = datetime.utcnow() - timedelta(days=retention_years * 365)

                    if entity.updated_at < retention_threshold:
                        violation = {
                            "policy_id": policy.policy_id,
                            "rule": policy_rule["rule"],
                            "threshold": retention_threshold.isoformat(),
                            "actual": entity.updated_at.isoformat(),
                            "severity": "low",
                            "description": f"保持期間違反: 最終更新が{retention_years}年以上前",
                        }
                        self._add_policy_violation(entity, violation)

        except Exception as e:
            logger.error(f"保持ポリシー適用エラー: {e}")

    async def _apply_access_policy(self, entity: MasterDataEntity, policy: DataGovernancePolicy):
        """アクセス制御ポリシー適用"""
        try:
            # アクセス制御情報をメタデータに設定
            access_control = entity.metadata.get("access_control", {})
            
            for policy_rule in policy.rules:
                if policy_rule["rule"] == "confidential_data_access":
                    access_control["confidential_roles"] = policy_rule["roles"]
                elif policy_rule["rule"] == "restricted_data_access":
                    access_control["restricted_roles"] = policy_rule["roles"]
                elif policy_rule["rule"] == "audit_requirement":
                    access_control["audit_required"] = policy_rule["enabled"]

            entity.metadata["access_control"] = access_control

        except Exception as e:
            logger.error(f"アクセス制御ポリシー適用エラー: {e}")

    async def _apply_approval_policy(self, entity: MasterDataEntity, policy: DataGovernancePolicy):
        """承認ポリシー適用"""
        try:
            approval_info = entity.metadata.get("approval_requirements", {})
            
            for policy_rule in policy.rules:
                if policy_rule["rule"] == "master_data_approval":
                    approval_info["high_impact_approvers"] = policy_rule["approvers"]
                elif policy_rule["rule"] == "bulk_update_approval":
                    approval_info["bulk_threshold"] = policy_rule["threshold"]

            entity.metadata["approval_requirements"] = approval_info

        except Exception as e:
            logger.error(f"承認ポリシー適用エラー: {e}")

    def _calculate_completeness(self, entity: MasterDataEntity) -> float:
        """完全性計算"""
        # エンティティタイプに基づく必須フィールド定義
        required_fields_map = {
            "stock": ["symbol", "company_name"],
            "company": ["name", "industry"],
            "currency": ["code", "name"],
            "exchange": ["code", "name", "country"],
        }
        
        required_fields = required_fields_map.get(entity.entity_type, [])
        if not required_fields:
            return 1.0

        filled_fields = sum(
            1 for field in required_fields
            if field in entity.attributes and entity.attributes[field]
        )

        return filled_fields / len(required_fields)

    def _add_policy_violation(self, entity: MasterDataEntity, violation: Dict[str, Any]):
        """ポリシー違反追加"""
        violations = entity.metadata.get("policy_violations", [])
        violations.append(violation)
        entity.metadata["policy_violations"] = violations

    def get_governance_status(self) -> Dict[str, Any]:
        """ガバナンス状況取得"""
        try:
            active_policies = [
                policy for policy in self.governance_policies.values()
                if policy.expiration_date is None or policy.expiration_date > datetime.utcnow()
            ]

            active_stewards = [
                steward for steward in self.data_stewards.values()
                if steward.active
            ]

            policy_by_type = {}
            for policy in active_policies:
                policy_type = policy.policy_type
                if policy_type not in policy_by_type:
                    policy_by_type[policy_type] = 0
                policy_by_type[policy_type] += 1

            stewards_by_role = {}
            for steward in active_stewards:
                role = steward.role.value
                if role not in stewards_by_role:
                    stewards_by_role[role] = 0
                stewards_by_role[role] += 1

            return {
                "total_policies": len(self.governance_policies),
                "active_policies": len(active_policies),
                "policies_by_type": policy_by_type,
                "total_stewards": len(self.data_stewards),
                "active_stewards": len(active_stewards),
                "stewards_by_role": stewards_by_role,
                "domains_covered": list(set(
                    domain.value for steward in active_stewards
                    for domain in steward.domains
                )),
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"ガバナンス状況取得エラー: {e}")
            return {"error": str(e)}

    def get_steward_for_domain(self, domain: DataDomain) -> List[str]:
        """ドメインのスチュワード取得"""
        stewards = []
        for steward in self.data_stewards.values():
            if domain in steward.domains and steward.active:
                stewards.append(steward.steward_id)
        return stewards

    def validate_policy_compliance(self, entity: MasterDataEntity) -> Dict[str, Any]:
        """ポリシー準拠性検証"""
        try:
            violations = entity.metadata.get("policy_violations", [])
            
            compliance_status = {
                "entity_id": entity.entity_id,
                "overall_compliant": len(violations) == 0,
                "violation_count": len(violations),
                "violations_by_severity": {
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
                "violations": violations,
                "last_checked": datetime.utcnow().isoformat(),
            }

            for violation in violations:
                severity = violation.get("severity", "medium")
                compliance_status["violations_by_severity"][severity] += 1

            return compliance_status

        except Exception as e:
            logger.error(f"ポリシー準拠性検証エラー: {e}")
            return {"error": str(e)}

    def add_custom_policy(self, policy: DataGovernancePolicy):
        """カスタムポリシー追加"""
        self.governance_policies[policy.policy_id] = policy
        logger.info(f"カスタムポリシー追加: {policy.policy_id}")

    def add_steward(self, steward: DataSteward):
        """スチュワード追加"""
        self.data_stewards[steward.steward_id] = steward
        logger.info(f"データスチュワード追加: {steward.steward_id}")
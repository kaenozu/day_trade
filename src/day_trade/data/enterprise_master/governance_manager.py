#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - ガバナンス管理

このモジュールは、データガバナンスポリシーの管理と適用を担当します。
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .enums import DataGovernanceLevel, MasterDataType
from .models import DataGovernancePolicy


class GovernanceManager:
    """データガバナンス管理クラス
    
    ガバナンスポリシーの作成、管理、適用を担当します。
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.policies: Dict[str, DataGovernancePolicy] = {}
        self._load_default_policies()

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
            DataGovernancePolicy(
                policy_id="currency_codes_policy",
                policy_name="通貨コードポリシー",
                entity_types=[MasterDataType.CURRENCY_CODES],
                rules=[
                    {"field": "code", "required": True, "unique": True, "pattern": "^[A-Z]{3}$"},
                    {"field": "name", "required": True, "min_length": 3, "max_length": 50},
                    {"field": "symbol", "max_length": 5},
                ],
                governance_level=DataGovernanceLevel.STANDARD,
                approval_roles=["data_steward"],
                quality_threshold=88.0,
            ),
            DataGovernancePolicy(
                policy_id="industry_codes_policy",
                policy_name="業界コードポリシー",
                entity_types=[MasterDataType.INDUSTRY_CODES],
                rules=[
                    {"field": "code", "required": True, "unique": True},
                    {"field": "name", "required": True, "min_length": 2},
                    {"field": "category", "required": True},
                ],
                governance_level=DataGovernanceLevel.STANDARD,
                approval_roles=["data_steward", "business_analyst"],
                quality_threshold=85.0,
            ),
        ]

        for policy in default_policies:
            self.policies[policy.policy_id] = policy
            self.logger.debug(f"デフォルトポリシー読み込み: {policy.policy_id}")

    def get_applicable_policy(
        self, entity_type: MasterDataType
    ) -> Optional[DataGovernancePolicy]:
        """適用可能なガバナンスポリシー取得
        
        Args:
            entity_type: エンティティタイプ
            
        Returns:
            Optional[DataGovernancePolicy]: 適用可能なポリシー
        """
        for policy in self.policies.values():
            if entity_type in policy.entity_types:
                return policy
        return None

    def add_policy(self, policy: DataGovernancePolicy) -> bool:
        """ガバナンスポリシー追加
        
        Args:
            policy: 追加するポリシー
            
        Returns:
            bool: 追加成功フラグ
        """
        try:
            if policy.policy_id in self.policies:
                self.logger.warning(f"ポリシーID '{policy.policy_id}' は既に存在します")
                return False

            # ポリシーバリデーション
            validation_result = self._validate_policy(policy)
            if not validation_result["is_valid"]:
                self.logger.error(f"ポリシーバリデーションエラー: {validation_result['errors']}")
                return False

            self.policies[policy.policy_id] = policy
            self.logger.info(f"ガバナンスポリシー追加: {policy.policy_id}")
            return True

        except Exception as e:
            self.logger.error(f"ポリシー追加エラー: {e}")
            return False

    def update_policy(self, policy: DataGovernancePolicy) -> bool:
        """ガバナンスポリシー更新
        
        Args:
            policy: 更新するポリシー
            
        Returns:
            bool: 更新成功フラグ
        """
        try:
            if policy.policy_id not in self.policies:
                self.logger.error(f"ポリシーID '{policy.policy_id}' が見つかりません")
                return False

            # ポリシーバリデーション
            validation_result = self._validate_policy(policy)
            if not validation_result["is_valid"]:
                self.logger.error(f"ポリシーバリデーションエラー: {validation_result['errors']}")
                return False

            self.policies[policy.policy_id] = policy
            self.logger.info(f"ガバナンスポリシー更新: {policy.policy_id}")
            return True

        except Exception as e:
            self.logger.error(f"ポリシー更新エラー: {e}")
            return False

    def remove_policy(self, policy_id: str) -> bool:
        """ガバナンスポリシー削除
        
        Args:
            policy_id: 削除するポリシーID
            
        Returns:
            bool: 削除成功フラグ
        """
        try:
            if policy_id not in self.policies:
                self.logger.error(f"ポリシーID '{policy_id}' が見つかりません")
                return False

            del self.policies[policy_id]
            self.logger.info(f"ガバナンスポリシー削除: {policy_id}")
            return True

        except Exception as e:
            self.logger.error(f"ポリシー削除エラー: {e}")
            return False

    def get_policy(self, policy_id: str) -> Optional[DataGovernancePolicy]:
        """ガバナンスポリシー取得
        
        Args:
            policy_id: ポリシーID
            
        Returns:
            Optional[DataGovernancePolicy]: 該当するポリシー
        """
        return self.policies.get(policy_id)

    def list_policies(
        self, 
        entity_type: Optional[MasterDataType] = None
    ) -> List[DataGovernancePolicy]:
        """ガバナンスポリシー一覧取得
        
        Args:
            entity_type: フィルタ対象のエンティティタイプ
            
        Returns:
            List[DataGovernancePolicy]: ポリシー一覧
        """
        if entity_type:
            return [
                policy for policy in self.policies.values()
                if entity_type in policy.entity_types
            ]
        else:
            return list(self.policies.values())

    def _validate_policy(self, policy: DataGovernancePolicy) -> Dict[str, any]:
        """ポリシーバリデーション
        
        Args:
            policy: バリデーション対象ポリシー
            
        Returns:
            Dict[str, any]: バリデーション結果
        """
        errors = []
        warnings = []

        try:
            # 必須フィールドチェック
            if not policy.policy_id:
                errors.append("policy_id は必須です")
            
            if not policy.policy_name:
                errors.append("policy_name は必須です")
            
            if not policy.entity_types:
                errors.append("entity_types は必須です")
            
            if not policy.rules:
                errors.append("rules は必須です")

            # ルールバリデーション
            for i, rule in enumerate(policy.rules):
                if not isinstance(rule, dict):
                    errors.append(f"ルール {i} は辞書である必要があります")
                    continue
                
                if "field" not in rule:
                    errors.append(f"ルール {i} に 'field' が定義されていません")
                
                # パターンバリデーション
                if "pattern" in rule:
                    try:
                        import re
                        re.compile(rule["pattern"])
                    except re.error as e:
                        errors.append(f"ルール {i} の正規表現パターンが無効です: {e}")

            # 品質しきい値チェック
            if not 0 <= policy.quality_threshold <= 100:
                errors.append("quality_threshold は 0-100 の範囲である必要があります")

            # 承認ロールチェック
            if policy.requires_approval and not policy.approval_roles:
                warnings.append("承認が必要ですが、承認ロールが定義されていません")

            # 保管期間チェック
            if policy.retention_period and policy.retention_period < 0:
                errors.append("retention_period は非負数である必要があります")

            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"ポリシーバリデーション実行エラー: {str(e)}"],
                "warnings": [],
            }

    def check_approval_required(
        self, entity_type: MasterDataType, change_type: str
    ) -> bool:
        """承認要否チェック
        
        Args:
            entity_type: エンティティタイプ
            change_type: 変更タイプ
            
        Returns:
            bool: 承認が必要かどうか
        """
        policy = self.get_applicable_policy(entity_type)
        if not policy:
            return True  # ポリシーがない場合はデフォルトで承認必要

        # 高リスクな変更は常に承認が必要
        high_risk_changes = ["delete", "merge", "split"]
        if change_type in high_risk_changes:
            return True

        return policy.requires_approval

    def get_required_approval_roles(
        self, entity_type: MasterDataType
    ) -> List[str]:
        """必要承認ロール取得
        
        Args:
            entity_type: エンティティタイプ
            
        Returns:
            List[str]: 承認が必要なロール一覧
        """
        policy = self.get_applicable_policy(entity_type)
        if not policy:
            return ["data_steward"]  # デフォルト承認ロール

        return policy.approval_roles

    def is_audit_required(self, entity_type: MasterDataType) -> bool:
        """監査要否チェック
        
        Args:
            entity_type: エンティティタイプ
            
        Returns:
            bool: 監査が必要かどうか
        """
        policy = self.get_applicable_policy(entity_type)
        if not policy:
            return False

        return policy.audit_required

    def get_retention_period(self, entity_type: MasterDataType) -> Optional[int]:
        """保管期間取得
        
        Args:
            entity_type: エンティティタイプ
            
        Returns:
            Optional[int]: 保管期間（日数）
        """
        policy = self.get_applicable_policy(entity_type)
        if not policy:
            return None

        return policy.retention_period

    def get_quality_threshold(self, entity_type: MasterDataType) -> float:
        """品質しきい値取得
        
        Args:
            entity_type: エンティティタイプ
            
        Returns:
            float: 品質しきい値
        """
        policy = self.get_applicable_policy(entity_type)
        if not policy:
            return 75.0  # デフォルトしきい値

        return policy.quality_threshold

    def export_policies(self) -> Dict[str, any]:
        """ポリシー設定のエクスポート
        
        Returns:
            Dict[str, any]: エクスポートデータ
        """
        try:
            export_data = {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_policies": len(self.policies),
                "policies": {}
            }

            for policy_id, policy in self.policies.items():
                export_data["policies"][policy_id] = policy.to_dict()

            return export_data

        except Exception as e:
            self.logger.error(f"ポリシーエクスポートエラー: {e}")
            return {"error": str(e)}

    def import_policies(self, import_data: Dict[str, any]) -> bool:
        """ポリシー設定のインポート
        
        Args:
            import_data: インポートデータ
            
        Returns:
            bool: インポート成功フラグ
        """
        try:
            if "policies" not in import_data:
                self.logger.error("インポートデータに 'policies' フィールドがありません")
                return False

            imported_count = 0
            for policy_id, policy_data in import_data["policies"].items():
                try:
                    # データからポリシーオブジェクトを復元
                    policy = DataGovernancePolicy(
                        policy_id=policy_data["policy_id"],
                        policy_name=policy_data["policy_name"],
                        entity_types=[
                            MasterDataType(et) for et in policy_data["entity_types"]
                        ],
                        rules=policy_data["rules"],
                        governance_level=DataGovernanceLevel(policy_data["governance_level"]),
                        requires_approval=policy_data.get("requires_approval", True),
                        approval_roles=policy_data.get("approval_roles", []),
                        retention_period=policy_data.get("retention_period"),
                        audit_required=policy_data.get("audit_required", True),
                        quality_threshold=policy_data.get("quality_threshold", 80.0),
                        created_by=policy_data.get("created_by", "system"),
                        created_at=datetime.fromisoformat(
                            policy_data.get("created_at", datetime.now(timezone.utc).isoformat())
                        ),
                    )

                    if self.add_policy(policy):
                        imported_count += 1

                except Exception as e:
                    self.logger.warning(f"ポリシー '{policy_id}' のインポートに失敗: {e}")

            self.logger.info(f"ポリシーインポート完了: {imported_count}/{len(import_data['policies'])}件")
            return imported_count > 0

        except Exception as e:
            self.logger.error(f"ポリシーインポートエラー: {e}")
            return False
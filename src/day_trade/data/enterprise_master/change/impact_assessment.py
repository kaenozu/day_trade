#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - 変更影響評価

変更リクエストの影響評価とリスクアセスメント機能を提供
"""

import logging
from typing import Any, Dict, List

from ..enums import ChangeType, RiskLevel
from ..models import MasterDataEntity


class ImpactAssessment:
    """変更影響評価クラス"""

    def __init__(self, governance_manager):
        self.logger = logging.getLogger(__name__)
        self.governance_manager = governance_manager

    async def assess_change_impact(
        self,
        entity: MasterDataEntity,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """変更影響評価"""
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
            self._assess_change_type_impact(impact, change_type)

            # 重要フィールドの変更チェック
            self._assess_critical_field_changes(
                impact, entity, proposed_changes
            )

            # ゴールデンレコードの変更
            if entity.is_golden_record:
                self._assess_golden_record_impact(impact)

            # 品質スコア予測
            if change_type == ChangeType.UPDATE:
                quality_impact = self._predict_quality_impact(entity, proposed_changes)
                impact.update(quality_impact)

            return impact

        except Exception as e:
            self.logger.error(f"影響評価エラー: {e}")
            return {
                "risk_level": RiskLevel.UNKNOWN.value,
                "error": str(e),
                "recommendations": ["影響評価を手動で実施してください"],
            }

    def _assess_change_type_impact(self, impact: Dict[str, Any], change_type: ChangeType):
        """変更タイプ別の影響評価"""
        if change_type == ChangeType.DELETE:
            impact["risk_level"] = RiskLevel.HIGH.value
            impact["business_impact"] = "high"
            impact["technical_impact"] = "high"
            impact["rollback_complexity"] = "complex"
            impact["recommendations"].extend([
                "削除前に関連データの確認が必要",
                "バックアップの作成を推奨",
            ])

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
            impact["recommendations"].append(
                "データ分割の設計を慎重に検討してください"
            )

        elif change_type == ChangeType.ARCHIVE:
            impact["risk_level"] = RiskLevel.MEDIUM.value
            impact["business_impact"] = "medium"
            impact["recommendations"].append(
                "アーカイブ前に使用状況の確認が必要"
            )

    def _assess_critical_field_changes(
        self, impact: Dict[str, Any], entity: MasterDataEntity, proposed_changes: Dict[str, Any]
    ):
        """重要フィールドの変更評価"""
        critical_fields = self._get_critical_fields(entity.entity_type)

        for field in critical_fields:
            if field in proposed_changes and field in entity.attributes:
                if entity.attributes[field] != proposed_changes[field]:
                    impact["risk_level"] = RiskLevel.HIGH.value
                    impact["business_impact"] = "high"
                    impact["recommendations"].append(
                        f"重要フィールド '{field}' の変更は慎重な確認が必要"
                    )

    def _assess_golden_record_impact(self, impact: Dict[str, Any]):
        """ゴールデンレコードの変更評価"""
        # 既存のリスクレベルを維持しつつ、最低でもMEDIUMに設定
        current_risk = RiskLevel(impact["risk_level"])
        if current_risk == RiskLevel.LOW:
            impact["risk_level"] = RiskLevel.MEDIUM.value

        impact["business_impact"] = "medium"
        impact["recommendations"].append(
            "ゴールデンレコードの変更は影響範囲が広いため注意が必要"
        )

    def _get_critical_fields(self, entity_type) -> List[str]:
        """エンティティタイプの重要フィールド取得"""
        critical_field_map = {
            "financial_instruments": ["symbol", "isin", "name", "market"],
            "exchange_codes": ["code", "description"],
            "currency_codes": ["code", "name"],
            "industry_codes": ["code", "name"],
            "regulatory_data": ["regulation_id", "authority", "effective_date"],
        }

        return critical_field_map.get(entity_type.value, ["primary_key", "name"])

    def _predict_quality_impact(
        self, entity: MasterDataEntity, proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """品質影響予測"""
        try:
            quality_impact = {
                "current_quality_score": entity.quality_score,
                "predicted_quality_change": 0,
                "quality_concerns": [],
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

            # パターンマッチングのバリデーション
            if policy:
                quality_impact["predicted_quality_change"] += self._validate_pattern_rules(
                    proposed_changes, policy.rules, quality_impact["quality_concerns"]
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
                "quality_concerns": ["品質影響の予測に失敗しました"],
            }

    def _validate_pattern_rules(
        self, proposed_changes: Dict[str, Any], rules: List[Dict[str, Any]], concerns: List[str]
    ) -> int:
        """パターンルールバリデーション"""
        quality_change = 0

        for rule in rules:
            field = rule.get("field")
            pattern = rule.get("pattern")

            if field in proposed_changes and pattern:
                import re

                new_value = proposed_changes[field]
                if isinstance(new_value, str):
                    if re.match(pattern, new_value):
                        quality_change += 5  # パターンマッチした場合の品質向上
                    else:
                        quality_change -= 10  # パターン不一致の場合の品質低下
                        concerns.append(
                            f"フィールド '{field}' がパターン '{pattern}' に一致しません"
                        )

        return quality_change

    async def assess_system_wide_impact(
        self, entity: MasterDataEntity, change_type: ChangeType
    ) -> Dict[str, Any]:
        """システム全体への影響評価"""
        try:
            system_impact = {
                "downstream_systems": [],
                "dependent_entities": [],
                "integration_points": [],
                "potential_failures": [],
                "mitigation_strategies": [],
            }

            # 下流システムの特定
            for source_system in entity.source_systems:
                system_impact["downstream_systems"].append({
                    "system": source_system,
                    "impact_level": self._assess_system_impact_level(
                        source_system, change_type
                    ),
                })

            # 依存エンティティの特定
            system_impact["dependent_entities"] = entity.related_entities.copy()

            # 統合ポイントの評価
            if entity.is_golden_record:
                system_impact["integration_points"].append({
                    "type": "golden_record",
                    "impact": "Master record changes affect all downstream consumers",
                })

            # 潜在的な障害シナリオ
            if change_type in [ChangeType.DELETE, ChangeType.MERGE]:
                system_impact["potential_failures"].append(
                    "Referenced data may become unavailable"
                )

            # 軽減戦略
            system_impact["mitigation_strategies"].extend([
                "Implement gradual rollout",
                "Monitor system health during change",
                "Prepare rollback procedures",
            ])

            return system_impact

        except Exception as e:
            self.logger.error(f"システム全体影響評価エラー: {e}")
            return {"error": str(e)}

    def _assess_system_impact_level(self, source_system: str, change_type: ChangeType) -> str:
        """システム影響レベル評価"""
        # 基本的な影響レベル判定ロジック
        high_impact_changes = [ChangeType.DELETE, ChangeType.MERGE, ChangeType.SPLIT]

        if change_type in high_impact_changes:
            return "high"
        elif change_type == ChangeType.UPDATE:
            return "medium"
        else:
            return "low"

    async def generate_risk_matrix(
        self, entities: List[MasterDataEntity], change_type: ChangeType
    ) -> Dict[str, Any]:
        """リスクマトリクス生成"""
        try:
            risk_matrix = {
                "low_risk": [],
                "medium_risk": [],
                "high_risk": [],
                "critical_risk": [],
                "summary": {
                    "total_entities": len(entities),
                    "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                },
            }

            for entity in entities:
                impact = await self.assess_change_impact(entity, change_type, {})
                risk_level = impact["risk_level"]

                entity_risk_info = {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type.value,
                    "quality_score": entity.quality_score,
                    "is_golden_record": entity.is_golden_record,
                    "risk_factors": impact.get("recommendations", []),
                }

                if risk_level == RiskLevel.LOW.value:
                    risk_matrix["low_risk"].append(entity_risk_info)
                elif risk_level == RiskLevel.MEDIUM.value:
                    risk_matrix["medium_risk"].append(entity_risk_info)
                elif risk_level == RiskLevel.HIGH.value:
                    risk_matrix["high_risk"].append(entity_risk_info)
                else:
                    risk_matrix["critical_risk"].append(entity_risk_info)

                risk_matrix["summary"]["risk_distribution"][risk_level] += 1

            return risk_matrix

        except Exception as e:
            self.logger.error(f"リスクマトリクス生成エラー: {e}")
            return {"error": str(e)}
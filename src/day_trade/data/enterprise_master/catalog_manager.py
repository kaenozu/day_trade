#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - カタログ・リネージュ管理

このモジュールは、データカタログとデータリネージュの管理を担当します。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .enums import MasterDataType, get_quality_level
from .database_operations import DatabaseOperations
from .governance_manager import GovernanceManager


class CatalogManager:
    """カタログ・リネージュ管理クラス
    
    データカタログの生成、メタデータ管理、データリネージュ追跡を担当します。
    """
    
    def __init__(
        self,
        database_operations: DatabaseOperations,
        governance_manager: GovernanceManager
    ):
        self.logger = logging.getLogger(__name__)
        self.db_ops = database_operations
        self.governance_manager = governance_manager

    async def generate_data_catalog(
        self, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """データカタログ生成
        
        Args:
            entity_type: フィルタ対象のエンティティタイプ
            
        Returns:
            Dict[str, Any]: データカタログ情報
        """
        try:
            # 基本カタログデータ取得
            catalog_data = await self.db_ops.get_catalog_data(entity_type)
            
            if "error" in catalog_data:
                return catalog_data

            # ガバナンス情報追加
            for item in catalog_data["catalog"]:
                entity_type_enum = MasterDataType(item["entity_type"])
                policy = self.governance_manager.get_applicable_policy(entity_type_enum)
                
                if policy:
                    item["governance_policy"] = policy.policy_name
                    item["governance_level"] = policy.governance_level.value
                    item["requires_approval"] = policy.requires_approval
                    item["quality_threshold"] = policy.quality_threshold
                else:
                    item["governance_policy"] = "デフォルト"
                    item["governance_level"] = "standard"
                    item["requires_approval"] = True
                    item["quality_threshold"] = 75.0

                # 品質レベル分類
                avg_quality = item.get("average_quality", 0)
                item["quality_level"] = get_quality_level(avg_quality).value

                # データ鮮度情報
                item["data_freshness"] = await self._calculate_data_freshness(
                    entity_type_enum, item.get("latest_updated")
                )

            # システム統計追加
            catalog_data["system_statistics"] = await self._generate_system_statistics()
            
            # メタデータ詳細追加
            catalog_data["metadata"] = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "EnterpriseMDM CatalogManager",
                "version": "1.0",
                "includes_governance": True,
                "includes_quality_metrics": True,
            }

            return catalog_data

        except Exception as e:
            self.logger.error(f"データカタログ生成エラー: {e}")
            return {"error": str(e)}

    async def get_entity_lineage(self, entity_id: str) -> Dict[str, Any]:
        """エンティティデータリネージュ取得
        
        Args:
            entity_id: エンティティID
            
        Returns:
            Dict[str, Any]: データリネージュ情報
        """
        try:
            # 基本リネージュデータ取得
            lineage_data = await self.db_ops.get_entity_lineage(entity_id)
            
            if "error" in lineage_data:
                return lineage_data

            # エンティティ基本情報追加
            entity = await self.db_ops.get_entity(entity_id)
            if entity:
                lineage_data["entity_info"] = {
                    "entity_type": entity.entity_type.value,
                    "primary_key": entity.primary_key,
                    "is_golden_record": entity.is_golden_record,
                    "quality_score": entity.quality_score,
                    "source_systems": entity.source_systems,
                    "governance_level": entity.governance_level.value,
                }

            # リネージュ分析
            lineage_data["lineage_analysis"] = await self._analyze_lineage(lineage_data)
            
            # 関連エンティティ情報
            if entity and entity.related_entities:
                lineage_data["related_entities"] = await self._get_related_entities_info(
                    entity.related_entities
                )

            return lineage_data

        except Exception as e:
            self.logger.error(f"データリネージュ取得エラー: {e}")
            return {"entity_id": entity_id, "error": str(e)}

    async def generate_impact_analysis(
        self, entity_id: str, proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """変更影響分析
        
        Args:
            entity_id: 対象エンティティID
            proposed_changes: 提案する変更内容
            
        Returns:
            Dict[str, Any]: 影響分析結果
        """
        try:
            # エンティティ情報取得
            entity = await self.db_ops.get_entity(entity_id)
            if not entity:
                return {"error": f"エンティティが見つかりません: {entity_id}"}

            impact_analysis = {
                "entity_id": entity_id,
                "entity_type": entity.entity_type.value,
                "is_golden_record": entity.is_golden_record,
                "current_quality_score": entity.quality_score,
                "proposed_changes": proposed_changes,
                "impact_areas": {},
                "risk_assessment": {},
                "affected_systems": [],
                "recommendations": [],
                "estimated_effort": "low",
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # 品質への影響
            quality_impact = await self._analyze_quality_impact(entity, proposed_changes)
            impact_analysis["impact_areas"]["quality"] = quality_impact

            # システムへの影響
            system_impact = await self._analyze_system_impact(entity, proposed_changes)
            impact_analysis["impact_areas"]["systems"] = system_impact
            impact_analysis["affected_systems"] = system_impact.get("affected_systems", [])

            # データ整合性への影響
            consistency_impact = await self._analyze_consistency_impact(entity, proposed_changes)
            impact_analysis["impact_areas"]["consistency"] = consistency_impact

            # 総合リスク評価
            risk_assessment = await self._assess_overall_risk(
                entity, proposed_changes, impact_analysis["impact_areas"]
            )
            impact_analysis["risk_assessment"] = risk_assessment

            # 推奨事項生成
            recommendations = await self._generate_recommendations(
                entity, impact_analysis["impact_areas"], risk_assessment
            )
            impact_analysis["recommendations"] = recommendations

            # 作業工数推定
            effort_estimation = await self._estimate_effort(
                entity, proposed_changes, risk_assessment
            )
            impact_analysis["estimated_effort"] = effort_estimation

            return impact_analysis

        except Exception as e:
            self.logger.error(f"変更影響分析エラー: {e}")
            return {"entity_id": entity_id, "error": str(e)}

    async def generate_quality_report(
        self, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """品質レポート生成
        
        Args:
            entity_type: エンティティタイプフィルタ
            
        Returns:
            Dict[str, Any]: 品質レポート
        """
        try:
            # エンティティ一覧取得
            entities = await self.db_ops.list_entities(entity_type)

            quality_report = {
                "report_type": "quality_assessment",
                "entity_type_filter": entity_type.value if entity_type else "all",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_entities": len(entities),
                "quality_distribution": {
                    "excellent": 0,  # 86-100
                    "good": 0,       # 71-85
                    "fair": 0,       # 51-70
                    "poor": 0        # 0-50
                },
                "governance_compliance": {},
                "top_quality_issues": [],
                "improvement_opportunities": [],
                "quality_trends": {},
            }

            # 品質分布計算
            total_quality_score = 0
            quality_scores = []
            governance_stats = {}
            quality_issues = {}

            for entity in entities:
                quality_level = get_quality_level(entity.quality_score)
                quality_report["quality_distribution"][quality_level.value] += 1
                
                total_quality_score += entity.quality_score
                quality_scores.append(entity.quality_score)

                # ガバナンス統計
                gov_level = entity.governance_level.value
                if gov_level not in governance_stats:
                    governance_stats[gov_level] = {"total": 0, "compliant": 0}
                governance_stats[gov_level]["total"] += 1

                # ガバナンス準拠チェック
                policy = self.governance_manager.get_applicable_policy(entity.entity_type)
                if policy and entity.quality_score >= policy.quality_threshold:
                    governance_stats[gov_level]["compliant"] += 1

            # 平均品質スコア
            if entities:
                quality_report["average_quality_score"] = total_quality_score / len(entities)
            else:
                quality_report["average_quality_score"] = 0

            # ガバナンス準拠率
            for gov_level, stats in governance_stats.items():
                compliance_rate = (stats["compliant"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                quality_report["governance_compliance"][gov_level] = {
                    "total_entities": stats["total"],
                    "compliant_entities": stats["compliant"],
                    "compliance_rate": round(compliance_rate, 2)
                }

            # 改善機会の特定
            improvement_opportunities = await self._identify_improvement_opportunities(entities)
            quality_report["improvement_opportunities"] = improvement_opportunities

            return quality_report

        except Exception as e:
            self.logger.error(f"品質レポート生成エラー: {e}")
            return {"error": str(e)}

    async def _calculate_data_freshness(
        self, entity_type: MasterDataType, latest_updated: Optional[str]
    ) -> Dict[str, Any]:
        """データ鮮度計算"""
        try:
            if not latest_updated:
                return {"status": "unknown", "message": "更新日時が不明です"}

            last_update = datetime.fromisoformat(latest_updated.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            days_since_update = (now - last_update).days

            # エンティティタイプ別の鮮度基準
            freshness_thresholds = {
                MasterDataType.FINANCIAL_INSTRUMENTS: {"fresh": 1, "stale": 7},
                MasterDataType.EXCHANGE_CODES: {"fresh": 30, "stale": 90},
                MasterDataType.CURRENCY_CODES: {"fresh": 7, "stale": 30},
            }

            thresholds = freshness_thresholds.get(entity_type, {"fresh": 7, "stale": 30})

            if days_since_update <= thresholds["fresh"]:
                status = "fresh"
            elif days_since_update <= thresholds["stale"]:
                status = "moderate"
            else:
                status = "stale"

            return {
                "status": status,
                "days_since_update": days_since_update,
                "last_updated": latest_updated,
                "threshold_fresh": thresholds["fresh"],
                "threshold_stale": thresholds["stale"],
            }

        except Exception as e:
            self.logger.error(f"データ鮮度計算エラー: {e}")
            return {"status": "error", "message": str(e)}

    async def _generate_system_statistics(self) -> Dict[str, Any]:
        """システム統計情報生成"""
        try:
            catalog_data = await self.db_ops.get_catalog_data()
            
            total_entities = sum(item["total_count"] for item in catalog_data.get("catalog", []))
            total_active = sum(item["active_count"] for item in catalog_data.get("catalog", []))
            total_golden = sum(item["golden_records"] for item in catalog_data.get("catalog", []))

            return {
                "total_entities": total_entities,
                "active_entities": total_active,
                "golden_records": total_golden,
                "entity_types": len(catalog_data.get("catalog", [])),
                "golden_record_ratio": (total_golden / total_entities) * 100 if total_entities > 0 else 0,
                "active_ratio": (total_active / total_entities) * 100 if total_entities > 0 else 0,
            }

        except Exception as e:
            self.logger.error(f"システム統計情報生成エラー: {e}")
            return {"error": str(e)}

    async def _analyze_lineage(self, lineage_data: Dict[str, Any]) -> Dict[str, Any]:
        """リネージュ分析"""
        try:
            analysis = {
                "total_changes": len(lineage_data.get("change_history", [])),
                "total_quality_checks": len(lineage_data.get("quality_history", [])),
                "change_frequency": "unknown",
                "quality_trend": "stable",
                "most_common_change_type": "update",
                "data_stewards": set(),
            }

            # 変更頻度分析
            change_history = lineage_data.get("change_history", [])
            if change_history:
                # 最も多い変更タイプを特定
                change_types = {}
                data_stewards = set()
                
                for change in change_history:
                    change_type = change.get("change_type", "unknown")
                    change_types[change_type] = change_types.get(change_type, 0) + 1
                    
                    changed_by = change.get("changed_by")
                    if changed_by:
                        data_stewards.add(changed_by)

                analysis["most_common_change_type"] = max(change_types, key=change_types.get)
                analysis["data_stewards"] = list(data_stewards)

            # 品質トレンド分析
            quality_history = lineage_data.get("quality_history", [])
            if len(quality_history) >= 2:
                recent_scores = [item["quality_score"] for item in quality_history[:5]]
                if len(recent_scores) >= 2:
                    if recent_scores[0] > recent_scores[-1]:
                        analysis["quality_trend"] = "improving"
                    elif recent_scores[0] < recent_scores[-1]:
                        analysis["quality_trend"] = "declining"
                    else:
                        analysis["quality_trend"] = "stable"

            return analysis

        except Exception as e:
            self.logger.error(f"リネージュ分析エラー: {e}")
            return {"error": str(e)}

    async def _get_related_entities_info(self, related_entity_ids: List[str]) -> List[Dict[str, Any]]:
        """関連エンティティ情報取得"""
        try:
            related_info = []
            
            for entity_id in related_entity_ids:
                entity = await self.db_ops.get_entity(entity_id)
                if entity:
                    related_info.append({
                        "entity_id": entity_id,
                        "entity_type": entity.entity_type.value,
                        "primary_key": entity.primary_key,
                        "quality_score": entity.quality_score,
                        "is_golden_record": entity.is_golden_record,
                        "is_active": entity.is_active,
                    })

            return related_info

        except Exception as e:
            self.logger.error(f"関連エンティティ情報取得エラー: {e}")
            return []

    async def _analyze_quality_impact(
        self, entity, proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """品質への影響分析"""
        # 簡略化された実装
        return {
            "current_score": entity.quality_score,
            "predicted_change": 0,  # 実際の実装では詳細な予測ロジック
            "risk_level": "low",
            "affected_metrics": []
        }

    async def _analyze_system_impact(
        self, entity, proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """システムへの影響分析"""
        return {
            "affected_systems": entity.source_systems,
            "integration_impact": "minimal",
            "downstream_effects": []
        }

    async def _analyze_consistency_impact(
        self, entity, proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """データ整合性への影響分析"""
        return {
            "consistency_risk": "low",
            "validation_required": False,
            "dependent_entities": entity.related_entities
        }

    async def _assess_overall_risk(
        self, entity, proposed_changes: Dict[str, Any], impact_areas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """総合リスク評価"""
        return {
            "overall_risk": "low",
            "risk_factors": [],
            "mitigation_strategies": []
        }

    async def _generate_recommendations(
        self, entity, impact_areas: Dict[str, Any], risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if entity.quality_score < 80:
            recommendations.append("データ品質の改善を検討してください")
        
        if not entity.is_golden_record and entity.quality_score > 90:
            recommendations.append("ゴールデンレコード候補として検討してください")
        
        return recommendations

    async def _estimate_effort(
        self, entity, proposed_changes: Dict[str, Any], risk_assessment: Dict[str, Any]
    ) -> str:
        """作業工数推定"""
        # 簡略化された実装
        if len(proposed_changes) > 5:
            return "high"
        elif len(proposed_changes) > 2:
            return "medium"
        else:
            return "low"

    async def _identify_improvement_opportunities(self, entities) -> List[Dict[str, Any]]:
        """改善機会の特定"""
        opportunities = []
        
        # 低品質エンティティの特定
        low_quality_count = sum(1 for e in entities if e.quality_score < 70)
        if low_quality_count > 0:
            opportunities.append({
                "type": "quality_improvement",
                "description": f"{low_quality_count}件のエンティティで品質改善が可能です",
                "priority": "high" if low_quality_count > len(entities) * 0.2 else "medium",
                "affected_entities": low_quality_count
            })

        # ゴールデンレコード候補の特定
        golden_candidates = sum(1 for e in entities if not e.is_golden_record and e.quality_score >= 90)
        if golden_candidates > 0:
            opportunities.append({
                "type": "golden_record_promotion",
                "description": f"{golden_candidates}件のエンティティがゴールデンレコード候補です",
                "priority": "medium",
                "affected_entities": golden_candidates
            })

        return opportunities
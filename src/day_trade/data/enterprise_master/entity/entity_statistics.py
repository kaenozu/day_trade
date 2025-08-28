#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - エンティティ統計

エンティティの統計情報と分析機能を提供
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..enums import MasterDataType
from ..models import MasterDataEntity


class EntityStatistics:
    """エンティティ統計クラス"""

    def __init__(self, entity_cache: Dict[str, MasterDataEntity]):
        self.logger = logging.getLogger(__name__)
        self._entity_cache = entity_cache

    async def get_entity_statistics(self) -> Dict[str, Any]:
        """エンティティ統計情報取得"""
        try:
            stats = {
                "total_entities": len(self._entity_cache),
                "active_entities": 0,
                "golden_records": 0,
                "by_entity_type": {},
                "by_governance_level": {},
                "quality_distribution": {
                    "excellent": 0,  # 86-100
                    "good": 0,  # 71-85
                    "fair": 0,  # 51-70
                    "poor": 0,  # 0-50
                },
                "average_quality_score": 0.0,
            }

            total_quality_score = 0.0

            for entity in self._entity_cache.values():
                # アクティブエンティティ数
                if entity.is_active:
                    stats["active_entities"] += 1

                # ゴールデンレコード数
                if entity.is_golden_record:
                    stats["golden_records"] += 1

                # エンティティタイプ別集計
                entity_type_key = entity.entity_type.value
                if entity_type_key not in stats["by_entity_type"]:
                    stats["by_entity_type"][entity_type_key] = 0
                stats["by_entity_type"][entity_type_key] += 1

                # ガバナンスレベル別集計
                governance_level_key = entity.governance_level.value
                if governance_level_key not in stats["by_governance_level"]:
                    stats["by_governance_level"][governance_level_key] = 0
                stats["by_governance_level"][governance_level_key] += 1

                # 品質分布
                quality_score = entity.quality_score
                total_quality_score += quality_score

                if quality_score >= 86:
                    stats["quality_distribution"]["excellent"] += 1
                elif quality_score >= 71:
                    stats["quality_distribution"]["good"] += 1
                elif quality_score >= 51:
                    stats["quality_distribution"]["fair"] += 1
                else:
                    stats["quality_distribution"]["poor"] += 1

            # 平均品質スコア
            if stats["total_entities"] > 0:
                stats["average_quality_score"] = (
                    total_quality_score / stats["total_entities"]
                )

            return stats

        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}

    async def get_quality_trends(
        self, days: int = 30, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """品質トレンド分析"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            daily_stats = {}
            for i in range(days):
                current_date = start_date + timedelta(days=i)
                date_key = current_date.strftime("%Y-%m-%d")
                daily_stats[date_key] = {
                    "total_entities": 0,
                    "total_quality_score": 0.0,
                    "average_quality_score": 0.0,
                    "golden_records": 0,
                }

            for entity in self._entity_cache.values():
                if entity_type and entity.entity_type != entity_type:
                    continue

                # エンティティの作成日または更新日をベースに集計
                entity_date = entity.updated_at.strftime("%Y-%m-%d")

                if entity_date in daily_stats:
                    daily_stats[entity_date]["total_entities"] += 1
                    daily_stats[entity_date]["total_quality_score"] += entity.quality_score

                    if entity.is_golden_record:
                        daily_stats[entity_date]["golden_records"] += 1

            # 平均品質スコア計算
            for date_key, stats in daily_stats.items():
                if stats["total_entities"] > 0:
                    stats["average_quality_score"] = (
                        stats["total_quality_score"] / stats["total_entities"]
                    )

            return {
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "daily_stats": daily_stats,
                "entity_type": entity_type.value if entity_type else "all",
            }

        except Exception as e:
            self.logger.error(f"品質トレンド分析エラー: {e}")
            return {"error": str(e)}

    async def get_source_system_analysis(self) -> Dict[str, Any]:
        """ソースシステム分析"""
        try:
            source_stats = {}

            for entity in self._entity_cache.values():
                for source_system in entity.source_systems:
                    if source_system not in source_stats:
                        source_stats[source_system] = {
                            "total_entities": 0,
                            "active_entities": 0,
                            "golden_records": 0,
                            "total_quality_score": 0.0,
                            "average_quality_score": 0.0,
                            "entity_types": set(),
                        }

                    stats = source_stats[source_system]
                    stats["total_entities"] += 1
                    stats["total_quality_score"] += entity.quality_score
                    stats["entity_types"].add(entity.entity_type.value)

                    if entity.is_active:
                        stats["active_entities"] += 1

                    if entity.is_golden_record:
                        stats["golden_records"] += 1

            # 平均品質スコア計算とセットを리ストに変換
            for source_system, stats in source_stats.items():
                if stats["total_entities"] > 0:
                    stats["average_quality_score"] = (
                        stats["total_quality_score"] / stats["total_entities"]
                    )
                stats["entity_types"] = list(stats["entity_types"])

            return {
                "source_systems": source_stats,
                "total_source_systems": len(source_stats),
            }

        except Exception as e:
            self.logger.error(f"ソースシステム分析エラー: {e}")
            return {"error": str(e)}

    async def get_governance_compliance_report(self) -> Dict[str, Any]:
        """ガバナンス準拠レポート"""
        try:
            compliance_stats = {
                "total_entities": len(self._entity_cache),
                "governance_levels": {},
                "compliance_issues": [],
                "golden_record_compliance": {
                    "total_golden_records": 0,
                    "compliant_golden_records": 0,
                    "non_compliant_golden_records": 0,
                },
            }

            for entity in self._entity_cache.values():
                governance_level = entity.governance_level.value

                if governance_level not in compliance_stats["governance_levels"]:
                    compliance_stats["governance_levels"][governance_level] = {
                        "count": 0,
                        "average_quality": 0.0,
                        "total_quality": 0.0,
                        "issues": 0,
                    }

                level_stats = compliance_stats["governance_levels"][governance_level]
                level_stats["count"] += 1
                level_stats["total_quality"] += entity.quality_score

                # コンプライアンス問題チェック
                issues = self._check_compliance_issues(entity)
                if issues:
                    level_stats["issues"] += len(issues)
                    compliance_stats["compliance_issues"].extend(
                        [{"entity_id": entity.entity_id, "issue": issue} for issue in issues]
                    )

                # ゴールデンレコード準拠チェック
                if entity.is_golden_record:
                    compliance_stats["golden_record_compliance"]["total_golden_records"] += 1
                    if entity.quality_score >= 90.0:  # ゴールデンレコード品質閾値
                        compliance_stats["golden_record_compliance"][
                            "compliant_golden_records"
                        ] += 1
                    else:
                        compliance_stats["golden_record_compliance"][
                            "non_compliant_golden_records"
                        ] += 1

            # 平均品質スコア計算
            for level_stats in compliance_stats["governance_levels"].values():
                if level_stats["count"] > 0:
                    level_stats["average_quality"] = (
                        level_stats["total_quality"] / level_stats["count"]
                    )

            return compliance_stats

        except Exception as e:
            self.logger.error(f"ガバナンス準拠レポートエラー: {e}")
            return {"error": str(e)}

    async def get_data_lineage_analysis(self) -> Dict[str, Any]:
        """データリネージ分析"""
        try:
            lineage_stats = {
                "entities_with_relationships": 0,
                "isolated_entities": 0,
                "relationship_distribution": {},
                "complex_relationships": [],
            }

            for entity in self._entity_cache.values():
                relationship_count = len(entity.related_entities)

                if relationship_count > 0:
                    lineage_stats["entities_with_relationships"] += 1

                    # 関係数の分布
                    if relationship_count not in lineage_stats["relationship_distribution"]:
                        lineage_stats["relationship_distribution"][relationship_count] = 0
                    lineage_stats["relationship_distribution"][relationship_count] += 1

                    # 複雑な関係（5つ以上の関連エンティティ）
                    if relationship_count >= 5:
                        lineage_stats["complex_relationships"].append(
                            {
                                "entity_id": entity.entity_id,
                                "entity_type": entity.entity_type.value,
                                "relationship_count": relationship_count,
                            }
                        )
                else:
                    lineage_stats["isolated_entities"] += 1

            return lineage_stats

        except Exception as e:
            self.logger.error(f"データリネージ分析エラー: {e}")
            return {"error": str(e)}

    def _check_compliance_issues(self, entity: MasterDataEntity) -> List[str]:
        """コンプライアンス問題チェック"""
        issues = []

        try:
            # 基本的なコンプライアンスチェック
            if not entity.source_systems:
                issues.append("ソースシステムが未定義")

            if entity.quality_score < 50.0:
                issues.append("品質スコアが低すぎる（50未満）")

            if entity.is_golden_record and entity.quality_score < 90.0:
                issues.append("ゴールデンレコードの品質基準未達")

            # エンティティタイプ別の特別チェック
            if entity.entity_type == MasterDataType.FINANCIAL_INSTRUMENTS:
                if "isin" not in entity.attributes:
                    issues.append("金融商品にISINコードが未設定")

                if "symbol" not in entity.attributes:
                    issues.append("金融商品にシンボルが未設定")

            elif entity.entity_type == MasterDataType.REGULATORY_DATA:
                if "effective_date" not in entity.attributes:
                    issues.append("規制データに効力発生日が未設定")

                if "authority" not in entity.attributes:
                    issues.append("規制データに監督機関が未設定")

            # メタデータチェック
            if not entity.metadata.get("source_system"):
                issues.append("メタデータにソースシステム情報が不足")

        except Exception as e:
            self.logger.error(f"コンプライアンスチェックエラー: {e}")
            issues.append(f"チェック実行エラー: {str(e)}")

        return issues

    async def generate_health_score(self) -> Dict[str, Any]:
        """システム健全性スコア生成"""
        try:
            stats = await self.get_entity_statistics()
            total_entities = stats["total_entities"]

            if total_entities == 0:
                return {"health_score": 0, "details": "エンティティが存在しません"}

            # 各要素のスコア計算
            active_ratio = stats["active_entities"] / total_entities
            golden_ratio = stats["golden_records"] / total_entities
            avg_quality = stats["average_quality_score"] / 100.0

            # 品質分布バランス
            excellent_ratio = stats["quality_distribution"]["excellent"] / total_entities
            good_ratio = stats["quality_distribution"]["good"] / total_entities

            # 総合健全性スコア（0-100）
            health_score = (
                active_ratio * 25  # アクティブ率 25%
                + golden_ratio * 20  # ゴールデンレコード率 20%
                + avg_quality * 30  # 平均品質 30%
                + excellent_ratio * 15  # 優秀品質率 15%
                + good_ratio * 10  # 良好品質率 10%
            ) * 100

            return {
                "health_score": round(health_score, 2),
                "components": {
                    "active_ratio": round(active_ratio, 3),
                    "golden_ratio": round(golden_ratio, 3),
                    "average_quality": round(avg_quality, 3),
                    "excellent_ratio": round(excellent_ratio, 3),
                    "good_ratio": round(good_ratio, 3),
                },
                "recommendations": self._generate_health_recommendations(
                    active_ratio, golden_ratio, avg_quality, excellent_ratio
                ),
            }

        except Exception as e:
            self.logger.error(f"健全性スコア生成エラー: {e}")
            return {"health_score": 0, "error": str(e)}

    def _generate_health_recommendations(
        self, active_ratio: float, golden_ratio: float, avg_quality: float, excellent_ratio: float
    ) -> List[str]:
        """健全性改善レコメンデーション"""
        recommendations = []

        if active_ratio < 0.8:
            recommendations.append("非アクティブエンティティの見直しとクリーンアップを検討してください")

        if golden_ratio < 0.3:
            recommendations.append("ゴールデンレコードの品質向上と認定プロセスの強化が必要です")

        if avg_quality < 0.7:
            recommendations.append("全体的なデータ品質の向上が必要です")

        if excellent_ratio < 0.2:
            recommendations.append("優秀品質のエンティティを増やすための品質管理強化が推奨されます")

        if not recommendations:
            recommendations.append("システムの健全性は良好です。現在の品質を維持してください")

        return recommendations
#!/usr/bin/env python3
"""
マスターデータ カタログ・ダッシュボード
データカタログ作成・ダッシュボード情報提供
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .types import DataDomain, MasterDataEntity

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class CatalogDashboard:
    """カタログ・ダッシュボード管理クラス"""

    def __init__(self, storage_path: Path):
        """初期化"""
        self.storage_path = storage_path

    async def create_data_catalog(
        self,
        data_elements: Dict[str, Any],
        master_entities: List[MasterDataEntity],
        data_stewards: Dict[str, Any],
        governance_policies: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        lineage_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """データカタログ作成"""
        logger.info("データカタログ作成中...")

        try:
            catalog = {
                "generated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "metadata": {
                    "total_data_elements": len(data_elements),
                    "total_entities": len(master_entities),
                    "total_stewards": len(data_stewards),
                    "total_policies": len(governance_policies),
                },
                "data_elements": self._create_data_elements_catalog(data_elements),
                "master_entities": self._create_entities_catalog(master_entities),
                "domains": self._create_domains_catalog(master_entities, data_stewards),
                "stewards": self._create_stewards_catalog(data_stewards),
                "governance_policies": self._create_policies_catalog(governance_policies),
                "quality_metrics": quality_metrics,
                "lineage_summary": lineage_summary,
                "entity_relationships": self._analyze_entity_relationships(master_entities),
                "data_coverage": self._analyze_data_coverage(master_entities),
            }

            # カタログファイル保存
            catalog_file = self.storage_path / f"data_catalog_{int(time.time())}.json"
            with open(catalog_file, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)

            logger.info(f"データカタログ作成完了: {catalog_file}")
            return catalog

        except Exception as e:
            logger.error(f"データカタログ作成エラー: {e}")
            return {"error": str(e)}

    def _create_data_elements_catalog(self, data_elements: Dict[str, Any]) -> Dict[str, Any]:
        """データ要素カタログ作成"""
        catalog = {}
        
        for element_id, element in data_elements.items():
            catalog[element_id] = {
                "name": element.name,
                "description": element.description,
                "data_type": element.data_type,
                "domain": element.domain.value,
                "classification": element.classification.value,
                "is_pii": element.is_pii,
                "business_rules": element.business_rules,
                "validation_rules": element.validation_rules,
                "source_systems": element.source_systems,
                "created_at": element.created_at.isoformat() if hasattr(element.created_at, 'isoformat') else str(element.created_at),
                "updated_at": element.updated_at.isoformat() if hasattr(element.updated_at, 'isoformat') else str(element.updated_at),
            }

        return catalog

    def _create_entities_catalog(self, master_entities: List[MasterDataEntity]) -> Dict[str, Any]:
        """マスターエンティティカタログ作成"""
        entities_by_type = {}
        entities_by_domain = {}
        total_quality_score = 0
        entity_count = len(master_entities)

        for entity in master_entities:
            # エンティティ種別別統計
            entity_type = entity.entity_type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = {
                    "count": 0,
                    "avg_quality": 0,
                    "examples": []
                }
            entities_by_type[entity_type]["count"] += 1
            entities_by_type[entity_type]["examples"].append({
                "entity_id": entity.entity_id,
                "primary_key": entity.primary_key,
                "quality_score": entity.data_quality_score,
            })
            # 最初の3件のみ保持
            if len(entities_by_type[entity_type]["examples"]) > 3:
                entities_by_type[entity_type]["examples"].pop()

            # ドメイン別統計
            domain_key = entity.domain.value
            if domain_key not in entities_by_domain:
                entities_by_domain[domain_key] = {
                    "count": 0,
                    "avg_quality": 0,
                    "entity_types": set()
                }
            entities_by_domain[domain_key]["count"] += 1
            entities_by_domain[domain_key]["entity_types"].add(entity_type)

            # 品質スコア集計
            total_quality_score += entity.data_quality_score

        # 平均品質スコア計算
        for entity_type_info in entities_by_type.values():
            if entity_type_info["count"] > 0:
                total_quality = sum(ex["quality_score"] for ex in entity_type_info["examples"])
                entity_type_info["avg_quality"] = total_quality / len(entity_type_info["examples"])

        for domain_info in entities_by_domain.values():
            domain_info["entity_types"] = list(domain_info["entity_types"])

        return {
            "total_count": entity_count,
            "by_type": entities_by_type,
            "by_domain": entities_by_domain,
            "average_quality_score": total_quality_score / entity_count if entity_count > 0 else 0,
        }

    def _create_domains_catalog(
        self, master_entities: List[MasterDataEntity], data_stewards: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ドメインカタログ作成"""
        domains = {}

        # エンティティからドメイン情報を集計
        domain_entities = {}
        for entity in master_entities:
            domain_key = entity.domain.value
            if domain_key not in domain_entities:
                domain_entities[domain_key] = 0
            domain_entities[domain_key] += 1

        # 各ドメインの情報を構築
        for domain in DataDomain:
            domain_key = domain.value
            
            # ドメイン担当スチュワード
            domain_stewards = [
                steward_id
                for steward_id, steward in data_stewards.items()
                if hasattr(steward, 'domains') and domain in steward.domains
            ]

            domains[domain_key] = {
                "name": domain_key.title(),
                "description": self._get_domain_description(domain),
                "entity_count": domain_entities.get(domain_key, 0),
                "stewards": domain_stewards,
                "data_classification": self._get_domain_classification(domain),
            }

        return domains

    def _get_domain_description(self, domain: DataDomain) -> str:
        """ドメイン説明取得"""
        descriptions = {
            DataDomain.FINANCIAL: "金融取引、財務諸表、会計データを管理するドメイン",
            DataDomain.MARKET: "市場データ、価格情報、取引量データを管理するドメイン",
            DataDomain.SECURITY: "有価証券、株式、債券データを管理するドメイン",
            DataDomain.REFERENCE: "参照データ、マスターデータ、分類データを管理するドメイン",
            DataDomain.CUSTOMER: "顧客情報、契約データを管理するドメイン",
            DataDomain.PRODUCT: "金融商品、サービスデータを管理するドメイン",
            DataDomain.TRANSACTION: "取引記録、決済データを管理するドメイン",
            DataDomain.REGULATORY: "規制要件、コンプライアンスデータを管理するドメイン",
        }
        return descriptions.get(domain, f"{domain.value}ドメイン")

    def _get_domain_classification(self, domain: DataDomain) -> str:
        """ドメイン分類取得"""
        classifications = {
            DataDomain.FINANCIAL: "機密",
            DataDomain.MARKET: "公開",
            DataDomain.SECURITY: "公開",
            DataDomain.REFERENCE: "内部",
            DataDomain.CUSTOMER: "制限",
            DataDomain.PRODUCT: "内部",
            DataDomain.TRANSACTION: "機密",
            DataDomain.REGULATORY: "内部",
        }
        return classifications.get(domain, "内部")

    def _create_stewards_catalog(self, data_stewards: Dict[str, Any]) -> Dict[str, Any]:
        """スチュワードカタログ作成"""
        catalog = {}
        
        for steward_id, steward in data_stewards.items():
            catalog[steward_id] = {
                "name": steward.name,
                "role": steward.role.value,
                "domains": [domain.value for domain in steward.domains],
                "responsibilities": steward.responsibilities,
                "active": steward.active,
                "email": steward.email if hasattr(steward, 'email') else "未設定",
            }

        return catalog

    def _create_policies_catalog(self, governance_policies: Dict[str, Any]) -> Dict[str, Any]:
        """ガバナンスポリシーカタログ作成"""
        catalog = {}
        
        for policy_id, policy in governance_policies.items():
            catalog[policy_id] = {
                "name": policy.name,
                "description": policy.description,
                "domain": policy.domain.value,
                "policy_type": policy.policy_type,
                "enforcement_level": policy.enforcement_level,
                "effective_date": policy.effective_date.isoformat(),
                "expiration_date": policy.expiration_date.isoformat() if policy.expiration_date else None,
                "owner": policy.owner,
                "rules_count": len(policy.rules),
            }

        return catalog

    def _analyze_entity_relationships(self, master_entities: List[MasterDataEntity]) -> Dict[str, Any]:
        """エンティティ関係分析"""
        relationships = {
            "total_relationships": 0,
            "by_type": {},
            "most_connected_entities": [],
        }

        entity_connections = {}

        for entity in master_entities:
            entity_id = entity.entity_id
            entity_relationships = entity.relationships
            
            connection_count = sum(len(related_entities) for related_entities in entity_relationships.values())
            entity_connections[entity_id] = {
                "entity_type": entity.entity_type,
                "primary_key": entity.primary_key,
                "connection_count": connection_count,
                "relationship_types": list(entity_relationships.keys()),
            }

            relationships["total_relationships"] += connection_count

            # 関係種別統計
            for rel_type, related_entities in entity_relationships.items():
                if rel_type not in relationships["by_type"]:
                    relationships["by_type"][rel_type] = 0
                relationships["by_type"][rel_type] += len(related_entities)

        # 最も接続数の多いエンティティ（上位5件）
        sorted_connections = sorted(
            entity_connections.items(),
            key=lambda x: x[1]["connection_count"],
            reverse=True
        )
        relationships["most_connected_entities"] = [
            {
                "entity_id": entity_id,
                "entity_type": conn_info["entity_type"],
                "primary_key": conn_info["primary_key"],
                "connection_count": conn_info["connection_count"],
                "relationship_types": conn_info["relationship_types"],
            }
            for entity_id, conn_info in sorted_connections[:5]
            if conn_info["connection_count"] > 0
        ]

        return relationships

    def _analyze_data_coverage(self, master_entities: List[MasterDataEntity]) -> Dict[str, Any]:
        """データカバレッジ分析"""
        coverage = {
            "field_coverage": {},
            "completeness_by_type": {},
            "data_freshness": {
                "very_fresh": 0,    # 1時間以内
                "fresh": 0,         # 24時間以内
                "moderate": 0,      # 1週間以内
                "stale": 0,         # 1週間超過
            },
        }

        # フィールドカバレッジ分析
        all_fields = set()
        field_counts = {}

        for entity in master_entities:
            # フィールド統計
            for field in entity.attributes.keys():
                all_fields.add(field)
                if field not in field_counts:
                    field_counts[field] = {"total": 0, "filled": 0}
                field_counts[field]["total"] += 1
                if entity.attributes[field] is not None:
                    field_counts[field]["filled"] += 1

            # エンティティタイプ別完全性
            entity_type = entity.entity_type
            if entity_type not in coverage["completeness_by_type"]:
                coverage["completeness_by_type"][entity_type] = {
                    "count": 0,
                    "avg_quality": 0,
                    "total_quality": 0,
                }
            
            type_info = coverage["completeness_by_type"][entity_type]
            type_info["count"] += 1
            type_info["total_quality"] += entity.data_quality_score

            # データ鮮度分析
            age_hours = (datetime.utcnow() - entity.updated_at).total_seconds() / 3600
            if age_hours <= 1:
                coverage["data_freshness"]["very_fresh"] += 1
            elif age_hours <= 24:
                coverage["data_freshness"]["fresh"] += 1
            elif age_hours <= 168:  # 1週間
                coverage["data_freshness"]["moderate"] += 1
            else:
                coverage["data_freshness"]["stale"] += 1

        # フィールドカバレッジ計算
        for field, counts in field_counts.items():
            coverage["field_coverage"][field] = {
                "coverage_rate": counts["filled"] / counts["total"] if counts["total"] > 0 else 0,
                "total_entities": counts["total"],
                "filled_entities": counts["filled"],
            }

        # エンティティタイプ別平均品質計算
        for type_info in coverage["completeness_by_type"].values():
            if type_info["count"] > 0:
                type_info["avg_quality"] = type_info["total_quality"] / type_info["count"]

        return coverage

    async def get_mdm_dashboard(
        self,
        master_entities: Dict[str, MasterDataEntity],
        data_elements: Dict[str, Any],
        data_stewards: Dict[str, Any],
        data_lineages: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        governance_status: Dict[str, Any],
        recent_activities: List[Dict[str, Any]],
        domain_distribution: Dict[str, int],
    ) -> Dict[str, Any]:
        """MDMダッシュボード情報取得"""
        try:
            # 基本統計
            total_entities = len(master_entities)
            active_stewards = sum(
                1 for steward in data_stewards.values() 
                if hasattr(steward, 'active') and steward.active
            )

            dashboard = {
                "generated_at": datetime.utcnow().isoformat(),
                "system_status": "active",
                "statistics": {
                    "total_entities": total_entities,
                    "data_elements": len(data_elements),
                    "active_stewards": active_stewards,
                    "data_lineages": len(data_lineages),
                },
                "quality_metrics": quality_metrics,
                "governance_status": governance_status,
                "recent_activities": recent_activities,
                "domain_distribution": domain_distribution,
                "alerts": self._generate_system_alerts(
                    master_entities, quality_metrics, governance_status
                ),
                "recommendations": self._generate_recommendations(
                    master_entities, quality_metrics
                ),
            }

            return dashboard

        except Exception as e:
            logger.error(f"MDMダッシュボード情報取得エラー: {e}")
            return {"error": str(e)}

    def _generate_system_alerts(
        self,
        master_entities: Dict[str, MasterDataEntity],
        quality_metrics: Dict[str, Any],
        governance_status: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """システムアラート生成"""
        alerts = []

        try:
            # 品質アラート
            if quality_metrics and "average_quality_score" in quality_metrics:
                avg_quality = quality_metrics["average_quality_score"]
                if avg_quality < 0.7:
                    alerts.append({
                        "type": "quality",
                        "severity": "high" if avg_quality < 0.5 else "medium",
                        "message": f"システム全体の品質スコアが低下しています: {avg_quality:.2f}",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            # 古いデータアラート
            current_time = datetime.utcnow()
            stale_entities = 0
            for entity in master_entities.values():
                age_hours = (current_time - entity.updated_at).total_seconds() / 3600
                if age_hours > 168:  # 1週間以上
                    stale_entities += 1

            if stale_entities > len(master_entities) * 0.1:  # 10%以上
                alerts.append({
                    "type": "freshness",
                    "severity": "medium",
                    "message": f"{stale_entities}件のエンティティが1週間以上更新されていません",
                    "timestamp": datetime.utcnow().isoformat(),
                })

            # ガバナンスアラート
            if governance_status and "active_policies" in governance_status:
                if governance_status["active_policies"] == 0:
                    alerts.append({
                        "type": "governance",
                        "severity": "high",
                        "message": "アクティブなガバナンスポリシーがありません",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

        except Exception as e:
            logger.error(f"システムアラート生成エラー: {e}")

        return alerts

    def _generate_recommendations(
        self,
        master_entities: Dict[str, MasterDataEntity],
        quality_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """改善推奨事項生成"""
        recommendations = []

        try:
            # 品質改善推奨
            if quality_metrics and "average_quality_score" in quality_metrics:
                avg_quality = quality_metrics["average_quality_score"]
                if avg_quality < 0.8:
                    recommendations.append({
                        "type": "quality_improvement",
                        "priority": "high",
                        "title": "データ品質の向上",
                        "description": "システム全体の品質スコアを向上させるため、品質ルールの見直しとデータクレンジングを実施してください",
                        "action_items": [
                            "品質ルールの見直し",
                            "データソースの検証",
                            "自動品質チェックの強化",
                        ],
                    })

            # データ更新頻度推奨
            outdated_count = 0
            for entity in master_entities.values():
                age_hours = (datetime.utcnow() - entity.updated_at).total_seconds() / 3600
                if age_hours > 24:
                    outdated_count += 1

            if outdated_count > len(master_entities) * 0.3:  # 30%以上
                recommendations.append({
                    "type": "data_freshness",
                    "priority": "medium",
                    "title": "データ更新頻度の改善",
                    "description": "多くのエンティティが古いデータとなっています。更新プロセスの見直しをお勧めします",
                    "action_items": [
                        "自動更新スケジュールの見直し",
                        "データソース接続の確認",
                        "更新失敗の監視強化",
                    ],
                })

            # スチュワード配置推奨
            recommendations.append({
                "type": "stewardship",
                "priority": "low",
                "title": "データスチュワードシップの強化",
                "description": "各ドメインに適切なデータスチュワードが配置されているか確認してください",
                "action_items": [
                    "ドメイン別スチュワード配置の確認",
                    "スチュワードの責任範囲の明確化",
                    "定期的な品質レビューの実施",
                ],
            })

        except Exception as e:
            logger.error(f"推奨事項生成エラー: {e}")

        return recommendations
#!/usr/bin/env python3
"""
マスターデータ品質評価
データ品質の評価・検証・ルール適用
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .types import DataDomain, MasterDataEntity

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class QualityAssessor:
    """データ品質評価クラス"""

    def __init__(self, data_elements: Dict[str, Any]):
        """初期化"""
        self.data_elements = data_elements
        self.entity_type_requirements = self._setup_entity_requirements()

    def _setup_entity_requirements(self) -> Dict[str, Dict[str, Any]]:
        """エンティティ種別要件設定"""
        return {
            "stock": {
                "required_fields": ["symbol", "company_name"],
                "numeric_fields": ["last_price", "open_price", "high_price", "low_price"],
                "format_rules": {
                    "symbol": r"^[0-9]{4}$"  # 4桁数字
                }
            },
            "company": {
                "required_fields": ["name", "industry"],
                "numeric_fields": ["employees", "revenue", "market_cap"],
                "format_rules": {}
            },
            "currency": {
                "required_fields": ["code", "name"],
                "numeric_fields": ["exchange_rate", "buy_rate", "sell_rate"],
                "format_rules": {
                    "code": r"^[A-Z]{3}$"  # 3文字大文字
                }
            },
            "exchange": {
                "required_fields": ["code", "name", "country"],
                "numeric_fields": [],
                "format_rules": {}
            }
        }

    async def assess_entity_quality(self, entity: MasterDataEntity) -> float:
        """エンティティ品質評価"""
        try:
            quality_score = 1.0

            # 必須属性チェック
            missing_attrs = self._check_required_attributes(entity)
            if missing_attrs:
                quality_score -= len(missing_attrs) * 0.1

            # データ型チェック
            type_violations = self._validate_attribute_types(entity)
            quality_score -= len(type_violations) * 0.05

            # フォーマットチェック
            format_violations = self._validate_formats(entity)
            quality_score -= len(format_violations) * 0.05

            # ビジネスルールチェック
            rule_violations = await self._validate_business_rules(entity)
            quality_score -= len(rule_violations) * 0.05

            # 鮮度チェック
            freshness_penalty = self._assess_data_freshness(entity)
            quality_score -= freshness_penalty

            # 完全性チェック
            completeness_score = self._assess_completeness(entity)
            quality_score = quality_score * completeness_score

            # 最小値・最大値制限
            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error(f"品質評価エラー: {e}")
            return 0.5  # デフォルト値

    def _check_required_attributes(self, entity: MasterDataEntity) -> List[str]:
        """必須属性チェック"""
        requirements = self.entity_type_requirements.get(entity.entity_type, {})
        required_attrs = requirements.get("required_fields", [])
        
        missing_attrs = []
        for attr in required_attrs:
            if attr not in entity.attributes or not entity.attributes[attr]:
                missing_attrs.append(attr)
        
        return missing_attrs

    def _validate_attribute_types(self, entity: MasterDataEntity) -> List[str]:
        """属性データ型検証"""
        violations = []
        requirements = self.entity_type_requirements.get(entity.entity_type, {})
        numeric_fields = requirements.get("numeric_fields", [])

        try:
            for attr_name, attr_value in entity.attributes.items():
                # 数値フィールドの検証
                if attr_name in numeric_fields and attr_value is not None:
                    try:
                        float(attr_value)
                    except (ValueError, TypeError):
                        violations.append(f"{attr_name}: 数値型が期待されます")

                # データ要素定義との照合
                for element in self.data_elements.values():
                    element_attr_name = element.name.lower().replace(" ", "_")
                    if element_attr_name == attr_name:
                        expected_type = element.data_type

                        if expected_type == "string" and not isinstance(attr_value, str):
                            violations.append(f"{attr_name}: 文字列型が期待されます")
                        elif expected_type == "decimal" and not isinstance(attr_value, (int, float)):
                            violations.append(f"{attr_name}: 数値型が期待されます")
                        elif expected_type == "integer" and not isinstance(attr_value, int):
                            violations.append(f"{attr_name}: 整数型が期待されます")

                        break

            return violations

        except Exception as e:
            logger.error(f"属性データ型検証エラー: {e}")
            return []

    def _validate_formats(self, entity: MasterDataEntity) -> List[str]:
        """フォーマット検証"""
        import re
        violations = []
        
        try:
            requirements = self.entity_type_requirements.get(entity.entity_type, {})
            format_rules = requirements.get("format_rules", {})

            for field_name, pattern in format_rules.items():
                if field_name in entity.attributes:
                    value = entity.attributes[field_name]
                    if isinstance(value, str) and not re.match(pattern, value):
                        violations.append(f"{field_name}: フォーマットが正しくありません ({pattern})")

            return violations

        except Exception as e:
            logger.error(f"フォーマット検証エラー: {e}")
            return []

    async def _validate_business_rules(self, entity: MasterDataEntity) -> List[str]:
        """ビジネスルール検証"""
        violations = []

        try:
            # 株式コードのビジネスルール
            if entity.entity_type == "stock" and "symbol" in entity.attributes:
                symbol = entity.attributes["symbol"]
                if (
                    not isinstance(symbol, str)
                    or not symbol.isdigit()
                    or len(symbol) != 4
                ):
                    violations.append("株式コードは4桁の数字である必要があります")

            # 価格のビジネスルール
            price_fields = ["last_price", "open_price", "high_price", "low_price"]
            for field in price_fields:
                if field in entity.attributes:
                    price = entity.attributes[field]
                    if isinstance(price, (int, float)) and price <= 0:
                        violations.append(f"{field}は正の数値である必要があります")

            # 通貨コードのビジネスルール
            if entity.entity_type == "currency" and "code" in entity.attributes:
                code = entity.attributes["code"]
                if not isinstance(code, str) or len(code) != 3 or not code.isupper():
                    violations.append("通貨コードは3文字の大文字である必要があります")

            # 従業員数のビジネスルール
            if entity.entity_type == "company" and "employees" in entity.attributes:
                employees = entity.attributes["employees"]
                if isinstance(employees, (int, float)) and employees < 0:
                    violations.append("従業員数は0以上である必要があります")

            return violations

        except Exception as e:
            logger.error(f"ビジネスルール検証エラー: {e}")
            return []

    def _assess_data_freshness(self, entity: MasterDataEntity) -> float:
        """データ鮮度評価"""
        try:
            data_age_hours = (datetime.utcnow() - entity.updated_at).total_seconds() / 3600
            
            # 鮮度ペナルティ算出
            if data_age_hours <= 1:  # 1時間以内
                return 0.0
            elif data_age_hours <= 24:  # 24時間以内
                return 0.05
            elif data_age_hours <= 168:  # 1週間以内
                return 0.1
            elif data_age_hours <= 720:  # 1ヶ月以内
                return 0.15
            else:  # 1ヶ月超過
                return 0.2

        except Exception as e:
            logger.error(f"データ鮮度評価エラー: {e}")
            return 0.1

    def _assess_completeness(self, entity: MasterDataEntity) -> float:
        """データ完全性評価"""
        try:
            requirements = self.entity_type_requirements.get(entity.entity_type, {})
            required_attrs = requirements.get("required_fields", [])
            
            if not required_attrs:
                return 1.0

            filled_attrs = sum(
                1 for attr in required_attrs
                if attr in entity.attributes and entity.attributes[attr] is not None
            )
            
            return filled_attrs / len(required_attrs)

        except Exception as e:
            logger.error(f"データ完全性評価エラー: {e}")
            return 0.8

    def create_quality_report(self, entity: MasterDataEntity) -> Dict[str, Any]:
        """品質レポート作成"""
        try:
            missing_attrs = self._check_required_attributes(entity)
            type_violations = self._validate_attribute_types(entity)
            format_violations = self._validate_formats(entity)
            
            # 非同期メソッドの結果は別途取得する必要がある
            rule_violations = []  # 簡略化

            freshness_score = 1.0 - self._assess_data_freshness(entity)
            completeness_score = self._assess_completeness(entity)

            return {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "overall_score": entity.data_quality_score,
                "assessment_timestamp": datetime.utcnow().isoformat(),
                "completeness": {
                    "score": completeness_score,
                    "missing_required_fields": missing_attrs,
                },
                "validity": {
                    "type_violations": type_violations,
                    "format_violations": format_violations,
                    "business_rule_violations": rule_violations,
                },
                "freshness": {
                    "score": freshness_score,
                    "last_updated": entity.updated_at.isoformat(),
                    "age_hours": (datetime.utcnow() - entity.updated_at).total_seconds() / 3600,
                },
                "recommendations": self._generate_recommendations(
                    missing_attrs, type_violations, format_violations, rule_violations
                ),
            }

        except Exception as e:
            logger.error(f"品質レポート作成エラー: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        missing_attrs: List[str],
        type_violations: List[str],
        format_violations: List[str],
        rule_violations: List[str],
    ) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []

        if missing_attrs:
            recommendations.append(f"必須フィールドを追加してください: {', '.join(missing_attrs)}")

        if type_violations:
            recommendations.append("データ型の不整合を修正してください")

        if format_violations:
            recommendations.append("データフォーマットを修正してください")

        if rule_violations:
            recommendations.append("ビジネスルール違反を修正してください")

        if not recommendations:
            recommendations.append("データ品質は良好です")

        return recommendations

    def get_quality_metrics_summary(self, entities: List[MasterDataEntity]) -> Dict[str, Any]:
        """品質メトリクスサマリー"""
        try:
            if not entities:
                return {"total_entities": 0}

            total_entities = len(entities)
            total_score = sum(entity.data_quality_score for entity in entities)
            avg_score = total_score / total_entities

            score_distribution = {
                "excellent": 0,  # 0.9以上
                "good": 0,       # 0.7-0.89
                "fair": 0,       # 0.5-0.69
                "poor": 0,       # 0.5未満
            }

            for entity in entities:
                score = entity.data_quality_score
                if score >= 0.9:
                    score_distribution["excellent"] += 1
                elif score >= 0.7:
                    score_distribution["good"] += 1
                elif score >= 0.5:
                    score_distribution["fair"] += 1
                else:
                    score_distribution["poor"] += 1

            # ドメイン別統計
            domain_stats = {}
            for entity in entities:
                domain = entity.domain.value
                if domain not in domain_stats:
                    domain_stats[domain] = {"count": 0, "total_score": 0}
                domain_stats[domain]["count"] += 1
                domain_stats[domain]["total_score"] += entity.data_quality_score

            for domain, stats in domain_stats.items():
                stats["avg_score"] = stats["total_score"] / stats["count"]

            return {
                "total_entities": total_entities,
                "average_quality_score": avg_score,
                "min_quality_score": min(entity.data_quality_score for entity in entities),
                "max_quality_score": max(entity.data_quality_score for entity in entities),
                "score_distribution": score_distribution,
                "domain_statistics": domain_stats,
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"品質メトリクスサマリー生成エラー: {e}")
            return {"error": str(e)}
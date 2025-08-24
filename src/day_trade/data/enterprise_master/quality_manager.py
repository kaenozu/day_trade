#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - データ品質管理

このモジュールは、マスターデータの品質評価、チェック、改善機能を提供します。
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .enums import (
    DataGovernanceLevel,
    MasterDataType,
    QualityScoreLevel,
    get_quality_level,
)
from .models import (
    DataGovernancePolicy,
    MasterDataEntity,
    QualityHistory,
    QualityMetric,
)


class QualityManager:
    """データ品質管理クラス
    
    マスターデータの品質評価、バリデーション、品質履歴管理を担当します。
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._quality_rules = self._initialize_quality_rules()
        self._validation_cache = {}

    def _initialize_quality_rules(self) -> Dict[str, Dict[str, Any]]:
        """品質ルール初期化
        
        Returns:
            Dict[str, Dict[str, Any]]: エンティティタイプ別の品質ルール
        """
        return {
            MasterDataType.FINANCIAL_INSTRUMENTS.value: {
                "critical_fields": ["symbol", "name", "isin", "market"],
                "optional_fields": ["sector", "industry", "currency"],
                "patterns": {
                    "isin": r"^[A-Z]{2}[A-Z0-9]{10}$",
                    "symbol": r"^[A-Z0-9]{1,10}$",
                },
                "length_constraints": {
                    "name": {"min": 2, "max": 200},
                    "symbol": {"min": 1, "max": 10},
                },
            },
            MasterDataType.EXCHANGE_CODES.value: {
                "critical_fields": ["code", "description"],
                "optional_fields": ["country", "timezone"],
                "patterns": {
                    "code": r"^[A-Z]{2,10}$",
                },
                "length_constraints": {
                    "description": {"min": 5, "max": 100},
                },
            },
            MasterDataType.CURRENCY_CODES.value: {
                "critical_fields": ["code", "name"],
                "optional_fields": ["symbol"],
                "patterns": {
                    "code": r"^[A-Z]{3}$",
                },
                "length_constraints": {
                    "name": {"min": 3, "max": 50},
                },
            },
        }

    async def calculate_entity_quality(
        self,
        entity: MasterDataEntity,
        policy: Optional[DataGovernancePolicy] = None
    ) -> Tuple[float, Dict[str, QualityMetric]]:
        """エンティティ品質スコア計算
        
        Args:
            entity: 評価対象エンティティ
            policy: 適用するガバナンスポリシー
            
        Returns:
            Tuple[float, Dict[str, QualityMetric]]: 品質スコアと詳細メトリック
        """
        try:
            metrics = {}
            total_weighted_score = 0.0
            total_weight = 0.0

            # 完全性チェック
            completeness_metric = await self._evaluate_completeness(entity, policy)
            metrics["completeness"] = completeness_metric
            total_weighted_score += completeness_metric.score * completeness_metric.weight
            total_weight += completeness_metric.weight

            # 正確性チェック
            accuracy_metric = await self._evaluate_accuracy(entity, policy)
            metrics["accuracy"] = accuracy_metric
            total_weighted_score += accuracy_metric.score * accuracy_metric.weight
            total_weight += accuracy_metric.weight

            # 一貫性チェック
            consistency_metric = await self._evaluate_consistency(entity)
            metrics["consistency"] = consistency_metric
            total_weighted_score += consistency_metric.score * consistency_metric.weight
            total_weight += consistency_metric.weight

            # 適時性チェック
            timeliness_metric = await self._evaluate_timeliness(entity)
            metrics["timeliness"] = timeliness_metric
            total_weighted_score += timeliness_metric.score * timeliness_metric.weight
            total_weight += timeliness_metric.weight

            # 有効性チェック
            validity_metric = await self._evaluate_validity(entity, policy)
            metrics["validity"] = validity_metric
            total_weighted_score += validity_metric.score * validity_metric.weight
            total_weight += validity_metric.weight

            # 加重平均スコア計算
            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

            self.logger.debug(f"品質スコア計算完了: {entity.entity_id} -> {overall_score:.2f}")
            return overall_score, metrics

        except Exception as e:
            self.logger.error(f"品質スコア計算エラー: {e}")
            return 0.0, {}

    async def _evaluate_completeness(
        self,
        entity: MasterDataEntity,
        policy: Optional[DataGovernancePolicy]
    ) -> QualityMetric:
        """完全性評価
        
        Args:
            entity: 評価対象エンティティ
            policy: ガバナンスポリシー
            
        Returns:
            QualityMetric: 完全性メトリック
        """
        issues = []
        score = 100.0
        
        # 必須フィールドチェック
        required_fields = []
        if policy:
            for rule in policy.rules:
                if rule.get("required", False):
                    required_fields.append(rule.get("field"))
        
        # デフォルト必須フィールド（ポリシーがない場合）
        if not required_fields:
            quality_rules = self._quality_rules.get(entity.entity_type.value, {})
            required_fields = quality_rules.get("critical_fields", [])

        missing_fields = []
        for field in required_fields:
            value = entity.attributes.get(field)
            if value is None or value == "":
                missing_fields.append(field)
                score -= 20  # 必須フィールド不足は重大な減点

        if missing_fields:
            issues.append(f"必須フィールドが不足: {', '.join(missing_fields)}")

        # オプションフィールドの充実度チェック
        quality_rules = self._quality_rules.get(entity.entity_type.value, {})
        optional_fields = quality_rules.get("optional_fields", [])
        
        if optional_fields:
            filled_optional = sum(
                1 for field in optional_fields
                if entity.attributes.get(field) not in [None, ""]
            )
            optional_ratio = filled_optional / len(optional_fields)
            bonus_score = optional_ratio * 15  # 最大15点のボーナス
            score = min(100.0, score + bonus_score)

        return QualityMetric(
            metric_name="completeness",
            score=max(0.0, score),
            weight=3.0,  # 完全性は重要度が高い
            issues=issues,
            recommendations=[
                f"不足している必須フィールドを入力してください: {', '.join(missing_fields)}"
            ] if missing_fields else [],
        )

    async def _evaluate_accuracy(
        self,
        entity: MasterDataEntity,
        policy: Optional[DataGovernancePolicy]
    ) -> QualityMetric:
        """正確性評価
        
        Args:
            entity: 評価対象エンティティ
            policy: ガバナンスポリシー
            
        Returns:
            QualityMetric: 正確性メトリック
        """
        issues = []
        score = 100.0
        recommendations = []

        # パターンマッチング検証
        patterns = {}
        if policy:
            for rule in policy.rules:
                if rule.get("pattern"):
                    patterns[rule.get("field")] = rule["pattern"]
        else:
            quality_rules = self._quality_rules.get(entity.entity_type.value, {})
            patterns = quality_rules.get("patterns", {})

        for field, pattern in patterns.items():
            value = entity.attributes.get(field)
            if value and isinstance(value, str):
                if not re.match(pattern, value):
                    issues.append(f"フィールド '{field}' がパターン '{pattern}' と一致しません")
                    score -= 15
                    recommendations.append(f"'{field}' フィールドの形式を確認してください")

        # 長さ制約検証
        quality_rules = self._quality_rules.get(entity.entity_type.value, {})
        length_constraints = quality_rules.get("length_constraints", {})

        for field, constraints in length_constraints.items():
            value = entity.attributes.get(field)
            if value and isinstance(value, str):
                if len(value) < constraints.get("min", 0):
                    issues.append(f"'{field}' の長さが最小値 {constraints['min']} 未満です")
                    score -= 10
                elif len(value) > constraints.get("max", 1000):
                    issues.append(f"'{field}' の長さが最大値 {constraints['max']} を超えています")
                    score -= 5

        # データ型検証
        if policy:
            for rule in policy.rules:
                field = rule.get("field")
                expected_type = rule.get("type")
                if field and expected_type:
                    value = entity.attributes.get(field)
                    if value is not None:
                        if not self._validate_data_type(value, expected_type):
                            issues.append(f"'{field}' のデータ型が不正です")
                            score -= 12
                            recommendations.append(f"'{field}' の値を {expected_type} 形式で入力してください")

        return QualityMetric(
            metric_name="accuracy",
            score=max(0.0, score),
            weight=2.5,
            issues=issues,
            recommendations=recommendations,
        )

    async def _evaluate_consistency(self, entity: MasterDataEntity) -> QualityMetric:
        """一貫性評価
        
        Args:
            entity: 評価対象エンティティ
            
        Returns:
            QualityMetric: 一貫性メトリック
        """
        issues = []
        score = 100.0
        recommendations = []

        # 関連フィールド間の一貫性チェック
        if entity.entity_type == MasterDataType.FINANCIAL_INSTRUMENTS:
            # 金融商品の場合の一貫性チェック例
            market = entity.attributes.get("market")
            currency = entity.attributes.get("currency")
            
            if market == "TSE" and currency and currency not in ["JPY", None]:
                issues.append("東証上場銘柄の通貨がJPYでない")
                score -= 15
                recommendations.append("東証上場銘柄の通貨をJPYに設定してください")

        # メタデータ一貫性チェック
        if entity.created_at > entity.updated_at:
            issues.append("作成日時が更新日時より後になっています")
            score -= 20

        # バージョン一貫性チェック
        if entity.version < 1:
            issues.append("バージョン番号が無効です")
            score -= 10

        return QualityMetric(
            metric_name="consistency",
            score=max(0.0, score),
            weight=2.0,
            issues=issues,
            recommendations=recommendations,
        )

    async def _evaluate_timeliness(self, entity: MasterDataEntity) -> QualityMetric:
        """適時性評価
        
        Args:
            entity: 評価対象エンティティ
            
        Returns:
            QualityMetric: 適時性メトリック
        """
        issues = []
        score = 100.0
        recommendations = []

        now = datetime.now(entity.updated_at.tzinfo or datetime.now().astimezone().tzinfo)
        days_since_update = (now - entity.updated_at).days

        # エンティティタイプ別の更新頻度要件
        update_thresholds = {
            MasterDataType.FINANCIAL_INSTRUMENTS: 30,  # 30日
            MasterDataType.EXCHANGE_CODES: 365,        # 1年
            MasterDataType.CURRENCY_CODES: 180,        # 6ヶ月
        }

        threshold = update_thresholds.get(entity.entity_type, 90)  # デフォルト90日

        if days_since_update > threshold:
            penalty = min(50, (days_since_update - threshold) * 2)  # 最大50点減点
            score -= penalty
            issues.append(f"{days_since_update}日間更新されていません（推奨: {threshold}日以内）")
            recommendations.append("データの最新性を確認し、必要に応じて更新してください")

        return QualityMetric(
            metric_name="timeliness",
            score=max(0.0, score),
            weight=1.5,
            issues=issues,
            recommendations=recommendations,
        )

    async def _evaluate_validity(
        self,
        entity: MasterDataEntity,
        policy: Optional[DataGovernancePolicy]
    ) -> QualityMetric:
        """有効性評価
        
        Args:
            entity: 評価対象エンティティ
            policy: ガバナンスポリシー
            
        Returns:
            QualityMetric: 有効性メトリック
        """
        issues = []
        score = 100.0
        recommendations = []

        # アクティブ状態チェック
        if not entity.is_active:
            issues.append("エンティティが非アクティブ状態です")
            score -= 30

        # ガバナンスレベルと品質スコアの整合性
        if policy:
            min_quality = policy.quality_threshold
            if entity.quality_score < min_quality:
                issues.append(f"品質スコアが要求レベル {min_quality} を下回っています")
                score -= 20
                recommendations.append("データ品質の改善が必要です")

        # ソースシステム情報の有効性
        if not entity.source_systems:
            issues.append("ソースシステム情報が不足しています")
            score -= 15

        return QualityMetric(
            metric_name="validity",
            score=max(0.0, score),
            weight=2.0,
            issues=issues,
            recommendations=recommendations,
        )

    def _validate_data_type(self, value: Any, expected_type: str) -> bool:
        """データ型バリデーション
        
        Args:
            value: 検証対象の値
            expected_type: 期待されるデータ型
            
        Returns:
            bool: バリデーション結果
        """
        try:
            if expected_type == "date":
                datetime.fromisoformat(str(value))
                return True
            elif expected_type == "number":
                return isinstance(value, (int, float))
            elif expected_type == "string":
                return isinstance(value, str)
            elif expected_type == "boolean":
                return isinstance(value, bool)
            else:
                return True  # 不明な型は通す
        except (ValueError, TypeError):
            return False

    async def validate_entity_attributes(
        self,
        attributes: Dict[str, Any],
        entity_type: MasterDataType,
        policy: Optional[DataGovernancePolicy] = None
    ) -> Dict[str, Any]:
        """エンティティ属性バリデーション
        
        Args:
            attributes: バリデーション対象属性
            entity_type: エンティティタイプ
            policy: ガバナンスポリシー
            
        Returns:
            Dict[str, Any]: バリデーション結果
        """
        errors = []
        warnings = []

        try:
            if policy:
                # ポリシーベースのバリデーション
                for rule in policy.rules:
                    field_name = rule.get("field")
                    if not field_name:
                        continue

                    field_value = attributes.get(field_name)

                    # 必須フィールドチェック
                    if rule.get("required", False):
                        if field_value is None or field_value == "":
                            errors.append(f"必須フィールド '{field_name}' が未設定です")
                            continue

                    if field_value is None:
                        continue

                    # パターンマッチングチェック
                    if rule.get("pattern") and isinstance(field_value, str):
                        if not re.match(rule["pattern"], field_value):
                            errors.append(
                                f"フィールド '{field_name}' がパターン '{rule['pattern']}' と一致しません"
                            )

                    # 長さチェック
                    if rule.get("min_length") and isinstance(field_value, str):
                        if len(field_value) < rule["min_length"]:
                            errors.append(
                                f"フィールド '{field_name}' の長さが最小値 {rule['min_length']} 未満です"
                            )

                    if rule.get("max_length") and isinstance(field_value, str):
                        if len(field_value) > rule["max_length"]:
                            warnings.append(
                                f"フィールド '{field_name}' の長さが推奨最大値 {rule['max_length']} を超えています"
                            )

                    # 一意性チェック（簡易実装）
                    if rule.get("unique", False):
                        # 実際の実装では、データベースでの重複チェックが必要
                        pass

            else:
                # デフォルトルールベースのバリデーション
                quality_rules = self._quality_rules.get(entity_type.value, {})
                critical_fields = quality_rules.get("critical_fields", [])
                
                for field in critical_fields:
                    if field not in attributes or attributes[field] in [None, ""]:
                        errors.append(f"重要フィールド '{field}' が未設定です")

            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "validation_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"バリデーションエラー: {e}")
            return {
                "is_valid": False,
                "errors": [f"バリデーション実行エラー: {str(e)}"],
                "warnings": [],
                "validation_timestamp": datetime.now().isoformat(),
            }

    def get_quality_recommendations(
        self, 
        entity: MasterDataEntity,
        metrics: Dict[str, QualityMetric]
    ) -> List[str]:
        """品質改善推奨事項の生成
        
        Args:
            entity: 対象エンティティ
            metrics: 品質メトリック
            
        Returns:
            List[str]: 推奨事項リスト
        """
        recommendations = []
        
        for metric in metrics.values():
            recommendations.extend(metric.recommendations)
        
        # 全体的な推奨事項
        quality_level = get_quality_level(entity.quality_score)
        
        if quality_level == QualityScoreLevel.POOR:
            recommendations.append("データの包括的な見直しが必要です")
        elif quality_level == QualityScoreLevel.FAIR:
            recommendations.append("主要フィールドの充実化を検討してください")
        elif quality_level == QualityScoreLevel.GOOD:
            recommendations.append("さらなる品質向上のため、オプション情報の追加を検討してください")
        
        return list(set(recommendations))  # 重複除去
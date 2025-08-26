#!/usr/bin/env python3
"""
データ品質ダッシュボード - 品質メトリクス管理
Quality metrics calculation and KPI management functionality
"""

import logging
from typing import Any, Dict

from .models import QualityKPI


logger = logging.getLogger(__name__)


class QualityMetricsManager:
    """品質メトリクス管理クラス"""

    def __init__(self, data_source_manager=None):
        self.data_source_manager = data_source_manager
        self.quality_kpis: Dict[str, QualityKPI] = {}
        self._setup_default_kpis()

    def _setup_default_kpis(self):
        """デフォルト品質KPI設定"""
        # データ完全性KPI
        completeness_kpi = QualityKPI(
            kpi_id="data_completeness",
            name="データ完全性",
            description="必須フィールドの完全性率",
            target_value=0.95,
            current_value=0.92,
            unit="percentage",
            status="warning",
        )
        self.quality_kpis[completeness_kpi.kpi_id] = completeness_kpi

        # データ正確性KPI
        accuracy_kpi = QualityKPI(
            kpi_id="data_accuracy",
            name="データ正確性",
            description="ビジネスルール準拠率",
            target_value=0.98,
            current_value=0.96,
            unit="percentage",
            status="warning",
        )
        self.quality_kpis[accuracy_kpi.kpi_id] = accuracy_kpi

        # SLA準拠KPI
        sla_kpi = QualityKPI(
            kpi_id="sla_compliance",
            name="SLA準拠率",
            description="データSLA要件準拠率",
            target_value=0.999,
            current_value=0.995,
            unit="percentage",
            status="healthy",
        )
        self.quality_kpis[sla_kpi.kpi_id] = sla_kpi

    async def get_quality_kpis_data(self) -> Dict[str, Any]:
        """品質KPIデータ取得"""
        try:
            kpi_data = {}
            for kpi_id, kpi in self.quality_kpis.items():
                kpi_data[kpi_id] = {
                    "name": kpi.name,
                    "current_value": kpi.current_value,
                    "target_value": kpi.target_value,
                    "unit": kpi.unit,
                    "status": kpi.status,
                    "trend": kpi.trend,
                    "achievement_rate": (
                        (kpi.current_value / kpi.target_value) * 100
                        if kpi.target_value > 0
                        else 0
                    ),
                }

            return {"kpis": kpi_data}

        except Exception as e:
            logger.error(f"品質KPIデータ取得エラー: {e}")
            return {"kpis": {}}

    def get_kpi(self, kpi_id: str) -> QualityKPI:
        """KPI取得"""
        return self.quality_kpis.get(kpi_id)

    def update_kpi_value(self, kpi_id: str, new_value: float) -> bool:
        """KPI値更新"""
        try:
            if kpi_id in self.quality_kpis:
                kpi = self.quality_kpis[kpi_id]
                old_value = kpi.current_value
                kpi.current_value = new_value

                # トレンド計算
                if new_value > old_value:
                    kpi.trend = "up"
                elif new_value < old_value:
                    kpi.trend = "down"
                else:
                    kpi.trend = "stable"

                # ステータス更新
                achievement_rate = (
                    (new_value / kpi.target_value) if kpi.target_value > 0 else 0
                )

                if achievement_rate >= 0.95:
                    kpi.status = "healthy"
                elif achievement_rate >= 0.85:
                    kpi.status = "warning"
                else:
                    kpi.status = "critical"

                return True

            return False

        except Exception as e:
            logger.error(f"KPI値更新エラー ({kpi_id}): {e}")
            return False

    def add_custom_kpi(self, kpi: QualityKPI) -> bool:
        """カスタムKPI追加"""
        try:
            self.quality_kpis[kpi.kpi_id] = kpi
            return True
        except Exception as e:
            logger.error(f"カスタムKPI追加エラー: {e}")
            return False

    def remove_kpi(self, kpi_id: str) -> bool:
        """KPI削除"""
        try:
            if kpi_id in self.quality_kpis:
                del self.quality_kpis[kpi_id]
                return True
            return False
        except Exception as e:
            logger.error(f"KPI削除エラー ({kpi_id}): {e}")
            return False

    async def calculate_overall_score(self) -> float:
        """総合品質スコア計算"""
        try:
            if not self.quality_kpis:
                return 0.0

            total_score = 0.0
            total_weight = 0.0

            for kpi in self.quality_kpis.values():
                achievement_rate = (
                    (kpi.current_value / kpi.target_value)
                    if kpi.target_value > 0
                    else 0
                )
                weight = 1.0  # 等重み（将来的に重み設定機能を追加可能）

                total_score += achievement_rate * weight
                total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"総合品質スコア計算エラー: {e}")
            return 0.0

    def get_kpi_summary(self) -> Dict[str, Any]:
        """KPI概要取得"""
        try:
            healthy_count = len([kpi for kpi in self.quality_kpis.values() if kpi.status == "healthy"])
            warning_count = len([kpi for kpi in self.quality_kpis.values() if kpi.status == "warning"])
            critical_count = len([kpi for kpi in self.quality_kpis.values() if kpi.status == "critical"])

            return {
                "total_kpis": len(self.quality_kpis),
                "healthy_kpis": healthy_count,
                "warning_kpis": warning_count,
                "critical_kpis": critical_count,
                "health_percentage": (healthy_count / len(self.quality_kpis) * 100) if self.quality_kpis else 0,
            }

        except Exception as e:
            logger.error(f"KPI概要取得エラー: {e}")
            return {
                "total_kpis": 0,
                "healthy_kpis": 0,
                "warning_kpis": 0,
                "critical_kpis": 0,
                "health_percentage": 0,
            }
#!/usr/bin/env python3
"""
データ品質ダッシュボード - ウィジェット管理
Widget and layout management functionality
"""

import logging
from typing import Dict, List

from .enums import ChartType, DashboardComponentType
from .models import DashboardLayout, DashboardWidget, QualityKPI


logger = logging.getLogger(__name__)


class WidgetManager:
    """ダッシュボードウィジェット管理クラス"""

    def __init__(self):
        self.dashboard_layouts: Dict[str, DashboardLayout] = {}
        self.active_widgets: Dict[str, DashboardWidget] = {}
        self.quality_kpis: Dict[str, QualityKPI] = {}

        # デフォルト設定の初期化
        self._setup_default_layouts()
        self._setup_default_kpis()

    def _setup_default_layouts(self):
        """デフォルトダッシュボードレイアウト設定"""
        # メインダッシュボードレイアウト
        main_widgets = [
            # 品質スコアメトリクスカード
            DashboardWidget(
                widget_id="quality_score_card",
                title="総合データ品質スコア",
                component_type=DashboardComponentType.METRIC_CARD,
                data_source="quality_metrics",
                position={"x": 0, "y": 0, "w": 3, "h": 2},
                config={
                    "metric": "overall_quality_score",
                    "format": "percentage",
                    "color_scheme": "quality",
                },
            ),
            # 鮮度メトリクスカード
            DashboardWidget(
                widget_id="freshness_card",
                title="データ鮮度",
                component_type=DashboardComponentType.METRIC_CARD,
                data_source="freshness_metrics",
                position={"x": 3, "y": 0, "w": 3, "h": 2},
                config={
                    "metric": "average_data_age",
                    "format": "minutes",
                    "color_scheme": "freshness",
                },
            ),
            # 可用性メトリクスカード
            DashboardWidget(
                widget_id="availability_card",
                title="システム可用性",
                component_type=DashboardComponentType.METRIC_CARD,
                data_source="availability_metrics",
                position={"x": 6, "y": 0, "w": 3, "h": 2},
                config={
                    "metric": "system_availability",
                    "format": "percentage",
                    "color_scheme": "availability",
                },
            ),
            # アクティブアラート数
            DashboardWidget(
                widget_id="alerts_card",
                title="アクティブアラート",
                component_type=DashboardComponentType.METRIC_CARD,
                data_source="alert_metrics",
                position={"x": 9, "y": 0, "w": 3, "h": 2},
                config={
                    "metric": "active_alerts_count",
                    "format": "number",
                    "color_scheme": "alerts",
                },
            ),
            # 品質トレンドチャート
            DashboardWidget(
                widget_id="quality_trend_chart",
                title="データ品質トレンド",
                component_type=DashboardComponentType.CHART,
                data_source="quality_history",
                position={"x": 0, "y": 2, "w": 6, "h": 4},
                config={
                    "chart_type": ChartType.LINE.value,
                    "metrics": [
                        "overall_quality",
                        "completeness",
                        "accuracy",
                        "consistency",
                    ],
                    "time_range": "24h",
                },
            ),
            # データソースヘルスマップ
            DashboardWidget(
                widget_id="datasource_heatmap",
                title="データソースヘルスマップ",
                component_type=DashboardComponentType.HEATMAP,
                data_source="datasource_health",
                position={"x": 6, "y": 2, "w": 6, "h": 4},
                config={
                    "x_axis": "data_source",
                    "y_axis": "metric_type",
                    "color_metric": "health_score",
                    "color_scheme": "RdYlGn",
                },
            ),
            # 最近のアラートテーブル
            DashboardWidget(
                widget_id="recent_alerts_table",
                title="最近のアラート",
                component_type=DashboardComponentType.TABLE,
                data_source="recent_alerts",
                position={"x": 0, "y": 6, "w": 6, "h": 3},
                config={
                    "columns": ["timestamp", "severity", "source", "message"],
                    "sort_by": "timestamp",
                    "sort_order": "desc",
                    "page_size": 10,
                },
            ),
            # 品質KPIゲージ
            DashboardWidget(
                widget_id="quality_kpi_gauge",
                title="品質KPI",
                component_type=DashboardComponentType.GAUGE,
                data_source="quality_kpis",
                position={"x": 6, "y": 6, "w": 6, "h": 3},
                config={
                    "kpis": ["data_completeness", "data_accuracy", "sla_compliance"],
                    "gauge_type": "semi_circle",
                },
            ),
        ]

        main_layout = DashboardLayout(
            layout_id="main_dashboard",
            name="メインダッシュボード",
            description="統合データ品質監視メインダッシュボード",
            widgets=main_widgets,
            is_default=True,
        )

        self.dashboard_layouts[main_layout.layout_id] = main_layout

        # MDM専用ダッシュボード
        mdm_widgets = [
            DashboardWidget(
                widget_id="master_entities_card",
                title="マスターエンティティ数",
                component_type=DashboardComponentType.METRIC_CARD,
                data_source="mdm_metrics",
                position={"x": 0, "y": 0, "w": 3, "h": 2},
            ),
            DashboardWidget(
                widget_id="entity_quality_chart",
                title="エンティティ品質分布",
                component_type=DashboardComponentType.CHART,
                data_source="mdm_quality",
                position={"x": 0, "y": 2, "w": 6, "h": 4},
                config={"chart_type": ChartType.HISTOGRAM.value},
            ),
            DashboardWidget(
                widget_id="domain_distribution_pie",
                title="ドメイン分布",
                component_type=DashboardComponentType.CHART,
                data_source="mdm_domains",
                position={"x": 6, "y": 2, "w": 6, "h": 4},
                config={"chart_type": ChartType.PIE.value},
            ),
        ]

        mdm_layout = DashboardLayout(
            layout_id="mdm_dashboard",
            name="MDMダッシュボード",
            description="マスターデータ管理専用ダッシュボード",
            widgets=mdm_widgets,
        )

        self.dashboard_layouts[mdm_layout.layout_id] = mdm_layout

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

    def get_layout(self, layout_id: str) -> DashboardLayout:
        """レイアウト取得"""
        return self.dashboard_layouts.get(layout_id)

    def get_widget(self, widget_id: str) -> DashboardWidget:
        """ウィジェット取得"""
        return self.active_widgets.get(widget_id)

    def get_kpi(self, kpi_id: str) -> QualityKPI:
        """KPI取得"""
        return self.quality_kpis.get(kpi_id)

    def list_layouts(self) -> List[str]:
        """レイアウト一覧取得"""
        return list(self.dashboard_layouts.keys())

    def list_widgets_for_layout(self, layout_id: str) -> List[DashboardWidget]:
        """レイアウト内ウィジェット一覧取得"""
        layout = self.get_layout(layout_id)
        return layout.widgets if layout else []
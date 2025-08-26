#!/usr/bin/env python3
"""
データ品質ダッシュボード - コアクラス
Core dashboard functionality and orchestration
"""

import asyncio
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .data_sources import DataSourceManager
from .mdm_integration import MDMIntegrationManager
from .models import DashboardWidget
from .quality_metrics import QualityMetricsManager
from .report_generator import ReportGenerator
from .widget_manager import WidgetManager

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class DataQualityDashboard:
    """統合データ品質ダッシュボードシステム"""

    def __init__(
        self,
        storage_path: str = "data/dashboard",
        enable_cache: bool = True,
        refresh_interval_seconds: int = 300,
        retention_days: int = 90,
    ):
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.refresh_interval_seconds = refresh_interval_seconds
        self.retention_days = retention_days

        # ディレクトリ初期化
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "reports").mkdir(exist_ok=True)
        (self.storage_path / "exports").mkdir(exist_ok=True)

        # キャッシュマネージャー初期化
        self.cache_manager = self._initialize_cache()

        # データ品質コンポーネント初期化
        self._initialize_components()

        # データキャッシュ
        self.dashboard_data_cache: Dict[str, Any] = {}
        self.last_refresh_time: Dict[str, datetime] = {}

        logger.info("統合データ品質ダッシュボード初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - リフレッシュ間隔: {refresh_interval_seconds}秒")
        logger.info(f"  - データ保持期間: {retention_days}日")

    def _initialize_cache(self):
        """キャッシュマネージャー初期化"""
        if self.enable_cache:
            try:
                # キャッシュマネージャーの動的インポートと初期化
                from ...utils.unified_cache_manager import UnifiedCacheManager
                cache_manager = UnifiedCacheManager(
                    l1_memory_mb=64, l2_memory_mb=256, l3_disk_mb=512
                )
                logger.info("ダッシュボードキャッシュシステム初期化完了")
                return cache_manager
            except ImportError as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                return None
        else:
            return None

    def _initialize_components(self):
        """データ品質コンポーネント初期化"""
        try:
            # 外部依存関係の動的ロード
            version_manager = self._load_version_manager()
            freshness_monitor = self._load_freshness_monitor()
            mdm_manager = self._load_mdm_manager()
            data_validator = self._load_data_validator()

            # コアコンポーネント初期化
            self.data_source_manager = DataSourceManager(
                data_validator=data_validator,
                freshness_monitor=freshness_monitor,
                mdm_manager=mdm_manager,
            )

            self.widget_manager = WidgetManager()

            self.quality_metrics_manager = QualityMetricsManager(
                data_source_manager=self.data_source_manager
            )

            self.mdm_integration_manager = MDMIntegrationManager(
                mdm_manager=mdm_manager
            )

            self.report_generator = ReportGenerator(
                storage_path=str(self.storage_path / "reports"),
                data_source_manager=self.data_source_manager,
                quality_metrics_manager=self.quality_metrics_manager,
            )

            # 外部コンポーネント参照保持（クリーンアップ用）
            self.version_manager = version_manager
            self.freshness_monitor = freshness_monitor
            self.mdm_manager = mdm_manager
            self.data_validator = data_validator

            logger.info("データ品質コンポーネント初期化完了")

        except Exception as e:
            logger.error(f"コンポーネント初期化エラー: {e}")
            # フォールバックモードで続行
            self._initialize_fallback_mode()

    def _load_version_manager(self):
        """データバージョン管理の動的ロード"""
        try:
            from ..data_version_manager import create_data_version_manager
            return create_data_version_manager(
                repository_path=str(self.storage_path / "versions"),
                enable_cache=True,
            )
        except ImportError:
            return None

    def _load_freshness_monitor(self):
        """データ鮮度監視の動的ロード"""
        try:
            from ..data_freshness_monitor import create_data_freshness_monitor
            return create_data_freshness_monitor(
                storage_path=str(self.storage_path / "monitoring"),
                enable_cache=True,
            )
        except ImportError:
            return None

    def _load_mdm_manager(self):
        """MDMマネージャーの動的ロード"""
        try:
            from ..data.master_data import create_master_data_manager
            return create_master_data_manager(
                storage_path=str(self.storage_path / "mdm"),
                enable_cache=True,
            )
        except ImportError:
            return None

    def _load_data_validator(self):
        """データバリデーターの動的ロード"""
        try:
            from ..real_data_validator import create_real_data_validator
            return create_real_data_validator(cache_manager=self.cache_manager)
        except ImportError:
            return None

    def _initialize_fallback_mode(self):
        """フォールバックモード初期化"""
        logger.warning("フォールバックモードで初期化中...")
        
        # モックコンポーネント作成
        self.data_source_manager = DataSourceManager()
        self.widget_manager = WidgetManager()
        self.quality_metrics_manager = QualityMetricsManager()
        self.mdm_integration_manager = MDMIntegrationManager()
        self.report_generator = ReportGenerator(
            storage_path=str(self.storage_path / "reports")
        )

        # 外部コンポーネント
        self.version_manager = None
        self.freshness_monitor = None
        self.mdm_manager = None
        self.data_validator = None

    async def get_dashboard_data(
        self, layout_id: str = "main_dashboard"
    ) -> Dict[str, Any]:
        """ダッシュボードデータ取得"""
        logger.info(f"ダッシュボードデータ取得: {layout_id}")

        try:
            layout = self.widget_manager.get_layout(layout_id)
            if not layout:
                return {"error": f"レイアウトが見つかりません: {layout_id}"}

            dashboard_data = {
                "layout": {
                    "id": layout.layout_id,
                    "name": layout.name,
                    "description": layout.description,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "widgets": {},
                "metadata": {
                    "last_refresh": datetime.utcnow().isoformat(),
                    "refresh_interval": self.refresh_interval_seconds,
                    "data_sources_status": {},
                },
            }

            # 各ウィジェットのデータ取得
            for widget in layout.widgets:
                widget_data = await self._get_widget_data(widget)
                dashboard_data["widgets"][widget.widget_id] = {
                    "config": {
                        "title": widget.title,
                        "type": widget.component_type.value,
                        "position": widget.position,
                        "refresh_interval": widget.refresh_interval,
                        **widget.config,
                    },
                    "data": widget_data,
                    "last_updated": datetime.utcnow().isoformat(),
                }

            # グローバルメタデータ追加
            dashboard_data["global_metrics"] = await self.data_source_manager.get_global_metrics()
            dashboard_data["system_health"] = await self.data_source_manager.get_system_health()

            return dashboard_data

        except Exception as e:
            logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return {"error": str(e)}

    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """ウィジェット固有データ取得"""
        try:
            # キャッシュチェック
            cache_key = f"widget_{widget.widget_id}_{widget.data_source}"
            last_refresh = self.last_refresh_time.get(cache_key, datetime.min)

            if (
                datetime.utcnow() - last_refresh
            ).total_seconds() < widget.refresh_interval:
                if cache_key in self.dashboard_data_cache:
                    return self.dashboard_data_cache[cache_key]

            # データソース別データ取得
            widget_data = await self._get_data_by_source(widget.data_source)

            # キャッシュ更新
            self.dashboard_data_cache[cache_key] = widget_data
            self.last_refresh_time[cache_key] = datetime.utcnow()

            return widget_data

        except Exception as e:
            logger.error(f"ウィジェットデータ取得エラー {widget.widget_id}: {e}")
            return {"error": str(e)}

    async def _get_data_by_source(self, data_source: str) -> Dict[str, Any]:
        """データソース別データ取得"""
        data_source_methods = {
            "quality_metrics": self.data_source_manager.get_quality_metrics_data,
            "freshness_metrics": self.data_source_manager.get_freshness_metrics_data,
            "availability_metrics": self.data_source_manager.get_availability_metrics_data,
            "alert_metrics": self.data_source_manager.get_alert_metrics_data,
            "quality_history": self.data_source_manager.get_quality_history_data,
            "datasource_health": self.data_source_manager.get_datasource_health_data,
            "recent_alerts": self.data_source_manager.get_recent_alerts_data,
            "quality_kpis": self.quality_metrics_manager.get_quality_kpis_data,
            "mdm_metrics": self.data_source_manager.get_mdm_metrics_data,
            "mdm_quality": self.data_source_manager.get_mdm_quality_data,
            "mdm_domains": self.data_source_manager.get_mdm_domains_data,
        }

        method = data_source_methods.get(data_source)
        if method:
            return await method()
        else:
            return {"error": f"未知のデータソース: {data_source}"}

    async def generate_quality_report(
        self,
        report_type: str = "daily",
        include_charts: bool = True,
        export_format: str = "json",
    ) -> str:
        """データ品質レポート生成"""
        return await self.report_generator.generate_quality_report(
            report_type=report_type,
            include_charts=include_charts,
            export_format=export_format,
        )

    async def export_dashboard_data(
        self, layout_id: str = "main_dashboard", format: str = "json"
    ) -> str:
        """ダッシュボードデータエクスポート"""
        logger.info(f"ダッシュボードデータエクスポート: {layout_id} ({format})")

        try:
            dashboard_data = await self.get_dashboard_data(layout_id)
            return await self.report_generator.export_dashboard_data(
                dashboard_data, format=format
            )

        except Exception as e:
            logger.error(f"ダッシュボードエクスポートエラー: {e}")
            raise

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("統合データ品質ダッシュボード クリーンアップ開始")

        # キャッシュクリア
        self.dashboard_data_cache.clear()
        self.last_refresh_time.clear()

        # レポートクリーンアップ
        self.report_generator.cleanup_old_reports(self.retention_days)

        # コンポーネントクリーンアップ
        cleanup_tasks = []
        
        if self.version_manager and hasattr(self.version_manager, 'cleanup'):
            cleanup_tasks.append(self.version_manager.cleanup())

        if self.freshness_monitor and hasattr(self.freshness_monitor, 'cleanup'):
            cleanup_tasks.append(self.freshness_monitor.cleanup())

        if self.mdm_manager and hasattr(self.mdm_manager, 'cleanup'):
            cleanup_tasks.append(self.mdm_manager.cleanup())

        # 並列クリーンアップ実行
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("統合データ品質ダッシュボード クリーンアップ完了")
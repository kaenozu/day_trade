"""
FastAPIアプリケーション設定とCORS設定

【重要】自動取引機能は完全に無効化されています
分析・情報提供・教育支援のみを行うセーフモードサーバー
"""

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ...automation.analysis_only_engine import AnalysisOnlyEngine
from ...config.trading_mode_config import get_current_trading_config, is_safe_mode
from ...utils.logging_config import get_context_logger
from ..custom_reports import CustomReportManager
from ..educational_system import EducationalSystem
from ..interactive_charts import InteractiveChartManager

logger = get_context_logger(__name__)

# グローバル変数
analysis_engine: Optional[AnalysisOnlyEngine] = None
chart_manager: Optional[InteractiveChartManager] = None
report_manager: Optional[CustomReportManager] = None
educational_system: Optional[EducationalSystem] = None


def create_app() -> FastAPI:
    """FastAPIアプリケーションの作成と設定"""
    app = FastAPI(
        title="Day Trade 分析専用ダッシュボード",
        description="完全セーフモード - 分析・教育・研究専用システム",
        version="1.0.0",
    )

    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 開発環境用、本番では制限
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 静的ファイル配信（存在する場合のみ）
    static_dir = "src/day_trade/dashboard/static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # イベントハンドラーの設定
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)

    return app


async def startup_event():
    """サーバー起動時の初期化"""
    global analysis_engine, chart_manager, report_manager, educational_system

    # セーフモード確認
    if not is_safe_mode():
        raise RuntimeError("セーフモードでない場合はサーバーを起動できません")

    logger.info("分析専用ダッシュボードサーバー起動中...")
    logger.info("※ 自動取引機能は完全に無効化されています")
    logger.info("※ 分析・教育・研究専用モードで動作します")

    # 分析エンジンの初期化
    test_symbols = ["7203", "8306", "9984", "6758", "4689"]  # サンプル銘柄
    analysis_engine = AnalysisOnlyEngine(test_symbols, update_interval=10.0)

    # インタラクティブチャートマネージャーの初期化
    chart_manager = InteractiveChartManager()
    logger.info("インタラクティブチャート機能を有効化しました")

    # カスタムレポートマネージャーの初期化
    report_manager = CustomReportManager()
    logger.info("カスタムレポート機能を有効化しました")

    # 教育システムの初期化
    educational_system = EducationalSystem()
    logger.info("教育・学習支援機能を有効化しました")

    logger.info("分析専用ダッシュボードサーバー起動完了")


async def shutdown_event():
    """サーバー終了時のクリーンアップ"""
    global analysis_engine

    if analysis_engine and analysis_engine.status.value == "running":
        await analysis_engine.stop()

    logger.info("分析専用ダッシュボードサーバー停止")


def get_system_status():
    """システム状態取得用の共通関数"""
    config = get_current_trading_config()
    
    if analysis_engine:
        engine_status = analysis_engine.get_status()
    else:
        engine_status = {"status": "not_initialized"}

    return {
        "safe_mode": is_safe_mode(),
        "trading_disabled": not config.enable_automatic_trading,
        "analysis_engine": engine_status,
        "server_time": None,  # 呼び出し元で設定
        "system_type": "analysis_only",
        "warning": "自動取引機能は完全に無効化されています",
    }


def get_analysis_engine() -> Optional[AnalysisOnlyEngine]:
    """分析エンジンインスタンスの取得"""
    return analysis_engine


def get_chart_manager() -> Optional[InteractiveChartManager]:
    """チャートマネージャーインスタンスの取得"""
    return chart_manager


def get_report_manager() -> Optional[CustomReportManager]:
    """レポートマネージャーインスタンスの取得"""
    return report_manager


def get_educational_system() -> Optional[EducationalSystem]:
    """教育システムインスタンスの取得"""
    return educational_system


def print_server_info():
    """サーバー情報の表示"""
    print("=" * 80)
    print("🔒 Day Trade 分析専用ダッシュボード")
    print("=" * 80)
    print("⚠️  重要: 自動取引機能は完全に無効化されています")
    print("✅  分析・教育・研究専用システムとして動作します")
    print("🌐  ダッシュボード: http://localhost:8000")
    print("=" * 80)
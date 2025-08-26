"""
Orchestra Package - Next-Gen AI Trading Engine モジュール
機能別に分割されたオーケストレーションシステム

このパッケージは元の大きなorchestrator.py（1477行）ファイルを
機能別に分割したモジュールから構成されています。

バックワード互換性を維持しながら、
保守性と拡張性を向上させています。
"""

# バックワード互換性のためのメインエクスポート
from .main_orchestrator import NextGenAIOrchestrator, DayTradeOrchestrator

# 設定・データクラス
from .types import (
    AIAnalysisResult,
    ExecutionReport,
    OrchestrationConfig,
)

# サブコンポーネント（必要に応じて直接利用可能）
from .ai_analyzer import AIAnalyzer
from .signal_generator import SignalGenerator
from .parallel_processor import ParallelProcessor
from .data_processor import DataProcessor
from .portfolio_analyzer import PortfolioAnalyzer
from .resource_manager import ResourceManager
from .core import OrchestratorCore

# CI環境検出
import os
CI_MODE = os.getenv("CI", "false").lower() == "true"

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "Next-Gen AI Trading Engine Orchestra - Modular Architecture"

# パッケージメタデータ
__all__ = [
    # メインクラス
    "NextGenAIOrchestrator",
    "DayTradeOrchestrator",  # バックワード互換性用エイリアス
    
    # 設定・データクラス
    "AIAnalysisResult",
    "ExecutionReport", 
    "OrchestrationConfig",
    "CI_MODE",
    
    # サブコンポーネント
    "AIAnalyzer",
    "SignalGenerator", 
    "ParallelProcessor",
    "DataProcessor",
    "PortfolioAnalyzer",
    "ResourceManager",
    "OrchestratorCore",
    
    # メタデータ
    "__version__",
    "__author__",
    "__description__",
]

# 初期化ログ
import logging

logger = logging.getLogger(__name__)
logger.info(f"Orchestra Package {__version__} 初期化完了")
logger.info("モジュール分割構成でロードされました")

# CI環境での動作確認
if CI_MODE:
    logger.info("CI軽量モード: 高度機能は無効化されています")

# モジュール構成の確認関数
def get_module_info() -> dict:
    """
    モジュール構成情報を取得
    
    Returns:
        dict: モジュール構成情報
    """
    return {
        "package": __name__,
        "version": __version__,
        "description": __description__,
        "modules": {
            "main_orchestrator": "メインオーケストレータークラス",
            "types": "設定・データクラス定義",
            "core": "コア機能・初期化管理",
            "ai_analyzer": "AI分析エンジン",
            "signal_generator": "シグナル・アラート生成",
            "parallel_processor": "並列処理管理",
            "data_processor": "データ処理・取得",
            "portfolio_analyzer": "ポートフォリオ・システム分析",
            "resource_manager": "リソース管理・クリーンアップ",
        },
        "features": {
            "safe_mode_only": True,
            "trading_disabled": True,
            "modular_architecture": True,
            "backward_compatible": True,
        },
        "ci_mode": CI_MODE,
        "total_lines_reduced": "1477 → 約300行×9ファイル",
    }


def validate_installation() -> bool:
    """
    インストール状態の確認
    
    Returns:
        bool: インストールが正常かどうか
    """
    try:
        # 必須コンポーネントの確認
        required_components = [
            NextGenAIOrchestrator,
            OrchestrationConfig,
            AIAnalyzer,
            SignalGenerator,
            ParallelProcessor,
            DataProcessor,
            PortfolioAnalyzer,
            ResourceManager,
        ]
        
        for component in required_components:
            if not hasattr(component, '__name__'):
                return False
        
        # 設定の基本チェック
        default_config = OrchestrationConfig()
        if not hasattr(default_config, 'max_workers'):
            logger.warning("設定オブジェクトの妥当性に問題があります")
            
        logger.info("Orchestra Package インストール確認完了")
        return True
        
    except Exception as e:
        logger.error(f"インストール確認エラー: {e}")
        return False


# 起動時インストール確認
if __name__ != "__main__":
    validate_installation()


# 使用例とドキュメント
def get_usage_example() -> str:
    """
    使用例を取得
    
    Returns:
        str: 使用例のコード
    """
    return '''
# 基本的な使用方法
from day_trade.automation.orchestrator import NextGenAIOrchestrator

# デフォルト設定で初期化
with NextGenAIOrchestrator() as orchestrator:
    # AI分析実行
    report = orchestrator.run_advanced_analysis(
        symbols=["7203", "8306", "9984"],
        analysis_type="comprehensive",
        include_predictions=True
    )
    
    print(f"分析完了: {report.successful_symbols}/{report.total_symbols} 成功")
    print(f"生成シグナル数: {len(report.generated_signals)}")
    print(f"発動アラート数: {len(report.triggered_alerts)}")

# カスタム設定での使用
from day_trade.automation.orchestrator import OrchestrationConfig

config = OrchestrationConfig(
    max_workers=4,
    confidence_threshold=0.8,
    enable_ml_engine=True
)

with NextGenAIOrchestrator(config=config) as orchestrator:
    report = orchestrator.run_advanced_analysis(["7203"])
    # ... 処理続行
'''


def print_architecture_info():
    """アーキテクチャ情報を表示"""
    info = get_module_info()
    print(f"\n{'='*60}")
    print(f"  {info['description']}")
    print(f"{'='*60}")
    print(f"バージョン: {info['version']}")
    print(f"CI モード: {info['ci_mode']}")
    print(f"コード削減: {info['total_lines_reduced']}")
    print(f"\nモジュール構成:")
    for module, desc in info['modules'].items():
        print(f"  {module:20} - {desc}")
    print(f"\n特徴:")
    for feature, enabled in info['features'].items():
        print(f"  {feature:20} - {enabled}")
    print(f"{'='*60}\n")


# デバッグ・開発時の情報表示
if __name__ == "__main__":
    print_architecture_info()
    print("使用例:")
    print(get_usage_example())
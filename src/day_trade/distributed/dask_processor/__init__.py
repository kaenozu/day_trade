#!/usr/bin/env python3
"""
Dask Data Processor Package
Issue #384: 並列処理のさらなる強化 - Modular Package

元のdask_data_processor.pyから分割されたモジュールパッケージ
後方互換性を保つため、すべてのクラスと関数を再エクスポートする
"""

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "Dask分散処理による高性能株価データ分析システム"

# 各モジュールからのインポート
try:
    # コアプロセッサー
    from .core_processor import DaskDataProcessor

    # 株価分析器
    from .stock_analyzer import DaskStockAnalyzer

    # バッチプロセッサー
    from .batch_processor import DaskBatchProcessor

    # ユーティリティ関数
    from .utils import (
        create_dask_data_processor,
        create_complete_dask_system,
        demonstrate_dask_system,
        validate_dask_environment,
        get_system_performance_metrics,
        run_performance_benchmark,
    )

    # 利用可能性フラグ
    MODULES_AVAILABLE = True

except ImportError as e:
    # フォールバック: インポートエラー時の対応
    import warnings
    
    warnings.warn(
        f"Daskプロセッサーモジュールのインポートに失敗しました: {e}",
        UserWarning,
        stacklevel=2
    )

    # ダミークラス（後方互換性のため）
    class DaskDataProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("DaskDataProcessorが利用できません")

    class DaskStockAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("DaskStockAnalyzerが利用できません")

    class DaskBatchProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("DaskBatchProcessorが利用できません")

    def create_dask_data_processor(*args, **kwargs):
        raise ImportError("create_dask_data_processor関数が利用できません")

    def create_complete_dask_system(*args, **kwargs):
        raise ImportError("create_complete_dask_system関数が利用できません")

    async def demonstrate_dask_system():
        raise ImportError("demonstrate_dask_system関数が利用できません")

    def validate_dask_environment():
        return {"error": "環境検証が利用できません"}

    def get_system_performance_metrics():
        return {"error": "パフォーマンス測定が利用できません"}

    async def run_performance_benchmark(*args, **kwargs):
        return {"error": "ベンチマークが利用できません"}

    MODULES_AVAILABLE = False

# パブリックAPI
__all__ = [
    # メインクラス
    "DaskDataProcessor",
    "DaskStockAnalyzer", 
    "DaskBatchProcessor",
    
    # ファクトリ関数
    "create_dask_data_processor",
    "create_complete_dask_system",
    
    # ユーティリティ関数
    "demonstrate_dask_system",
    "validate_dask_environment",
    "get_system_performance_metrics",
    "run_performance_benchmark",
    
    # メタデータ
    "MODULES_AVAILABLE",
    "__version__",
    "__author__",
    "__description__",
]

# モジュール情報
def get_module_info():
    """
    モジュール情報を取得
    
    Returns:
        モジュール情報辞書
    """
    return {
        "name": "dask_processor",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules_available": MODULES_AVAILABLE,
        "available_classes": [
            "DaskDataProcessor",
            "DaskStockAnalyzer", 
            "DaskBatchProcessor"
        ],
        "available_functions": [
            "create_dask_data_processor",
            "create_complete_dask_system",
            "demonstrate_dask_system",
            "validate_dask_environment",
            "get_system_performance_metrics",
            "run_performance_benchmark"
        ],
        "submodules": [
            "core_processor",
            "stock_analyzer",
            "batch_processor",
            "utils"
        ]
    }

def check_dependencies():
    """
    依存関係チェック
    
    Returns:
        依存関係チェック結果
    """
    dependencies = {
        "dask": False,
        "pandas": False,
        "numpy": False,
        "psutil": False,
    }
    
    try:
        import dask
        dependencies["dask"] = True
    except ImportError:
        pass
    
    try:
        import pandas
        dependencies["pandas"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        pass
    
    return {
        "dependencies": dependencies,
        "all_satisfied": all(dependencies.values()),
        "missing": [name for name, available in dependencies.items() if not available]
    }

# パッケージ初期化時のセルフチェック
def _initialize_package():
    """パッケージ初期化処理"""
    if MODULES_AVAILABLE:
        # 依存関係チェック
        deps = check_dependencies()
        if not deps["all_satisfied"]:
            import warnings
            warnings.warn(
                f"一部の依存関係が不足しています: {deps['missing']}",
                UserWarning,
                stacklevel=2
            )

# 初期化実行
_initialize_package()

# 後方互換性のための旧関数名（廃止予定）
def create_dask_processor(*args, **kwargs):
    """
    廃止予定: create_dask_data_processorを使用してください
    """
    import warnings
    warnings.warn(
        "create_dask_processorは廃止予定です。create_dask_data_processorを使用してください",
        DeprecationWarning,
        stacklevel=2
    )
    return create_dask_data_processor(*args, **kwargs)
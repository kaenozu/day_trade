#!/usr/bin/env python3
"""
量子化モジュール統合エントリーポイント
Issue #379: ML Model Inference Performance Optimization

分割されたモジュールの統合インポートとバックワード互換性提供
- 全てのクラス・関数のエクスポート
- 元のmodel_quantization_engine.pyとの完全互換性
- 適切な__all__定義
"""

# データ構造
from .data_structures import (
    CompressionConfig,
    CompressionResult,
    HardwareTarget,
    PruningType,
    QuantizationType,
)

# ハードウェア検出
from .hardware_detector import HardwareDetector

# 量子化エンジン
from .quantization_engine import ONNXQuantizationEngine

# プルーニングエンジン
from .pruning_engine import ModelPruningEngine

# パフォーマンス分析
from .performance_analyzer import PerformanceAnalyzer

# 統合圧縮エンジン（メインエンジン）
from .compression_engine import (
    ModelCompressionEngine,
    create_model_compression_engine,
)

# バックワード互換性のため、元のファイルで使用されていた名前を維持
ModelCompressionEngine = ModelCompressionEngine
ONNXQuantizationEngine = ONNXQuantizationEngine
ModelPruningEngine = ModelPruningEngine
HardwareDetector = HardwareDetector
PerformanceAnalyzer = PerformanceAnalyzer

# ファクトリ関数エイリアス
create_model_compression_engine = create_model_compression_engine

# 公開API定義
__all__ = [
    # データ構造・列挙型
    "QuantizationType",
    "PruningType", 
    "HardwareTarget",
    "CompressionConfig",
    "CompressionResult",
    
    # エンジンクラス
    "ModelCompressionEngine",
    "ONNXQuantizationEngine", 
    "ModelPruningEngine",
    "HardwareDetector",
    "PerformanceAnalyzer",
    
    # ファクトリ関数
    "create_model_compression_engine",
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade ML Team"
__description__ = "Advanced model compression and optimization system"

# モジュール初期化時のログ
import warnings
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存関係チェック
_dependencies_available = {}

try:
    import onnxruntime
    _dependencies_available["onnxruntime"] = True
except ImportError:
    _dependencies_available["onnxruntime"] = False
    warnings.warn(
        "ONNX Runtime not available. Quantization features will be limited.",
        ImportWarning,
        stacklevel=2
    )

try:
    import cpuinfo
    _dependencies_available["cpuinfo"] = True  
except ImportError:
    _dependencies_available["cpuinfo"] = False
    warnings.warn(
        "py-cpuinfo not available. Hardware detection will be limited.",
        ImportWarning,
        stacklevel=2
    )

try:
    import psutil
    _dependencies_available["psutil"] = True
except ImportError:
    _dependencies_available["psutil"] = False
    warnings.warn(
        "psutil not available. Memory detection will use defaults.",
        ImportWarning,
        stacklevel=2
    )

logger.info(
    f"量子化モジュール初期化完了 - 依存関係: {_dependencies_available}"
)

def get_available_features() -> dict:
    """利用可能な機能一覧を取得
    
    Returns:
        機能別利用可能性辞書
    """
    features = {
        "dynamic_quantization": _dependencies_available.get("onnxruntime", False),
        "static_quantization": _dependencies_available.get("onnxruntime", False), 
        "fp16_quantization": _dependencies_available.get("onnxruntime", False),
        "hardware_detection": _dependencies_available.get("cpuinfo", False),
        "memory_detection": _dependencies_available.get("psutil", False),
        "magnitude_pruning": True,  # NumPyベースなので常に利用可能
        "structured_pruning": True,  # NumPyベースなので常に利用可能
        "performance_analysis": True,  # 基本機能なので常に利用可能
    }
    
    return features

def check_system_requirements() -> bool:
    """システム要件チェック
    
    Returns:
        要件充足フラグ
    """
    required_features = ["onnxruntime"]
    available_deps = _dependencies_available
    
    missing_deps = [
        dep for dep in required_features 
        if not available_deps.get(dep, False)
    ]
    
    if missing_deps:
        logger.warning(f"必須依存関係が不足: {missing_deps}")
        return False
        
    return True

# 初期化時のシステムチェック実行
_system_ready = check_system_requirements()

def is_system_ready() -> bool:
    """システム準備状況取得
    
    Returns:
        システム準備完了フラグ
    """
    return _system_ready
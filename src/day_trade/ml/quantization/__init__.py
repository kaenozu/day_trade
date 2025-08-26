#!/usr/bin/env python3
"""
モデル量子化・プルーニングモジュール
Issue #379: ML Model Inference Performance Optimization

後方互換性を保つための統合エクスポート
"""

# コア定義のエクスポート
from .core import (
    CompressionConfig,
    CompressionResult,
    QuantizationType,
    PruningType,
    HardwareTarget,
    check_dependencies,
    get_model_size_mb,
    validate_compression_config,
)

# ハードウェア検出
from .hardware_detector import HardwareDetector

# ONNX量子化エンジン
from .onnx_quantization import ONNXQuantizationEngine

# プルーニングエンジン
from .pruning import ModelPruningEngine

# 統合圧縮エンジン
from .compression_engine import ModelCompressionEngine

# ファクトリ関数
from .factory import (
    create_model_compression_engine,
    create_hardware_detector,
    get_recommended_config,
    quick_compress,
)

# 後方互換性のための元のクラス・関数名の再エクスポート
# 元の model_quantization_engine.py のメインクラス・関数
ModelQuantizationEngine = ModelCompressionEngine  # エイリアス
create_model_quantization_engine = create_model_compression_engine  # エイリアス

# バージョン情報
__version__ = "1.0.0"

# 公開API
__all__ = [
    # コア定義
    "CompressionConfig",
    "CompressionResult", 
    "QuantizationType",
    "PruningType",
    "HardwareTarget",
    "check_dependencies",
    "get_model_size_mb",
    "validate_compression_config",
    
    # エンジンクラス
    "HardwareDetector",
    "ONNXQuantizationEngine", 
    "ModelPruningEngine",
    "ModelCompressionEngine",
    
    # ファクトリ関数
    "create_model_compression_engine",
    "create_hardware_detector",
    "get_recommended_config",
    "quick_compress",
    
    # 後方互換性エイリアス
    "ModelQuantizationEngine",
    "create_model_quantization_engine",
    
    # バージョン
    "__version__",
]

# モジュール初期化時の情報ログ
import logging
logger = logging.getLogger(__name__)
logger.info("モデル量子化・プルーニングモジュール初期化完了 (分割版)")
logger.debug(f"利用可能なクラス・関数: {__all__}")

# 依存関係チェック実行（初期化時）
try:
    deps = check_dependencies()
    logger.info(f"依存関係チェック完了: {deps}")
except Exception as e:
    logger.warning(f"依存関係チェック警告: {e}")
#!/usr/bin/env python3
"""
ML Engine Module - 機械学習エンジンパッケージ
Advanced ML Engine を機能別に分割したモジュール群

このパッケージは元々のadvanced_ml_engine.pyを
以下の機能別モジュールに分割して提供します：

- config: 設定クラスとデータクラス
- models: PyTorchニューラルネットワークモデル  
- data_processor: データ前処理機能
- technical_indicators: テクニカル指標計算
- inference_engine: 推論エンジン
- next_gen_engine: 次世代AI取引エンジン
- utils: ユーティリティ関数

バックワード互換性のため、元のクラスと関数を
全て同一インターフェースで提供します。
"""

# 基本設定とデータ構造
from .config import (
    ModelConfig,
    PredictionResult,
    create_default_config,
    create_lightweight_config,
    validate_config,
)

# PyTorchモデル
from .models import (
    LSTMTransformerHybrid,
    PositionalEncoding,
    PYTORCH_AVAILABLE,
    create_model,
    get_model_summary,
)

# データ処理
from .data_processor import (
    DataProcessor,
    prepare_data,
)

# テクニカル指標
from .technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    extract_fft_features,
    calculate_ml_trend_strength,
    calculate_volatility_score,
    calculate_pattern_recognition_score,
    calculate_advanced_technical_indicators,
    get_default_ml_scores,
    calculate_williams_r,
    calculate_stochastic,
    calculate_atr,
)

# 推論エンジン
from .inference_engine import (
    AdvancedMLEngine,
    create_advanced_ml_engine,
)

# 次世代エンジン
from .next_gen_engine import (
    NextGenAITradingEngine,
    create_next_gen_engine,
)

# ユーティリティ関数
from .utils import (
    extract_fft_features_async,
    measure_inference_time_optimized,
    measure_inference_time_async,
    validate_data_shape,
    normalize_features,
    calculate_correlation_matrix,
    detect_outliers,
    smooth_time_series,
    calculate_feature_importance,
    memory_usage_info,
)

# バックワード互換性のための関数エイリアス
def create_hybrid_model(config):
    """元のcreate_hybrid_model関数との互換性"""
    return create_model(config)


# パッケージ情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "Advanced ML Engine - 機械学習エンジンパッケージ"

# パッケージレベルのエクスポート
__all__ = [
    # 設定関連
    "ModelConfig",
    "PredictionResult",
    "create_default_config",
    "create_lightweight_config", 
    "validate_config",
    
    # モデル関連
    "LSTMTransformerHybrid",
    "PositionalEncoding",
    "PYTORCH_AVAILABLE",
    "create_model",
    "get_model_summary",
    "create_hybrid_model",  # バックワード互換性
    
    # データ処理関連
    "DataProcessor",
    "prepare_data",
    
    # テクニカル指標関連
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands", 
    "extract_fft_features",
    "calculate_ml_trend_strength",
    "calculate_volatility_score",
    "calculate_pattern_recognition_score",
    "calculate_advanced_technical_indicators",
    "get_default_ml_scores",
    "calculate_williams_r",
    "calculate_stochastic",
    "calculate_atr",
    
    # エンジン関連
    "AdvancedMLEngine",
    "create_advanced_ml_engine",
    "NextGenAITradingEngine", 
    "create_next_gen_engine",
    
    # ユーティリティ関連
    "extract_fft_features_async",
    "measure_inference_time_optimized",
    "measure_inference_time_async",
    "validate_data_shape",
    "normalize_features",
    "calculate_correlation_matrix",
    "detect_outliers",
    "smooth_time_series",
    "calculate_feature_importance",
    "memory_usage_info",
    
    # 定数
    "PYTORCH_AVAILABLE",
]


# モジュール情報の動的取得
def get_module_info():
    """
    モジュール情報を取得
    
    Returns:
        dict: モジュール情報
    """
    import importlib.util
    
    # 各依存関係の確認
    deps_status = {
        "pytorch": importlib.util.find_spec("torch") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "mlflow": importlib.util.find_spec("mlflow") is not None,
        "psutil": importlib.util.find_spec("psutil") is not None,
    }
    
    # 利用可能モジュール数
    modules_available = [
        "config",
        "models", 
        "data_processor",
        "technical_indicators",
        "inference_engine",
        "next_gen_engine",
        "utils",
    ]
    
    return {
        "version": __version__,
        "description": __description__,
        "modules_count": len(modules_available),
        "modules": modules_available,
        "dependencies": deps_status,
        "pytorch_available": PYTORCH_AVAILABLE,
        "backward_compatible": True,
    }


# 初期化時の警告
import warnings
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# PyTorchの可用性チェック
if not PYTORCH_AVAILABLE:
    logger.warning(
        "PyTorch が利用できません。機械学習機能は制限モードで動作します。"
    )
    warnings.warn(
        "PyTorch is not available. ML features will run in limited mode.",
        UserWarning
    )

# 初期化完了ログ
logger.info(f"ML Engine Package 初期化完了 (version: {__version__})")

# 使用例とドキュメントの提供
def show_usage_examples():
    """
    使用例を表示
    """
    examples = """
    # ML Engine パッケージ使用例:
    
    # 1. 基本的な使用方法
    from day_trade.data.ml_engine import AdvancedMLEngine, ModelConfig
    
    config = ModelConfig()
    engine = AdvancedMLEngine(config)
    
    # 2. データ処理
    from day_trade.data.ml_engine import DataProcessor
    
    processor = DataProcessor(config)
    X, y = processor.prepare_data(market_data)
    
    # 3. テクニカル指標計算
    from day_trade.data.ml_engine import calculate_advanced_technical_indicators
    
    indicators = calculate_advanced_technical_indicators(price_data, "USDJPY")
    
    # 4. 次世代エンジン使用
    from day_trade.data.ml_engine import NextGenAITradingEngine
    
    next_gen_engine = NextGenAITradingEngine()
    results = next_gen_engine.train_next_gen_model(data)
    
    # 5. バックワード互換性
    # 元のadvanced_ml_engine.pyと同じインターフェースで使用可能
    from day_trade.data.ml_engine import create_advanced_ml_engine
    
    engine = create_advanced_ml_engine()
    """
    print(examples)


def verify_installation():
    """
    インストール状況の確認
    
    Returns:
        dict: 確認結果
    """
    info = get_module_info()
    
    print(f"ML Engine Package v{info['version']}")
    print(f"モジュール数: {info['modules_count']}")
    print(f"PyTorch利用可能: {info['pytorch_available']}")
    print(f"バックワード互換性: {info['backward_compatible']}")
    
    print("\n依存関係:")
    for dep, status in info['dependencies'].items():
        status_str = "✓" if status else "✗"
        print(f"  {dep}: {status_str}")
    
    print(f"\n利用可能モジュール:")
    for module in info['modules']:
        print(f"  - {module}")
    
    return info
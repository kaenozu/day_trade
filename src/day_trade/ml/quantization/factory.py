#!/usr/bin/env python3
"""
モデル量子化・プルーニング - ファクトリ関数
Issue #379: ML Model Inference Performance Optimization

エンジンインスタンスの作成と便利な関数
"""

import asyncio
from typing import Optional

from .core import CompressionConfig, QuantizationType, PruningType
from .compression_engine import ModelCompressionEngine
from .hardware_detector import HardwareDetector
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


# エクスポート用ファクトリ関数
async def create_model_compression_engine(
    quantization_type: QuantizationType = QuantizationType.DYNAMIC_INT8,
    pruning_type: PruningType = PruningType.MAGNITUDE_BASED,
    auto_hardware_detection: bool = True,
    config: Optional[CompressionConfig] = None,
) -> ModelCompressionEngine:
    """モデル圧縮エンジン作成"""
    if config is None:
        config = CompressionConfig(
            quantization_type=quantization_type, 
            pruning_type=pruning_type
        )

    engine = ModelCompressionEngine(config)

    if auto_hardware_detection:
        optimal_config = engine.hardware_detector.get_optimal_config()
        engine.config = optimal_config
        logger.info("ハードウェア自動最適化設定適用")

    return engine


def create_hardware_detector() -> HardwareDetector:
    """ハードウェア検出器作成"""
    return HardwareDetector()


def get_recommended_config() -> CompressionConfig:
    """推奨設定取得"""
    detector = HardwareDetector()
    return detector.get_optimal_config()


async def quick_compress(
    model_path: str,
    output_dir: str = "./compressed_models",
    model_name: str = "model",
    auto_optimize: bool = True,
) -> dict:
    """クイック圧縮 - 簡単な1行呼び出し"""
    try:
        # エンジン作成
        engine = await create_model_compression_engine(
            auto_hardware_detection=auto_optimize
        )
        
        # 圧縮実行
        result = await engine.compress_model(
            model_path, output_dir, model_name=model_name
        )
        
        logger.info(f"クイック圧縮完了: {result}")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"クイック圧縮エラー: {e}")
        return {"error": str(e), "success": False}


# テスト関数
async def test_compression_engine():
    """圧縮エンジンのテスト"""
    print("=== モデル量子化・プルーニングエンジンテスト ===")

    # エンジン作成
    engine = await create_model_compression_engine(auto_hardware_detection=True)

    # 統計表示
    stats = engine.get_compression_stats()
    print(f"初期化完了: {stats}")

    print("✅ モデル圧縮エンジンテスト完了")


async def test_hardware_detection():
    """ハードウェア検出のテスト"""
    print("=== ハードウェア検出テスト ===")

    # 検出器作成
    detector = create_hardware_detector()

    # ハードウェア情報表示
    hardware_summary = detector.get_hardware_summary()
    print(f"ハードウェア情報: {hardware_summary}")

    # 推奨設定表示
    recommended_config = detector.get_optimal_config()
    print(f"推奨設定: {recommended_config.to_dict()}")

    print("✅ ハードウェア検出テスト完了")


if __name__ == "__main__":
    # テスト実行
    async def run_tests():
        await test_hardware_detection()
        await test_compression_engine()

    asyncio.run(run_tests())
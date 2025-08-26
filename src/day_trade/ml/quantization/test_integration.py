#!/usr/bin/env python3
"""
分割されたモジュールの動作確認テスト
Issue #379: ML Model Inference Performance Optimization

分割後の全モジュールが正常にインポート・動作することを確認
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


async def test_module_imports():
    """モジュールインポートテスト"""
    logger.info("=== モジュールインポートテスト開始 ===")
    
    try:
        # データ構造
        from . import (
            QuantizationType,
            PruningType,
            HardwareTarget,
            CompressionConfig,
            CompressionResult,
        )
        logger.info("✅ データ構造インポート成功")
        
        # ハードウェア検出
        from . import HardwareDetector
        logger.info("✅ ハードウェア検出インポート成功")
        
        # 量子化エンジン
        from . import ONNXQuantizationEngine
        logger.info("✅ 量子化エンジンインポート成功")
        
        # プルーニングエンジン
        from . import ModelPruningEngine
        logger.info("✅ プルーニングエンジンインポート成功")
        
        # パフォーマンス分析
        from . import PerformanceAnalyzer
        logger.info("✅ パフォーマンス分析インポート成功")
        
        # 統合エンジン
        from . import ModelCompressionEngine, create_model_compression_engine
        logger.info("✅ 統合エンジンインポート成功")
        
        # バックワード互換性チェック
        from . import get_available_features, check_system_requirements
        logger.info("✅ 互換性関数インポート成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ インポートエラー: {e}")
        return False


async def test_basic_functionality():
    """基本機能テスト"""
    logger.info("=== 基本機能テスト開始 ===")
    
    try:
        from . import (
            CompressionConfig,
            QuantizationType,
            PruningType,
            HardwareDetector,
            ModelCompressionEngine,
            get_available_features,
        )
        
        # 設定作成
        config = CompressionConfig(
            quantization_type=QuantizationType.DYNAMIC_INT8,
            pruning_type=PruningType.MAGNITUDE_BASED,
        )
        logger.info(f"✅ 設定作成成功: {config.to_dict()}")
        
        # ハードウェア検出
        detector = HardwareDetector()
        hardware_info = detector.get_hardware_info()
        logger.info(f"✅ ハードウェア検出成功: CPU={hardware_info['cpu_features']['vendor']}")
        
        # 圧縮エンジン作成
        engine = ModelCompressionEngine(config)
        stats = engine.get_compression_stats()
        logger.info(f"✅ 圧縮エンジン作成成功: {len(stats)}個の統計項目")
        
        # 利用可能機能チェック
        features = get_available_features()
        available_count = sum(1 for available in features.values() if available)
        logger.info(f"✅ 機能チェック成功: {available_count}/{len(features)}機能が利用可能")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 基本機能テストエラー: {e}")
        return False


async def test_backward_compatibility():
    """バックワード互換性テスト"""
    logger.info("=== バックワード互換性テスト開始 ===")
    
    try:
        # 元のmodel_quantization_engine.pyと同じインポート方法をテスト
        from . import (
            ModelCompressionEngine,
            ONNXQuantizationEngine,
            ModelPruningEngine,
            HardwareDetector,
            create_model_compression_engine,
        )
        
        # 元のファクトリ関数が動作することを確認
        engine = await create_model_compression_engine(auto_hardware_detection=True)
        logger.info("✅ ファクトリ関数互換性確認")
        
        # 元の統計取得方法が動作することを確認
        stats = engine.get_compression_stats()
        assert "models_compressed" in stats
        assert "hardware_info" in stats
        logger.info("✅ 統計取得互換性確認")
        
        # 元のエンジン個別取得が動作することを確認
        quant_engine = engine.get_quantization_engine()
        prune_engine = engine.get_pruning_engine()
        hw_detector = engine.get_hardware_detector()
        perf_analyzer = engine.get_performance_analyzer()
        
        assert isinstance(quant_engine, ONNXQuantizationEngine)
        assert isinstance(prune_engine, ModelPruningEngine)
        assert isinstance(hw_detector, HardwareDetector)
        logger.info("✅ エンジン個別取得互換性確認")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ バックワード互換性テストエラー: {e}")
        return False


async def run_integration_tests():
    """統合テスト実行"""
    logger.info("🚀 分割モジュール統合テスト開始")
    
    test_results = []
    
    # インポートテスト
    import_success = await test_module_imports()
    test_results.append(("モジュールインポート", import_success))
    
    # 基本機能テスト
    functionality_success = await test_basic_functionality()
    test_results.append(("基本機能", functionality_success))
    
    # 互換性テスト
    compatibility_success = await test_backward_compatibility()
    test_results.append(("バックワード互換性", compatibility_success))
    
    # 結果サマリー
    logger.info("=== テスト結果サマリー ===")
    passed = 0
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    total = len(test_results)
    logger.info(f"テスト結果: {passed}/{total} 合格")
    
    if passed == total:
        logger.info("🎉 全テスト合格！モジュール分割成功")
    else:
        logger.error("❌ テスト失敗。モジュール分割に問題があります")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
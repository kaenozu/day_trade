#!/usr/bin/env python3
"""
GPU モジュール基本テスト
"""

import sys
sys.path.append("src")

try:
    from day_trade.ml.gpu import (
        GPUBackend,
        GPUInferenceConfig,
        GPUInferenceResult,
        GPUDeviceManager,
        GPUMonitoringData,
        TensorRTEngine,
        GPUInferenceSession,
        GPUStreamManager,
        GPUBatchProcessor,
        GPUAcceleratedInferenceEngine
    )
    import numpy as np
    
    print("GPU モジュールインポート成功")
    
    # 設定テスト
    config = GPUInferenceConfig(
        backend=GPUBackend.CUDA,
        device_ids=[0],
        memory_pool_size_mb=1024
    )
    print(f"設定作成成功: {config.backend.value}")
    
    # 設定検証テスト
    errors = config.validate()
    print(f"設定検証: {'問題なし' if not errors else f'{len(errors)}個のエラー'}")
    
    # デバイスマネージャーテスト
    device_manager = GPUDeviceManager()
    print(f"デバイスマネージャー初期化成功")
    
    devices = device_manager.list_devices()
    print(f"検出されたデバイス数: {len(devices)}")
    
    for device in devices:
        print(f"  - {device['name']} ({device['backend'].value}) - {device['memory_mb']}MB")
    
    # サマリーテスト
    summary = device_manager.get_device_summary()
    print(f"デバイスサマリー: {summary['total_devices']}台、総メモリ{summary['total_memory_mb']}MB")
    
    # TensorRTエンジンテスト（初期化のみ）
    trt_engine = TensorRTEngine(config)
    print(f"TensorRTエンジン初期化成功")
    
    engine_info = trt_engine.get_engine_info()
    print(f"エンジンステータス: {engine_info['status']}")
    
    # ストリームマネージャーテスト
    stream_manager = GPUStreamManager(config)
    stream_stats = stream_manager.get_stream_stats()
    print(f"ストリームマネージャー初期化成功: {stream_stats['total_streams']}ストリーム")
    
    # バッチプロセッサーテスト
    batch_processor = GPUBatchProcessor(config)
    batch_stats = batch_processor.get_batch_stats()
    print(f"バッチプロセッサー初期化成功: バッチ統計取得完了")
    
    # メインエンジンテスト
    engine = GPUAcceleratedInferenceEngine(config)
    engine_stats = engine.get_comprehensive_stats()
    print(f"GPU推論エンジン初期化成功: {engine_stats['models_loaded']}モデル読み込み済み")
    
    # クリーンアップ
    stream_manager.cleanup()
    batch_processor.cleanup()
    engine.cleanup()
    
    print("\nGPU モジュール基本テスト完了 - すべて成功")
    
except ImportError as e:
    print(f"インポートエラー: {e}")
except Exception as e:
    print(f"予期しないエラー: {e}")
    import traceback
    traceback.print_exc()
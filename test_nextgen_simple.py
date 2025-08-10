#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - 簡易テスト
Unicode問題を避けた基本動作確認
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.day_trade.data.advanced_ml_engine import (
        NextGenAITradingEngine,
        create_next_gen_engine,
        PYTORCH_AVAILABLE
    )
    from src.day_trade.ml.hybrid_lstm_transformer import HybridModelConfig

    print("Next-Gen AI Trading Engine - 簡易テスト開始")
    print("=" * 50)

    # 1. 初期化テスト
    print("1. 初期化テスト")
    config = HybridModelConfig(
        sequence_length=30,
        prediction_horizon=3,
        lstm_hidden_size=64,
        transformer_d_model=32,
        epochs=2,
        batch_size=8
    )

    engine = NextGenAITradingEngine(config)
    print(f"   エンジン初期化: 成功")
    print(f"   PyTorch利用可能: {PYTORCH_AVAILABLE}")

    # 2. データ生成
    print("\n2. テストデータ生成")
    np.random.seed(42)
    data = pd.DataFrame({
        'Open': np.random.randn(500) + 100,
        'High': np.random.randn(500) + 102,
        'Low': np.random.randn(500) + 98,
        'Close': np.random.randn(500) + 101,
        'Volume': np.random.randint(1000, 10000, 500)
    })
    print(f"   データ生成: {data.shape[0]}行 x {data.shape[1]}列")

    # 3. 訓練テスト
    print("\n3. ハイブリッドモデル訓練テスト")
    try:
        start_time = time.time()
        result = engine.train_next_gen_model(data, enable_ensemble=False)
        training_time = time.time() - start_time

        print(f"   訓練時間: {training_time:.2f}秒")

        if 'performance_summary' in result:
            perf = result['performance_summary']
            print(f"   精度: {perf.get('accuracy', 0):.4f}")
            print(f"   MAE: {perf.get('mae', 1.0):.6f}")
            print(f"   RMSE: {perf.get('rmse', 1.0):.6f}")

        print("   訓練: 成功")
    except Exception as e:
        print(f"   訓練エラー: {e}")

    # 4. 予測テスト
    print("\n4. 予測テスト")
    try:
        test_data = data.tail(50)
        start_time = time.time()
        pred_result = engine.predict_next_gen(test_data, use_uncertainty=False, use_ensemble=False)
        inference_time = time.time() - start_time

        print(f"   推論時間: {inference_time*1000:.2f}ms")

        if 'predictions' in pred_result and 'hybrid_lstm_transformer' in pred_result['predictions']:
            predictions = pred_result['predictions']['hybrid_lstm_transformer']['predictions']
            print(f"   予測数: {len(predictions)}")
            print(f"   予測サンプル: {predictions[:3] if len(predictions) >= 3 else predictions}")

        print("   予測: 成功")
    except Exception as e:
        print(f"   予測エラー: {e}")

    # 5. システム概要
    print("\n5. システム概要")
    try:
        summary = engine.get_comprehensive_summary()
        print(f"   名前: {summary['engine_info']['name']}")
        print(f"   バージョン: {summary['engine_info']['version']}")
        print(f"   アーキテクチャ: {summary['engine_info']['architecture']}")
        print(f"   ハイブリッドモデル初期化済み: {summary['system_status']['hybrid_model_initialized']}")
        print(f"   モデル訓練済み: {summary['system_status']['model_trained']}")
    except Exception as e:
        print(f"   概要取得エラー: {e}")

    print("\n" + "=" * 50)
    print("Next-Gen AI Trading Engine テスト完了")

except ImportError as e:
    print(f"インポートエラー: {e}")
except Exception as e:
    print(f"テスト実行エラー: {e}")
    import traceback
    traceback.print_exc()

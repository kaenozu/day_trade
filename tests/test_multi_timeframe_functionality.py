#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタイムフレーム予測システム動作確認テスト
Issue #882対応: 週次・月次予測機能のテスト
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from multi_timeframe_predictor import MultiTimeframePredictor, TimeFrame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_multi_timeframe_functionality():
    """マルチタイムフレーム予測機能のテスト"""

    print("=" * 80)
    print("マルチタイムフレーム予測システム 動作確認テスト")
    print("=" * 80)

    try:
        # 設定ファイルパスの確認
        config_path = Path("config/multi_timeframe_config.yaml")
        if not config_path.exists():
            print(f"❌ 設定ファイルが見つかりません: {config_path}")
            return False

        # システム初期化
        print("\n1. システム初期化中...")
        predictor = MultiTimeframePredictor(config_path=config_path)
        print("✅ システム初期化完了")

        # テスト銘柄
        test_symbols = ["7203", "8306", "9984"]

        # 各タイムフレームでのテスト
        timeframes = [TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY]

        for timeframe in timeframes:
            print(f"\n2. {timeframe.value}予測のテスト中...")

            for symbol in test_symbols:
                try:
                    print(f"   - 銘柄: {symbol}")

                    # 予測実行
                    result = await predictor.predict_all_timeframes(symbol=symbol)

                    if result:
                        print(f"     ✅ 予測成功: 方向={result.integrated_direction}, 信頼度={result.integrated_confidence:.3f}")
                    else:
                        print(f"     ⚠️ 予測データなし")

                except Exception as e:
                    print(f"     ❌ 予測エラー: {e}")

        # 統合予測テスト
        print(f"\n3. 統合予測テスト中...")

        test_symbol = "7203"  # トヨタ
        try:
            result = await predictor.predict_all_timeframes(symbol=test_symbol)

            if result:
                print(f"✅ 統合予測成功:")
                print(f"   - 統合方向: {result.integrated_direction}")
                print(f"   - 統合信頼度: {result.integrated_confidence:.3f}")
                print(f"   - 一貫性スコア: {result.consistency_score:.3f}")
            else:
                print(f"⚠️ 統合予測データなし")

        except Exception as e:
            print(f"❌ 統合予測エラー: {e}")

        # データ準備テスト
        print(f"\n4. データ準備機能テスト中...")

        for timeframe in timeframes:
            try:
                data, targets = await predictor.prepare_timeframe_data(test_symbol, timeframe)

                if data is not None and len(data) > 0:
                    print(f"✅ {timeframe.value}データ準備成功: {len(data)}件")
                else:
                    print(f"⚠️ {timeframe.value}データ準備: データなし")

            except Exception as e:
                print(f"❌ {timeframe.value}データ準備エラー: {e}")

        # システム情報表示
        print(f"\n5. システム情報:")
        print(f"   - 利用可能タイムフレーム: {[tf.value for tf in timeframes]}")
        print(f"   - ML予測モデル: {'利用可能' if hasattr(predictor, 'ml_models') else '利用不可'}")
        print(f"   - 特徴量エンジニアリング: {'利用可能' if hasattr(predictor, 'feature_engineer') else '利用不可'}")
        print(f"   - リアルデータプロバイダー: {'利用可能' if hasattr(predictor, 'data_provider') else '利用不可'}")

        print("\n" + "=" * 80)
        print("✅ マルチタイムフレーム予測システム動作確認完了")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """メイン実行関数"""
    success = await test_multi_timeframe_functionality()

    if success:
        print("\n🎉 すべてのテストが完了しました")
    else:
        print("\n⚠️ テストでエラーが発生しました")

if __name__ == "__main__":
    asyncio.run(main())
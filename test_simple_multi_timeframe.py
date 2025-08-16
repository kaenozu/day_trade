#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルなマルチタイムフレーム予測テスト
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_multi_timeframe():
    """マルチタイムフレーム予測テスト"""
    print("マルチタイムフレーム予測テスト開始")
    print("=" * 50)

    try:
        from multi_timeframe_prediction_engine import MultiTimeframePredictionEngine

        # エンジン初期化
        print("1. エンジン初期化中...")
        engine = MultiTimeframePredictionEngine()
        print("エンジン初期化成功")

        # テスト銘柄で予測実行
        test_symbol = "^N225"
        print(f"\n2. テスト予測実行中: {test_symbol}")

        # マルチタイムフレーム予測
        prediction = await engine.generate_multi_timeframe_prediction(test_symbol)

        if prediction:
            print("マルチタイムフレーム予測成功")
            print(f"   銘柄: {prediction.symbol}")
            print(f"   統合方向: {prediction.consensus_direction}")
            print(f"   統合信頼度: {prediction.consensus_confidence:.1f}%")

            # 利用可能な属性を確認
            print(f"   予測オブジェクト型: {type(prediction)}")
            available_attrs = [attr for attr in dir(prediction) if not attr.startswith('_')]
            print(f"   利用可能属性: {available_attrs}")

            # 期間別結果表示
            if hasattr(prediction, 'timeframe_results'):
                print(f"\n   期間別予測:")
                for tf_result in prediction.timeframe_results:
                    print(f"     {tf_result.timeframe.value}: {tf_result.direction} ({tf_result.confidence:.1f}%)")

            return True
        else:
            print("マルチタイムフレーム予測失敗")
            return False

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """メインテスト実行"""
    print("Issue #882対応テスト: マルチタイムフレーム予測")
    print("=" * 60)

    # 基本機能テスト
    success = await test_multi_timeframe()

    print("\n" + "=" * 60)
    if success:
        print("テスト成功: マルチタイムフレーム予測が正常動作")
        print("デフォルト動作変更: --symbolでマルチタイムフレーム予測実行")
        print("高速モード: --quickで従来のデイトレード予測")
    else:
        print("テスト失敗")

    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        sys.exit(1)
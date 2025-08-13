#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - 簡易リアルタイムシステムテスト
基本的なシステム動作確認
"""

import asyncio
import logging
from datetime import datetime

import numpy as np

from src.day_trade.realtime.live_prediction_engine import create_live_prediction_engine
from src.day_trade.realtime.websocket_stream import MarketTick

# プロジェクト内インポート
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


async def simple_system_test():
    """簡易システムテスト"""

    print("=== Next-Gen AI Trading Engine - Simple Test ===")

    try:
        # 1. ライブ予測エンジン作成
        print("1. Creating live prediction engine...")
        engine = await create_live_prediction_engine(["AAPL", "MSFT"])
        print("OK Live prediction engine created")

        # 2. 模擬市場データ作成
        print("2. Creating mock market data...")
        mock_ticks = []

        for i in range(50):
            for symbol in ["AAPL", "MSFT"]:
                tick = MarketTick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=150.0 + np.random.uniform(-5, 5),
                    volume=1000 + int(np.random.uniform(0, 500)),
                )
                mock_ticks.append(tick)

        print(f"OK Created {len(mock_ticks)} mock market ticks")

        # 3. 市場データ更新
        print("3. Updating market data...")
        await engine.update_market_data(mock_ticks)
        print("OK Market data updated")

        # 4. 予測生成
        print("4. Generating predictions...")
        predictions = await engine.generate_predictions()

        if predictions:
            print(f"OK Generated {len(predictions)} predictions:")
            for symbol, prediction in predictions.items():
                print(
                    f"  {symbol}: {prediction.final_action} "
                    f"(confidence: {prediction.confidence:.2%}, "
                    f"target: ${prediction.predicted_price:.2f})"
                )
        else:
            print("ERROR No predictions generated")

        # 5. 統計確認
        print("5. Checking statistics...")
        stats = engine.get_statistics()
        print(f"OK Engine stats: {stats}")

        # 6. クリーンアップ
        print("6. Cleaning up...")
        await engine.cleanup()
        print("OK Cleanup completed")

        print("\n=== Simple Test Completed Successfully ===")
        return True

    except Exception as e:
        print(f"\nERROR Simple test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """メイン実行"""

    # ログレベル設定
    logging.getLogger().setLevel(logging.WARNING)  # エラーログのみ表示

    success = await simple_system_test()

    if success:
        print("System is ready for full integration!")
        return 0
    else:
        print("System has issues that need fixing.")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
推奨エンジンの簡単なテスト
"""

import asyncio
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.day_trade.recommendation.recommendation_engine import get_daily_recommendations

async def simple_test():
    print("推奨銘柄選定エンジン テスト開始...")
    
    try:
        # TOP3の推奨銘柄を取得
        recommendations = await get_daily_recommendations(3)
        
        print(f"\nTOP {len(recommendations)} 推奨銘柄:")
        
        if not recommendations:
            print("推奨銘柄が見つかりませんでした。")
            return
            
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec.symbol} ({rec.name})")
            print(f"   総合スコア: {rec.composite_score:.1f}点")
            print(f"   テクニカル: {rec.technical_score:.1f}点")
            print(f"   ML予測: {rec.ml_score:.1f}点")
            print(f"   アクション: {rec.action.value}")
            print(f"   信頼度: {rec.confidence:.1f}%")
            print(f"   リスク: {rec.risk_level}")
            if rec.reasons:
                print(f"   理由: {', '.join(rec.reasons[:3])}")
            if rec.price_target:
                print(f"   目標価格: {rec.price_target:.0f}円")
            if rec.stop_loss:
                print(f"   ストップロス: {rec.stop_loss:.0f}円")
                
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_test())
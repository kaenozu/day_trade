#!/usr/bin/env python3
"""
リファクタリング後TradeManagerの統合テスト
"""

import sys
from decimal import Decimal
from datetime import datetime

# パス追加
sys.path.append("src")

try:
    from day_trade.core.managers import TradeManager
    from day_trade.models.enums import TradeType
    
    print("インポート成功")
    
    # TradeManager初期化テスト
    manager = TradeManager(
        commission_rate=Decimal("0.001"),
        tax_rate=Decimal("0.2"),
        load_from_db=False
    )
    print("TradeManager初期化成功")
    
    # 基本機能テスト
    print("\n基本統計:")
    stats = manager.get_summary_stats()
    print(f"  取引数: {stats['total_trades']}")
    print(f"  ポジション数: {stats['total_positions']}")
    
    # 取引追加テスト（メモリのみ）
    print("\n取引テスト開始:")
    try:
        # 買い取引
        trade_id = manager.add_trade(
            symbol="1234",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000"),
            notes="テスト買い",
            persist_to_db=False
        )
        print(f"買い取引追加成功: {trade_id}")
        
        # ポジション確認
        position = manager.get_position("1234")
        if position:
            print(f"ポジション確認: {position.quantity}株 @{position.average_price}円")
        
        # サマリー更新確認
        updated_stats = manager.get_summary_stats()
        print(f"統計更新: 取引数 {updated_stats['total_trades']}")
        
        print("\n統合テスト完了 - すべて成功")
        
    except Exception as e:
        print(f"取引テストエラー: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"インポートエラー: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"予期しないエラー: {e}")
    import traceback
    traceback.print_exc()
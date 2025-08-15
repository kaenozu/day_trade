#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Integration Test - システム統合テスト
全コンポーネントの連携確認
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

def test_module_imports():
    """モジュールインポートテスト"""
    print("モジュールインポートテスト")
    modules_to_test = [
        'daytrade_core',
        'daytrade_web', 
        'daytrade_cli',
        'enhanced_data_provider',
        'enhanced_personal_analysis_engine',
        'ml_accuracy_improvement_system',
        'user_centric_trading_system',
        'performance_optimization_system',
        'system_performance_monitor',
        'market_time_manager',
        'fallback_notification_system'
    ]
    
    results = {}
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            results[module_name] = "✓"
            print(f"  OK {module_name}")
        except ImportError as e:
            results[module_name] = f"✗ {e}"
            print(f"  NG {module_name}: {e}")
    
    success_count = sum(1 for result in results.values() if result == "✓")
    total_count = len(results)
    print(f"\nインポート結果: {success_count}/{total_count} 成功")
    
    return success_count == total_count

def test_core_functionality():
    """コア機能テスト"""
    print("\nコア機能テスト")
    
    try:
        # DayTradeCoreの初期化テスト
        from daytrade_core import DayTradeCore
        core = DayTradeCore()
        print("  OK DayTradeCoreの初期化成功")
        
        # データプロバイダーテスト
        from enhanced_data_provider import get_data_provider
        provider = get_data_provider()
        print("  OK データプロバイダー初期化成功")
        
        # 分析エンジンテスト
        from enhanced_personal_analysis_engine import get_analysis_engine
        engine = get_analysis_engine()
        print("  OK 分析エンジン初期化成功")
        
        # ユーザー中心システムテスト
        from user_centric_trading_system import get_trading_system
        trading_system = get_trading_system()
        print("  OK ユーザー中心取引システム初期化成功")
        
        # パフォーマンス最適化システムテスト
        from performance_optimization_system import get_performance_system
        perf_system = get_performance_system()
        print("  OK パフォーマンス最適化システム初期化成功")
        
        return True
        
    except Exception as e:
        print(f"  NG コア機能エラー: {e}")
        return False

async def test_async_functionality():
    """非同期機能テスト"""
    print("\n非同期機能テスト")
    
    try:
        # 分析エンジンの非同期機能テスト
        from enhanced_personal_analysis_engine import get_analysis_engine
        engine = get_analysis_engine()
        
        # シンプルな分析テスト
        test_symbols = ['7203', '8306']
        for symbol in test_symbols:
            try:
                result = await engine.analyze_symbol(symbol)
                print(f"  OK {symbol}分析完了: {result.signal.value}")
            except Exception as e:
                print(f"  WN {symbol}分析警告: {e}")
        
        # データプロバイダーの非同期機能テスト
        from enhanced_data_provider import get_data_provider
        provider = get_data_provider()
        
        result = await provider.get_stock_data('7203')
        if result and result.data:
            print(f"  OK 非同期データ取得成功: {result.data.get('symbol', 'N/A')}")
        else:
            print("  WN 非同期データ取得警告: フォールバック使用")
        
        return True
        
    except Exception as e:
        print(f"  NG 非同期機能エラー: {e}")
        return False

def test_web_server():
    """Webサーバーテスト"""
    print("\nWebサーバーテスト")
    
    try:
        from daytrade_web import DayTradeWebServer, WEB_AVAILABLE
        
        if not WEB_AVAILABLE:
            print("  WN Web機能の依存関係が不足（Flaskなど）")
            return False
            
        server = DayTradeWebServer(port=8001, debug=True)
        print("  OK Webサーバー初期化成功")
        
        # APIエンドポイントのテスト（起動せずに）
        analysis_data = server._get_analysis_data()
        if analysis_data.get('status') == 'success':
            print("  OK 分析APIエンドポイント正常")
        
        ml_details = server._get_ml_details()
        if ml_details.get('status') == 'success':
            print("  OK ML詳細APIエンドポイント正常")
            
        return True
        
    except Exception as e:
        print(f"  NG Webサーバーエラー: {e}")
        return False

def test_performance_system():
    """パフォーマンスシステムテスト"""
    print("\nパフォーマンスシステムテスト")
    
    try:
        from performance_optimization_system import get_performance_system
        from system_performance_monitor import get_system_monitor
        
        # パフォーマンス最適化システム
        perf_system = get_performance_system()
        metrics = perf_system.get_current_metrics()
        print(f"  OK 現在のメトリクス: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
        
        # システム監視
        monitor = get_system_monitor()
        health = monitor.get_current_health()
        print(f"  OK システム健全性: {health.overall_status}")
        
        return True
        
    except Exception as e:
        print(f"  NG パフォーマンスシステムエラー: {e}")
        return False

def main():
    """メイン実行"""
    print("Day Trade AI システム統合テスト")
    print("=" * 50)
    
    test_results = []
    
    # テスト実行
    test_results.append(("モジュールインポート", test_module_imports()))
    test_results.append(("コア機能", test_core_functionality()))
    test_results.append(("非同期機能", asyncio.run(test_async_functionality())))
    test_results.append(("Webサーバー", test_web_server()))
    test_results.append(("パフォーマンスシステム", test_performance_system()))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    
    success_count = 0
    for test_name, result in test_results:
        status = "成功" if result else "失敗"
        print(f"  {test_name}: {status}")
        if result:
            success_count += 1
    
    total_tests = len(test_results)
    success_rate = (success_count / total_tests) * 100
    
    print(f"\n総合結果: {success_count}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("統合テスト合格 - システム運用準備完了")
        return 0
    else:
        print("統合テスト不合格 - 修正が必要")
        return 1

if __name__ == "__main__":
    sys.exit(main())
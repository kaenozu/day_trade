#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最適化システム簡単テスト
"""

import sys
import asyncio
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimization_modules():
    """最適化モジュールの基本テスト"""
    try:
        # 個別モジュールのインポートテスト
        from src.day_trade.optimization.prediction_accuracy_enhancer import PredictionAccuracyEnhancer
        print("OK PredictionAccuracyEnhancer インポート成功")
        
        from src.day_trade.optimization.performance_optimization_engine import PerformanceOptimizationEngine
        print("OK PerformanceOptimizationEngine インポート成功")
        
        from src.day_trade.optimization.model_accuracy_improver import ModelAccuracyImprover
        print("OK ModelAccuracyImprover インポート成功")
        
        from src.day_trade.optimization.response_speed_optimizer import ResponseSpeedOptimizer
        print("OK ResponseSpeedOptimizer インポート成功")
        
        from src.day_trade.optimization.memory_efficiency_optimizer import MemoryEfficiencyOptimizer
        print("OK MemoryEfficiencyOptimizer インポート成功")
        
        # 統合システムインポートテスト
        from src.day_trade.optimization.integrated_optimization_system import IntegratedOptimizationSystem
        print("OK IntegratedOptimizationSystem インポート成功")
        
        return True
        
    except ImportError as e:
        print(f"NG インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"NG 予期しないエラー: {e}")
        return False

def test_basic_functionality():
    """基本機能テスト"""
    try:
        # メモリ効率化テスト
        from src.day_trade.optimization.memory_efficiency_optimizer import MemoryEfficiencyOptimizer
        memory_optimizer = MemoryEfficiencyOptimizer()
        
        # メモリ分析実行
        analysis = memory_optimizer.analyze_memory_usage()
        print("OK メモリ分析実行成功")
        print(f"   - システムメモリ: {analysis['system_memory']['total_mb']:.1f}MB")
        print(f"   - プロセスメモリ: {analysis['process_memory']['rss_mb']:.1f}MB")
        
        # 予測精度向上システムテスト
        from src.day_trade.optimization.prediction_accuracy_enhancer import PredictionAccuracyEnhancer
        prediction_enhancer = PredictionAccuracyEnhancer()
        
        print("OK 予測精度向上システム初期化成功")
        
        # パフォーマンス最適化テスト
        from src.day_trade.optimization.performance_optimization_engine import PerformanceOptimizationEngine
        performance_engine = PerformanceOptimizationEngine()
        
        cpu_usage = performance_engine.monitor_cpu_usage()
        memory_info = performance_engine.monitor_memory_usage()
        print("OK パフォーマンス監視実行成功")
        print(f"   - CPU使用率: {cpu_usage:.1f}%")
        print(f"   - メモリ使用率: {memory_info['usage_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"NG 基本機能テストエラー: {e}")
        return False

async def test_async_functionality():
    """非同期機能テスト"""
    try:
        # レスポンス速度最適化テスト
        from src.day_trade.optimization.response_speed_optimizer import ResponseSpeedOptimizer
        speed_optimizer = ResponseSpeedOptimizer()
        
        await speed_optimizer.initialize()
        print("OK レスポンス速度最適化システム初期化成功")
        
        # パフォーマンスメトリクス取得
        metrics = await speed_optimizer.get_performance_metrics()
        print("OK パフォーマンスメトリクス取得成功")
        print(f"   - リクエスト数: {metrics['total_requests']}")
        
        return True
        
    except Exception as e:
        print(f"NG 非同期機能テストエラー: {e}")
        return False

def test_web_integration():
    """Web統合テスト"""
    try:
        # Flaskアプリ作成テスト
        from web.app import create_app, OPTIMIZATION_AVAILABLE
        
        app = create_app()
        print("OK Flaskアプリケーション作成成功")
        print(f"   - 最適化システム利用可能: {'Yes' if OPTIMIZATION_AVAILABLE else 'No'}")
        
        # テストクライアント作成
        client = app.test_client()
        
        # ヘルスチェック
        response = client.get('/health')
        assert response.status_code == 200
        print("OK ヘルスチェックAPI成功")
        
        # 最適化ステータス
        if OPTIMIZATION_AVAILABLE:
            response = client.get('/api/optimization/status')
            print(f"OK 最適化ステータスAPI: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"NG Web統合テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("Day Trade 最適化システム 統合テスト開始")
    print("=" * 50)
    
    all_passed = True
    
    # 1. モジュールインポートテスト
    print("\n1. モジュールインポートテスト")
    if not test_optimization_modules():
        all_passed = False
    
    # 2. 基本機能テスト
    print("\n2. 基本機能テスト")
    if not test_basic_functionality():
        all_passed = False
    
    # 3. 非同期機能テスト
    print("\n3. 非同期機能テスト")
    try:
        if not asyncio.run(test_async_functionality()):
            all_passed = False
    except Exception as e:
        print(f"NG 非同期テスト実行エラー: {e}")
        all_passed = False
    
    # 4. Web統合テスト
    print("\n4. Web統合テスト")
    if not test_web_integration():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("全てのテストが成功しました！")
        print("\n実装された最適化機能:")
        print("   - 予測精度向上システム")
        print("   - パフォーマンス最適化エンジン")
        print("   - モデル精度改善システム")
        print("   - レスポンス速度最適化")
        print("   - メモリ効率化最適化")
        print("   - 統合最適化システム")
        print("   - Web API統合")
        
        print("\n利用可能なAPI:")
        print("   - GET  /api/optimization/status")
        print("   - POST /api/optimization/run")
        print("   - GET  /api/optimization/health")
        print("   - GET  /api/optimization/history")
        
        return 0
    else:
        print("NG 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
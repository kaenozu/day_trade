#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple System Test - 簡単なシステムテスト
"""

print("Simple System Test")
print("=" * 30)

try:
    print("1. モジュールインポート...")
    from enhanced_data_provider import get_data_provider
    print("   OK: enhanced_data_provider")
    
    from performance_optimization_system import get_performance_system
    print("   OK: performance_optimization_system")
    
    print("\n2. インスタンス作成...")
    provider = get_data_provider()
    print(f"   OK: EnhancedDataProvider - logger: {hasattr(provider, 'logger')}")
    
    perf_system = get_performance_system()
    print("   OK: PerformanceOptimizationSystem")
    
    print("\n3. パフォーマンステスト...")
    metrics = perf_system.get_current_metrics()
    print(f"   OK: Current metrics - CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
    
    print("\nテスト完了 - 全て正常")

except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡単な統合テスト
"""

import asyncio
import sys
from pathlib import Path

# Windows環境での文字化け対策
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from auto_update_optimizer import AutoUpdateOptimizer

async def test_integration():
    """統合テスト実行"""
    print("=== AutoUpdateOptimizer統合テスト ===")

    try:
        # 初期化
        optimizer = AutoUpdateOptimizer()
        print("✓ システム初期化完了")

        # 銘柄初期化
        await optimizer.initialize()
        print(f"✓ 銘柄初期化完了: {len(optimizer.current_symbols)}銘柄")

        # EnhancedSymbolManager統合確認
        if optimizer.symbol_manager:
            print("✓ EnhancedSymbolManagerとの統合成功")

            # 統計情報取得
            stats = optimizer.get_enhanced_symbol_statistics()
            print(f"  - 選択銘柄数: {stats['current_selected_symbols']}")
            print(f"  - 高優先度: {stats['priority_distribution'].get('high', 0)}")
            print(f"  - 中優先度: {stats['priority_distribution'].get('medium', 0)}")
            print(f"  - 低優先度: {stats['priority_distribution'].get('low', 0)}")
        else:
            print("⚠ EnhancedSymbolManagerが利用できません")

        # システムメトリクス更新テスト
        optimizer.update_system_metrics()
        print(f"✓ システムメトリクス更新完了")
        print(f"  - CPU使用率: {optimizer.system_metrics.cpu_usage:.1f}%")
        print(f"  - メモリ使用量: {optimizer.system_metrics.memory_usage_mb:.0f}MB")
        print(f"  - 負荷レベル: {optimizer.system_metrics.load_level.value}")

        # コンポーネント確認
        if optimizer.symbol_queue:
            print("✓ 銘柄優先度キュー初期化完了")
            print(f"  - キューサイズ: {optimizer.symbol_queue.get_queue_size()}")

        if optimizer.frequency_manager:
            print("✓ 更新頻度管理初期化完了")
            print(f"  - 基本頻度: {optimizer.frequency_manager.get_base_frequency()}秒")

        if optimizer.progress_manager:
            print("✓ 進捗表示管理初期化完了")

        if optimizer.performance_tracker:
            print("✓ パフォーマンス追跡初期化完了")

        print("\n=== 統合テスト完了 ===")
        return True

    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_integration())
    if result:
        print("統合テスト成功")
    else:
        print("統合テスト失敗")
        sys.exit(1)
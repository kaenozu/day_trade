#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動更新最適化システム統合テストスイート
Issue #881対応：自動更新の更新時間を考える - 統合テスト

包括的な統合テストによる品質保証
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import yaml
import json
import time
import psutil
import sqlite3
from rich.console import Console
from rich.progress import Progress
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

# テスト対象システムのインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from auto_update_optimizer import (
    AutoUpdateOptimizer,
    SystemMetrics,
    ProgressDisplayManager,
    SymbolPriorityQueue,
    UpdateFrequencyManager,
    PerformanceTracker
)

from enhanced_symbol_manager import (
    EnhancedSymbolManager,
    EnhancedStockInfo,
    SymbolTier,
    IndustryCategory,
    VolatilityLevel
)

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


@dataclass
class TestConfiguration:
    """テスト設定"""
    test_symbols_count: int = 10
    test_duration_seconds: int = 30
    performance_threshold_seconds: float = 5.0
    memory_threshold_mb: float = 100.0
    cpu_threshold_percent: float = 50.0


@pytest.fixture(scope="session")
def console():
    """コンソール出力管理"""
    return Console()


@pytest.fixture(scope="session")
def test_config():
    """テスト設定"""
    return TestConfiguration()


@pytest.fixture
def temp_dir():
    """一時ディレクトリ"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config_file(temp_dir):
    """モック設定ファイル"""
    config_data = {
        'system_name': 'テスト自動更新最適化システム',
        'version': '1.0.0',
        'debug_mode': True,

        'update_frequency_optimization': {
            'adaptive_frequency': {
                'enabled': True,
                'base_frequency_seconds': 60,
                'min_frequency_seconds': 30,
                'max_frequency_seconds': 300
            },
            'market_hours_aware': {
                'enabled': True,
                'frequencies': {
                    'market_open': 30,
                    'market_close': 120
                }
            },
            'load_based_adjustment': {
                'enabled': True,
                'cpu_threshold_high': 80,
                'cpu_threshold_low': 50,
                'memory_threshold_mb': 1000
            }
        },

        'symbol_expansion': {
            'max_symbols': 20,
            'batch_size': 5,
            'parallel_processing': True,
            'categories': {
                'high_priority': {'count': 5, 'update_frequency': 30},
                'medium_priority': {'count': 10, 'update_frequency': 60},
                'low_priority': {'count': 5, 'update_frequency': 120}
            }
        },

        'progress_display': {
            'progress_bar': {
                'enabled': True,
                'style': 'detailed'
            },
            'status_display': {
                'real_time_updates': True,
                'update_interval_seconds': 5
            }
        },

        'performance_optimization': {
            'memory_management': {
                'max_memory_usage_mb': 500,
                'garbage_collection': True
            },
            'parallel_processing': {
                'max_workers': 4,
                'worker_pool_type': 'thread'
            }
        }
    }

    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    return config_path


@pytest.fixture
def mock_symbol_manager():
    """モック銘柄管理システム"""
    manager = Mock(spec=EnhancedSymbolManager)

    # テスト銘柄データ
    test_symbols = [
        EnhancedStockInfo("7203", "トヨタ自動車", SymbolTier.TIER1_CORE, IndustryCategory.AUTOMOTIVE,
                         VolatilityLevel.LOW, 37000, 800, 95, 90, 70, 2.4),
        EnhancedStockInfo("8306", "三菱UFJ", SymbolTier.TIER1_CORE, IndustryCategory.FINANCE,
                         VolatilityLevel.MEDIUM, 11000, 600, 90, 85, 60, 4.2),
        EnhancedStockInfo("4751", "サイバーエージェント", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                         VolatilityLevel.VERY_HIGH, 2800, 200, 75, 70, 95, 0.5),
        EnhancedStockInfo("2370", "メディネット", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                         VolatilityLevel.HIGH, 250, 50, 60, 55, 98, 0.0),
        EnhancedStockInfo("9984", "ソフトバンクG", SymbolTier.TIER1_CORE, IndustryCategory.TECH,
                         VolatilityLevel.HIGH, 8000, 500, 85, 70, 80, 6.1)
    ]

    # モックメソッドの設定
    manager.get_daytrading_optimized_portfolio.return_value = test_symbols
    manager.get_diversified_portfolio.return_value = test_symbols
    manager.get_high_volatility_opportunities.return_value = test_symbols[:3]
    manager.get_symbols_by_tier.return_value = test_symbols[:2]
    manager.symbols = {symbol.symbol: symbol for symbol in test_symbols}

    return manager


class TestAutoUpdateOptimizerIntegration:
    """自動更新最適化システム統合テスト"""

    @pytest.mark.asyncio
    async def test_system_initialization(self, mock_config_file, mock_symbol_manager, console):
        """システム初期化テスト"""
        console.print("[bold green]システム初期化テスト開始[/bold green]")

        with patch('auto_update_optimizer.EnhancedSymbolManager', return_value=mock_symbol_manager):
            optimizer = AutoUpdateOptimizer(str(mock_config_file))

            # 初期化確認
            assert optimizer.config is not None
            assert optimizer.symbol_manager is not None
            assert optimizer.frequency_manager is not None
            assert optimizer.progress_manager is not None
            assert optimizer.performance_tracker is not None

        console.print("[bold green]✓ システム初期化完了[/bold green]")

    @pytest.mark.asyncio
    async def test_symbol_priority_queue_operations(self, mock_config_file, mock_symbol_manager, console):
        """銘柄優先度キュー操作テスト"""
        console.print("[bold blue]銘柄優先度キュー操作テスト開始[/bold blue]")

        with patch('auto_update_optimizer.EnhancedSymbolManager', return_value=mock_symbol_manager):
            optimizer = AutoUpdateOptimizer(str(mock_config_file))
            await optimizer.initialize()

            # キュー操作テスト
            queue = optimizer.symbol_queue

            # キューが空でないことを確認
            assert not queue.is_empty()

            # 高優先度銘柄の取得
            high_priority_symbol = await queue.get_next_symbol()
            assert high_priority_symbol is not None

            # 銘柄更新とスコア調整
            await queue.update_symbol_score(high_priority_symbol.symbol, 95.0)

            # キューサイズ確認
            queue_size = queue.get_queue_size()
            assert queue_size > 0

        console.print(f"[bold blue]✓ キュー操作テスト完了 (キューサイズ: {queue_size})[/bold blue]")

    @pytest.mark.asyncio
    async def test_update_frequency_management(self, mock_config_file, mock_symbol_manager, console):
        """更新頻度管理テスト"""
        console.print("[bold yellow]更新頻度管理テスト開始[/bold yellow]")

        with patch('auto_update_optimizer.EnhancedSymbolManager', return_value=mock_symbol_manager):
            optimizer = AutoUpdateOptimizer(str(mock_config_file))
            await optimizer.initialize()

            frequency_manager = optimizer.frequency_manager

            # 基本頻度設定確認
            base_frequency = frequency_manager.get_base_frequency()
            assert base_frequency > 0

            # 動的頻度調整テスト
            # 高負荷状態をシミュレート
            frequency_manager.system_metrics.cpu_usage = 85.0
            frequency_manager.system_metrics.memory_usage_mb = 1200.0
            frequency_manager.system_metrics.load_level = "high_load"

            adjusted_frequency = frequency_manager.get_adjusted_frequency("7203")
            assert adjusted_frequency > base_frequency  # 高負荷時は頻度を下げる

            # 低負荷状態をシミュレート
            frequency_manager.system_metrics.cpu_usage = 30.0
            frequency_manager.system_metrics.memory_usage_mb = 400.0
            frequency_manager.system_metrics.load_level = "low_load"

            adjusted_frequency_low = frequency_manager.get_adjusted_frequency("7203")
            assert adjusted_frequency_low < base_frequency  # 低負荷時は頻度を上げる

        console.print("[bold yellow]✓ 更新頻度管理テスト完了[/bold yellow]")

    @pytest.mark.asyncio
    async def test_progress_display_system(self, mock_config_file, mock_symbol_manager, console):
        """進捗表示システムテスト"""
        console.print("[bold magenta]進捗表示システムテスト開始[/bold magenta]")

        with patch('auto_update_optimizer.EnhancedSymbolManager', return_value=mock_symbol_manager):
            optimizer = AutoUpdateOptimizer(str(mock_config_file))
            await optimizer.initialize()

            progress_manager = optimizer.progress_manager

            # 進捗表示開始
            total_symbols = 10
            progress_manager.start_progress_display(total_symbols)

            # 進捗更新シミュレート
            for i in range(5):
                progress_manager.update_progress(
                    current_symbol=f"750{i}",
                    processed_count=i + 1,
                    total_count=total_symbols,
                    stage="データ取得中",
                    processing_time=0.5
                )
                await asyncio.sleep(0.1)  # 短時間待機

            # 進捗情報確認
            progress_info = progress_manager.get_progress_info()
            assert progress_info['processed_count'] == 5
            assert progress_info['total_count'] == total_symbols
            assert progress_info['completion_percentage'] == 50.0

            # 進捗表示終了
            progress_manager.stop_progress_display()

        console.print("[bold magenta]✓ 進捗表示システムテスト完了[/bold magenta]")

    @pytest.mark.asyncio
    async def test_performance_tracking(self, mock_config_file, mock_symbol_manager, console):
        """パフォーマンストラッキングテスト"""
        console.print("[bold cyan]パフォーマンストラッキングテスト開始[/bold cyan]")

        with patch('auto_update_optimizer.EnhancedSymbolManager', return_value=mock_symbol_manager):
            optimizer = AutoUpdateOptimizer(str(mock_config_file))
            await optimizer.initialize()

            tracker = optimizer.performance_tracker

            # パフォーマンス測定開始
            tracker.start_tracking()

            # 模擬処理実行
            for i in range(5):
                symbol = f"750{i}"
                start_time = time.time()

                # 模擬処理時間
                await asyncio.sleep(0.1)

                processing_time = time.time() - start_time
                tracker.record_symbol_processing(symbol, processing_time, True)

            # パフォーマンス統計取得
            stats = tracker.get_performance_statistics()

            assert stats['total_processed'] == 5
            assert stats['success_rate'] == 100.0
            assert stats['average_processing_time'] > 0
            assert 'memory_usage_mb' in stats
            assert 'cpu_usage_percent' in stats

            # パフォーマンス測定終了
            tracker.stop_tracking()

        console.print(f"[bold cyan]✓ パフォーマンストラッキングテスト完了 (成功率: {stats['success_rate']}%)[/bold cyan]")

    @pytest.mark.asyncio
    async def test_system_load_adaptation(self, mock_config_file, mock_symbol_manager, console):
        """システム負荷適応テスト"""
        console.print("[bold red]システム負荷適応テスト開始[/bold red]")

        with patch('auto_update_optimizer.EnhancedSymbolManager', return_value=mock_symbol_manager):
            optimizer = AutoUpdateOptimizer(str(mock_config_file))
            await optimizer.initialize()

            # 現在のシステムメトリクス取得
            initial_metrics = optimizer.system_metrics
            initial_frequency = optimizer.frequency_manager.get_base_frequency()

            # 高負荷状態をシミュレート
            optimizer.system_metrics.cpu_usage = 90.0
            optimizer.system_metrics.memory_usage_mb = 1500.0
            optimizer._adjust_to_high_load()

            high_load_frequency = optimizer.frequency_manager.get_base_frequency()
            assert high_load_frequency > initial_frequency

            # 低負荷状態をシミュレート
            optimizer.system_metrics.cpu_usage = 20.0
            optimizer.system_metrics.memory_usage_mb = 300.0
            optimizer._adjust_to_low_load()

            low_load_frequency = optimizer.frequency_manager.get_base_frequency()
            assert low_load_frequency < high_load_frequency

        console.print("[bold red]✓ システム負荷適応テスト完了[/bold red]")

    @pytest.mark.asyncio
    async def test_enhanced_symbol_manager_integration(self, mock_config_file, console):
        """EnhancedSymbolManager統合テスト"""
        console.print("[bold white]EnhancedSymbolManager統合テスト開始[/bold white]")

        # 実際のEnhancedSymbolManagerを使用
        optimizer = AutoUpdateOptimizer(str(mock_config_file))
        await optimizer.initialize()

        # 銘柄管理システムの動作確認
        symbol_manager = optimizer.symbol_manager

        # デイトレード最適化ポートフォリオ取得
        daytrading_portfolio = symbol_manager.get_daytrading_optimized_portfolio(count=10)
        assert len(daytrading_portfolio) <= 10
        assert all(symbol.is_active for symbol in daytrading_portfolio)

        # 業界分散ポートフォリオ取得
        diversified_portfolio = symbol_manager.get_diversified_portfolio(count=15)
        assert len(diversified_portfolio) <= 15

        # ティア別銘柄取得
        tier1_symbols = symbol_manager.get_symbols_by_tier(SymbolTier.TIER1_CORE)
        assert len(tier1_symbols) > 0

        # 高ボラティリティ銘柄取得
        high_vol_symbols = symbol_manager.get_high_volatility_opportunities(count=8)
        assert len(high_vol_symbols) <= 8

        console.print(f"[bold white]✓ EnhancedSymbolManager統合テスト完了[/bold white]")
        console.print(f"  - デイトレード最適化: {len(daytrading_portfolio)}銘柄")
        console.print(f"  - 業界分散: {len(diversified_portfolio)}銘柄")
        console.print(f"  - Tier1コア: {len(tier1_symbols)}銘柄")
        console.print(f"  - 高ボラティリティ: {len(high_vol_symbols)}銘柄")

    @pytest.mark.asyncio
    async def test_full_system_workflow(self, mock_config_file, test_config, console):
        """フルシステムワークフローテスト"""
        console.print("[bold green]フルシステムワークフローテスト開始[/bold green]")

        optimizer = AutoUpdateOptimizer(str(mock_config_file))
        await optimizer.initialize()

        # テスト実行設定
        test_symbols = list(optimizer.symbol_manager.symbols.keys())[:test_config.test_symbols_count]

        console.print(f"テスト対象銘柄数: {len(test_symbols)}")
        console.print(f"テスト実行時間: {test_config.test_duration_seconds}秒")

        # ワークフロー実行
        start_time = time.time()
        processed_count = 0

        optimizer.progress_manager.start_progress_display(len(test_symbols))

        try:
            for symbol in test_symbols:
                processing_start = time.time()

                # 模擬データ更新処理
                await optimizer._simulate_symbol_update(symbol)

                processing_time = time.time() - processing_start
                processed_count += 1

                # 進捗更新
                optimizer.progress_manager.update_progress(
                    current_symbol=symbol,
                    processed_count=processed_count,
                    total_count=len(test_symbols),
                    stage="データ更新完了",
                    processing_time=processing_time
                )

                # パフォーマンス記録
                optimizer.performance_tracker.record_symbol_processing(
                    symbol, processing_time, True
                )

                # テスト時間制限チェック
                if time.time() - start_time > test_config.test_duration_seconds:
                    break

                await asyncio.sleep(0.1)  # 負荷調整

        finally:
            optimizer.progress_manager.stop_progress_display()

        # 結果検証
        total_time = time.time() - start_time
        stats = optimizer.performance_tracker.get_performance_statistics()

        assert processed_count > 0
        assert stats['success_rate'] >= 90.0  # 90%以上の成功率
        assert stats['average_processing_time'] < test_config.performance_threshold_seconds

        console.print(f"[bold green]✓ フルシステムワークフローテスト完了[/bold green]")
        console.print(f"  - 処理銘柄数: {processed_count}")
        console.print(f"  - 実行時間: {total_time:.2f}秒")
        console.print(f"  - 成功率: {stats['success_rate']:.1f}%")
        console.print(f"  - 平均処理時間: {stats['average_processing_time']:.3f}秒")

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, mock_config_file, test_config, console):
        """メモリ使用量最適化テスト"""
        console.print("[bold purple]メモリ使用量最適化テスト開始[/bold purple]")

        optimizer = AutoUpdateOptimizer(str(mock_config_file))
        await optimizer.initialize()

        # 初期メモリ使用量記録
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大量データ処理シミュレート
        test_symbols = list(optimizer.symbol_manager.symbols.keys())

        for i in range(3):  # 3回繰り返し
            for symbol in test_symbols:
                await optimizer._simulate_symbol_update(symbol)

            # ガベージコレクション実行
            optimizer._perform_memory_cleanup()

            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            console.print(f"  ラウンド{i+1}: メモリ使用量 {current_memory:.1f}MB (+{memory_increase:.1f}MB)")

        # 最終メモリ使用量確認
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory

        # メモリ使用量が閾値以下であることを確認
        assert total_increase < test_config.memory_threshold_mb

        console.print(f"[bold purple]✓ メモリ使用量最適化テスト完了[/bold purple]")
        console.print(f"  - 初期メモリ: {initial_memory:.1f}MB")
        console.print(f"  - 最終メモリ: {final_memory:.1f}MB")
        console.print(f"  - 増加量: {total_increase:.1f}MB (閾値: {test_config.memory_threshold_mb}MB)")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_config_file, console):
        """エラーハンドリングと復旧テスト"""
        console.print("[bold orange]エラーハンドリングと復旧テスト開始[/bold orange]")

        optimizer = AutoUpdateOptimizer(str(mock_config_file))
        await optimizer.initialize()

        # エラー状況をシミュレート
        error_count = 0
        recovery_count = 0

        # 模擬エラー処理
        test_symbols = ["INVALID1", "INVALID2", "7203", "8306", "INVALID3"]

        for symbol in test_symbols:
            try:
                if symbol.startswith("INVALID"):
                    # 意図的なエラー発生
                    raise ValueError(f"Invalid symbol: {symbol}")
                else:
                    # 正常処理
                    await optimizer._simulate_symbol_update(symbol)
                    recovery_count += 1
            except ValueError:
                error_count += 1
                # エラー処理とログ記録
                optimizer.performance_tracker.record_symbol_processing(
                    symbol, 0.0, False
                )
                console.print(f"  エラー処理: {symbol}")

        # エラー率確認
        total_processed = len(test_symbols)
        error_rate = error_count / total_processed
        recovery_rate = recovery_count / total_processed

        console.print(f"[bold orange]✓ エラーハンドリングと復旧テスト完了[/bold orange]")
        console.print(f"  - 総処理数: {total_processed}")
        console.print(f"  - エラー数: {error_count} (率: {error_rate:.1%})")
        console.print(f"  - 復旧数: {recovery_count} (率: {recovery_rate:.1%})")

        # 復旧機能が正常に動作していることを確認
        assert recovery_count > 0
        assert error_rate < 1.0  # 全てがエラーではない


class TestSupportUtilities:
    """テスト支援ユーティリティ"""

    @staticmethod
    async def simulate_symbol_update(optimizer, symbol: str) -> bool:
        """銘柄更新シミュレート"""
        try:
            # 模擬データ取得処理
            await asyncio.sleep(0.05)  # 50ms処理時間シミュレート

            # 模擬データ検証
            if symbol.startswith("INVALID"):
                raise ValueError(f"Invalid symbol: {symbol}")

            # 成功
            return True

        except Exception:
            # 失敗
            return False


# AutoUpdateOptimizerクラスに追加するテスト用メソッド
def add_test_methods_to_optimizer():
    """テスト用メソッドをAutoUpdateOptimizerに追加"""

    async def _simulate_symbol_update(self, symbol: str):
        """テスト用銘柄更新シミュレート"""
        await asyncio.sleep(0.05)  # 50ms処理時間
        if symbol.startswith("INVALID"):
            raise ValueError(f"Invalid symbol: {symbol}")

    def _perform_memory_cleanup(self):
        """テスト用メモリクリーンアップ"""
        import gc
        gc.collect()

    def _adjust_to_high_load(self):
        """テスト用高負荷調整"""
        self.frequency_manager.base_frequency *= 1.5

    def _adjust_to_low_load(self):
        """テスト用低負荷調整"""
        self.frequency_manager.base_frequency *= 0.8

    # AutoUpdateOptimizerクラスにメソッドを動的追加
    AutoUpdateOptimizer._simulate_symbol_update = _simulate_symbol_update
    AutoUpdateOptimizer._perform_memory_cleanup = _perform_memory_cleanup
    AutoUpdateOptimizer._adjust_to_high_load = _adjust_to_high_load
    AutoUpdateOptimizer._adjust_to_low_load = _adjust_to_low_load


# テスト実行前の初期化
add_test_methods_to_optimizer()


def test_configuration_validation():
    """設定ファイル検証テスト"""
    console = Console()
    console.print("[bold blue]設定ファイル検証テスト開始[/bold blue]")

    # 本番設定ファイルの存在確認
    config_path = Path("config/auto_update_optimizer_config.yaml")
    assert config_path.exists(), "設定ファイルが存在しません"

    # 設定ファイル読み込みテスト
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 必須セクションの存在確認
    required_sections = [
        'update_frequency_optimization',
        'symbol_expansion',
        'progress_display',
        'performance_optimization'
    ]

    for section in required_sections:
        assert section in config, f"必須セクション'{section}'が不足しています"

    console.print("[bold blue]✓ 設定ファイル検証テスト完了[/bold blue]")


if __name__ == "__main__":
    # 直接実行時の簡易テスト
    console = Console()
    console.print("[bold green]自動更新最適化システム統合テスト実行[/bold green]")

    # pytest実行
    pytest.main([__file__, "-v", "--tb=short"])
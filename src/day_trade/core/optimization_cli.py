#!/usr/bin/env python3
"""
最適化設定管理CLI

Strategy Pattern統合システムの設定管理と動作テスト機能
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import psutil
import pandas as pd
import numpy as np

from .optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategyFactory
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class OptimizationCLI:
    """最適化設定CLI管理クラス"""

    def __init__(self):
        self.config_dir = Path(__file__).parent.parent.parent.parent / "config"
        self.config_file = self.config_dir / "optimization_config.json"

    def create_config_template(self, output_path: Optional[str] = None) -> None:
        """設定テンプレートファイルの作成"""
        if output_path:
            template_path = Path(output_path)
        else:
            template_path = self.config_file

        # ディレクトリが存在しない場合は作成
        template_path.parent.mkdir(parents=True, exist_ok=True)

        OptimizationStrategyFactory.create_config_template(str(template_path))
        print(f"設定テンプレート作成完了: {template_path}")

    def show_current_config(self) -> None:
        """現在の設定を表示"""
        try:
            config = OptimizationConfig.from_file(str(self.config_file))
            print("現在の最適化設定:")
            print(f"  最適化レベル: {config.level.value}")
            print(f"  自動フォールバック: {config.auto_fallback}")
            print(f"  パフォーマンス監視: {config.performance_monitoring}")
            print(f"  キャッシュ有効: {config.cache_enabled}")
            print(f"  並列処理: {config.parallel_processing}")
            print(f"  バッチサイズ: {config.batch_size}")
            print(f"  メモリ制限: {config.memory_limit_mb}MB")
            print(f"  タイムアウト: {config.timeout_seconds}秒")

        except Exception as e:
            print(f"設定ファイル読み込み失敗: {e}")
            print("環境変数からの設定を使用:")
            config = OptimizationConfig.from_env()
            print(f"  最適化レベル: {config.level.value}")

    def list_components(self) -> None:
        """登録済みコンポーネント一覧を表示"""
        components = OptimizationStrategyFactory.get_registered_components()

        if not components:
            print("登録済みコンポーネントはありません")
            return

        print("登録済みコンポーネント:")
        for component, levels in components.items():
            print(f"  {component}: {', '.join(levels)}")

    def test_component(self, component_name: str, level: Optional[str] = None) -> None:
        """特定コンポーネントのテスト"""
        print(f"コンポーネントテスト開始: {component_name}")

        # 設定の準備
        if level:
            try:
                optimization_level = OptimizationLevel(level.lower())
            except ValueError:
                print(f"無効な最適化レベル: {level}")
                return

            config = OptimizationConfig(level=optimization_level)
        else:
            config = OptimizationConfig.from_file(str(self.config_file))

        try:
            # 戦略の取得とテスト
            strategy = OptimizationStrategyFactory.get_strategy(component_name, config)
            print(f"戦略取得成功: {strategy.get_strategy_name()}")

            # コンポーネント別テスト実行
            if component_name == "technical_indicators":
                self._test_technical_indicators(strategy)
            elif component_name == "feature_engineering":
                self._test_feature_engineering(strategy)
            elif component_name == "database":
                self._test_database(strategy)
            else:
                print(f"テスト未実装: {component_name}")

            # パフォーマンス指標の表示
            metrics = strategy.get_performance_metrics()
            print(f"パフォーマンス指標:")
            print(f"  実行回数: {metrics.get('execution_count', 0)}")
            print(f"  成功回数: {metrics.get('success_count', 0)}")
            print(f"  平均実行時間: {metrics.get('average_time', 0):.3f}秒")

        except Exception as e:
            print(f"コンポーネントテスト失敗: {e}")

    def _test_technical_indicators(self, strategy) -> None:
        """テクニカル指標のテスト"""
        print("テクニカル指標テストデータ生成...")

        # テストデータ生成
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 1000 + np.cumsum(np.random.randn(100) * 10)

        test_data = pd.DataFrame({
            'Date': dates,
            '終値': prices,
            '高値': prices + np.random.rand(100) * 5,
            '安値': prices - np.random.rand(100) * 5,
            '出来高': np.random.randint(1000, 10000, 100)
        }).set_index('Date')

        # テスト実行
        test_indicators = ['sma', 'bollinger_bands', 'rsi']

        start_time = time.time()
        result = strategy.execute(test_data, test_indicators, period=20)
        execution_time = time.time() - start_time

        print(f"テクニカル指標計算完了: {len(test_indicators)}指標, {execution_time:.3f}秒")

        for indicator, indicator_result in result.items():
            print(f"  {indicator}: 計算時間 {indicator_result.calculation_time:.3f}秒")

    def _test_feature_engineering(self, strategy) -> None:
        """特徴量エンジニアリングのテスト"""
        print("特徴量エンジニアリングテストデータ生成...")

        # テストデータ生成
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        prices = 1000 + np.cumsum(np.random.randn(200) * 10)

        test_data = pd.DataFrame({
            'Date': dates,
            '終値': prices,
            '高値': prices + np.random.rand(200) * 5,
            '安値': prices - np.random.rand(200) * 5,
            '出来高': np.random.randint(1000, 10000, 200)
        }).set_index('Date')

        # テスト実行
        start_time = time.time()
        result = strategy.execute(test_data)
        execution_time = time.time() - start_time

        print(f"特徴量生成完了: {result.features.shape[1]}特徴量, {execution_time:.3f}秒")
        print(f"生成特徴量: {len(result.feature_names)}個")

    def _test_database(self, strategy) -> None:
        """データベースのテスト"""
        print("データベーステスト実行...")

        # テストクエリ
        test_queries = [
            "SELECT 1 as test_value",
            "SELECT datetime('now') as current_time",
        ]

        for query in test_queries:
            start_time = time.time()
            result = strategy.execute("execute_query", query)
            execution_time = time.time() - start_time

            if result.success:
                print(f"クエリ成功: {execution_time:.3f}秒")
            else:
                print(f"クエリ失敗: {result.error_message}")

    def benchmark_all(self) -> None:
        """全コンポーネントのベンチマーク"""
        print("全コンポーネントベンチマーク開始...")

        components = OptimizationStrategyFactory.get_registered_components()
        levels = [OptimizationLevel.STANDARD, OptimizationLevel.OPTIMIZED]

        results = {}

        for component in components.keys():
            results[component] = {}

            for level in levels:
                if level.value not in components[component]:
                    continue

                print(f"\nベンチマーク: {component} - {level.value}")

                config = OptimizationConfig(level=level)

                try:
                    strategy = OptimizationStrategyFactory.get_strategy(component, config)

                    # 簡易ベンチマーク実行
                    start_time = time.time()
                    self.test_component(component, level.value)
                    execution_time = time.time() - start_time

                    metrics = strategy.get_performance_metrics()
                    results[component][level.value] = {
                        "total_time": execution_time,
                        "avg_time": metrics.get('average_time', 0),
                        "success_count": metrics.get('success_count', 0),
                    }

                except Exception as e:
                    print(f"ベンチマーク失敗: {e}")
                    results[component][level.value] = {"error": str(e)}

        # 結果サマリー表示
        print("\n" + "=" * 50)
        print("ベンチマーク結果サマリー")
        print("=" * 50)

        for component, level_results in results.items():
            print(f"\n{component}:")
            for level, metrics in level_results.items():
                if "error" in metrics:
                    print(f"  {level}: エラー - {metrics['error']}")
                else:
                    print(f"  {level}: 総時間 {metrics['total_time']:.3f}秒, "
                          f"平均時間 {metrics['avg_time']:.3f}秒")

    def system_info(self) -> None:
        """システム情報の表示"""
        print("システム情報:")

        # CPU情報
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        print(f"  CPU使用率: {cpu_percent}% ({cpu_count}コア)")

        # メモリ情報
        memory = psutil.virtual_memory()
        print(f"  メモリ使用率: {memory.percent}%")
        print(f"  利用可能メモリ: {memory.available / (1024**3):.1f}GB")
        print(f"  総メモリ: {memory.total / (1024**3):.1f}GB")

        # ディスク情報
        disk = psutil.disk_usage('.')
        print(f"  ディスク使用率: {disk.percent}%")
        print(f"  利用可能容量: {disk.free / (1024**3):.1f}GB")

        # Python環境
        print(f"  Python版: {sys.version.split()[0]}")

        # 依存ライブラリ確認
        optional_libs = {
            'numba': 'Numba高速化',
            'talib': 'TA-Lib指標',
            'psutil': 'システム監視',
            'sqlalchemy': 'データベース',
            'pandas': 'データ処理',
            'numpy': '数値計算'
        }

        print("\n依存ライブラリ:")
        for lib, description in optional_libs.items():
            try:
                __import__(lib)
                print(f"  {lib}: 利用可能 ({description})")
            except ImportError:
                print(f"  {lib}: 未インストール ({description})")


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Day Trade最適化設定管理CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')

    # 設定関連コマンド
    config_parser = subparsers.add_parser('config', help='設定管理')
    config_subparsers = config_parser.add_subparsers(dest='config_action')

    config_subparsers.add_parser('show', help='現在の設定を表示')

    template_parser = config_subparsers.add_parser('template', help='設定テンプレート作成')
    template_parser.add_argument('--output', '-o', help='出力ファイルパス')

    # コンポーネント関連コマンド
    comp_parser = subparsers.add_parser('component', help='コンポーネント管理')
    comp_subparsers = comp_parser.add_subparsers(dest='component_action')

    comp_subparsers.add_parser('list', help='登録済みコンポーネント一覧')

    test_parser = comp_subparsers.add_parser('test', help='コンポーネントテスト')
    test_parser.add_argument('name', help='コンポーネント名')
    test_parser.add_argument('--level', '-l', help='最適化レベル (standard/optimized/adaptive)')

    # ベンチマーク関連コマンド
    subparsers.add_parser('benchmark', help='全コンポーネントベンチマーク')

    # システム情報
    subparsers.add_parser('system', help='システム情報表示')

    args = parser.parse_args()

    cli = OptimizationCLI()

    if args.command == 'config':
        if args.config_action == 'show':
            cli.show_current_config()
        elif args.config_action == 'template':
            cli.create_config_template(args.output)

    elif args.command == 'component':
        if args.component_action == 'list':
            cli.list_components()
        elif args.component_action == 'test':
            cli.test_component(args.name, args.level)

    elif args.command == 'benchmark':
        cli.benchmark_all()

    elif args.command == 'system':
        cli.system_info()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

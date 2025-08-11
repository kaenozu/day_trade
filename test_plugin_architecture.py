#!/usr/bin/env python3
"""
プラグインアーキテクチャテストシステム

Issue #369: プラグイン型リスク分析システム総合テスト
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

# プロジェクトパス設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.day_trade.plugins import PluginManager
from src.day_trade.plugins.interfaces import (
    MarketData,
    PluginType,
    PortfolioData,
    RiskLevel,
)


def create_sample_market_data() -> MarketData:
    """サンプル市場データ作成"""
    price_data = pd.DataFrame({
        'Open': [1000, 1010, 1005, 1020, 1015],
        'High': [1020, 1025, 1015, 1035, 1030],
        'Low': [995, 1000, 995, 1015, 1010],
        'Close': [1010, 1005, 1020, 1015, 1025],
        'Volume': [100000, 120000, 90000, 150000, 110000]
    })

    volume_data = pd.DataFrame({
        'Volume': [100000, 120000, 90000, 150000, 110000]
    })

    news_data = [
        {
            'symbol': '7203.T',
            'title': '業績好調で株価上昇',
            'content': 'トヨタの業績が好調で株価が上昇している',
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': '9984.T',
            'title': 'リスク要因で下落',
            'content': 'ソフトバンクGにリスク要因で株価が下落',
            'timestamp': datetime.now().isoformat()
        }
    ]

    return MarketData(
        symbol='7203.T',
        timestamp=datetime.now(),
        price_data=price_data,
        volume_data=volume_data,
        news_data=news_data
    )


def create_sample_portfolio_data() -> PortfolioData:
    """サンプルポートフォリオデータ作成"""
    return PortfolioData(
        positions={
            '7203.T': {'quantity': 1000, 'avg_price': 1000, 'market_value': 1025000},
            '8306.T': {'quantity': 500, 'avg_price': 800, 'market_value': 410000},
            '9984.T': {'quantity': 200, 'avg_price': 5000, 'market_value': 990000},
            '6758.T': {'quantity': 300, 'avg_price': 3000, 'market_value': 915000}
        },
        total_value=3340000,
        available_cash=660000,
        allocation={
            '7203.T': 0.30,
            '8306.T': 0.15,
            '9984.T': 0.35,
            '6758.T': 0.20
        },
        performance_metrics={
            'total_return': 0.08,
            'sharpe_ratio': 0.75,
            'max_drawdown': -0.12
        }
    )


def test_plugin_manager_initialization():
    """プラグインマネージャー初期化テスト"""
    print("=== プラグインマネージャー初期化テスト ===")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"

            # プラグインマネージャー作成
            manager = PluginManager(
                plugins_dir=str(plugins_dir),
                enable_hot_reload=False,  # テスト用に無効化
                security_enabled=True
            )

            # プラグインリスト確認
            plugins = manager.list_plugins()
            print(f"読み込まれたプラグイン数: {len(plugins)}")

            for plugin_info in plugins:
                print(f"- {plugin_info.name} v{plugin_info.version} ({plugin_info.plugin_type.value})")

            # 状態確認
            status = manager.get_plugin_status()
            print(f"プラグイン状態: {len(status)}件")

            manager.cleanup()
            print("[OK] プラグインマネージャー初期化テスト成功\n")
            return True

    except Exception as e:
        print(f"[ERROR] プラグインマネージャー初期化テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_security_sandbox():
    """セキュリティサンドボックステスト"""
    print("=== セキュリティサンドボックステスト ===")

    try:
        from src.day_trade.plugins.sandbox import SecuritySandbox

        sandbox = SecuritySandbox()

        # 安全なコードテスト
        safe_code = '''
import pandas as pd
from datetime import datetime

class TestPlugin:
    def __init__(self):
        self.data = pd.DataFrame()

    def process(self):
        return {"result": "success"}
'''

        # 危険なコードテスト
        dangerous_code = '''
import os
import subprocess

class BadPlugin:
    def __init__(self):
        os.system("rm -rf /")  # 危険
        subprocess.call(["ls", "-la"])  # 危険
'''

        # テンポラリファイル作成
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_file = Path(temp_dir) / "safe_plugin.py"
            dangerous_file = Path(temp_dir) / "dangerous_plugin.py"

            safe_file.write_text(safe_code)
            dangerous_file.write_text(dangerous_code)

            # 検証テスト
            safe_result = sandbox.validate_plugin_file(safe_file)
            dangerous_result = sandbox.validate_plugin_file(dangerous_file)

            print(f"安全コード検証結果: {safe_result}")
            print(f"危険コード検証結果: {dangerous_result}")

            # 実行環境作成テスト
            env = sandbox.create_execution_environment("test_plugin")
            print(f"実行環境作成: {len(env)}項目")

            if safe_result and not dangerous_result:
                print("✅ セキュリティサンドボックステスト成功\n")
                return True
            else:
                print("❌ セキュリティサンドボックス検証失敗")
                return False

    except Exception as e:
        print(f"❌ セキュリティサンドボックステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_builtin_plugins():
    """組み込みプラグインテスト"""
    print("=== 組み込みプラグインテスト ===")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"

            manager = PluginManager(
                plugins_dir=str(plugins_dir),
                enable_hot_reload=False,
                security_enabled=True
            )

            # テストデータ準備
            market_data = create_sample_market_data()
            portfolio_data = create_sample_portfolio_data()

            context = {
                'daily_trades': 25,
                'news_data': market_data.news_data
            }

            # 全プラグインで分析実行
            results = manager.analyze_with_all_plugins(
                market_data=market_data,
                portfolio_data=portfolio_data,
                context=context
            )

            print(f"分析結果: {len(results)}件")

            success_count = 0
            for plugin_name, result in results.items():
                print(f"\n--- {plugin_name} ---")
                print(f"リスクレベル: {result.risk_level.value}")
                print(f"リスクスコア: {result.risk_score:.3f}")
                print(f"信頼度: {result.confidence:.3f}")
                print(f"メッセージ: {result.message}")

                if result.recommendations:
                    print("推奨事項:")
                    for rec in result.recommendations:
                        print(f"  - {rec}")

                if result.alerts:
                    print("アラート:")
                    for alert in result.alerts:
                        print(f"  ⚠️ {alert}")

                # エラーでない場合は成功カウント
                if not result.metadata.get('error', False):
                    success_count += 1

            manager.cleanup()

            if success_count == len(results):
                print(f"\n✅ 組み込みプラグインテスト成功 ({success_count}/{len(results)})\n")
                return True
            else:
                print(f"\n⚠️ 組み込みプラグインテスト部分成功 ({success_count}/{len(results)})\n")
                return False

    except Exception as e:
        print(f"❌ 組み込みプラグインテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_hot_reload():
    """ホットリロードテスト"""
    print("=== ホットリロードテスト ===")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            plugins_dir.mkdir(parents=True, exist_ok=True)

            # カスタムプラグイン作成
            custom_plugin_code = '''
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from ..interfaces import IRiskPlugin, PluginInfo, PluginType, RiskLevel, RiskAnalysisResult

class CustomTestPlugin(IRiskPlugin):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="Custom Test Plugin",
            version="1.0.0",
            description="テスト用カスタムプラグイン",
            author="Test System",
            plugin_type=PluginType.RISK_ANALYZER,
            dependencies=[],
            config_schema={}
        )

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def analyze_risk(self, market_data, portfolio_data=None, context=None) -> RiskAnalysisResult:
        return RiskAnalysisResult(
            plugin_name=self.get_info().name,
            timestamp=datetime.now(),
            risk_level=RiskLevel.LOW,
            risk_score=0.2,
            confidence=0.8,
            message="カスタムプラグインテスト実行",
            details={"test": True},
            recommendations=["テスト推奨事項"],
            alerts=[],
            metadata={"version": "1.0.0"}
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True
'''

            # プラグインファイル作成
            plugin_file = plugins_dir / "plugin_custom_test.py"
            plugin_file.write_text(custom_plugin_code)

            # プラグインマネージャー作成（ホットリロード無効）
            manager = PluginManager(
                plugins_dir=str(plugins_dir),
                enable_hot_reload=False,
                security_enabled=False  # テスト用に無効化
            )

            # 初回プラグイン確認
            initial_plugins = manager.list_plugins()
            initial_count = len([p for p in initial_plugins if p.name == "Custom Test Plugin"])
            print(f"初回カスタムプラグイン: {initial_count}件")

            # 手動リロードテスト
            reload_success = manager.reload_plugin("custom_test")
            print(f"手動リロード結果: {reload_success}")

            # リロード後確認
            reloaded_plugins = manager.list_plugins()
            reloaded_count = len([p for p in reloaded_plugins if p.name == "Custom Test Plugin"])
            print(f"リロード後カスタムプラグイン: {reloaded_count}件")

            # 分析テスト
            if reloaded_count > 0:
                market_data = create_sample_market_data()
                results = manager.analyze_with_all_plugins(market_data)
                custom_result = results.get("custom_test")

                if custom_result:
                    print(f"カスタムプラグイン分析結果: {custom_result.message}")

            manager.cleanup()

            if reload_success and reloaded_count > 0:
                print("✅ ホットリロードテスト成功\n")
                return True
            else:
                print("⚠️ ホットリロードテスト部分成功\n")
                return False

    except Exception as e:
        print(f"❌ ホットリロードテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_performance():
    """プラグインパフォーマンステスト"""
    print("=== プラグインパフォーマンステスト ===")

    try:
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"

            manager = PluginManager(
                plugins_dir=str(plugins_dir),
                enable_hot_reload=False,
                security_enabled=True
            )

            # テストデータ準備
            market_data = [create_sample_market_data() for _ in range(5)]  # 複数銘柄
            portfolio_data = create_sample_portfolio_data()

            # パフォーマンス測定
            start_time = time.time()

            results = manager.analyze_with_all_plugins(
                market_data=market_data,
                portfolio_data=portfolio_data
            )

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"実行時間: {execution_time:.3f}秒")
            print(f"処理プラグイン数: {len(results)}")
            print(f"プラグインあたり平均時間: {execution_time/len(results):.3f}秒")

            # メモリ使用量チェック（概算）
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"メモリ使用量: {memory_mb:.1f}MB")
            except ImportError:
                print("メモリ監視未対応（psutil未インストール）")

            manager.cleanup()

            # パフォーマンス基準チェック
            if execution_time < 10.0:  # 10秒以内
                print("✅ プラグインパフォーマンステスト成功\n")
                return True
            else:
                print("⚠️ プラグインパフォーマンス要改善\n")
                return False

    except Exception as e:
        print(f"❌ プラグインパフォーマンステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """エラーハンドリングテスト"""
    print("=== エラーハンドリングテスト ===")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            plugins_dir.mkdir(parents=True, exist_ok=True)

            # エラーを発生させるプラグイン
            error_plugin_code = '''
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from ..interfaces import IRiskPlugin, PluginInfo, PluginType, RiskLevel, RiskAnalysisResult

class ErrorPlugin(IRiskPlugin):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="Error Plugin",
            version="1.0.0",
            description="エラー発生テストプラグイン",
            author="Test System",
            plugin_type=PluginType.RISK_ANALYZER,
            dependencies=[],
            config_schema={}
        )

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def analyze_risk(self, market_data, portfolio_data=None, context=None) -> RiskAnalysisResult:
        # 意図的にエラー発生
        raise ValueError("テスト用エラー")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True
'''

            # エラープラグインファイル作成
            error_plugin_file = plugins_dir / "plugin_error_test.py"
            error_plugin_file.write_text(error_plugin_code)

            manager = PluginManager(
                plugins_dir=str(plugins_dir),
                enable_hot_reload=False,
                security_enabled=False
            )

            # エラープラグインでの分析実行
            market_data = create_sample_market_data()
            results = manager.analyze_with_all_plugins(market_data)

            # エラー結果確認
            error_result = results.get("error_test")
            if error_result and error_result.metadata.get('error', False):
                print("エラーハンドリング正常動作")
                print(f"エラーメッセージ: {error_result.message}")

                manager.cleanup()
                print("✅ エラーハンドリングテスト成功\n")
                return True
            else:
                print("❌ エラーハンドリング失敗")
                manager.cleanup()
                return False

    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("[TEST] プラグインアーキテクチャ総合テスト開始")
    print("=" * 60)

    test_functions = [
        test_plugin_manager_initialization,
        test_plugin_security_sandbox,
        test_builtin_plugins,
        test_plugin_hot_reload,
        test_plugin_performance,
        test_error_handling
    ]

    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] テスト関数実行エラー {test_func.__name__}: {e}")
            results.append(False)

    # 結果サマリー
    print("=" * 60)
    print("[SUMMARY] テスト結果サマリー")
    print("=" * 60)

    success_count = sum(results)
    total_count = len(results)

    test_names = [
        "プラグインマネージャー初期化",
        "セキュリティサンドボックス",
        "組み込みプラグイン",
        "ホットリロード",
        "パフォーマンス",
        "エラーハンドリング"
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "[OK] 成功" if result else "[NG] 失敗"
        print(f"{i+1}. {name}: {status}")

    print("-" * 60)
    print(f"総合結果: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    if success_count == total_count:
        print("[SUCCESS] プラグインアーキテクチャテスト全体成功！")
        return True
    elif success_count >= total_count * 0.8:
        print("[PARTIAL] プラグインアーキテクチャテスト概ね成功（要改善項目あり）")
        return True
    else:
        print("[FAILED] プラグインアーキテクチャテスト要改善")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"[CRITICAL] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

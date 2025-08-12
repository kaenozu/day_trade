#!/usr/bin/env python3
"""
プラグインアーキテクチャ簡易テスト
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

# プロジェクトパス設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import pandas as pd

    from src.day_trade.plugins import PluginManager
    from src.day_trade.plugins.interfaces import MarketData, PortfolioData

    def test_basic_functionality():
        """基本機能テスト"""
        print("=== 基本機能テスト開始 ===")

        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"

            # プラグインマネージャー初期化
            manager = PluginManager(
                plugins_dir=str(plugins_dir),
                enable_hot_reload=False,
                security_enabled=False,  # テスト用に無効化
            )

            print("プラグイン読み込み完了")

            # プラグインリスト確認
            plugins = manager.list_plugins()
            print(f"プラグイン数: {len(plugins)}")

            for plugin in plugins:
                print(f"- {plugin.name} ({plugin.plugin_type.value})")

            # テストデータ作成
            price_data = pd.DataFrame({"Close": [1000, 1010, 1005, 1020, 1015]})

            market_data = MarketData(
                symbol="TEST",
                timestamp=datetime.now(),
                price_data=price_data,
                volume_data=price_data,
                news_data=[
                    {
                        "title": "テストニュース",
                        "content": "テスト内容",
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
            )

            portfolio_data = PortfolioData(
                positions={"TEST": {"quantity": 100}},
                total_value=100000,
                available_cash=50000,
                allocation={"TEST": 1.0},
                performance_metrics={},
            )

            # 分析実行
            results = manager.analyze_with_all_plugins(
                market_data=market_data, portfolio_data=portfolio_data
            )

            print(f"分析結果: {len(results)}件")

            for name, result in results.items():
                print(f"- {name}: {result.risk_level.value} (スコア: {result.risk_score:.2f})")

            manager.cleanup()

            print("=== 基本機能テスト完了 ===")
            return len(results) > 0

    if __name__ == "__main__":
        success = test_basic_functionality()
        print(f"\n結果: {'成功' if success else '失敗'}")
        sys.exit(0 if success else 1)

except Exception as e:
    print(f"テストエラー: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

#!/usr/bin/env python3
"""
全自動最適化機能の簡単なテスト
"""

import sys
from pathlib import Path

# プロジェクトのrootディレクトリをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from day_trade.automation.auto_optimizer import AutoOptimizer
    from day_trade.config.config_manager import ConfigManager

    print(">> 全自動最適化機能テスト開始")
    print("=" * 50)

    # 設定テスト
    try:
        config_manager = ConfigManager()
        auto_settings = config_manager.get_auto_optimizer_settings()
        print("<< 設定読み込み成功")
        print(f"   - 有効: {auto_settings.enabled}")
        print(f"   - デフォルト銘柄数: {auto_settings.default_max_symbols}")
        print(f"   - 最適化深度: {auto_settings.default_optimization_depth}")
    except Exception as e:
        print(f"!! 設定読み込みエラー（フォールバック設定使用）: {e}")

    # AutoOptimizer初期化テスト
    try:
        optimizer = AutoOptimizer()
        print("<< AutoOptimizer初期化成功")

        # デフォルト銘柄ユニバース確認
        print(
            f"   - デフォルト銘柄ユニバース: {len(optimizer.default_symbol_universe)}銘柄"
        )
        print(f"   - 先頭3銘柄: {optimizer.default_symbol_universe[:3]}")

        # スクリーナー可用性確認
        print(f"   - スクリーナー利用可能: {optimizer.screener_available}")

    except Exception as e:
        print(f"XX AutoOptimizer初期化失敗: {e}")
        sys.exit(1)

    # 簡単な最適化テスト（高速モード）
    try:
        print("\n>> 高速最適化テスト実行中...")
        result = optimizer.run_auto_optimization(
            max_symbols=3, optimization_depth="fast", show_progress=False
        )

        print("<< 高速最適化テスト成功")
        print(f"   - 選択銘柄: {result.best_symbols}")
        print(f"   - 推奨戦略: {result.best_strategy}")
        print(f"   - 期待リターン: {result.expected_return:.2%}")
        print(f"   - 信頼度: {result.confidence:.1%}")
        print(f"   - 最適化時間: {result.optimization_time:.1f}秒")

    except Exception as e:
        print(f"XX 高速最適化テスト失敗: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print(">> テスト完了")
    print("\n|| 使用方法:")
    print("   python -m day_trade.cli.main auto --help")
    print("   python -m day_trade.cli.main auto --symbols 5 --depth balanced")

except ImportError as e:
    print(f"XX インポートエラー: {e}")
    print("必要なモジュールがインストールされていません。")
    sys.exit(1)
except Exception as e:
    print(f"XX 予期しないエラー: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

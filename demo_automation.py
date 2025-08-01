#!/usr/bin/env python3
"""
全自動化機能のデモンストレーション
daytrade.pyの動作確認とテスト
"""
import sys
import logging
from pathlib import Path
from datetime import datetime
import traceback

from src.day_trade.automation.orchestrator import DayTradeOrchestrator  # Moved to top
from src.day_trade.config.config_manager import ConfigManager  # Moved to top

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_demo_logging():
    """デモ用ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def demo_config_manager():
    """設定管理機能のデモ"""
    print("=" * 60)
    print("[設定] 設定管理システムのデモ")
    print("=" * 60)

    try:
        config_manager = ConfigManager()

        print(f"[ファイル] 設定ファイル: {config_manager.config_path}")

        # 監視銘柄情報
        symbols = config_manager.get_watchlist_symbols()
        print(f"\n[銘柄] 監視銘柄 ({len(symbols)}銘柄):")
        for symbol in symbols:
            print(
                f"  {symbol.code}: {symbol.name} ({symbol.group}) [優先度: {symbol.priority}]"
            )

        # 高優先度銘柄
        high_priority = config_manager.get_high_priority_symbols()
        print(f"\n[高優先度] 高優先度銘柄: {high_priority}")

        # 市場時間
        market_hours = config_manager.get_market_hours()
        print("\n[時間] 市場営業時間:")
        print(f"  開始: {market_hours.start}")
        print(f"  終了: {market_hours.end}")
        print(f"  昼休み: {market_hours.lunch_start} - {market_hours.lunch_end}")
        print(f"  現在オープン中: {config_manager.is_market_open()}")

        # 各種設定
        tech_settings = config_manager.get_technical_indicator_settings()
        print("\n[分析] テクニカル分析:")
        print(f"  有効: {tech_settings.enabled}")
        print(f"  SMA期間: {tech_settings.sma_periods}")
        print(f"  RSI期間: {tech_settings.rsi_period}")

        alert_settings = config_manager.get_alert_settings()
        print("\n[アラート] アラート設定:")
        print(f"  有効: {alert_settings.enabled}")
        print(f"  通知方法: {alert_settings.notification_methods}")

        report_settings = config_manager.get_report_settings()
        print("\n[レポート] レポート設定:")
        print(f"  有効: {report_settings.enabled}")
        print(f"  出力形式: {report_settings.formats}")
        print(f"  出力先: {report_settings.output_directory}")

        print("[OK] 設定管理システムのテスト完了")

    except Exception as e:
        print(f"[NG] 設定管理システムエラー: {e}")
        traceback.print_exc()


def demo_orchestrator():
    """オーケストレーターのデモ"""
    print("\n" + "=" * 60)
    print(" 全自動化オーケストレーターのデモ")
    print("=" * 60)

    try:
        # テスト用の少数銘柄
        test_symbols = ["7203", "8306"]  # トヨタ、三菱UFJ

        print(f"[対象] テスト対象: {test_symbols}")
        print("[設定] 小規模テスト実行中...")

        # オーケストレーター初期化
        orchestrator = DayTradeOrchestrator()

        # 実行時間測定
        start_time = datetime.now()

        # 自動化実行
        report = orchestrator.run_full_automation(symbols=test_symbols)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 結果表示
        print("\n[レポート] 実行結果:")
        print(f"  実行時間: {execution_time:.2f}秒")
        print(f"  対象銘柄: {report.total_symbols}")
        print(f"  成功銘柄: {report.successful_symbols}")
        print(f"  失敗銘柄: {report.failed_symbols}")
        print(f"  生成シグナル: {len(report.generated_signals)}")
        print(f"  発生アラート: {len(report.triggered_alerts)}")
        print(f"  エラー数: {len(report.errors)}")

        # 詳細結果
        if report.execution_results:
            print("\n[銘柄] 銘柄別実行結果:")
            for result in report.execution_results:
                status = "[OK]" if result.success else "[NG]"
                print(f"  {status} {result.symbol}: {result.execution_time:.2f}秒")
                if result.error:
                    print(f"    エラー: {result.error}")

        # 生成されたシグナル
        if report.generated_signals:
            print("\n[対象] 生成シグナル:")
            for signal in report.generated_signals:
                print(
                    f"  {signal['symbol']}: {signal['type']} - {signal['reason']} (信頼度: {signal['confidence']:.2f})"
                )

        # 発生したアラート
        if report.triggered_alerts:
            print("\n[アラート] 発生アラート:")
            for alert in report.triggered_alerts:
                print(f"  {alert['symbol']}: {alert['type']} - {alert['message']}")

        # ポートフォリオ情報
        if report.portfolio_summary:
            print("\n[ポートフォリオ] ポートフォリオ:")
            if "metrics" in report.portfolio_summary:
                metrics = report.portfolio_summary["metrics"]
                print(f"  総資産: {metrics.get('total_value', 'N/A')}円")
                print(f"  総損益: {metrics.get('total_pnl', 'N/A')}円")

        # エラー情報
        if report.errors:
            print("\n[警告]  発生エラー:")
            for error in report.errors:
                print(f"  - {error}")

        print("[OK] オーケストレーターのテスト完了")

    except Exception as e:
        print(f"[NG] オーケストレーターエラー: {e}")
        traceback.print_exc()


def demo_command_line():
    """コマンドライン実行のデモ"""
    print("\n" + "=" * 60)
    print("[コマンド] コマンドライン実行のデモ")
    print("=" * 60)

    print("以下のコマンドでdaytrade.pyを実行できます:")
    print()

    # 基本的な使用例
    examples = [
        {"command": "python daytrade.py", "description": "デフォルト設定で全自動実行"},
        {
            "command": "python daytrade.py --symbols 7203,8306,9984",
            "description": "特定銘柄のみ分析",
        },
        {
            "command": "python daytrade.py --report-only",
            "description": "レポート生成のみ実行",
        },
        {
            "command": "python daytrade.py --config config/custom.json",
            "description": "カスタム設定ファイルを使用",
        },
        {
            "command": "python daytrade.py --log-level DEBUG",
            "description": "デバッグログ出力",
        },
        {
            "command": "python daytrade.py --no-banner --symbols 7203",
            "description": "バナー非表示で特定銘柄のみ",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   $ {example['command']}")
        print()

    print("[設定] オプション:")
    print("  --symbols       対象銘柄をカンマ区切りで指定")
    print("  --config        設定ファイルのパス")
    print("  --report-only   レポート生成のみ")
    print("  --log-level     ログレベル (DEBUG/INFO/WARNING/ERROR)")
    print("  --no-banner     バナー表示を無効")
    print("  --version       バージョン表示")
    print("  --help          ヘルプ表示")


def demo_report_generation():
    """レポート生成のデモ"""
    print("\n" + "=" * 60)
    print("[レポート] レポート生成機能のデモ")
    print("=" * 60)

    try:
        # reportsディレクトリの確認
        reports_dir = Path("reports")

        print(f"[ファイル] レポート出力先: {reports_dir.absolute()}")

        if reports_dir.exists():
            report_files = list(reports_dir.glob("*"))
            if report_files:
                print(f"\n[ファイル] 既存レポートファイル ({len(report_files)}件):")
                for report_file in sorted(report_files)[-5:]:  # 最新5件
                    print(f"  {report_file.name}")
                if len(report_files) > 5:
                    print(f"  ... 他{len(report_files) - 5}件")
            else:
                print("\n[ファイル] 既存レポートファイル: なし")
        else:
            print(
                "\n[ファイル] reportsディレクトリは未作成（初回実行時に作成されます）"
            )

        print("\n[レポート] レポート形式:")
        print("  [OK] JSON形式 - 機械読み取り用の詳細データ")
        print("  [OK] CSV形式 - Excel等での分析用")
        print("  [OK] HTML形式 - ブラウザで見やすい表示")

        print("\n[銘柄] レポート内容:")
        print("  • 実行サマリー（成功/失敗銘柄数、実行時間）")
        print("  • 生成シグナル一覧（BUY/SELL推奨）")
        print("  • アラート発生履歴")
        print("  • ポートフォリオ状況")
        print("  • エラー詳細")

    except Exception as e:
        print(f"[NG] レポートデモエラー: {e}")


def main():
    """メインデモ関数"""
    setup_demo_logging()

    print(
        """
    ========================================================
                  DayTrade自動化システム
                    デモンストレーション
    ========================================================
    """
    )

    print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}")

    try:
        # 1. 設定管理システムのデモ
        demo_config_manager()

        # 2. オーケストレーターのデモ
        demo_orchestrator()

        # 3. コマンドライン実行のデモ
        demo_command_line()

        # 4. レポート生成のデモ
        demo_report_generation()

        print("\n" + "=" * 60)
        print("[完了] 全自動化システムのデモンストレーション完了！")
        print("=" * 60)
        print("\n実際にdaytrade.pyを実行するには:")
        print("$ python daytrade.py --symbols 7203,8306")
        print("\nまたは設定ファイルの銘柄で全自動実行:")
        print("$ python daytrade.py")

    except Exception as e:
        print(f"\n[NG] デモ実行エラー: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
DayTrade全自動化メインスクリプト
ワンクリックでデイトレードの全工程を自動実行
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import traceback

from src.day_trade.automation.orchestrator import DayTradeOrchestrator  # Moved to top
from src.day_trade.config.config_manager import ConfigManager  # Moved to top

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(log_level: str = "INFO"):
    """ログ設定をセットアップ"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f'daytrade_{datetime.now().strftime("%Y%m%d")}.log', encoding="utf-8"
            ),
        ],
    )


def print_banner():
    """バナーを表示"""
    banner = """
    ========================================
          DayTrade Auto Engine
        全自動デイトレードシステム
    ========================================
    """
    print(banner)


def print_summary(report):
    """実行サマリーを表示"""
    execution_time = (report.end_time - report.start_time).total_seconds()

    print("\n" + "=" * 50)
    print("[実行サマリー]")
    print("=" * 50)
    print(f"実行時間: {execution_time:.2f}秒")
    print(f"対象銘柄: {report.total_symbols}銘柄")
    print(f"成功: {report.successful_symbols}銘柄")
    print(f"失敗: {report.failed_symbols}銘柄")
    print(f"生成シグナル: {len(report.generated_signals)}個")
    print(f"発生アラート: {len(report.triggered_alerts)}個")

    if report.generated_signals:
        print("\n[シグナル] 生成されたシグナル:")
        for i, signal in enumerate(report.generated_signals[:5], 1):  # 上位5件
            print(
                f"  {i}. {signal['symbol']} - {signal['type']} ({signal['reason']}) [信頼度: {signal['confidence']:.2f}]"
            )

        if len(report.generated_signals) > 5:
            print(f"  ... 他{len(report.generated_signals) - 5}件")

    if report.triggered_alerts:
        print("\n[アラート] 発生したアラート:")
        for i, alert in enumerate(report.triggered_alerts[:3], 1):  # 上位3件
            print(f"  {i}. {alert['symbol']} - {alert['type']} ({alert['message']})")

        if len(report.triggered_alerts) > 3:
            print(f"  ... 他{len(report.triggered_alerts) - 3}件")

    if report.portfolio_summary and "metrics" in report.portfolio_summary:
        metrics = report.portfolio_summary["metrics"]
        print("\n[ポートフォリオ] ポートフォリオ:")
        print(f"  総資産: {metrics.get('total_value', 'N/A')}円")
        print(
            f"  総損益: {metrics.get('total_pnl', 'N/A')}円 ({metrics.get('total_pnl_percent', 'N/A')}%)"
        )

    if report.errors:
        print(f"\n[エラー]  エラー ({len(report.errors)}件):")
        for i, error in enumerate(report.errors[:3], 1):
            print(f"  {i}. {error}")
        if len(report.errors) > 3:
            print(f"  ... 他{len(report.errors) - 3}件")

    print("=" * 50)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="DayTrade全自動化システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python daytrade.py                          # 全自動実行
  python daytrade.py --symbols 7203,8306     # 特定銘柄のみ
  python daytrade.py --report-only            # レポート生成のみ
  python daytrade.py --config custom.json    # カスタム設定ファイル
  python daytrade.py --log-level DEBUG       # デバッグモード
        """,
    )

    parser.add_argument(
        "--symbols", type=str, help="対象銘柄をカンマ区切りで指定 (例: 7203,8306,9984)"
    )

    parser.add_argument("--config", type=str, help="設定ファイルのパスを指定")

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="データ取得・分析をスキップし、レポート生成のみ実行",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="ログレベルを指定 (デフォルト: INFO)",
    )

    parser.add_argument(
        "--no-banner", action="store_true", help="バナー表示を無効にする"
    )

    parser.add_argument(
        "--version", action="version", version="DayTrade Auto Engine v1.0.0"
    )

    args = parser.parse_args()

    # ログ設定
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # バナー表示
        if not args.no_banner:
            print_banner()

        # 引数処理
        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",")]
            logger.info(f"指定銘柄: {symbols}")

        # 設定ファイル確認
        config_path = args.config
        if config_path and not Path(config_path).exists():
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return 1

        # 設定の表示
        try:
            config_manager = ConfigManager(config_path)
            if not symbols:
                symbols = config_manager.get_symbol_codes()

            print("[設定] 設定情報:")
            print(f"   設定ファイル: {config_manager.config_path}")
            print(f"   対象銘柄数: {len(symbols)}")
            print(f"   銘柄コード: {', '.join(symbols)}")
            print(f"   レポートのみ: {'はい' if args.report_only else 'いいえ'}")

            # 市場時間チェック
            if config_manager.is_market_open():
                print("   [オープン] 市場オープン中")
            else:
                print("   [クローズ] 市場クローズ中")

        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            return 1

        # 実行確認
        if not args.report_only:
            print(f"\n {len(symbols)}銘柄の自動分析を開始します...")
        else:
            print("\n[レポート] レポート生成を開始します...")

        # オーケストレーター初期化・実行
        orchestrator = DayTradeOrchestrator(config_path)

        start_time = datetime.now()
        print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

        # メイン処理実行
        report = orchestrator.run_full_automation(
            symbols=symbols, report_only=args.report_only
        )

        # 結果表示
        print_summary(report)

        # 成功/失敗判定
        if report.failed_symbols == 0 and not report.errors:
            print("\n[完了] 全自動化処理が正常に完了しました！")
            return 0
        elif report.successful_symbols > 0:
            print(
                f"\n[エラー]  部分的に成功しました ({report.successful_symbols}/{report.total_symbols})"
            )
            return 0
        else:
            print("\n[失敗] 処理に失敗しました")
            return 1

    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
        print("\n\n[中断]  処理が中断されました")
        return 130

    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[失敗] エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

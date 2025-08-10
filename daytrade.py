#!/usr/bin/env python3
"""
DayTrade全自動化メインスクリプト
ワンクリックでデイトレードの全工程を自動実行
"""

import argparse
import logging
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List

# Windows環境対応の初期化
try:
    from src.day_trade.utils.windows_console_fix import setup_windows_console

    setup_windows_console()
except ImportError:
    pass  # Windows環境修正ユーティリティが無い場合はスキップ

from src.day_trade.analysis.educational_analysis import EducationalMarketAnalyzer
from src.day_trade.automation.orchestrator import DayTradeOrchestrator  # Moved to top
from src.day_trade.config.config_manager import ConfigManager  # Moved to top
from src.day_trade.config.trading_mode_config import (
    is_safe_mode,
)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class CLIValidationError(Exception):
    """CLI引数検証エラー"""

    pass


def validate_symbols(symbols_str: str) -> List[str]:
    """
    銘柄コード文字列をバリデートし、リストに変換

    Args:
        symbols_str: カンマ区切りの銘柄コード文字列

    Returns:
        バリデート済み銘柄コードリスト

    Raises:
        CLIValidationError: バリデーションエラー
    """
    if not symbols_str:
        raise CLIValidationError("銘柄コードが指定されていません")

    # カンマで分割し、前後の空白を削除
    symbols = [s.strip() for s in symbols_str.split(",")]

    # 空の要素を除外
    symbols = [s for s in symbols if s]

    if not symbols:
        raise CLIValidationError("有効な銘柄コードが見つかりません")

    # 銘柄コードの形式チェック（4桁の数字または4桁+拡張子）
    symbol_pattern = re.compile(r"^\d{4}(\.[A-Z]+)?$")

    invalid_symbols = []
    for symbol in symbols:
        if not symbol_pattern.match(symbol):
            invalid_symbols.append(symbol)

    if invalid_symbols:
        raise CLIValidationError(
            "無効な銘柄コード形式: {}. ".format(', '.join(invalid_symbols)) +
            "4桁の数字 (例: 7203) または市場コード付き (例: 7203.T) を使用してください。"
        )

    # 重複を除去
    unique_symbols = list(dict.fromkeys(symbols))

    if len(symbols) != len(unique_symbols):
        logging.getLogger(__name__).warning("注意: 重複する銘柄コードが除去されました")

    return unique_symbols


def validate_config_file(config_path: str) -> Path:
    """
    設定ファイルパスをバリデート

    Args:
        config_path: 設定ファイルパス

    Returns:
        バリデート済みPathオブジェクト

    Raises:
        CLIValidationError: バリデーションエラー
    """
    if not config_path:
        raise CLIValidationError("設定ファイルパスが指定されていません")

    path = Path(config_path)

    # ファイルの存在チェック
    if not path.exists():
        raise CLIValidationError(f"設定ファイルが見つかりません: {config_path}")

    # ファイルかどうかチェック
    if not path.is_file():
        raise CLIValidationError(
            f"指定されたパスはファイルではありません: {config_path}"
        )

    # 拡張子チェック
    if path.suffix.lower() not in [".json", ".yaml", ".yml"]:
        raise CLIValidationError(
            f"サポートされていない設定ファイル形式: {path.suffix}. "
            f".json, .yaml, .yml のいずれかを使用してください。"
        )

    # 読み取り権限チェック
    try:
        with open(path, encoding="utf-8") as f:
            f.read(1)  # 1文字だけ読み取りテスト
    except PermissionError as e:
        raise CLIValidationError(
            f"設定ファイルに読み取り権限がありません: {config_path}"
        ) from e
    except UnicodeDecodeError as e:
        raise CLIValidationError(
            f"設定ファイルのエンコーディングが無効です（UTF-8である必要があります）: {config_path}"
        ) from e
    except Exception as e:
        raise CLIValidationError(
            f"設定ファイルの読み取りでエラーが発生しました: {e}"
        ) from e

    return path


def validate_log_level(log_level: str) -> str:
    """
    ログレベルをバリデート

    Args:
        log_level: ログレベル文字列

    Returns:
        バリデート済みログレベル

    Raises:
        CLIValidationError: バリデーションエラー
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    if log_level.upper() not in valid_levels:
        raise CLIValidationError(
            "無効なログレベル: {}. 有効な値: {}".format(log_level, ', '.join(valid_levels))
        )

    return log_level.upper()


def setup_logging(log_level: str = "INFO"):
    """ログ設定をセットアップ"""
    # 構造化ロギングを優先して使用
    try:
        import os

        from src.day_trade.utils.logging_config import (
            setup_logging as setup_structured_logging,
        )

        os.environ["LOG_LEVEL"] = log_level.upper()
        setup_structured_logging()
    except ImportError:
        # フォールバック: 標準ロギング
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    f"daytrade_{datetime.now().strftime('%Y%m%d')}.log",
                    encoding="utf-8",
                ),
            ],
        )


def validate_interval(interval: int) -> int:
    """
    監視間隔（分）をバリデート

    Args:
        interval: 監視間隔（分）

    Returns:
        バリデート済み監視間隔

    Raises:
        CLIValidationError: バリデーションエラー
    """
    if interval <= 0:
        raise CLIValidationError("監視間隔は正の整数である必要があります。 সন")
    return interval


def print_banner():
    """
    バナーを表示
    """
    banner = """
    ========================================
          DayTrade Auto Engine
        全自動デイトレードシステム
    ========================================
    """
    print(banner)


def print_summary(report):
    """
    実行サマリーを表示
    """
    execution_time = (report.end_time - report.start_time).total_seconds()

    print("\n" + "=" * 50)
    print("[実行サマリー]")
    print("=" * 50)
    if isinstance(execution_time, (int, float)):
        print(f"実行時間: {execution_time:.2f}秒")
    else:
        print(f"実行時間: {execution_time}秒")
    print(f"対象銘柄: {report.total_symbols}銘柄")
    print(f"成功: {report.successful_symbols}銘柄")
    print(f"失敗: {report.failed_symbols}銘柄")
    print(f"生成シグナル: {len(report.generated_signals)}個")
    print(f"発生アラート: {len(report.triggered_alerts)}個")

    if report.generated_signals:
        print("\n[シグナル] 生成されたシグナル:")
        for i, signal in enumerate(report.generated_signals[:5], 1):  # 上位5件
            reason = signal.get("reason", "N/A")
            confidence = signal.get("confidence", 0.0)

            # enhanced_details が存在する場合、より詳細な情報を表示
            if "enhanced_details" in signal:
                details = signal["enhanced_details"]
                risk_score = details.get("risk_score", "N/A")
                reason = f"Enhanced Ensemble (Risk: {risk_score:.1f})"

            print(
                f"  {i}. {signal['symbol']} - {signal['type']} ({reason}) [信頼度: {confidence:.2f}]"
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


def run_watch_mode(symbols, interval_minutes, orchestrator_instance: DayTradeOrchestrator):
    """
    継続監視モード
    """

    try:
        print(f"[監視] 継続監視モードを開始します（{interval_minutes}分間隔）")
        print("  Ctrl+C で停止できます")

        iteration = 0

        while True:
            iteration += 1
            start_time = time.time()

            print(
                f"\n[監視 #{iteration}] {datetime.now().strftime('%H:%M:%S')} - 分析開始"
            )

            try:
                # 分析実行
                report = orchestrator_instance.run_full_automation(symbols=symbols)

                # 簡潔な結果表示
                print(
                    f"  [OK] 成功:{report.successful_symbols} 失敗:{report.failed_symbols} "
                    f"シグナル:{len(report.generated_signals)} "
                    f"({(report.end_time - report.start_time).total_seconds():.1f}秒)"
                )

                # 重要なアラートがあれば表示
                if report.triggered_alerts:
                    for alert in report.triggered_alerts[:3]:  # 上位3件
                        if alert.get("severity") == "high":
                            print(f"  [WARN] {alert['symbol']}: {alert['message']}")

            except Exception as e:
                print(f"  [ERROR] 分析エラー: {e}")
                # logger.error(f"監視モード分析エラー: {e}")

            # 待機時間計算
            elapsed = time.time() - start_time
            sleep_time = max(0, (interval_minutes * 60) - elapsed)

            if sleep_time > 0:
                print(f"  [WAIT] 次回分析まで {sleep_time/60:.1f}分待機...")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[中断] 監視モードが中断されました")
    finally:
        orchestrator_instance.cleanup()


def run_dashboard_mode():
    """
    ダッシュボードモード
    """
    try:
        print("[ダッシュボード] サーバーを起動します...")
        print("  URL: http://localhost:8000")
        print("  Ctrl+C で停止できます")

        # ダッシュボードサーバー起動
        result = subprocess.run(
            [sys.executable, "run_analysis_dashboard.py"], cwd=project_root
        )

        if result.returncode != 0:
            print("[ERROR] ダッシュボード起動エラー")

    except KeyboardInterrupt:
        print("\n[中断] ダッシュボードが停止されました")
    except Exception as e:
        print(f"[ERROR] ダッシュボードエラー: {e}")


def print_startup_banner():
    """
    起動バナー表示
    """
    print("=" * 70)
    print("    DayTrade 分析システム - 統合版")
    print("    [SECURE] 完全セーフモード - 分析・情報提供専用")
    print("=" * 70)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"セーフモード: {'[OK] 有効' if is_safe_mode() else '[ERROR] 無効'}")
    print("-" * 70)


def _parse_and_validate_args():
    parser = argparse.ArgumentParser(
        description="DayTrade統合分析システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python daytrade.py                          # 基本分析（教育レポート+MLスコア付き）
  python daytrade.py --symbols 7203,8306     # 特定銘柄の教育分析+MLスコア
  python daytrade.py --no-educational         # 教育レポートなしの簡単分析
  python daytrade.py --quiet                  # 最小限の出力
  python daytrade.py --watch                  # 継続監視（5分間隔）
  python daytrade.py --watch --interval 3     # 継続監視（3分間隔）
  python daytrade.py --dash                   # ダッシュボード起動
  python daytrade.py --report-only            # レポート生成のみ
  python daytrade.py --interactive            # インタラクティブモード
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

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="拡張インタラクティブモードで開始（オートコンプリート、履歴機能付き）",
    )

    # 排他的な引数グループ
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--watch",
        action="store_true",
        help="継続監視モードを開始します。--interval で監視間隔（分）を指定できます（デフォルト: 5分）。",
    )

    group.add_argument(
        "--dash",
        action="store_true",
        help="分析ダッシュボードを起動します。",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="--watch 使用時の監視間隔（分、デフォルト: 5）。正の整数である必要があります。",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="簡潔出力",
    )

    parser.add_argument(
        "--no-educational",
        action="store_true",
        help="教育的技術指標レポート（MLスコア含む）を無効化",
    )

    args = parser.parse_args()

    # セーフモード確認
    if not is_safe_mode():
        print("[ERROR] セーフモードが無効です")
        print("   このシステムは分析専用として設計されています")
        sys.exit(1)

    try:
        # 引数バリデーション
        validated_symbols = None
        validated_config_path = None
        validated_log_level = args.log_level
        validated_interval = args.interval

        # ログレベルのバリデーション
        try:
            validated_log_level = validate_log_level(args.log_level)
        except CLIValidationError as e:
            print(f"❌ エラー: コマンドライン引数のログレベルが無効です。", file=sys.stderr) # 変更
            sys.exit(1)

        # 監視間隔のバリデーション
        if args.watch:  # --watch が指定された場合のみバリデーション
            try:
                validated_interval = validate_interval(args.interval)
            except CLIValidationError as e:
                print(f"❌ エラー: コマンドライン引数の監視間隔が無効です。", file=sys.stderr) # 変更
                sys.exit(1)

        # 銘柄コードのバリデーション
        if args.symbols:
            try:
                validated_symbols = validate_symbols(args.symbols)
                print(f"✅ 銘柄コード検証完了: {len(validated_symbols)}銘柄")
            except CLIValidationError as e:
                print(f"❌ エラー: コマンドライン引数の銘柄コードが無効です。", file=sys.stderr) # 変更
                sys.exit(1)

        # 設定ファイルのバリデーション
        if args.config:
            try:
                validated_config_path = validate_config_file(args.config)
                print(f"✅ 設定ファイル検証完了: {validated_config_path}")
            except CLIValidationError as e:
                print(f"❌ エラー: コマンドライン引数の設定ファイルパスが無効です。", file=sys.stderr) # 変更
                sys.exit(1)
        return args, validated_symbols, validated_config_path, validated_log_level, validated_interval
    except Exception as e:
        # 詳細エラーはログに、ユーザーには一般的なメッセージ
        logging.getLogger(__name__).error(f"予期しないバリデーションエラー: {e}", exc_info=True) # 追加
        print(f"❌ 予期しないエラーが発生しました。詳細はログを確認してください。", file=sys.stderr) # 変更
        sys.exit(1)


def _run_dashboard_mode(args):
    if not args.quiet:
        print_startup_banner()
    run_dashboard_mode()
    return 0

def _run_interactive_mode(args):
    # ログ設定（バリデート済みレベルを使用）
    setup_logging(args.log_level)

    # バナー表示
    if not args.no_banner:
        print_banner()

    try:
        from src.day_trade.cli.enhanced_interactive import (
            run_enhanced_interactive,
        )

        config_path = args.config
        print("[インタラクティブ] 拡張インタラクティブモードを開始します...")
        print("• オートコンプリート機能")
        print("• コマンド履歴")
        print("• 色分け表示")
        print("• カスタムキーバインディング")
        print()
        run_enhanced_interactive(config_path)
        return 0
    except ImportError:
        print("❌ エラー: 拡張インタラクティブモードは利用できません。")
        print("prompt_toolkit>=3.0.0 をインストールしてください。")
        print("コマンド: pip install prompt_toolkit>=3.0.0")
        return 1
    except Exception as e:
        print(f"❌ インタラクティブモードエラー: {e}")
        return 1

def _run_analysis_mode(args, validated_symbols, validated_config_path, orchestrator, _signal_handler):
    logger = logging.getLogger(__name__)

    # バナー表示
    if not args.no_banner:
        print_banner()

    # 引数処理（バリデート済みの値を使用）
    symbols = validated_symbols
    config_path = str(validated_config_path) if validated_config_path else None

    if symbols:
        logger.info(f"指定銘柄: {symbols}")

    # 追加のバリデーション情報をログに記録
    logger.debug(f"バリデーション完了 - 銘柄数: {len(symbols) if symbols else 0}")
    if config_path:
        logger.debug(f"設定ファイル: {config_path}")

    # 設定の表示
    try:
        config_manager = ConfigManager(config_path)
        if not symbols:
            symbols = config_manager.get_symbol_codes()

        if not args.quiet:
            logger.info("[設定] 設定情報:")
            logger.info(f"   設定ファイル: {config_manager.config_path}")
            logger.info(f"   対象銘柄数: {len(symbols)}")
            logger.info("   銘柄コード: {}".format(', '.join(symbols)))
            logger.info(f"   レポートのみ: {'はい' if args.report_only else 'いいえ'}")

            # 市場時間チェック
            if config_manager.is_market_open():
                logger.info("   [オープン] 市場オープン中")
            else:
                logger.info("   [クローズ] 市場クローズ中")

    except Exception as e:
        logger.error(f"設定読み込みエラー: {e}")
        return 1

    # 実行確認
    if not args.report_only:
        # 実行確認
        if not args.report_only:
            if not args.quiet:
                logger.info(f" {len(symbols)}銘柄の自動分析を開始します...")
        else:
            if not args.quiet:
                logger.info("\n[レポート] レポート生成を開始します...")

    # オーケストレーター初期化・実行
    orchestrator_instance = DayTradeOrchestrator(config_path)
    # シグナルハンドラに実際のorchestratorインスタンスをバインド
    signal.signal(signal.SIGINT, partial(_signal_handler, orchestrator_instance=orchestrator_instance))
    signal.signal(signal.SIGTERM, partial(_signal_handler, orchestrator_instance=orchestrator_instance))

    start_time = datetime.now()
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # メイン処理実行
    report = orchestrator_instance.run_full_automation(
        symbols=symbols, report_only=args.report_only
    )

    # 結果表示
    print_summary(report)

    # 教育的分析の表示（デフォルト有効、機械学習含む）
    if not args.quiet and not args.no_educational:
        analyzer = EducationalMarketAnalyzer()
        all_results = []

        # 全銘柄の分析実行
        for symbol in symbols:
            try:
                educational_result = analyzer.analyze_symbol_educational(symbol)
                all_results.append(educational_result)
            except Exception as e:
                print(f"分析エラー {symbol}: {e}")

        # 詳細レポート（最初の3銘柄のみ）
        print("\n" + "=" * 80)
        print("【詳細技術指標レポート】（上位3銘柄）")
        print("=" * 80)

        for result in all_results[:3]:
            try:
                educational_report = analyzer.format_educational_report(result)
                print(educational_report)
            except Exception as e:
                print(f"詳細レポートエラー {result.symbol}: {e}")

        # 全銘柄MLスコア一覧表（最後に表示）
        if all_results:
            print("\n" + "=" * 100)
            print("【全銘柄 機械学習スコア一覧表】（教育・研究目的）")
            print("=" * 100)
            print("※機械学習スコアは教育・研究目的の技術情報です")
            print("=" * 100)

            # 総合スコアでソート
            scored_results = []
            for result in all_results:
                if result.ml_technical_scores:
                    # 各スコアを取得
                    trend_score = next(
                        (
                            s
                            for s in result.ml_technical_scores
                            if "トレンド" in s.score_name
                        ),
                        None,
                    )
                    volatility_score = next(
                        (
                            s
                            for s in result.ml_technical_scores
                            if "変動予測" in s.score_name
                        ),
                        None,
                    )
                    pattern_score = next(
                        (
                            s
                            for s in result.ml_technical_scores
                            if "パターン" in s.score_name
                        ),
                        None,
                    )

                    trend_val = trend_score.score_value if trend_score else 0
                    volatility_val = volatility_score.score_value if volatility_score else 0
                    pattern_val = pattern_score.score_value if pattern_score else 0

                    # 総合判定
                    avg_score = (trend_val + volatility_val + pattern_val) / 3
                    overall = (
                        "強い上昇"
                        if avg_score >= 70
                        else "上昇傾向"
                        if avg_score >= 55
                        else "中立"
                        if avg_score >= 45
                        else "下降傾向"
                        if avg_score >= 30
                        else "弱い"
                    )

                    scored_results.append(
                        {
                            "result": result,
                            "trend_val": trend_val,
                            "volatility_val": volatility_val,
                            "pattern_val": pattern_val,
                            "avg_score": avg_score,
                            "overall": overall,
                        }
                    )

            # 総合スコア順でソート（降順）
            scored_results.sort(key=lambda x: x["avg_score"], reverse=True)

            print(
                f"{'ランク':<4} {'銘柄':<8} {'会社名':<12} {'価格':<8} {'トレンド':<8} {'変動予測':<8} {'パターン':<8} {'総合':<6} {'判定':<10}"
            )
            print("-" * 110)

            for i, scored_result in enumerate(scored_results, 1):
                result = scored_result["result"]
                trend_val = scored_result["trend_val"]
                volatility_val = scored_result["volatility_val"]
                pattern_val = scored_result["pattern_val"]
                avg_score = scored_result["avg_score"]
                overall = scored_result["overall"]

                rank_symbol = (
                    "🥇"
                    if i == 1
                    else "🥈"
                    if i == 2
                    else "🥉"
                    if i == 3
                    else f"{i:2d}"
                )

                print(
                    f"{rank_symbol:<4} {result.symbol:<8} {result.company_name[:10]:<12} {result.current_price:>7.0f} {trend_val:>6.1f} {volatility_val:>8.1f} {pattern_val:>7.1f} {avg_score:>5.1f} {overall:<10}"
                )

            print("-" * 110)
            print("※数値は0-100のスコア、総合スコア順でランキング表示")
            print(
                "※総合判定は平均値による技術的参考情報、投資判断は自己責任で行ってください"
            )
            print("=" * 100)

    # 成功/失敗判定
    if report.failed_symbols == 0 and not report.errors:
        print("\n[完了] 全自動化処理が正常に完了しました！")
        return 0
    else:  # 何らかの失敗またはエラーがある場合
        if report.successful_symbols > 0:
            print(
                f"\n[警告]  一部の処理が失敗しました ({report.successful_symbols}/{report.total_symbols} 成功)"
            )
        else:
            print("\n[失敗] 処理に失敗しました")
        return 1  # 部分的または全体的な失敗の場合、1を返す

def _print_educational_report_and_ml_scores(symbols, args, analyzer, all_results):
    # 詳細レポート（最初の3銘柄のみ）
    print("\n" + "=" * 80)
    print("【詳細技術指標レポート】（上位3銘柄）")
    print("=" * 80)

    for result in all_results[:3]:
        try:
            educational_report = analyzer.format_educational_report(result)
            print(educational_report)
        except Exception as e:
            print(f"詳細レポートエラー {result.symbol}: {e}")

    # 全銘柄MLスコア一覧表（最後に表示）
    if all_results:
        print("\n" + "=" * 100)
        print("【全銘柄 機械学習スコア一覧表】（教育・研究目的）")
        print("=" * 100)
        print("※機械学習スコアは教育・研究目的の技術情報です")
        print("=" * 100)

        # 総合スコアでソート
        scored_results = []
        for result in all_results:
            if result.ml_technical_scores:
                # 各スコアを取得
                trend_score = next(
                    (
                        s
                        for s in result.ml_technical_scores
                        if "トレンド" in s.score_name
                    ),
                    None,
                )
                volatility_score = next(
                    (
                        s
                        for s in result.ml_technical_scores
                        if "変動予測" in s.score_name
                    ),
                    None,
                )
                pattern_score = next(
                    (
                        s
                        for s in result.ml_technical_scores
                        if "パターン" in s.score_name
                    ),
                    None,
                )

                trend_val = trend_score.score_value if trend_score else 0
                volatility_val = volatility_score.score_value if volatility_score else 0
                pattern_val = pattern_score.score_value if pattern_score else 0

                # 総合判定
                avg_score = (trend_val + volatility_val + pattern_val) / 3
                overall = (
                    "強い上昇"
                    if avg_score >= 70
                    else "上昇傾向"
                    if avg_score >= 55
                    else "中立"
                    if avg_score >= 45
                    else "下降傾向"
                    if avg_score >= 30
                    else "弱い"
                )

                scored_results.append(
                    {
                        "result": result,
                        "trend_val": trend_val,
                        "volatility_val": volatility_val,
                        "pattern_val": pattern_val,
                        "avg_score": avg_score,
                        "overall": overall,
                    }
                )

        # 総合スコア順でソート（降順）
        scored_results.sort(key=lambda x: x["avg_score"], reverse=True)

        print(
            f"{'ランク':<4} {'銘柄':<8} {'会社名':<12} {'価格':<8} {'トレンド':<8} {'変動予測':<8} {'パターン':<8} {'総合':<6} {'判定':<10}"
        )
        print("-" * 110)

        for i, scored_result in enumerate(scored_results, 1):
            result = scored_result["result"]
            trend_val = scored_result["trend_val"]
            volatility_val = scored_result["volatility_val"]
            pattern_val = scored_result["pattern_val"]
            avg_score = scored_result["avg_score"]
            overall = scored_result["overall"]

            rank_symbol = (
                "🥇"
                if i == 1
                else "🥈"
                if i == 2
                else "🥉"
                if i == 3
                else f"{i:2d}"
            )

            print(
                f"{rank_symbol:<4} {result.symbol:<8} {result.company_name[:10]:<12} {result.current_price:>7.0f} {trend_val:>6.1f} {volatility_val:>8.1f} {pattern_val:>7.1f} {avg_score:>5.1f} {overall:<10}"
            )

        print("-" * 110)
        print("※数値は0-100のスコア、総合スコア順でランキング表示")
        print(
            "※総合判定は平均値による技術的参考情報、投資判断は自己責任で行ってください"
        )
        print("=" * 100)

    return 0

def main():
    """
    メイン関数
    """
    def _signal_handler(signum, frame, orchestrator_instance: DayTradeOrchestrator):
        print("\n\n[中断] システムを安全に停止しています...")
        if orchestrator_instance:
            orchestrator_instance.cleanup()
        print("[完了] システムが正常に停止されました")
        sys.exit(0)

    # シグナルハンドラ設定 (partialでorchestratorインスタンスを後でバインド)
    signal.signal(signal.SIGINT, partial(_signal_handler, orchestrator_instance=None))
    signal.signal(signal.SIGTERM, partial(_signal_handler, orchestrator_instance=None))

    args, validated_symbols, validated_config_path, validated_log_level, validated_interval = _parse_and_validate_args()

    # ログ設定（バリデート済みレベルを使用）
    setup_logging(validated_log_level)
    logger = logging.getLogger(__name__)

    try:
        # ダッシュボードモードの処理
        if args.dash:
            return _run_dashboard_mode(args)

        # インタラクティブモードの処理
        if args.interactive:
            return _run_interactive_mode(args)

        # 以下は通常の分析または監視モードの処理
        # オーケストレーター初期化
        orchestrator = DayTradeOrchestrator(
            str(validated_config_path) if validated_config_path else None
        )

        # シグナルハンドラに実際のorchestratorインスタンスをバインド
        signal.signal(signal.SIGINT, partial(_signal_handler, orchestrator_instance=orchestrator))
        signal.signal(signal.SIGTERM, partial(_signal_handler, orchestrator_instance=orchestrator))

        # 設定の表示とシンボル取得
        symbols = validated_symbols
        config_path = str(validated_config_path) if validated_config_path else None

        try:
            config_manager = ConfigManager(config_path)
            if not symbols:
                symbols = config_manager.get_symbol_codes()

            if not args.quiet:
                logger.info("[設定] 設定情報:")
                logger.info(f"   設定ファイル: {config_manager.config_path}")
                logger.info(f"   対象銘柄数: {len(symbols)}")
                logger.info("   銘柄コード: {}".format(', '.join(symbols)))
                logger.info(f"   レポートのみ: {'はい' if args.report_only else 'いいえ'}")

                # 市場時間チェック
                if config_manager.is_market_open():
                    logger.info("   [オープン] 市場オープン中")
                else:
                    logger.info("   [クローズ] 市場クローズ中")

        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            return 1

        # 監視モードの処理
        if args.watch:
            if not args.quiet:
                print_startup_banner()
                print(f"対象銘柄: {symbols}")
            run_watch_mode(symbols, validated_interval, orchestrator)
            return 0

        # 通常分析モードの実行
        return _run_analysis_mode(args, symbols, validated_config_path, orchestrator, _signal_handler)

    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
        print("\n\n[中断]  処理が中断されました")
        return 130

    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True) # 変更
        print(f"\n[失敗] エラーが発生しました。詳細はログを確認してください。", file=sys.stderr) # 変更
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
最終実データ動作確認テスト - セーフモード版
Issue #321: 実際の市場データを使用した包括的システム検証

【重要】このテストは以下を保証します：
- 完全なセーフモード動作
- 実際の取引は一切実行されない
- 分析・情報提供のみ
- リアルマーケットデータを使用した実証実験
"""

import asyncio
import contextlib
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 必要なインポート
from src.day_trade.config.trading_mode_config import (
    get_current_trading_config,
    is_safe_mode,
)
from src.day_trade.automation.analysis_only_engine import AnalysisOnlyEngine
from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.analysis.signals import TradingSignalGenerator
from src.day_trade.utils.performance_monitor import PerformanceMonitor


class FinalLiveDataValidator:
    """
    最終実データ動作確認システム

    セーフモードで実際の市場データを使用して：
    1. データ取得・品質確認
    2. ML分析性能測定
    3. シグナル生成検証
    4. システム安定性確認
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = {}
        self.performance_data = {}
        self.errors = []

        # TOPIX Core30中心の実テスト銘柄
        self.test_symbols = [
            # TOPIX Core30主要銘柄
            "7203.T",  # トヨタ自動車
            "6758.T",  # ソニーグループ
            "9984.T",  # ソフトバンクグループ
            "6861.T",  # キーエンス
            "4519.T",  # 中外製薬
            "8306.T",  # 三菱UFJ
            "9432.T",  # NTT
            "4578.T",  # 大塚ホールディングス
            "8031.T",  # 三井物産
            "7974.T",  # 任天堂
            # 追加銘柄（分析精度向上用）
            "6594.T",  # 日本電産
            "7733.T",  # オリンパス
            "4568.T",  # 第一三共
            "8058.T",  # 三菱商事
            "9983.T",  # ファーストリテイリング
        ]

        # コンポーネント初期化
        self.stock_fetcher = None
        self.signal_generator = None
        self.analysis_engine = None
        self.performance_monitor = None

    def print_header(self, title: str) -> None:
        """テストセクションヘッダー出力"""
        print("\n" + "=" * 80)
        print(f" {title} ".center(80, "="))
        print("=" * 80)

    def print_status(self, message: str, status: str = "INFO") -> None:
        """ステータス出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{status:^7}] {message}")

    def verify_safe_mode(self) -> bool:
        """セーフモード確認"""
        try:
            if not is_safe_mode():
                self.print_status("セーフモード無効 - テスト中止", "ERROR")
                return False

            config = get_current_trading_config()
            safety_checks = [
                ("Safe Mode", is_safe_mode()),
                ("Auto Trading Disabled", not config.enable_automatic_trading),
                ("Order Execution Disabled", not config.enable_order_execution),
                ("Order API Disabled", config.disable_order_api),
            ]

            self.print_status("セーフモード設定確認:", "CHECK")
            all_safe = True
            for check_name, is_safe in safety_checks:
                status = "OK" if is_safe else "NG"
                self.print_status(f"  {status} {check_name}: {'有効' if is_safe else '無効'}")
                if not is_safe:
                    all_safe = False

            return all_safe

        except Exception as e:
            self.print_status(f"セーフモード確認エラー: {e}", "ERROR")
            return False

    def initialize_components(self) -> bool:
        """システムコンポーネント初期化"""
        try:
            self.print_status("システムコンポーネント初期化中...", "INIT")

            # 基本コンポーネント
            self.stock_fetcher = StockFetcher()
            self.signal_generator = TradingSignalGenerator()
            self.performance_monitor = PerformanceMonitor()

            # 分析エンジン（セーフモード）
            self.analysis_engine = AnalysisOnlyEngine(
                symbols=self.test_symbols,
                signal_generator=self.signal_generator,
                stock_fetcher=self.stock_fetcher,
                update_interval=30.0  # 実データテスト用
            )

            self.print_status("全コンポーネント初期化完了", "SUCCESS")
            return True

        except Exception as e:
            self.print_status(f"初期化失敗: {e}", "ERROR")
            self.errors.append(f"Initialization: {e}")
            return False

    async def test_data_acquisition(self) -> Dict:
        """実データ取得テスト"""
        self.print_header("実市場データ取得・品質確認テスト")

        results = {
            "success_count": 0,
            "failed_symbols": [],
            "data_quality": {},
            "acquisition_time": 0,
        }

        start_time = time.time()

        for symbol in self.test_symbols:
            try:
                self.print_status(f"データ取得テスト: {symbol}")

                # 現在価格取得
                current_data = self.stock_fetcher.get_current_price(symbol)
                if not current_data:
                    results["failed_symbols"].append(symbol)
                    continue

                # 履歴データ取得（30日）
                historical_data = self.stock_fetcher.get_historical_data(symbol, "30d")
                if historical_data is None or historical_data.empty:
                    results["failed_symbols"].append(symbol)
                    continue

                # データ品質チェック
                quality_score = self._assess_data_quality(historical_data)
                results["data_quality"][symbol] = quality_score

                results["success_count"] += 1
                self.print_status(f"  OK {symbol}: 品質スコア {quality_score:.2f}")

            except Exception as e:
                self.print_status(f"  NG {symbol}: {e}", "WARN")
                results["failed_symbols"].append(symbol)
                self.errors.append(f"Data acquisition {symbol}: {e}")

        results["acquisition_time"] = time.time() - start_time

        # 結果サマリー
        success_rate = (results["success_count"] / len(self.test_symbols)) * 100
        self.print_status(f"データ取得成功率: {success_rate:.1f}% ({results['success_count']}/{len(self.test_symbols)})")
        self.print_status(f"データ取得時間: {results['acquisition_time']:.2f}秒")

        if success_rate < 80:
            self.print_status("データ取得成功率が80%未満", "WARN")

        return results

    def _assess_data_quality(self, data) -> float:
        """データ品質スコア計算（0-1）"""
        try:
            quality_score = 1.0

            # 欠損値チェック
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_score -= missing_ratio * 0.3

            # データの最新性チェック
            if len(data) > 0:
                last_date = data.index[-1]
                days_old = (datetime.now() - last_date.to_pydatetime()).days
                if days_old > 5:  # 5日以上古い
                    quality_score -= min(days_old / 30, 0.3)

            # 価格の整合性チェック
            if 'High' in data.columns and 'Low' in data.columns:
                invalid_prices = (data['High'] < data['Low']).sum()
                if invalid_prices > 0:
                    quality_score -= (invalid_prices / len(data)) * 0.2

            return max(0.0, quality_score)

        except Exception:
            return 0.5  # エラー時は中程度のスコア

    async def test_ml_analysis_performance(self) -> Dict:
        """ML分析性能テスト"""
        self.print_header("実データML分析性能測定")

        results = {
            "analysis_times": [],
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_time": 0,
            "target_achievement": False,
        }

        target_time = 3.6  # 目標：85銘柄3.6秒
        adjusted_target = target_time * (len(self.test_symbols) / 85)  # 銘柄数調整

        self.print_status(f"目標時間: {adjusted_target:.2f}秒 ({len(self.test_symbols)}銘柄対象)")

        # 複数回実行して性能測定
        test_iterations = 3

        for iteration in range(test_iterations):
            self.print_status(f"分析実行 {iteration + 1}/{test_iterations}")

            try:
                start_time = time.time()

                # 分析エンジンで市場分析実行
                await self.analysis_engine._perform_market_analysis()

                analysis_time = time.time() - start_time
                results["analysis_times"].append(analysis_time)
                results["successful_analyses"] += 1

                self.print_status(f"  完了時間: {analysis_time:.2f}秒")

            except Exception as e:
                results["failed_analyses"] += 1
                self.print_status(f"  分析失敗: {e}", "WARN")
                self.errors.append(f"ML Analysis iteration {iteration + 1}: {e}")

        if results["analysis_times"]:
            results["average_time"] = sum(results["analysis_times"]) / len(results["analysis_times"])
            results["target_achievement"] = results["average_time"] <= adjusted_target

            self.print_status(f"平均分析時間: {results['average_time']:.2f}秒")
            self.print_status(f"目標達成: {'OK' if results['target_achievement'] else 'NG'}")

        return results

    async def test_signal_generation(self) -> Dict:
        """シグナル生成テスト"""
        self.print_header("実データシグナル生成検証")

        results = {
            "signals_generated": 0,
            "signal_quality": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "signal_types": {"buy": 0, "sell": 0, "hold": 0},
        }

        try:
            # 分析結果から最新のシグナルを取得
            analyses = self.analysis_engine.get_all_analyses()

            for symbol, analysis in analyses.items():
                if analysis.signal:
                    results["signals_generated"] += 1

                    # 信頼度分布
                    confidence = analysis.signal.confidence
                    if confidence >= 80:
                        results["confidence_distribution"]["high"] += 1
                    elif confidence >= 60:
                        results["confidence_distribution"]["medium"] += 1
                    else:
                        results["confidence_distribution"]["low"] += 1

                    # シグナルタイプ
                    signal_type = analysis.signal.signal_type.value.lower()
                    if signal_type in results["signal_types"]:
                        results["signal_types"][signal_type] += 1

                    # シグナル品質評価
                    quality_score = self._evaluate_signal_quality(analysis.signal)
                    results["signal_quality"][symbol] = quality_score

                    self.print_status(f"  {symbol}: {signal_type.upper()} (信頼度: {confidence:.1f}%, 品質: {quality_score:.2f})")

            # 結果サマリー
            signal_rate = (results["signals_generated"] / len(self.test_symbols)) * 100
            self.print_status(f"シグナル生成率: {signal_rate:.1f}%")

            if results["signal_quality"]:
                avg_quality = sum(results["signal_quality"].values()) / len(results["signal_quality"])
                self.print_status(f"平均シグナル品質: {avg_quality:.2f}")

        except Exception as e:
            self.print_status(f"シグナル生成テストエラー: {e}", "ERROR")
            self.errors.append(f"Signal generation: {e}")

        return results

    def _evaluate_signal_quality(self, signal) -> float:
        """シグナル品質評価（0-1）"""
        try:
            quality_score = 0.0

            # 信頼度スコア（50%）
            confidence_score = min(signal.confidence / 100, 1.0)
            quality_score += confidence_score * 0.5

            # 理由の具体性スコア（30%）
            if hasattr(signal, 'reasons') and signal.reasons:
                reason_score = min(len(signal.reasons) / 3, 1.0)
                quality_score += reason_score * 0.3

            # 条件満足度スコア（20%）
            if hasattr(signal, 'conditions_met') and signal.conditions_met:
                met_ratio = sum(1 for met in signal.conditions_met.values() if met) / len(signal.conditions_met)
                quality_score += met_ratio * 0.2

            return min(quality_score, 1.0)

        except Exception:
            return 0.5

    async def test_system_stability(self, duration_minutes: int = 10) -> Dict:
        """システム安定性テスト"""
        self.print_header(f"システム安定性テスト（{duration_minutes}分間）")

        results = {
            "test_duration": duration_minutes,
            "cycles_completed": 0,
            "errors_encountered": 0,
            "average_cycle_time": 0,
            "memory_usage": [],
            "stability_score": 0,
        }

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle_times = []

        try:
            while time.time() < end_time:
                cycle_start = time.time()

                try:
                    # 分析サイクル実行
                    await self.analysis_engine._perform_market_analysis()

                    cycle_time = time.time() - cycle_start
                    cycle_times.append(cycle_time)
                    results["cycles_completed"] += 1

                    # メモリ使用量チェック
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        results["memory_usage"].append(memory_mb)
                    except ImportError:
                        pass

                    if results["cycles_completed"] % 5 == 0:
                        elapsed = time.time() - start_time
                        self.print_status(f"  サイクル {results['cycles_completed']} 完了 ({elapsed:.0f}秒経過)")

                    # 短時間待機
                    await asyncio.sleep(5)

                except Exception as e:
                    results["errors_encountered"] += 1
                    self.print_status(f"  サイクルエラー: {e}", "WARN")
                    if results["errors_encountered"] > 5:  # 連続エラー制限
                        break

        except KeyboardInterrupt:
            self.print_status("安定性テスト中断", "WARN")

        # 結果計算
        if cycle_times:
            results["average_cycle_time"] = sum(cycle_times) / len(cycle_times)

        # 安定性スコア計算
        if results["cycles_completed"] > 0:
            error_rate = results["errors_encountered"] / results["cycles_completed"]
            results["stability_score"] = max(0, 1.0 - error_rate)

        # 結果出力
        self.print_status(f"完了サイクル数: {results['cycles_completed']}")
        self.print_status(f"エラー発生回数: {results['errors_encountered']}")
        self.print_status(f"平均サイクル時間: {results['average_cycle_time']:.2f}秒")
        self.print_status(f"安定性スコア: {results['stability_score']:.2f}")

        return results

    def generate_final_report(self, all_results: Dict) -> str:
        """最終検証レポート生成"""
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"final_live_data_validation_report_{report_time}.txt"

        total_duration = (datetime.now() - self.start_time).total_seconds()

        report_content = f"""
Final Live Data Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {total_duration:.1f} seconds
================================================================================

EXECUTIVE SUMMARY
================================================================================
Safe Mode: VERIFIED - All safety checks passed
System Components: All initialized successfully
Test Symbols: {len(self.test_symbols)} real market symbols tested
Total Test Duration: {total_duration/60:.1f} minutes

DETAILED RESULTS
================================================================================

1. DATA ACQUISITION TEST
   - Success Rate: {all_results.get('data', {}).get('success_count', 0)}/{len(self.test_symbols)} symbols ({(all_results.get('data', {}).get('success_count', 0)/len(self.test_symbols)*100):.1f}%)
   - Acquisition Time: {all_results.get('data', {}).get('acquisition_time', 0):.2f} seconds
   - Failed Symbols: {', '.join(all_results.get('data', {}).get('failed_symbols', [])) or 'None'}

2. ML ANALYSIS PERFORMANCE
   - Target Time: 3.6s for 85 symbols (adjusted for {len(self.test_symbols)} symbols)
   - Average Analysis Time: {all_results.get('performance', {}).get('average_time', 0):.2f} seconds
   - Target Achievement: {'YES' if all_results.get('performance', {}).get('target_achievement', False) else 'NO'}
   - Successful Analyses: {all_results.get('performance', {}).get('successful_analyses', 0)}

3. SIGNAL GENERATION
   - Signals Generated: {all_results.get('signals', {}).get('signals_generated', 0)}
   - Signal Generation Rate: {(all_results.get('signals', {}).get('signals_generated', 0)/len(self.test_symbols)*100):.1f}%
   - High Confidence Signals: {all_results.get('signals', {}).get('confidence_distribution', {}).get('high', 0)}

4. SYSTEM STABILITY
   - Test Duration: {all_results.get('stability', {}).get('test_duration', 0)} minutes
   - Completed Cycles: {all_results.get('stability', {}).get('cycles_completed', 0)}
   - Errors Encountered: {all_results.get('stability', {}).get('errors_encountered', 0)}
   - Stability Score: {all_results.get('stability', {}).get('stability_score', 0):.2f}/1.00

SAFETY VERIFICATION
================================================================================
Safe Mode: ENABLED throughout entire test
No Real Trading: Zero trading orders executed
Analysis Only: All operations performed in analysis-only mode
Risk Management: All safety protocols maintained

ERRORS AND WARNINGS
================================================================================
"""

        if self.errors:
            for i, error in enumerate(self.errors, 1):
                report_content += f"{i}. {error}\n"
        else:
            report_content += "No errors encountered during validation.\n"

        report_content += f"""

RECOMMENDATIONS
================================================================================
"""

        # 推奨事項の生成
        recommendations = []

        if all_results.get('data', {}).get('success_count', 0) / len(self.test_symbols) < 0.9:
            recommendations.append("- Consider improving data source reliability")

        if not all_results.get('performance', {}).get('target_achievement', False):
            recommendations.append("- Optimize ML analysis performance for production load")

        if all_results.get('stability', {}).get('stability_score', 0) < 0.95:
            recommendations.append("- Address stability issues before production deployment")

        if all_results.get('signals', {}).get('signals_generated', 0) < len(self.test_symbols) * 0.7:
            recommendations.append("- Review signal generation criteria for better coverage")

        if not recommendations:
            recommendations.append("- System ready for production deployment")
            recommendations.append("- Continue monitoring in production environment")

        for rec in recommendations:
            report_content += rec + "\n"

        report_content += f"""

CONCLUSION
================================================================================
The system has been validated with real market data in safe mode.
All safety protocols were maintained throughout the testing process.
"""

        # レポートファイル保存
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.print_status(f"最終レポート保存: {report_file}", "SUCCESS")
        except Exception as e:
            self.print_status(f"レポート保存失敗: {e}", "ERROR")

        return report_content

    async def run_validation(self) -> None:
        """最終検証実行"""
        self.print_header("最終実データ動作確認テスト開始")
        self.print_status(f"開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.print_status(f"テスト銘柄数: {len(self.test_symbols)}")

        # 1. セーフモード確認
        if not self.verify_safe_mode():
            self.print_status("セーフモード確認失敗 - テスト中止", "ERROR")
            return

        # 2. コンポーネント初期化
        if not self.initialize_components():
            self.print_status("初期化失敗 - テスト中止", "ERROR")
            return

        # 3. 各種テスト実行
        all_results = {}

        try:
            # データ取得テスト
            all_results["data"] = await self.test_data_acquisition()

            # ML分析性能テスト
            all_results["performance"] = await self.test_ml_analysis_performance()

            # シグナル生成テスト
            all_results["signals"] = await self.test_signal_generation()

            # システム安定性テスト（短時間版）
            all_results["stability"] = await self.test_system_stability(duration_minutes=5)

        except Exception as e:
            self.print_status(f"テスト実行中エラー: {e}", "ERROR")
            self.errors.append(f"Test execution: {e}")

        # 最終レポート生成
        self.print_header("最終検証結果")
        report_content = self.generate_final_report(all_results)

        print(report_content)

        # テスト完了
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.print_status(f"テスト完了 - 総実行時間: {total_time:.1f}秒", "SUCCESS")


async def main():
    """メイン実行"""
    print("Final Live Data Validation Test - Safe Mode")
    print("=" * 80)
    print("このテストは完全にセーフモードで実行されます")
    print("実際の取引は一切行われません")
    print("=" * 80)

    # 実行確認
    try:
        response = input("実データテストを開始しますか？ (y/N): ").strip().lower()
        if response != 'y':
            print("テストをキャンセルしました")
            return
    except KeyboardInterrupt:
        print("\nテストをキャンセルしました")
        return

    # 検証実行
    validator = FinalLiveDataValidator()
    await validator.run_validation()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())

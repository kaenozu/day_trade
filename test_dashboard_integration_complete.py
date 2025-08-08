#!/usr/bin/env python3
"""
Issue #319: 分析ダッシュボード強化 - 完全統合テスト
全ての新機能（インタラクティブチャート、カスタムレポート、教育システム）の統合テスト
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

from src.day_trade.dashboard.interactive_charts import InteractiveChartManager
from src.day_trade.dashboard.custom_reports import CustomReportManager
from src.day_trade.dashboard.educational_system import EducationalSystem


class DashboardIntegrationTester:
    """ダッシュボード統合テスト実行クラス"""

    def __init__(self):
        """テスト環境初期化"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dashboard_test_"))
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }
        print(f"テスト環境: {self.temp_dir}")

    def cleanup(self):
        """テスト環境クリーンアップ"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"テスト環境をクリーンアップしました: {self.temp_dir}")
        except Exception as e:
            print(f"クリーンアップエラー: {e}")

    def log_result(self, test_name: str, passed: bool, error: str = None):
        """テスト結果記録"""
        if passed:
            self.results["tests_passed"] += 1
            print(f"[OK] {test_name}")
        else:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"{test_name}: {error}")
            print(f"[NG] {test_name}: {error}")

    def create_sample_data(self) -> dict:
        """サンプルデータ作成"""
        # 価格データ
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        symbols = ['7203.T', '8306.T', '9984.T', '6758.T', '6861.T']

        price_data = {}
        for symbol in symbols:
            base_price = 2000 + hash(symbol) % 1000
            prices = base_price + np.cumsum(np.random.normal(0, 20, len(dates)))

            price_data[symbol] = pd.DataFrame({
                'Open': prices + np.random.normal(0, 10, len(dates)),
                'High': prices + np.abs(np.random.normal(10, 5, len(dates))),
                'Low': prices - np.abs(np.random.normal(10, 5, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

        # パフォーマンスデータ
        performance_data = {
            f'{symbol} 戦略': np.random.normal(0.001, 0.02, len(dates)).tolist()
            for symbol in symbols
        }

        # ベンチマークデータ
        benchmark_data = {
            'TOPIX': np.random.normal(0.0005, 0.018, len(dates)).tolist()
        }

        # リスクデータ
        risk_data = {}
        for symbol in symbols:
            base_return = 0.08 + (hash(symbol) % 20 - 10) / 100
            base_vol = 0.15 + (hash(symbol) % 10) / 100

            risk_data[f'{symbol} 戦略'] = {
                'return': base_return,
                'volatility': base_vol,
                'sharpe_ratio': base_return / base_vol,
                'max_drawdown': -0.05 - (hash(symbol) % 10) / 100,
                'sortino_ratio': (base_return / base_vol) * 1.2
            }

        # 市場データ
        market_data = {
            "sector_performance": {
                "テクノロジー": 0.15,
                "金融": 0.08,
                "ヘルスケア": 0.12,
                "消費財": 0.06,
                "エネルギー": -0.03
            },
            "volume_trend": {
                "dates": dates.strftime('%Y-%m-%d').tolist()[:30],
                "volumes": np.random.randint(1000000, 5000000, 30).tolist()
            },
            "price_distribution": np.random.normal(2500, 500, 1000).tolist(),
            "correlation_matrix": {
                "symbols": symbols[:3],
                "values": [
                    [1.0, 0.3, 0.1],
                    [0.3, 1.0, 0.6],
                    [0.1, 0.6, 1.0]
                ]
            }
        }

        return {
            'price_data': price_data,
            'performance_data': performance_data,
            'benchmark_data': benchmark_data,
            'risk_data': risk_data,
            'market_data': market_data,
            'dates': dates
        }

    def test_interactive_charts(self, data: dict):
        """インタラクティブチャート機能テスト"""
        print("\n=== インタラクティブチャート機能テスト ===")

        try:
            chart_manager = InteractiveChartManager()
            symbols = list(data['price_data'].keys())[:3]

            # 1. 価格チャート作成テスト
            try:
                price_chart = chart_manager.create_price_chart(
                    data=data['price_data'],
                    symbols=symbols,
                    chart_type="candlestick",
                    show_volume=True,
                    indicators=['sma_20', 'bollinger']
                )

                assert price_chart['type'] == 'price_chart'
                assert 'chart' in price_chart
                assert len(price_chart['chart']) > 1000  # JSONサイズチェック

                self.log_result("価格チャート作成（ローソク足・出来高・指標付き）", True)

            except Exception as e:
                self.log_result("価格チャート作成", False, str(e))

            # 2. パフォーマンスチャート作成テスト
            try:
                perf_chart = chart_manager.create_performance_chart(
                    performance_data=data['performance_data'],
                    dates=data['dates'].to_list(),
                    benchmark=data['benchmark_data'],
                    cumulative=True
                )

                assert perf_chart['type'] == 'performance_chart'
                assert 'chart' in perf_chart

                self.log_result("パフォーマンスチャート作成（ベンチマーク比較）", True)

            except Exception as e:
                self.log_result("パフォーマンスチャート作成", False, str(e))

            # 3. リスク分析チャート作成テスト
            try:
                # 散布図
                risk_scatter = chart_manager.create_risk_analysis_chart(
                    risk_data=data['risk_data'],
                    chart_type="scatter"
                )
                assert risk_scatter['type'] == 'risk_scatter'

                # レーダーチャート
                risk_radar = chart_manager.create_risk_analysis_chart(
                    risk_data=data['risk_data'],
                    chart_type="radar"
                )
                assert risk_radar['type'] == 'risk_radar'

                self.log_result("リスク分析チャート作成（散布図・レーダー）", True)

            except Exception as e:
                self.log_result("リスク分析チャート作成", False, str(e))

            # 4. 市場概要ダッシュボード作成テスト
            try:
                market_dashboard = chart_manager.create_market_overview_dashboard(
                    market_data=data['market_data'],
                    layout="grid"
                )

                assert market_dashboard['type'] == 'market_overview'
                assert 'dashboard' in market_dashboard

                self.log_result("市場概要ダッシュボード作成", True)

            except Exception as e:
                self.log_result("市場概要ダッシュボード作成", False, str(e))

        except Exception as e:
            self.log_result("インタラクティブチャートマネージャー初期化", False, str(e))

    def test_custom_reports(self, data: dict):
        """カスタムレポート機能テスト"""
        print("\n=== カスタムレポート機能テスト ===")

        try:
            report_manager = CustomReportManager(output_dir=str(self.temp_dir / "reports"))

            # サンプルデータ準備
            sample_data = {
                "portfolio_returns": data['performance_data'][list(data['performance_data'].keys())[0]],
                "benchmark_returns": data['benchmark_data']['TOPIX'],
                "symbols": list(data['price_data'].keys())[:3],
                "sector_weights": {
                    "technology": 0.35,
                    "finance": 0.25,
                    "healthcare": 0.20,
                    "consumer": 0.15,
                    "energy": 0.05
                }
            }

            # 1. 標準レポート作成テスト
            try:
                standard_report = report_manager.create_custom_report(
                    data=sample_data,
                    template="standard",
                    title="統合テスト用標準レポート"
                )

                assert standard_report['success'] is True
                assert 'report' in standard_report
                assert len(standard_report['report']['sections']) > 0

                self.log_result("標準レポート作成", True)

                # レポートデータを保存（後続テスト用）
                self.standard_report_data = standard_report['report']

            except Exception as e:
                self.log_result("標準レポート作成", False, str(e))

            # 2. 詳細レポート作成テスト
            try:
                detailed_report = report_manager.create_custom_report(
                    data=sample_data,
                    template="detailed",
                    title="統合テスト用詳細レポート"
                )

                assert detailed_report['success'] is True
                assert len(detailed_report['report']['sections']) >= 5

                self.log_result("詳細レポート作成", True)

            except Exception as e:
                self.log_result("詳細レポート作成", False, str(e))

            # 3. JSONエクスポートテスト
            try:
                if hasattr(self, 'standard_report_data'):
                    json_path = report_manager.export_to_json(
                        self.standard_report_data,
                        "integration_test_report.json"
                    )

                    assert Path(json_path).exists()
                    assert Path(json_path).stat().st_size > 1000

                    self.log_result("JSONエクスポート", True)
                else:
                    self.log_result("JSONエクスポート", False, "標準レポートデータがありません")

            except Exception as e:
                self.log_result("JSONエクスポート", False, str(e))

            # 4. Excelエクスポートテスト
            try:
                if hasattr(self, 'standard_report_data'):
                    excel_path = report_manager.export_to_excel(
                        self.standard_report_data,
                        "integration_test_report.xlsx"
                    )

                    assert Path(excel_path).exists()

                    self.log_result("Excelエクスポート", True)

            except ImportError:
                self.log_result("Excelエクスポート", False, "openpyxlライブラリが必要")

            except Exception as e:
                self.log_result("Excelエクスポート", False, str(e))

            # 5. テンプレート・指標取得テスト
            try:
                templates = report_manager.get_available_templates()
                metrics = report_manager.get_available_metrics()

                assert len(templates) >= 3
                assert len(metrics) >= 3

                self.log_result("テンプレート・指標取得", True)

            except Exception as e:
                self.log_result("テンプレート・指標取得", False, str(e))

        except Exception as e:
            self.log_result("カスタムレポートマネージャー初期化", False, str(e))

    def test_educational_system(self):
        """教育システム機能テスト"""
        print("\n=== 教育システム機能テスト ===")

        try:
            edu_system = EducationalSystem(content_dir=str(self.temp_dir / "educational_content"))

            # 1. 用語検索テスト
            try:
                # 完全一致検索
                term_result = edu_system.get_term_explanation("sharpe_ratio")
                assert term_result is not None
                assert term_result['term'] == 'シャープレシオ'
                assert 'definition' in term_result

                # 部分一致検索
                partial_result = edu_system.get_term_explanation("シャープ")
                assert partial_result is not None
                assert 'partial_matches' in partial_result

                self.log_result("用語検索（完全・部分一致）", True)

            except Exception as e:
                self.log_result("用語検索", False, str(e))

            # 2. カテゴリ検索テスト
            try:
                risk_terms = edu_system.search_glossary("", category="リスク指標")
                assert len(risk_terms) > 0

                # クエリ付き検索
                var_terms = edu_system.search_glossary("VaR")
                assert len(var_terms) > 0

                self.log_result("カテゴリ・クエリ検索", True)

            except Exception as e:
                self.log_result("カテゴリ・クエリ検索", False, str(e))

            # 3. 投資戦略ガイド取得テスト
            try:
                strategy = edu_system.get_strategy_guide("buy_and_hold")
                assert strategy is not None
                assert strategy['name'] == 'バイ・アンド・ホールド戦略'
                assert 'advantages' in strategy
                assert len(strategy['advantages']) > 0

                self.log_result("投資戦略ガイド取得", True)

            except Exception as e:
                self.log_result("投資戦略ガイド取得", False, str(e))

            # 4. 過去事例分析取得テスト
            try:
                case = edu_system.get_case_study("financial_crisis_2008")
                assert case is not None
                assert case['title'] == 'リーマンショック (2008)'
                assert 'lessons' in case
                assert len(case['lessons']) > 0

                self.log_result("過去事例分析取得", True)

            except Exception as e:
                self.log_result("過去事例分析取得", False, str(e))

            # 5. クイズ生成テスト
            try:
                quiz_questions = edu_system.get_quiz_questions(count=5, difficulty="初級")
                assert len(quiz_questions) > 0
                assert len(quiz_questions) <= 5

                # 各問題の構造確認
                for q in quiz_questions:
                    assert 'question' in q
                    assert 'correct_answer' in q
                    assert 'difficulty' in q

                self.log_result("クイズ生成", True)

            except Exception as e:
                self.log_result("クイズ生成", False, str(e))

            # 6. 学習進捗管理テスト
            try:
                user_id = "test_user_integration"

                # 進捗更新
                progress = edu_system.update_learning_progress(
                    user_id=user_id,
                    module="basic_concepts",
                    study_time=90
                )

                assert progress['study_time'] == 90
                assert 'basic_concepts' in progress['completed_modules']

                # 推奨取得
                recommendations = edu_system.get_recommendations(user_id)
                assert len(recommendations) > 0

                self.log_result("学習進捗管理・推奨", True)

            except Exception as e:
                self.log_result("学習進捗管理・推奨", False, str(e))

        except Exception as e:
            self.log_result("教育システム初期化", False, str(e))

    def test_cross_component_integration(self, data: dict):
        """コンポーネント間統合テスト"""
        print("\n=== コンポーネント間統合テスト ===")

        try:
            # 1. チャート→レポート統合テスト
            try:
                chart_manager = InteractiveChartManager()
                report_manager = CustomReportManager(output_dir=str(self.temp_dir / "integration_reports"))

                # チャートデータ生成
                symbols = list(data['price_data'].keys())[:2]
                price_chart = chart_manager.create_price_chart(
                    data=data['price_data'],
                    symbols=symbols,
                    chart_type="line"
                )

                # チャートデータをレポートに含める
                report_data = {
                    "portfolio_returns": data['performance_data'][list(data['performance_data'].keys())[0]],
                    "benchmark_returns": data['benchmark_data']['TOPIX'],
                    "symbols": symbols,
                    "chart_data": price_chart
                }

                custom_report = report_manager.create_custom_report(
                    data=report_data,
                    template="detailed",
                    title="チャート統合レポート"
                )

                assert custom_report['success'] is True
                self.log_result("チャート→レポート統合", True)

            except Exception as e:
                self.log_result("チャート→レポート統合", False, str(e))

            # 2. 教育システム→レポート統合テスト
            try:
                edu_system = EducationalSystem()

                # 用語解説をレポートデータに含める
                sharpe_explanation = edu_system.get_term_explanation("sharpe_ratio")
                var_explanation = edu_system.get_term_explanation("var")

                educational_report_data = {
                    "portfolio_returns": data['performance_data'][list(data['performance_data'].keys())[0]],
                    "educational_content": {
                        "sharpe_ratio": sharpe_explanation,
                        "var": var_explanation
                    }
                }

                educational_report = report_manager.create_custom_report(
                    data=educational_report_data,
                    template="standard",
                    title="教育コンテンツ付きレポート"
                )

                assert educational_report['success'] is True
                self.log_result("教育システム→レポート統合", True)

            except Exception as e:
                self.log_result("教育システム→レポート統合", False, str(e))

            # 3. 全コンポーネント統合テスト
            try:
                # 全システムの機能を組み合わせた包括的テスト

                # チャート生成
                comprehensive_chart = chart_manager.create_market_overview_dashboard(
                    market_data=data['market_data']
                )

                # 教育コンテンツ取得
                strategy_guide = edu_system.get_strategy_guide("momentum")
                case_study = edu_system.get_case_study("covid19_crash_recovery")

                # 統合レポート作成
                comprehensive_data = {
                    "portfolio_returns": data['performance_data'][list(data['performance_data'].keys())[0]],
                    "benchmark_returns": data['benchmark_data']['TOPIX'],
                    "symbols": list(data['price_data'].keys())[:3],
                    "market_dashboard": comprehensive_chart,
                    "strategy_guide": strategy_guide,
                    "case_study": case_study
                }

                comprehensive_report = report_manager.create_custom_report(
                    data=comprehensive_data,
                    template="detailed",
                    title="完全統合ダッシュボードレポート"
                )

                assert comprehensive_report['success'] is True

                # 複数形式でエクスポート
                json_export = report_manager.export_to_json(
                    comprehensive_report['report'],
                    "comprehensive_integration_test.json"
                )

                try:
                    excel_export = report_manager.export_to_excel(
                        comprehensive_report['report'],
                        "comprehensive_integration_test.xlsx"
                    )
                except ImportError:
                    print("   [SKIP] Excelエクスポートをスキップ（openpyxl未インストール）")

                assert Path(json_export).exists()

                self.log_result("全コンポーネント統合テスト", True)

            except Exception as e:
                self.log_result("全コンポーネント統合テスト", False, str(e))

        except Exception as e:
            self.log_result("コンポーネント間統合テスト初期化", False, str(e))

    def run_all_tests(self):
        """全テスト実行"""
        print("Issue #319: 分析ダッシュボード強化 - 完全統合テスト開始")
        print("=" * 80)

        start_time = datetime.now()

        try:
            # サンプルデータ作成
            print("サンプルデータ作成中...")
            data = self.create_sample_data()
            self.log_result("サンプルデータ作成", True)

            # 各コンポーネントのテスト実行
            self.test_interactive_charts(data)
            self.test_custom_reports(data)
            self.test_educational_system()
            self.test_cross_component_integration(data)

            # テスト結果サマリー
            print("\n" + "=" * 80)
            print("テスト結果サマリー")
            print("=" * 80)

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            total_tests = self.results["tests_passed"] + self.results["tests_failed"]
            success_rate = (self.results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0

            print(f"実行時間: {execution_time:.2f}秒")
            print(f"合計テスト数: {total_tests}")
            print(f"成功: {self.results['tests_passed']}")
            print(f"失敗: {self.results['tests_failed']}")
            print(f"成功率: {success_rate:.1f}%")

            if self.results["errors"]:
                print("\n失敗したテスト:")
                for error in self.results["errors"]:
                    print(f"  - {error}")

            if self.results["tests_failed"] == 0:
                print("\n[SUCCESS] 全てのテストが成功しました！")
                print("Issue #319の分析ダッシュボード強化が完全に実装されました。")
                return True
            else:
                print(f"\n[WARNING] {self.results['tests_failed']}個のテストが失敗しました。")
                return False

        except Exception as e:
            print(f"\n[ERROR] テスト実行中に予期しないエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.cleanup()


if __name__ == "__main__":
    tester = DashboardIntegrationTester()
    success = tester.run_all_tests()

    exit_code = 0 if success else 1
    sys.exit(exit_code)

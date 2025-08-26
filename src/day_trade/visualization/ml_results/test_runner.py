#!/usr/bin/env python3
"""
機械学習結果可視化システム - テスト実行
Issue #315: 高度テクニカル指標・ML機能拡張

可視化システムの動作確認・テスト実行機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, MATPLOTLIB_AVAILABLE, PLOTLY_AVAILABLE, logger
from .dashboard_interactive import InteractiveDashboardGenerator
from .dashboard_static import StaticDashboardGenerator
from .lstm_chart import LSTMChartGenerator
from .multiframe_chart import MultiframeChartGenerator
from .report_generator import ReportGenerator
from .volatility_chart import VolatilityChartGenerator


class MLVisualizationTestRunner(BaseMLVisualizer):
    """機械学習可視化システムテスト実行クラス"""

    def __init__(self, output_dir: str = "output/ml_visualizations"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        super().__init__(output_dir)
        
        # 各生成クラスのインスタンス作成
        self.lstm_generator = LSTMChartGenerator(output_dir)
        self.volatility_generator = VolatilityChartGenerator(output_dir)
        self.multiframe_generator = MultiframeChartGenerator(output_dir)
        self.static_dashboard_generator = StaticDashboardGenerator(output_dir)
        self.interactive_dashboard_generator = InteractiveDashboardGenerator(output_dir)
        self.report_generator = ReportGenerator(output_dir)

    def generate_sample_data(self) -> pd.DataFrame:
        """サンプルデータ生成"""
        dates = pd.date_range(start="2023-06-01", end="2024-12-31", freq="D")
        np.random.seed(42)

        base_price = 2800
        prices = [base_price]
        for i in range(1, len(dates)):
            change = np.random.normal(0.0008, 0.025)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))

        return pd.DataFrame(
            {
                "Open": [p * np.random.uniform(0.995, 1.005) for p in prices],
                "High": [p * np.random.uniform(1.000, 1.025) for p in prices],
                "Low": [p * np.random.uniform(0.975, 1.000) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(800000, 12000000, len(dates)),
            },
            index=dates,
        )

    def generate_sample_results(self) -> Dict:
        """サンプル分析結果データ生成"""
        return {
            "lstm": {
                "predicted_prices": [2850, 2880, 2920, 2950, 2970],
                "predicted_returns": [1.8, 1.05, 1.4, 1.0, 0.7],
                "prediction_dates": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                "confidence_score": 78.5,
            },
            "volatility": {
                "current_metrics": {
                    "realized_volatility": 0.22,
                    "vix_like_indicator": 28.5,
                    "volatility_regime": "medium_vol",
                },
                "ensemble_forecast": {
                    "ensemble_volatility": 26.8,
                    "ensemble_confidence": 0.72,
                    "individual_forecasts": {
                        "GARCH": 25.2,
                        "Machine Learning": 28.1,
                        "VIX-like": 27.1,
                    },
                },
                "risk_assessment": {
                    "risk_level": "MEDIUM",
                    "risk_score": 45,
                    "risk_factors": ["中程度のボラティリティ環境", "一部不確実性要因"],
                },
                "investment_implications": {
                    "portfolio_adjustments": ["適度なリスク管理"],
                    "trading_strategies": ["レンジトレーディング機会"],
                    "risk_management": ["標準的なリスク管理手法"],
                },
            },
            "multiframe": {
                "timeframes": {
                    "daily": {
                        "timeframe": "日足",
                        "trend_direction": "uptrend",
                        "trend_strength": 65.2,
                        "technical_indicators": {
                            "rsi": 58.3,
                            "macd": 0.15,
                            "bb_position": 0.67,
                        },
                        "support_level": 2750,
                        "resistance_level": 2950,
                    },
                    "weekly": {
                        "timeframe": "週足",
                        "trend_direction": "uptrend",
                        "trend_strength": 72.8,
                        "technical_indicators": {
                            "rsi": 62.1,
                            "macd": 0.22,
                            "bb_position": 0.71,
                        },
                    },
                },
                "integrated_analysis": {
                    "overall_trend": "uptrend",
                    "trend_confidence": 68.9,
                    "consistency_score": 78.4,
                    "integrated_signal": {
                        "action": "BUY",
                        "strength": "MODERATE",
                        "signal_score": 12.5,
                    },
                    "investment_recommendation": {
                        "recommendation": "BUY",
                        "position_size": "MODERATE",
                        "confidence": 68.9,
                        "reasons": ["複数時間軸で上昇トレンド確認", "技術指標の良好な位置"],
                    },
                },
            },
        }

    def run_comprehensive_test(self, symbol: str = "TEST_STOCK") -> Dict[str, str]:
        """包括的テスト実行"""
        print("=== 機械学習結果可視化システム テスト ===")

        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib未インストールのためテストを実行できません")
            return {"error": "matplotlib未インストール"}

        # サンプルデータ生成
        sample_data = self.generate_sample_data()
        sample_results = self.generate_sample_results()

        results = {}
        
        print(f"サンプルデータ: {len(sample_data)}日分")
        print(f"出力ディレクトリ: {self.output_dir}")

        # 1. LSTM予測チャートテスト
        print("\n1. LSTM予測チャート作成テスト")
        lstm_chart = self.lstm_generator.create_lstm_prediction_chart(
            sample_data, sample_results["lstm"], symbol=symbol
        )
        if lstm_chart:
            print(f"✅ LSTM予測チャート作成完了: {lstm_chart}")
            results["lstm_chart"] = lstm_chart
        else:
            print("❌ LSTM予測チャート作成失敗")

        # 2. ボラティリティ予測チャートテスト
        print("\n2. ボラティリティ予測チャート作成テスト")
        vol_chart = self.volatility_generator.create_volatility_forecast_chart(
            sample_data, sample_results["volatility"], symbol=symbol
        )
        if vol_chart:
            print(f"✅ ボラティリティ予測チャート作成完了: {vol_chart}")
            results["volatility_chart"] = vol_chart
        else:
            print("❌ ボラティリティ予測チャート作成失敗")

        # 3. マルチタイムフレーム分析チャートテスト
        print("\n3. マルチタイムフレーム分析チャート作成テスト")
        mf_chart = self.multiframe_generator.create_multiframe_analysis_chart(
            sample_data, sample_results["multiframe"], symbol=symbol
        )
        if mf_chart:
            print(f"✅ マルチタイムフレーム分析チャート作成完了: {mf_chart}")
            results["multiframe_chart"] = mf_chart
        else:
            print("❌ マルチタイムフレーム分析チャート作成失敗")

        # 4. 総合ダッシュボードテスト
        print("\n4. 総合ダッシュボード作成テスト")
        dashboard = self.static_dashboard_generator.create_comprehensive_dashboard(
            sample_data,
            lstm_results=sample_results["lstm"],
            volatility_results=sample_results["volatility"],
            multiframe_results=sample_results["multiframe"],
            symbol=symbol,
        )
        if dashboard:
            print(f"✅ 総合ダッシュボード作成完了: {dashboard}")
            results["static_dashboard"] = dashboard
        else:
            print("❌ 総合ダッシュボード作成失敗")

        # 5. インタラクティブダッシュボードテスト
        if PLOTLY_AVAILABLE:
            print("\n5. インタラクティブダッシュボード作成テスト")
            interactive_dashboard = self.interactive_dashboard_generator.create_interactive_plotly_dashboard(
                sample_data, sample_results, symbol=symbol
            )
            if interactive_dashboard:
                print(
                    f"✅ インタラクティブダッシュボード作成完了: {interactive_dashboard}"
                )
                results["interactive_dashboard"] = interactive_dashboard
            else:
                print("❌ インタラクティブダッシュボード作成失敗")
        else:
            print(
                "\n5. インタラクティブダッシュボード - スキップ（plotly未インストール）"
            )

        # 6. 分析レポートテスト
        print("\n6. 分析レポート生成テスト")
        report = self.report_generator.generate_analysis_report(symbol, sample_results)
        if report:
            print(f"✅ 分析レポート生成完了: {report}")
            results["analysis_report"] = report

            # レポート内容の一部表示
            with open(report, encoding="utf-8") as f:
                lines = f.readlines()
                print("📄 レポート内容（抜粋）:")
                for line in lines[:15]:  # 最初の15行表示
                    print(f"   {line.rstrip()}")
                if len(lines) > 15:
                    print(f"   ... (残り{len(lines) - 15}行)")
        else:
            print("❌ 分析レポート生成失敗")

        print("\n✅ 機械学習結果可視化システム テスト完了！")
        print(f"📁 生成ファイル保存場所: {self.output_dir}")

        # 生成されたファイル一覧
        output_files = list(self.output_dir.glob("*"))
        if output_files:
            print("📊 生成されたファイル:")
            for file_path in output_files:
                print(f"   - {file_path.name}")

        return results


def main():
    """メイン関数 - テスト実行"""
    try:
        test_runner = MLVisualizationTestRunner()
        results = test_runner.run_comprehensive_test()
        
        if "error" in results:
            print(f"❌ テストエラー: {results['error']}")
        else:
            print(f"\n✅ 成功したテスト数: {len(results)}")
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
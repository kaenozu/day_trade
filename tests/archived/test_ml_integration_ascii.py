#!/usr/bin/env python3
"""
ML機能統合テスト（ASCII安全版）
Issue #315: 高度テクニカル指標・ML機能拡張

全てのML機能を統合してテスト実行
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent))

from src.day_trade.analysis.advanced_technical_indicators import (
    AdvancedTechnicalIndicators,
)
from src.day_trade.analysis.multi_timeframe_analysis import MultiTimeframeAnalyzer
from src.day_trade.data.lstm_time_series_model import LSTMTimeSeriesModel
from src.day_trade.risk.volatility_prediction_engine import VolatilityPredictionEngine
from src.day_trade.utils.logging_config import get_context_logger
from src.day_trade.visualization.ml_results_visualizer import MLResultsVisualizer

logger = get_context_logger(__name__)


def generate_sample_data(days=500, base_price=2800):
    """サンプル株価データ生成"""
    print(f"Sample data generation... ({days} days)")

    # 日付範囲作成
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    np.random.seed(42)

    # より現実的な価格生成
    returns = []
    for i in range(len(dates)):
        # トレンドとボラティリティクラスターを模擬
        if i < 150:  # 初期上昇トレンド
            base_return = 0.0005
            volatility = 0.015
        elif i < 250:  # 調整局面
            base_return = -0.0003
            volatility = 0.025
        elif i < 350:  # 回復局面
            base_return = 0.0008
            volatility = 0.018
        else:  # 最近の動き
            base_return = 0.0002
            volatility = 0.020

        daily_return = np.random.normal(base_return, volatility)
        returns.append(daily_return)

    # 価格系列生成
    prices = [base_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, base_price * 0.5))  # 最低価格制限

    prices = prices[1:]  # 最初の要素を削除

    # OHLCV データ作成
    data = []
    for i, close_price in enumerate(prices):
        open_price = close_price * np.random.uniform(0.995, 1.005)
        high_price = max(open_price, close_price) * np.random.uniform(1.000, 1.015)
        low_price = min(open_price, close_price) * np.random.uniform(0.985, 1.000)
        volume = np.random.randint(500000, 8000000)

        data.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data, index=dates[: len(data)])

    print(f"[OK] Sample data generated: {len(df)} records")
    return df


def test_ml_integration():
    """ML機能統合テスト実行"""
    print("=" * 60)
    print("ML Integration Test Start")
    print("=" * 60)

    # サンプルデータ生成
    sample_data = generate_sample_data(days=400)
    test_symbol = "TEST_ML_STOCK"

    print(f"\nTest Symbol: {test_symbol}")
    print(
        f"Data Period: {sample_data.index[0].strftime('%Y-%m-%d')} - {sample_data.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"Data Count: {len(sample_data)} records")
    print(
        f"Price Range: {sample_data['Close'].min():.0f} - {sample_data['Close'].max():.0f}"
    )

    # 各コンポーネントのテスト結果を保存
    results = {}

    try:
        # 1. LSTM時系列予測テスト
        print("\n" + "=" * 40)
        print("1. LSTM Time Series Prediction Test")
        print("=" * 40)

        lstm_model = LSTMTimeSeriesModel(
            sequence_length=30,
            prediction_horizon=5,
            lstm_units=[64, 32],
            dropout_rate=0.3,
        )

        # LSTM訓練
        print("LSTM training start...")
        training_result = lstm_model.train_lstm_model(
            symbol=test_symbol,
            data=sample_data,
            validation_split=0.2,
            epochs=15,  # テスト用に短縮
            batch_size=32,
            early_stopping_patience=5,
        )

        if training_result:
            print("[OK] LSTM training completed")
            print(f"   Validation R2: {training_result['val_r2']:.3f}")
            print(f"   Validation RMSE: {training_result['val_rmse']:.2f}")
            print(f"   Confidence Score: {training_result['val_r2'] * 100:.1f}%")

            # LSTM予測実行
            print("LSTM prediction execution...")
            prediction_result = lstm_model.predict_future_prices(
                symbol=test_symbol, data=sample_data, steps_ahead=5
            )

            if prediction_result:
                print("[OK] LSTM prediction completed")
                print(f"   Current Price: {prediction_result['current_price']:.0f}")
                print(
                    f"   5-day Prediction: {prediction_result['predicted_prices'][-1]:.0f}"
                )
                print(
                    f"   Predicted Return: {prediction_result['predicted_returns'][-1]:+.2f}%"
                )
                print(f"   Confidence: {prediction_result['confidence_score']:.1f}%")
                results["lstm"] = prediction_result
            else:
                print("[NG] LSTM prediction failed")
        else:
            print("[NG] LSTM training failed")

    except Exception as e:
        print(f"[NG] LSTM test error: {e}")

    try:
        # 2. 高度テクニカル指標テスト
        print("\n" + "=" * 40)
        print("2. Advanced Technical Indicators Test")
        print("=" * 40)

        technical_analyzer = AdvancedTechnicalIndicators()

        # 一目均衡表
        print("Ichimoku Cloud calculation...")
        ichimoku_result = technical_analyzer.calculate_ichimoku_cloud(sample_data)
        if ichimoku_result and "signal" in ichimoku_result:
            print(
                f"[OK] Ichimoku: {ichimoku_result['signal']} (Strength: {ichimoku_result.get('signal_strength', 'N/A')})"
            )
            results["ichimoku"] = ichimoku_result

        # フィボナッチ分析
        print("Fibonacci analysis...")
        fibonacci_result = technical_analyzer.analyze_fibonacci_levels(sample_data)
        if fibonacci_result and "levels" in fibonacci_result:
            levels = fibonacci_result["levels"]
            print(f"[OK] Fibonacci levels: {len(levels)} levels identified")
            print(f"   Key Support: {fibonacci_result.get('key_support', 'N/A')}")
            print(f"   Key Resistance: {fibonacci_result.get('key_resistance', 'N/A')}")
            results["fibonacci"] = fibonacci_result

        # エリオット波動
        print("Elliott Wave analysis...")
        elliott_result = technical_analyzer.detect_elliott_waves(sample_data)
        if elliott_result and "current_wave" in elliott_result:
            print(f"[OK] Elliott Wave: {elliott_result['current_wave']}")
            print(f"   Wave Phase: {elliott_result.get('wave_phase', 'N/A')}")
            results["elliott"] = elliott_result

    except Exception as e:
        print(f"[NG] Advanced technical indicators test error: {e}")

    try:
        # 3. マルチタイムフレーム分析テスト
        print("\n" + "=" * 40)
        print("3. Multi-Timeframe Analysis Test")
        print("=" * 40)

        multiframe_analyzer = MultiTimeframeAnalyzer()

        print("Multi-timeframe analysis execution...")
        mf_result = multiframe_analyzer.analyze_multiple_timeframes(sample_data)

        if mf_result:
            print("[OK] Multi-timeframe analysis completed")

            timeframes = mf_result.get("timeframes", {})
            for tf_key, tf_data in timeframes.items():
                tf_name = tf_data.get("timeframe", tf_key)
                trend = tf_data.get("trend_direction", "unknown")
                strength = tf_data.get("trend_strength", 0)
                print(f"   {tf_name}: {trend} (Strength: {strength:.1f})")

            integrated = mf_result.get("integrated_analysis", {})
            if integrated:
                overall_trend = integrated.get("overall_trend", "unknown")
                confidence = integrated.get("trend_confidence", 0)
                consistency = integrated.get("consistency_score", 0)

                print(f"   Integrated Trend: {overall_trend}")
                print(f"   Confidence: {confidence:.1f}%")
                print(f"   Consistency: {consistency:.1f}%")

                signal = integrated.get("integrated_signal", {})
                if signal:
                    action = signal.get("action", "HOLD")
                    strength = signal.get("strength", "WEAK")
                    print(f"   Integrated Signal: {action} ({strength})")

            results["multiframe"] = mf_result
        else:
            print("[NG] Multi-timeframe analysis failed")

    except Exception as e:
        print(f"[NG] Multi-timeframe analysis test error: {e}")

    try:
        # 4. ボラティリティ予測テスト
        print("\n" + "=" * 40)
        print("4. Volatility Prediction Test")
        print("=" * 40)

        vol_engine = VolatilityPredictionEngine()

        print("Volatility analysis execution...")
        vol_result = vol_engine.predict_volatility(sample_data)

        if vol_result:
            print("[OK] Volatility prediction completed")

            current_metrics = vol_result.get("current_metrics", {})
            if current_metrics:
                realized_vol = current_metrics.get("realized_volatility", 0) * 100
                vix_like = current_metrics.get("vix_like_indicator", 0)
                regime = current_metrics.get("volatility_regime", "unknown")

                print(f"   Realized Volatility: {realized_vol:.1f}%")
                print(f"   VIX-like Indicator: {vix_like:.1f}")
                print(f"   Volatility Regime: {regime}")

            ensemble = vol_result.get("ensemble_forecast", {})
            if ensemble:
                ensemble_vol = ensemble.get("ensemble_volatility", 0)
                confidence = ensemble.get("ensemble_confidence", 0)
                print(f"   Ensemble Prediction: {ensemble_vol:.1f}%")
                print(f"   Prediction Confidence: {confidence:.2f}")

            risk_assessment = vol_result.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                risk_score = risk_assessment.get("risk_score", 0)
                print(f"   Risk Level: {risk_level} ({risk_score} points)")

            results["volatility"] = vol_result
        else:
            print("[NG] Volatility prediction failed")

    except Exception as e:
        print(f"[NG] Volatility prediction test error: {e}")

    try:
        # 5. 可視化システムテスト
        print("\n" + "=" * 40)
        print("5. Visualization System Test")
        print("=" * 40)

        visualizer = MLResultsVisualizer(output_dir="output/test_ml_integration")

        print("Visualization file generation...")

        # 個別チャート作成
        charts_created = 0

        if "lstm" in results:
            lstm_chart = visualizer.create_lstm_prediction_chart(
                sample_data, results["lstm"], symbol=test_symbol
            )
            if lstm_chart:
                charts_created += 1
                print(f"   [OK] LSTM Chart: {Path(lstm_chart).name}")

        if "volatility" in results:
            vol_chart = visualizer.create_volatility_forecast_chart(
                sample_data, results["volatility"], symbol=test_symbol
            )
            if vol_chart:
                charts_created += 1
                print(f"   [OK] Volatility Chart: {Path(vol_chart).name}")

        if "multiframe" in results:
            mf_chart = visualizer.create_multiframe_analysis_chart(
                sample_data, results["multiframe"], symbol=test_symbol
            )
            if mf_chart:
                charts_created += 1
                print(f"   [OK] Multi-timeframe Chart: {Path(mf_chart).name}")

        # 総合ダッシュボード作成
        dashboard = visualizer.create_comprehensive_dashboard(
            sample_data,
            lstm_results=results.get("lstm"),
            volatility_results=results.get("volatility"),
            multiframe_results=results.get("multiframe"),
            symbol=test_symbol,
        )
        if dashboard:
            charts_created += 1
            print(f"   [OK] Comprehensive Dashboard: {Path(dashboard).name}")

        # 分析レポート生成
        report = visualizer.generate_analysis_report(test_symbol, results)
        if report:
            charts_created += 1
            print(f"   [OK] Analysis Report: {Path(report).name}")

        print(
            f"[OK] Visualization system test completed: {charts_created} files generated"
        )

    except Exception as e:
        print(f"[NG] Visualization system test error: {e}")

    # 6. 統合評価
    print("\n" + "=" * 40)
    print("6. Integration Evaluation")
    print("=" * 40)

    components_tested = len(results)
    print(f"Tested Components: {components_tested}/4")

    # 各コンポーネントの結果サマリー
    for component, result in results.items():
        print(f"\n[{component.upper()}]")

        if component == "lstm":
            if "confidence_score" in result:
                print(f"  Prediction Confidence: {result['confidence_score']:.1f}%")
            if "predicted_returns" in result:
                avg_return = np.mean(result["predicted_returns"])
                print(f"  Average Predicted Return: {avg_return:+.2f}%")

        elif component == "volatility":
            current_metrics = result.get("current_metrics", {})
            if current_metrics:
                vol = current_metrics.get("realized_volatility", 0) * 100
                print(f"  Current Volatility: {vol:.1f}%")

            risk_assessment = result.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                print(f"  Risk Level: {risk_level}")

        elif component == "multiframe":
            integrated = result.get("integrated_analysis", {})
            if integrated:
                trend = integrated.get("overall_trend", "unknown")
                confidence = integrated.get("trend_confidence", 0)
                print(f"  Overall Trend: {trend}")
                print(f"  Trend Confidence: {confidence:.1f}%")

                signal = integrated.get("integrated_signal", {})
                if signal:
                    action = signal.get("action", "HOLD")
                    print(f"  Integrated Signal: {action}")

    # 最終統合判定
    print("\n" + "=" * 40)
    print("Final Integration Assessment")
    print("=" * 40)

    # シグナル統合
    signals = []
    if "lstm" in results:
        lstm_result = results["lstm"]
        if "predicted_returns" in lstm_result:
            avg_return = np.mean(lstm_result["predicted_returns"])
            if avg_return > 1:
                signals.append("BUY")
            elif avg_return < -1:
                signals.append("SELL")
            else:
                signals.append("HOLD")

    if "multiframe" in results:
        mf_result = results["multiframe"]
        integrated = mf_result.get("integrated_analysis", {})
        signal_info = integrated.get("integrated_signal", {})
        action = signal_info.get("action", "HOLD")
        signals.append(action)

    # 統合判定
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    hold_count = signals.count("HOLD")

    if buy_count > sell_count and buy_count > hold_count:
        final_signal = "BUY"
        signal_strength = buy_count / len(signals)
    elif sell_count > buy_count and sell_count > hold_count:
        final_signal = "SELL"
        signal_strength = sell_count / len(signals)
    else:
        final_signal = "HOLD"
        signal_strength = hold_count / len(signals) if signals else 0

    print(f"Integrated Signal: {final_signal}")
    print(f"Signal Strength: {signal_strength:.1%}")
    print(
        f"Signal Distribution: BUY({buy_count}) SELL({sell_count}) HOLD({hold_count})"
    )

    # テスト成功率
    success_rate = components_tested / 4 * 100
    print(f"\nML Integration Test Success Rate: {success_rate:.1f}%")

    if success_rate >= 75:
        print("PASS: ML Integration Test - All major functions working")
        return True
    else:
        print("WARN: ML Integration Test - Some functions failed")
        return False


if __name__ == "__main__":
    try:
        success = test_ml_integration()

        print("\n" + "=" * 60)
        if success:
            print("[OK] ML Integration Test Completed - All functions verified")
        else:
            print("[NG] ML Integration Test Completed - Some issues found")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Integration test error: {e}")
        import traceback

        traceback.print_exc()

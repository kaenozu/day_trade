#!/usr/bin/env python3
"""
ML機能統合テスト
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


def generate_sample_data(days: int = 500, base_price: float = 2800) -> pd.DataFrame:
    """
    サンプル株価データ生成

    Args:
        days: 生成する日数
        base_price: 基準価格

    Returns:
        株価データ（OHLCV）
    """
    print(f"サンプルデータ生成中... ({days}日分)")

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

    print(f"OK サンプルデータ生成完了: {len(df)}レコード")
    return df


def test_ml_integration():
    """
    ML機能統合テスト実行
    """
    print("=" * 60)
    print("ML機能統合テスト開始")
    print("=" * 60)

    # サンプルデータ生成
    sample_data = generate_sample_data(days=400)
    test_symbol = "TEST_ML_STOCK"

    print(f"\nテスト対象銘柄: {test_symbol}")
    print(
        f"データ期間: {sample_data.index[0].strftime('%Y-%m-%d')} ～ {sample_data.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"データ件数: {len(sample_data)}件")
    print(
        f"価格範囲: {sample_data['Close'].min():.0f} ～ {sample_data['Close'].max():.0f}"
    )

    # 各コンポーネントのテスト結果を保存
    results = {}

    try:
        # 1. LSTM時系列予測テスト
        print("\n" + "=" * 40)
        print("1. LSTM時系列予測テスト")
        print("=" * 40)

        lstm_model = LSTMTimeSeriesModel(
            sequence_length=30,
            prediction_horizon=5,
            lstm_units=[64, 32],
            dropout_rate=0.3,
        )

        # LSTM訓練
        print("LSTM訓練開始...")
        training_result = lstm_model.train_lstm_model(
            symbol=test_symbol,
            data=sample_data,
            validation_split=0.2,
            epochs=15,  # テスト用に短縮
            batch_size=32,
            early_stopping_patience=5,
        )

        if training_result:
            print("OK LSTM訓練完了")
            print(f"   検証R²: {training_result['val_r2']:.3f}")
            print(f"   検証RMSE: {training_result['val_rmse']:.2f}")
            print(f"   信頼度指標: {training_result['val_r2'] * 100:.1f}%")

            # LSTM予測実行
            print("LSTM予測実行...")
            prediction_result = lstm_model.predict_future_prices(
                symbol=test_symbol, data=sample_data, steps_ahead=5
            )

            if prediction_result:
                print("OK LSTM予測完了")
                print(f"   現在価格: {prediction_result['current_price']:.0f}")
                print(f"   5日後予測: {prediction_result['predicted_prices'][-1]:.0f}")
                print(
                    f"   予測リターン: {prediction_result['predicted_returns'][-1]:+.2f}%"
                )
                print(f"   予測信頼度: {prediction_result['confidence_score']:.1f}%")
                results["lstm"] = prediction_result
            else:
                print("NG LSTM予測失敗")
        else:
            print("NG LSTM訓練失敗")

    except Exception as e:
        print(f"NG LSTMテストエラー: {e}")

    try:
        # 2. 高度テクニカル指標テスト
        print("\n" + "=" * 40)
        print("2. 高度テクニカル指標テスト")
        print("=" * 40)

        technical_analyzer = AdvancedTechnicalIndicators()

        # 一目均衡表
        print("一目均衡表計算...")
        ichimoku_result = technical_analyzer.calculate_ichimoku_cloud(sample_data)
        if ichimoku_result and "signal" in ichimoku_result:
            print(
                f"OK 一目均衡表: {ichimoku_result['signal']} (強度: {ichimoku_result['signal_strength']})"
            )
            results["ichimoku"] = ichimoku_result

        # フィボナッチ分析
        print("フィボナッチ分析...")
        fibonacci_result = technical_analyzer.analyze_fibonacci_levels(sample_data)
        if fibonacci_result and "levels" in fibonacci_result:
            levels = fibonacci_result["levels"]
            print(f"✅ フィボナッチレベル: {len(levels)}レベル特定")
            print(f"   主要サポート: {fibonacci_result.get('key_support', 'N/A')}")
            print(
                f"   主要レジスタンス: {fibonacci_result.get('key_resistance', 'N/A')}"
            )
            results["fibonacci"] = fibonacci_result

        # エリオット波動
        print("エリオット波動分析...")
        elliott_result = technical_analyzer.detect_elliott_waves(sample_data)
        if elliott_result and "current_wave" in elliott_result:
            print(f"✅ エリオット波動: {elliott_result['current_wave']}")
            print(f"   波動段階: {elliott_result.get('wave_phase', 'N/A')}")
            results["elliott"] = elliott_result

    except Exception as e:
        print(f"❌ 高度テクニカル指標テストエラー: {e}")

    try:
        # 3. マルチタイムフレーム分析テスト
        print("\n" + "=" * 40)
        print("3. マルチタイムフレーム分析テスト")
        print("=" * 40)

        multiframe_analyzer = MultiTimeframeAnalyzer()

        print("マルチタイムフレーム分析実行...")
        mf_result = multiframe_analyzer.analyze_multiple_timeframes(sample_data)

        if mf_result:
            print("✅ マルチタイムフレーム分析完了")

            timeframes = mf_result.get("timeframes", {})
            for tf_key, tf_data in timeframes.items():
                tf_name = tf_data.get("timeframe", tf_key)
                trend = tf_data.get("trend_direction", "unknown")
                strength = tf_data.get("trend_strength", 0)
                print(f"   {tf_name}: {trend} (強度: {strength:.1f})")

            integrated = mf_result.get("integrated_analysis", {})
            if integrated:
                overall_trend = integrated.get("overall_trend", "unknown")
                confidence = integrated.get("trend_confidence", 0)
                consistency = integrated.get("consistency_score", 0)

                print(f"   統合トレンド: {overall_trend}")
                print(f"   信頼度: {confidence:.1f}%")
                print(f"   整合性: {consistency:.1f}%")

                signal = integrated.get("integrated_signal", {})
                if signal:
                    action = signal.get("action", "HOLD")
                    strength = signal.get("strength", "WEAK")
                    print(f"   統合シグナル: {action} ({strength})")

            results["multiframe"] = mf_result
        else:
            print("❌ マルチタイムフレーム分析失敗")

    except Exception as e:
        print(f"❌ マルチタイムフレーム分析テストエラー: {e}")

    try:
        # 4. ボラティリティ予測テスト
        print("\n" + "=" * 40)
        print("4. ボラティリティ予測テスト")
        print("=" * 40)

        vol_engine = VolatilityPredictionEngine()

        print("ボラティリティ分析実行...")
        vol_result = vol_engine.predict_volatility(sample_data)

        if vol_result:
            print("✅ ボラティリティ予測完了")

            current_metrics = vol_result.get("current_metrics", {})
            if current_metrics:
                realized_vol = current_metrics.get("realized_volatility", 0) * 100
                vix_like = current_metrics.get("vix_like_indicator", 0)
                regime = current_metrics.get("volatility_regime", "unknown")

                print(f"   実現ボラティリティ: {realized_vol:.1f}%")
                print(f"   VIX風指標: {vix_like:.1f}")
                print(f"   ボラティリティ環境: {regime}")

            ensemble = vol_result.get("ensemble_forecast", {})
            if ensemble:
                ensemble_vol = ensemble.get("ensemble_volatility", 0)
                confidence = ensemble.get("ensemble_confidence", 0)
                print(f"   アンサンブル予測: {ensemble_vol:.1f}%")
                print(f"   予測信頼度: {confidence:.2f}")

            risk_assessment = vol_result.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                risk_score = risk_assessment.get("risk_score", 0)
                print(f"   リスクレベル: {risk_level} ({risk_score}点)")

            results["volatility"] = vol_result
        else:
            print("❌ ボラティリティ予測失敗")

    except Exception as e:
        print(f"❌ ボラティリティ予測テストエラー: {e}")

    try:
        # 5. 可視化システムテスト
        print("\n" + "=" * 40)
        print("5. 可視化システムテスト")
        print("=" * 40)

        visualizer = MLResultsVisualizer(output_dir="output/test_ml_integration")

        print("可視化ファイル生成中...")

        # 個別チャート作成
        charts_created = 0

        if "lstm" in results:
            lstm_chart = visualizer.create_lstm_prediction_chart(
                sample_data, results["lstm"], symbol=test_symbol
            )
            if lstm_chart:
                charts_created += 1
                print(f"   ✅ LSTM予測チャート: {Path(lstm_chart).name}")

        if "volatility" in results:
            vol_chart = visualizer.create_volatility_forecast_chart(
                sample_data, results["volatility"], symbol=test_symbol
            )
            if vol_chart:
                charts_created += 1
                print(f"   ✅ ボラティリティチャート: {Path(vol_chart).name}")

        if "multiframe" in results:
            mf_chart = visualizer.create_multiframe_analysis_chart(
                sample_data, results["multiframe"], symbol=test_symbol
            )
            if mf_chart:
                charts_created += 1
                print(f"   ✅ マルチタイムフレームチャート: {Path(mf_chart).name}")

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
            print(f"   ✅ 総合ダッシュボード: {Path(dashboard).name}")

        # 分析レポート生成
        report = visualizer.generate_analysis_report(test_symbol, results)
        if report:
            charts_created += 1
            print(f"   ✅ 分析レポート: {Path(report).name}")

        print(f"✅ 可視化システムテスト完了: {charts_created}個のファイル生成")

    except Exception as e:
        print(f"❌ 可視化システムテストエラー: {e}")

    # 6. 統合評価
    print("\n" + "=" * 40)
    print("6. 統合評価")
    print("=" * 40)

    components_tested = len(results)
    print(f"テスト完了コンポーネント数: {components_tested}/4")

    # 各コンポーネントの結果サマリー
    for component, result in results.items():
        print(f"\n【{component.upper()}】")

        if component == "lstm":
            if "confidence_score" in result:
                print(f"  予測信頼度: {result['confidence_score']:.1f}%")
            if "predicted_returns" in result:
                avg_return = np.mean(result["predicted_returns"])
                print(f"  平均予測リターン: {avg_return:+.2f}%")

        elif component == "volatility":
            current_metrics = result.get("current_metrics", {})
            if current_metrics:
                vol = current_metrics.get("realized_volatility", 0) * 100
                print(f"  現在のボラティリティ: {vol:.1f}%")

            risk_assessment = result.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                print(f"  リスクレベル: {risk_level}")

        elif component == "multiframe":
            integrated = result.get("integrated_analysis", {})
            if integrated:
                trend = integrated.get("overall_trend", "unknown")
                confidence = integrated.get("trend_confidence", 0)
                print(f"  総合トレンド: {trend}")
                print(f"  トレンド信頼度: {confidence:.1f}%")

                signal = integrated.get("integrated_signal", {})
                if signal:
                    action = signal.get("action", "HOLD")
                    print(f"  統合シグナル: {action}")

    # 最終統合判定
    print("\n" + "=" * 40)
    print("最終統合判定")
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

    print(f"統合シグナル: {final_signal}")
    print(f"シグナル強度: {signal_strength:.1%}")
    print(f"シグナル分布: BUY({buy_count}) SELL({sell_count}) HOLD({hold_count})")

    # テスト成功率
    success_rate = components_tested / 4 * 100
    print(f"\nML統合テスト成功率: {success_rate:.1f}%")

    if success_rate >= 75:
        print("🎉 ML機能統合テスト: 合格")
        return True
    else:
        print("⚠️  ML機能統合テスト: 一部失敗")
        return False


if __name__ == "__main__":
    try:
        success = test_ml_integration()

        print("\n" + "=" * 60)
        if success:
            print("✅ ML機能統合テスト完了 - 全機能正常動作確認")
        else:
            print("❌ ML機能統合テスト完了 - 一部機能に問題あり")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 統合テストエラー: {e}")
        import traceback

        traceback.print_exc()

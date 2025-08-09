#!/usr/bin/env python3
"""
深層学習統合システム テストスクリプト
Phase F: 次世代機能拡張フェーズ

Transformer, LSTM統合システムの動作確認・ベンチマーク
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.day_trade.ml import (
    DeepLearningModelManager,
    TransformerModel,
    LSTMModel,
    DeepLearningConfig
)
from src.day_trade.core.optimization_strategy import OptimizationConfig, OptimizationLevel


class DeepLearningSystemTester:
    """深層学習システムテスター"""

    def __init__(self):
        self.config = DeepLearningConfig(
            sequence_length=60,
            prediction_horizon=5,
            hidden_dim=128,
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            epochs=10,  # テスト用に少なく設定
            batch_size=32,
            early_stopping_patience=3,
            use_pytorch=False  # NumPy fallbackでテスト
        )

        self.test_data = self._generate_test_data()

        print("[START] 深層学習統合システム テスト開始")
        print(f"設定: PyTorch利用={self.config.use_pytorch}")
        print(f"シーケンス長: {self.config.sequence_length}")
        print("=" * 60)

    def _generate_test_data(self) -> pd.DataFrame:
        """テスト用時系列データ生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')

        # より複雑な時系列パターン生成
        trend = np.linspace(100, 120, 500)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(500) / 252)  # 年次季節性
        noise = np.random.normal(0, 2, 500)

        # 価格系列生成
        prices = trend + seasonality + noise

        # OHLCV データ作成
        return pd.DataFrame({
            'Date': dates,
            'Open': prices + np.random.normal(0, 0.5, 500),
            'High': prices + np.abs(np.random.normal(0, 1, 500)),
            'Low': prices - np.abs(np.random.normal(0, 1, 500)),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 500)
        }).set_index('Date')

    def test_transformer_model(self):
        """Transformerモデルテスト"""
        print("[TRANSFORMER] Transformerモデルテスト")

        try:
            model = TransformerModel(self.config)

            # データ準備
            start_time = time.time()
            X, y = model.prepare_training_data(self.test_data)
            data_prep_time = time.time() - start_time

            print(f"データ準備時間: {data_prep_time:.4f}秒")
            print(f"特徴量形状: {X.shape}")
            print(f"ターゲット形状: {y.shape}")

            # モデル訓練
            start_time = time.time()
            training_result = model.train(self.test_data)
            training_time = time.time() - start_time

            print(f"訓練時間: {training_time:.4f}秒")
            print(f"最終損失: {training_result.final_loss:.6f}")
            print(f"最良損失: {training_result.best_loss:.6f}")
            print(f"実行エポック数: {training_result.epochs_run}")

            # 予測実行
            start_time = time.time()
            recent_data = self.test_data.tail(100)
            prediction_result = model.predict(recent_data)
            prediction_time = time.time() - start_time

            print(f"予測時間: {prediction_time:.4f}秒")
            print(f"予測値: {prediction_result.predictions[:5]}")  # 最初の5個
            print(f"信頼度: {prediction_result.confidence[:5]}")

            # 不確実性推定
            if prediction_result.uncertainty:
                print(f"予測不確実性: mean={prediction_result.uncertainty.mean:.4f}")
                print(f"            std={prediction_result.uncertainty.std:.4f}")

            print("[OK] Transformerモデルテスト完了\n")
            return True

        except Exception as e:
            print(f"[ERROR] Transformerモデルテスト失敗: {e}\n")
            return False

    def test_lstm_model(self):
        """LSTMモデルテスト"""
        print("[LSTM] LSTMモデルテスト")

        try:
            model = LSTMModel(self.config)

            # データ準備と訓練
            start_time = time.time()
            training_result = model.train(self.test_data)
            training_time = time.time() - start_time

            print(f"訓練時間: {training_time:.4f}秒")
            print(f"最終損失: {training_result.final_loss:.6f}")
            print(f"検証精度: {training_result.validation_metrics.get('mse', 'N/A')}")

            # 予測実行
            start_time = time.time()
            recent_data = self.test_data.tail(100)
            prediction_result = model.predict(recent_data)
            prediction_time = time.time() - start_time

            print(f"予測時間: {prediction_time:.4f}秒")
            print(f"予測値: {prediction_result.predictions[:5]}")
            print(f"予測精度: {prediction_result.metrics.get('mae', 'N/A')}")

            print("[OK] LSTMモデルテスト完了\n")
            return True

        except Exception as e:
            print(f"[ERROR] LSTMモデルテスト失敗: {e}\n")
            return False

    def test_model_manager(self):
        """モデル管理システムテスト"""
        print("[MANAGER] モデル管理システムテスト")

        try:
            # 最適化設定
            opt_config = OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                performance_monitoring=True
            )

            manager = DeepLearningModelManager(self.config, opt_config)

            # 複数モデル登録
            transformer = TransformerModel(self.config)
            lstm = LSTMModel(self.config)

            manager.register_model("transformer", transformer)
            manager.register_model("lstm", lstm)

            print(f"登録モデル数: {len(manager.models)}")
            print(f"利用可能モデル: {list(manager.models.keys())}")

            # アンサンブル訓練
            start_time = time.time()
            ensemble_results = manager.train_ensemble(self.test_data)
            ensemble_training_time = time.time() - start_time

            print(f"アンサンブル訓練時間: {ensemble_training_time:.4f}秒")
            print(f"訓練結果数: {len(ensemble_results)}")

            # アンサンブル予測
            start_time = time.time()
            recent_data = self.test_data.tail(100)
            ensemble_prediction = manager.predict_ensemble(recent_data)
            ensemble_prediction_time = time.time() - start_time

            print(f"アンサンブル予測時間: {ensemble_prediction_time:.4f}秒")
            print(f"アンサンブル予測: {ensemble_prediction.predictions[:3]}")
            print(f"予測重み: {ensemble_prediction.model_weights}")

            # 性能統計
            performance = manager.get_performance_summary()
            print(f"総実行回数: {performance.get('total_predictions', 0)}")
            print(f"平均予測時間: {performance.get('average_prediction_time', 0):.4f}秒")

            print("[OK] モデル管理システムテスト完了\n")
            return True

        except Exception as e:
            print(f"[ERROR] モデル管理システムテスト失敗: {e}\n")
            return False

    def test_uncertainty_estimation(self):
        """不確実性推定テスト"""
        print("[UNCERTAINTY] 不確実性推定テスト")

        try:
            # Monte Carlo Dropoutを有効にしてモデル作成
            config_with_uncertainty = self.config.__class__(
                **{**self.config.__dict__, 'dropout_rate': 0.3}
            )

            model = TransformerModel(config_with_uncertainty)

            # 訓練（短縮版）
            model.train(self.test_data)

            # 不確実性付き予測
            start_time = time.time()
            recent_data = self.test_data.tail(50)
            prediction_result = model.predict_with_uncertainty(
                recent_data,
                num_samples=10  # テスト用に少なく設定
            )
            uncertainty_time = time.time() - start_time

            print(f"不確実性推定時間: {uncertainty_time:.4f}秒")
            print(f"サンプル数: 10")

            if prediction_result.uncertainty:
                uncertainty = prediction_result.uncertainty
                print(f"不確実性統計:")
                print(f"  平均: {uncertainty.mean:.4f}")
                print(f"  標準偏差: {uncertainty.std:.4f}")
                print(f"  95%信頼区間: [{uncertainty.lower_bound[0]:.4f}, {uncertainty.upper_bound[0]:.4f}]")
                print(f"  エピステミック不確実性: {uncertainty.epistemic:.4f}")
                print(f"  アレアトリック不確実性: {uncertainty.aleatoric:.4f}")

            print("[OK] 不確実性推定テスト完了\n")
            return True

        except Exception as e:
            print(f"[ERROR] 不確実性推定テスト失敗: {e}\n")
            return False

    def test_feature_importance(self):
        """特徴量重要度テスト"""
        print("[FEATURE_IMPORTANCE] 特徴量重要度テスト")

        try:
            model = TransformerModel(self.config)
            model.train(self.test_data)

            # 特徴量重要度分析
            start_time = time.time()
            importance_result = model.get_feature_importance(self.test_data.tail(100))
            importance_time = time.time() - start_time

            print(f"特徴量重要度計算時間: {importance_time:.4f}秒")

            if importance_result:
                print("特徴量重要度ランキング:")
                for i, (feature, importance) in enumerate(importance_result.items()):
                    if i < 5:  # 上位5個表示
                        print(f"  {i+1}. {feature}: {importance:.4f}")

            print("[OK] 特徴量重要度テスト完了\n")
            return True

        except Exception as e:
            print(f"[ERROR] 特徴量重要度テスト失敗: {e}\n")
            return False

    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("[COMPREHENSIVE] 深層学習統合システム 包括的テスト")
        print("=" * 60)

        # 全テスト実行
        test_results = {}
        test_results['transformer'] = self.test_transformer_model()
        test_results['lstm'] = self.test_lstm_model()
        test_results['manager'] = self.test_model_manager()
        test_results['uncertainty'] = self.test_uncertainty_estimation()
        test_results['feature_importance'] = self.test_feature_importance()

        # 結果集計
        success_count = sum(test_results.values())
        total_tests = len(test_results)

        print("[SUMMARY] テスト結果サマリー")
        print("=" * 60)
        print(f"成功テスト: {success_count}/{total_tests} ({success_count/total_tests:.0%})")

        for test_name, result in test_results.items():
            status = "[OK]" if result else "[FAIL]"
            print(f"  {status} {test_name}")

        # 総合評価
        if success_count == total_tests:
            print("\n[SUCCESS] すべてのテストが成功！深層学習システムは正常動作中です。")
        elif success_count >= total_tests * 0.8:
            print("\n[GOOD] 主要テストが成功。深層学習システムは動作可能です。")
        else:
            print("\n[WARNING] 多くのテストが失敗。設定や依存関係を確認してください。")

        return {
            'success_count': success_count,
            'total_tests': total_tests,
            'test_results': test_results
        }


def main():
    """メインテスト実行"""
    try:
        tester = DeepLearningSystemTester()
        result = tester.run_comprehensive_test()

        # テスト結果の保存
        import json

        serializable_result = {
            'success_count': result['success_count'],
            'total_tests': result['total_tests'],
            'test_results': result['test_results'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'sequence_length': tester.config.sequence_length,
                'prediction_horizon': tester.config.prediction_horizon,
                'use_pytorch': tester.config.use_pytorch,
                'test_data_size': len(tester.test_data)
            }
        }

        with open('deep_learning_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVE] テスト結果をdeep_learning_test_results.jsonに保存しました。")

        return 0 if result['success_count'] >= result['total_tests'] * 0.8 else 1

    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

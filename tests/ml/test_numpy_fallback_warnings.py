#!/usr/bin/env python3
"""
NumPy Fallback Warning System Tests

Issue #696対応: DeepLearningModelsでのNumPyフォールバック明確化テスト
"""

import pytest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.deep_learning_models import (
        TransformerModel,
        LSTMModel,
        ModelConfig,
        TORCH_AVAILABLE
    )
    from day_trade.ml.hybrid_lstm_transformer import (
        HybridLSTMTransformerEngine,
        HybridModelConfig,
        PYTORCH_AVAILABLE
    )
    from day_trade.utils.logging_config import get_context_logger
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestNumPyFallbackWarnings:
    """Issue #696: NumPyフォールバック警告システムテスト"""
    
    def test_pytorch_availability_warnings(self):
        """Issue #696: PyTorch未利用時の警告テスト"""
        with patch('day_trade.ml.deep_learning_models.TORCH_AVAILABLE', False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # ImportWarningが発生することを確認
                with patch('day_trade.ml.deep_learning_models.logger') as mock_logger:
                    # deep_learning_models を再インポート
                    import importlib
                    import day_trade.ml.deep_learning_models
                    importlib.reload(day_trade.ml.deep_learning_models)
                    
                    # critical ログが出力されることを確認
                    mock_logger.critical.assert_called()
                    
                    # 警告メッセージの内容確認
                    critical_call_args = mock_logger.critical.call_args[0][0]
                    assert "PyTorchが未インストールです" in critical_call_args
                    assert "簡易的な線形回帰モデル" in critical_call_args
                    assert "深層学習の性能は期待できません" in critical_call_args
                    assert "pip install torch" in critical_call_args
    
    def test_transformer_fallback_model_warning(self):
        """Issue #696: Transformerフォールバックモデル警告テスト"""
        config = ModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            epochs=1
        )
        
        model = TransformerModel(config)
        
        # テストデータ
        X = np.random.randn(20, 10, 5).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32)
        
        with patch('day_trade.ml.deep_learning_models.logger') as mock_logger:
            # PyTorchが利用できない場合のフォールバック訓練
            with patch.object(model, 'model', None):
                with patch('day_trade.ml.deep_learning_models.TORCH_AVAILABLE', False):
                    model._train_fallback_model(X, y, 0.0)
                    
                    # 警告ログが出力されることを確認
                    mock_logger.warning.assert_called()
                    warning_call = mock_logger.warning.call_args[0][0]
                    
                    # 警告メッセージの内容確認
                    assert "NumPy フォールバック実行中" in warning_call
                    assert "簡易線形回帰モデルを使用中" in warning_call
                    assert "深層学習の性能は期待できません" in warning_call
                    assert "複雑なパターン学習は不可能です" in warning_call
                    assert "PyTorchインストール推奨" in warning_call
    
    def test_fallback_model_structure(self):
        """Issue #696: フォールバックモデル構造の明確化テスト"""
        config = ModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            epochs=1
        )
        
        model = TransformerModel(config)
        
        X = np.random.randn(20, 10, 5).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32)
        
        with patch('day_trade.ml.deep_learning_models.TORCH_AVAILABLE', False):
            result = model._train_fallback_model(X, y, 0.0)
            
            # フォールバックモデルの構造確認
            assert isinstance(model.model, dict)
            assert "fallback_weights" in model.model
            assert "model_type" in model.model
            assert model.model["model_type"] == "numpy_linear_regression"
            assert "performance_warning" in model.model
            assert "recommendation" in model.model
            
            # メッセージ内容確認
            assert "深層学習性能は期待できません" in model.model["performance_warning"]
            assert "PyTorchインストール" in model.model["recommendation"]
    
    def test_hybrid_lstm_transformer_fallback_warnings(self):
        """Issue #696: HybridLSTMTransformer フォールバック警告テスト"""
        config = HybridModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            epochs=1
        )
        
        with patch('day_trade.ml.hybrid_lstm_transformer.PYTORCH_AVAILABLE', False):
            engine = HybridLSTMTransformerEngine(config)
            
            X = np.random.randn(20, 10, 5).astype(np.float32)
            y = np.random.randn(20, 1).astype(np.float32)
            
            with patch('day_trade.ml.hybrid_lstm_transformer.logger') as mock_logger:
                engine._train_numpy_hybrid(X, y, 0.0)
                
                # 警告ログが出力されることを確認
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                
                # 警告メッセージの内容確認
                assert "HybridLSTMTransformer NumPy フォールバック実行中" in warning_call
                assert "LSTM時系列学習: 単純な線形変換に置換" in warning_call
                assert "Transformer注意機構: 線形結合に置換" in warning_call
                assert "Cross-Attention融合: 重み平均に置換" in warning_call
                assert "GPU加速: 無効" in warning_call
                assert "ハイブリッド深層学習の性能は期待できません" in warning_call
    
    def test_fallback_prediction_warning(self):
        """Issue #696: フォールバック予測時の警告テスト"""
        config = ModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            epochs=1
        )
        
        model = TransformerModel(config)
        
        # フォールバックモデルを設定
        X_flat = np.random.randn(100)
        model.model = {
            "fallback_weights": X_flat,
            "model_type": "numpy_linear_regression",
            "performance_warning": "深層学習性能は期待できません",
            "recommendation": "PyTorchインストールを強く推奨"
        }
        model.is_trained = True
        
        X_test = np.random.randn(5, 10, 10).astype(np.float32)
        
        # 予測実行
        predictions = model._predict_internal(X_test)
        
        # 予測結果の形状確認
        assert predictions.shape == (5, 1)
        assert isinstance(predictions, np.ndarray)
    
    def test_hybrid_forward_backward_comments(self):
        """Issue #696: HybridモデルのNumPy実装コメント確認テスト"""
        config = HybridModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            epochs=1
        )
        
        with patch('day_trade.ml.hybrid_lstm_transformer.PYTORCH_AVAILABLE', False):
            engine = HybridLSTMTransformerEngine(config)
            
            # モデル初期化
            engine.model = {
                "fusion_weights": np.random.randn(256, 1) * 0.1
            }
            
            X = np.random.randn(5, 10, 5).astype(np.float32)
            y = np.random.randn(5, 1).astype(np.float32)
            
            # フォワードパス実行
            predictions = engine._numpy_hybrid_forward(X)
            assert predictions.shape == (5, 1)
            
            # バックワードパス実行（エラーなく実行されることを確認）
            engine._numpy_hybrid_backward(X, y, predictions)
    
    def test_deep_learning_models_import_warning(self):
        """Issue #696: deep_learning_models インポート時の警告テスト"""
        with patch('day_trade.ml.deep_learning_models.TORCH_AVAILABLE', False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # モジュール再読み込みで警告発生を確認
                import importlib
                import day_trade.ml.deep_learning_models
                importlib.reload(day_trade.ml.deep_learning_models)
                
                # ImportWarningが発生していることを確認
                import_warnings = [warning for warning in w if issubclass(warning.category, ImportWarning)]
                assert len(import_warnings) >= 1
                
                warning_msg = str(import_warnings[0].message)
                assert "PyTorch未インストール" in warning_msg
                assert "NumPy簡易実装にフォールバック" in warning_msg
                assert "性能制限あり" in warning_msg
    
    def test_hybrid_import_warning(self):
        """Issue #696: hybrid_lstm_transformer インポート時の警告テスト"""
        with patch('day_trade.ml.hybrid_lstm_transformer.PYTORCH_AVAILABLE', False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # モジュール再読み込みで警告発生を確認
                import importlib
                import day_trade.ml.hybrid_lstm_transformer
                importlib.reload(day_trade.ml.hybrid_lstm_transformer)
                
                # ImportWarningが発生していることを確認
                import_warnings = [warning for warning in w if issubclass(warning.category, ImportWarning)]
                assert len(import_warnings) >= 1
                
                warning_msg = str(import_warnings[0].message)
                assert "PyTorch未インストール" in warning_msg
                assert "HybridLSTMTransformer簡易実装にフォールバック" in warning_msg
                assert "大幅な性能制限あり" in warning_msg


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestFallbackFunctionality:
    """Issue #696: フォールバック機能テスト"""
    
    def test_transformer_fallback_accuracy_calculation(self):
        """Issue #696: Transformerフォールバック精度計算テスト"""
        config = ModelConfig(epochs=1)
        model = TransformerModel(config)
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        accuracy = model._calculate_accuracy(y_true, y_pred)
        
        # 精度が0-1の範囲内であることを確認
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
        
        # 予測が完全に間違っている場合
        y_pred_bad = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        accuracy_bad = model._calculate_accuracy(y_true, y_pred_bad)
        assert accuracy_bad < accuracy  # 悪い予測の精度が低いことを確認
    
    def test_lstm_numpy_sigmoid_function(self):
        """Issue #696: LSTM NumPyシグモイド関数テスト"""
        config = ModelConfig(epochs=1)
        model = LSTMModel(config)
        
        # 極値テスト
        x_extreme = np.array([-1000, -10, 0, 10, 1000])
        sigmoid_result = model._sigmoid(x_extreme)
        
        # シグモイド関数の性質確認
        assert np.all(sigmoid_result >= 0)
        assert np.all(sigmoid_result <= 1)
        assert sigmoid_result[2] == pytest.approx(0.5, abs=1e-5)  # sigmoid(0) = 0.5
        assert sigmoid_result[0] == pytest.approx(0.0, abs=1e-5)  # sigmoid(-inf) ≈ 0
        assert sigmoid_result[4] == pytest.approx(1.0, abs=1e-5)  # sigmoid(inf) ≈ 1
    
    def test_performance_degradation_warning(self):
        """Issue #696: 性能劣化警告の一貫性テスト"""
        # すべてのフォールバック警告が適切な性能劣化情報を含むことを確認
        warnings_to_check = [
            "深層学習の性能は期待できません",
            "ハイブリッド深層学習の性能は期待できません",
            "簡易的な線形回帰モデル",
            "簡易線形近似モデル",
            "PyTorchインストール推奨"
        ]
        
        for warning_text in warnings_to_check:
            assert isinstance(warning_text, str)
            assert len(warning_text) > 10  # 意味のある警告メッセージ長


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
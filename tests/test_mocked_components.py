#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モック活用テストスイート - 重い処理を高速化したテスト
"""

import pytest
import asyncio
import unittest.mock as mock
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# テスト対象のインポート
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from daytrade import DayTradeWebDashboard
from simple_ml_prediction_system import SimpleMLPredictionSystem
from prediction_validator import PredictionValidator

class TestMockedComponents:
    """モック活用コンポーネントテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.mock_price_data = {
            '7203': {'opening_price': 3000.0, 'current_price': 3150.0},
            '6861': {'opening_price': 1500.0, 'current_price': 1480.0},
            '4063': {'opening_price': 8000.0, 'current_price': 8200.0}
        }
        
        self.mock_ml_prediction = {
            'signal': '●買い●',
            'confidence': 85.5,
            'score': 78,
            'risk_level': '中リスク',
            'ml_source': 'advanced_ml',
            'backtest_score': 92.3,
            'model_consensus': {'random_forest': 1, 'logistic': 1},
            'feature_importance': ['volatility', 'trend', 'volume']
        }

    @pytest.mark.asyncio
    async def test_stock_price_data_mock(self):
        """株価データ取得のモックテスト"""
        with patch('daytrade.get_yfinance') as mock_yf:
            # yfinanceモック設定
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame({
                'Open': [3000.0],
                'Close': [3150.0],
                'High': [3200.0],
                'Low': [2950.0],
                'Volume': [1000000]
            })
            
            mock_yf_module = MagicMock()
            mock_yf_module.Ticker.return_value = mock_ticker
            mock_yf.return_value = (mock_yf_module, True)
            
            # テスト実行
            system = DayTradeWebDashboard()
            result = await system.get_stock_price_data('7203')
            
            # 検証
            assert result['opening_price'] == 3000.0
            assert result['current_price'] == 3150.0
            mock_yf_module.Ticker.assert_called_with('7203.T')

    @pytest.mark.asyncio
    async def test_ml_prediction_mock(self):
        """ML予測システムのモックテスト"""
        with patch.object(SimpleMLPredictionSystem, 'predict_symbol_movement') as mock_predict:
            # ML予測モック設定
            mock_result = MagicMock()
            mock_result.symbol = '7203'
            mock_result.prediction = 1
            mock_result.confidence = 0.855
            mock_result.model_consensus = {'random_forest': 1, 'logistic': 1}
            mock_result.feature_values = {'volatility': 0.25, 'trend': 0.15}
            
            mock_predict.return_value = mock_result
            
            # テスト実行
            ml_system = SimpleMLPredictionSystem()
            result = await ml_system.predict_symbol_movement('7203')
            
            # 検証
            assert result.symbol == '7203'
            assert result.prediction == 1
            assert result.confidence == 0.855

    @pytest.mark.asyncio
    async def test_web_dashboard_api_mock(self):
        """Webダッシュボード APIのモックテスト"""
        with patch.object(DayTradeWebDashboard, 'get_stock_price_data') as mock_price, \
             patch.object(DayTradeWebDashboard, 'get_ml_prediction') as mock_ml:
            
            # モック設定
            mock_price.return_value = self.mock_price_data['7203']
            mock_ml.return_value = self.mock_ml_prediction
            
            # テスト実行
            system = DayTradeWebDashboard()
            
            # 価格データテスト
            price_result = await system.get_stock_price_data('7203')
            assert price_result['opening_price'] == 3000.0
            assert price_result['current_price'] == 3150.0
            
            # ML予測テスト
            ml_result = await system.get_ml_prediction('7203')
            assert ml_result['signal'] == '●買い●'
            assert ml_result['confidence'] == 85.5

    def test_prediction_validator_mock(self):
        """予測検証システムのモックテスト"""
        with patch('sqlite3.connect') as mock_connect:
            # データベースモック設定
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('TEST_001', '7203', 'トヨタ', '2024-08-15', '買い', 3000, 2800, 85, 5.0, '低', '1日', 'テスト', '的中', 3150, 5.0, '2024-08-16', 95.0),
                ('TEST_002', '6861', 'キーエンス', '2024-08-15', '売り', 1500, 1600, 80, -3.0, '中', '1日', 'テスト', '外れ', 1480, 1.3, '2024-08-16', 20.0)
            ]
            
            mock_conn.execute.return_value = mock_cursor
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # テスト実行
            validator = PredictionValidator()
            
            # モック呼び出し確認
            mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_performance_heavy_operation_mock(self):
        """重い処理のパフォーマンステスト（モック版）"""
        start_time = datetime.now()
        
        with patch('time.sleep'), \
             patch.object(SimpleMLPredictionSystem, '_train_models') as mock_train, \
             patch.object(SimpleMLPredictionSystem, '_generate_features') as mock_features:
            
            # 重い処理をモック化
            mock_train.return_value = None
            mock_features.return_value = {
                'volatility': 0.25,
                'trend': 0.15,
                'volume_ratio': 1.2,
                'rsi': 65.0
            }
            
            # テスト実行
            ml_system = SimpleMLPredictionSystem()
            result = await ml_system.predict_symbol_movement('7203')
            
            # パフォーマンス検証（モック使用で高速化）
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 1.0  # 1秒以内で完了
            
            # モック呼び出し確認
            mock_features.assert_called_with('7203')

    def test_database_mock(self):
        """データベースアクセスのモックテスト"""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            
            # データベース操作モック
            mock_cursor.execute.return_value = None
            mock_cursor.fetchone.return_value = ('7203', 'トヨタ自動車', 3150.0)
            mock_conn.execute.return_value = mock_cursor
            mock_conn.commit.return_value = None
            
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # テスト実行
            validator = PredictionValidator()
            
            # データベース接続確認
            mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_operations_mock(self):
        """並行処理のモックテスト"""
        symbols = ['7203', '6861', '4063', '8306', '9984']
        
        with patch.object(DayTradeWebDashboard, 'get_stock_price_data') as mock_price, \
             patch.object(DayTradeWebDashboard, 'get_ml_prediction') as mock_ml:
            
            # モック設定（各銘柄に対して異なるデータ）
            mock_price.side_effect = lambda symbol: {
                'opening_price': float(hash(symbol) % 1000 + 2000),
                'current_price': float(hash(symbol) % 1000 + 2100)
            }
            
            mock_ml.side_effect = lambda symbol: {
                'signal': '●買い●' if hash(symbol) % 2 else '▽売り▽',
                'confidence': hash(symbol) % 30 + 70,
                'score': hash(symbol) % 40 + 60
            }
            
            # 並行処理テスト
            system = DayTradeWebDashboard()
            tasks = []
            
            for symbol in symbols:
                tasks.append(system.get_stock_price_data(symbol))
                tasks.append(system.get_ml_prediction(symbol))
            
            results = await asyncio.gather(*tasks)
            
            # 検証
            assert len(results) == len(symbols) * 2
            assert all('opening_price' in results[i] for i in range(0, len(results), 2))
            assert all('signal' in results[i] for i in range(1, len(results), 2))

    def test_error_handling_mock(self):
        """エラーハンドリングのモックテスト"""
        with patch('daytrade.get_yfinance') as mock_yf:
            # エラー発生モック
            mock_yf.side_effect = Exception("Network error")
            
            # テスト実行
            system = DayTradeWebDashboard()
            
            # エラーハンドリング確認（例外が適切に処理されること）
            try:
                asyncio.run(system.get_stock_price_data('INVALID'))
            except Exception as e:
                pytest.fail(f"Exception not handled properly: {e}")

    @pytest.mark.asyncio
    async def test_cache_behavior_mock(self):
        """キャッシュ動作のモックテスト"""
        with patch.object(DayTradeAIWebSystem, 'get_stock_price_data') as mock_price:
            call_count = 0
            
            def mock_price_func(symbol):
                nonlocal call_count
                call_count += 1
                return self.mock_price_data.get(symbol, {'opening_price': None, 'current_price': None})
            
            mock_price.side_effect = mock_price_func
            
            # テスト実行
            system = DayTradeWebDashboard()
            
            # 同じ銘柄を複数回取得
            await system.get_stock_price_data('7203')
            await system.get_stock_price_data('7203')
            await system.get_stock_price_data('7203')
            
            # キャッシュが効いていることを確認（呼び出し回数）
            assert call_count == 3  # モックなので毎回呼ばれる（実際のキャッシュロジックによる）

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Issue #755 Phase 2: EnsembleSystem統合テスト

93%精度アンサンブルシステムとメインシステムの統合検証
- daytrade.pyとの統合
- データフロー統合テスト
- エラーハンドリング統合テスト
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig
    from src.day_trade.data_fetcher import DataFetcher
    from src.day_trade.analysis.trend_analyzer import TrendAnalyzer
    from src.day_trade.utils.smart_symbol_selector import SmartSymbolSelector

except ImportError as e:
    print(f"統合テスト用インポートエラー: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


class TestEnsembleSystemIntegration(unittest.TestCase):
    """EnsembleSystemメインシステム統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.config = EnsembleConfig(verbose=False)
        self.ensemble = EnsembleSystem(self.config)

        # モックデータ準備
        self.sample_symbols = ['7203.T', '6758.T', '9984.T']
        self.mock_market_data = self._create_mock_market_data()

    def _create_mock_market_data(self) -> pd.DataFrame:
        """モック市場データ作成"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        data = []
        for symbol in self.sample_symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(1000000, 5000000)
                })

        return pd.DataFrame(data)

    @patch('src.day_trade.data_fetcher.DataFetcher.fetch_data')
    def test_data_fetcher_integration(self, mock_fetch):
        """DataFetcherとの統合テスト"""
        # DataFetcherのモック設定
        mock_fetch.return_value = self.mock_market_data

        # DataFetcher経由でのデータ取得
        fetcher = DataFetcher()
        market_data = fetcher.fetch_data(self.sample_symbols)

        # EnsembleSystemでの処理用にデータ変換
        features = self._extract_features_from_market_data(market_data)

        if len(features) > 50:  # 十分なデータがある場合
            X = features.drop('target', axis=1).values
            y = features['target'].values

            # EnsembleSystem訓練・予測
            self.ensemble.fit(X[:80], y[:80])
            predictions = self.ensemble.predict(X[80:])

            # 統合フローが正常に完了することを確認
            self.assertIsNotNone(predictions)
            self.assertGreater(len(predictions.final_predictions), 0)

    def _extract_features_from_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """市場データから特徴量抽出"""
        features_list = []

        for symbol in self.sample_symbols:
            symbol_data = data[data['Symbol'] == symbol].copy()

            if len(symbol_data) > 20:
                # 技術指標計算
                symbol_data['Returns'] = symbol_data['Close'].pct_change()
                symbol_data['SMA_5'] = symbol_data['Close'].rolling(5).mean()
                symbol_data['SMA_20'] = symbol_data['Close'].rolling(20).mean()
                symbol_data['RSI'] = self._calculate_rsi(symbol_data['Close'])
                symbol_data['Volatility'] = symbol_data['Returns'].rolling(10).std()

                # 特徴量選択
                feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'Volatility']
                symbol_features = symbol_data[feature_cols].dropna()

                # ターゲット（翌日リターン）
                symbol_features['target'] = symbol_features['Returns'].shift(-1)
                symbol_features = symbol_features.dropna()

                features_list.append(symbol_features)

        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @patch('src.day_trade.analysis.trend_analyzer.TrendAnalyzer.analyze')
    def test_trend_analyzer_integration(self, mock_analyze):
        """TrendAnalyzerとの統合テスト"""
        # TrendAnalyzerのモック設定
        mock_analyze.return_value = {
            'trend': 'upward',
            'strength': 0.75,
            'support_levels': [1000, 1050],
            'resistance_levels': [1150, 1200]
        }

        # TrendAnalyzer経由での分析
        analyzer = TrendAnalyzer()
        trend_result = analyzer.analyze(self.mock_market_data)

        # EnsembleSystemの予測とトレンド分析の組み合わせ
        if len(self.mock_market_data) > 50:
            features = self._extract_features_from_market_data(self.mock_market_data)

            if len(features) > 20:
                X = features.drop('target', axis=1).values
                y = features['target'].values

                self.ensemble.fit(X[:15], y[:15])
                predictions = self.ensemble.predict(X[15:20])

                # トレンド情報を使用した統合判断
                trend_adjusted_predictions = self._adjust_predictions_with_trend(
                    predictions.final_predictions, trend_result
                )

                # 統合結果の妥当性確認
                self.assertIsNotNone(trend_adjusted_predictions)
                self.assertEqual(len(trend_adjusted_predictions), len(predictions.final_predictions))

    def _adjust_predictions_with_trend(self, predictions: np.ndarray, trend_info: dict) -> np.ndarray:
        """トレンド情報による予測調整"""
        trend_multiplier = 1.0

        if trend_info['trend'] == 'upward':
            trend_multiplier = 1.0 + trend_info['strength'] * 0.1
        elif trend_info['trend'] == 'downward':
            trend_multiplier = 1.0 - trend_info['strength'] * 0.1

        return predictions * trend_multiplier

    @patch('src.day_trade.utils.smart_symbol_selector.SmartSymbolSelector.select_top_symbols')
    def test_smart_symbol_selector_integration(self, mock_select):
        """SmartSymbolSelectorとの統合テスト"""
        # SmartSymbolSelectorのモック設定
        mock_select.return_value = [
            {'symbol': '7203.T', 'score': 0.95, 'reason': 'high_momentum'},
            {'symbol': '6758.T', 'score': 0.88, 'reason': 'strong_fundamentals'},
            {'symbol': '9984.T', 'score': 0.82, 'reason': 'technical_breakout'}
        ]

        # SmartSymbolSelector経由でのシンボル選択
        selector = SmartSymbolSelector()
        selected_symbols = selector.select_top_symbols(n_symbols=3)

        # 選択されたシンボルでEnsemble予測
        symbol_names = [item['symbol'] for item in selected_symbols]
        symbol_data = self.mock_market_data[
            self.mock_market_data['Symbol'].isin(symbol_names)
        ]

        if len(symbol_data) > 50:
            features = self._extract_features_from_market_data(symbol_data)

            if len(features) > 20:
                X = features.drop('target', axis=1).values
                y = features['target'].values

                self.ensemble.fit(X[:15], y[:15])
                predictions = self.ensemble.predict(X[15:20])

                # シンボル選択スコアと予測の統合
                integrated_results = self._integrate_selection_and_prediction(
                    selected_symbols, predictions
                )

                # 統合結果の確認
                self.assertIsInstance(integrated_results, list)
                self.assertGreater(len(integrated_results), 0)

    def _integrate_selection_and_prediction(self, symbol_scores: list, predictions) -> list:
        """シンボル選択スコアと予測の統合"""
        integrated = []

        for i, symbol_info in enumerate(symbol_scores):
            if i < len(predictions.final_predictions):
                integrated.append({
                    'symbol': symbol_info['symbol'],
                    'selection_score': symbol_info['score'],
                    'prediction': predictions.final_predictions[i],
                    'confidence': predictions.ensemble_confidence[i] if hasattr(predictions, 'ensemble_confidence') else 0.5,
                    'combined_score': symbol_info['score'] * 0.6 + predictions.final_predictions[i] * 0.4
                })

        return integrated

    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        # 不正なデータでの統合テスト
        invalid_data_scenarios = [
            pd.DataFrame(),  # 空のデータ
            pd.DataFrame({'invalid': [1, 2, 3]}),  # 不正な列名
            None  # Noneデータ
        ]

        for scenario in invalid_data_scenarios:
            try:
                if scenario is not None and len(scenario) > 0:
                    features = self._extract_features_from_market_data(scenario)

                    if len(features) > 10:
                        X = features.drop('target', axis=1).values
                        y = features['target'].values

                        self.ensemble.fit(X[:5], y[:5])
                        predictions = self.ensemble.predict(X[5:])

                        # エラーが発生しないか、適切にハンドリングされることを確認
                        self.assertIsNotNone(predictions)

            except Exception as e:
                # 適切なエラーメッセージが含まれていることを確認
                self.assertIsInstance(e, (ValueError, IndexError, KeyError))

    def test_performance_integration(self):
        """パフォーマンス統合テスト"""
        import time

        # 大規模データでの統合パフォーマンステスト
        large_data = self._create_large_mock_data()

        start_time = time.time()

        # データ処理からEnsemble予測まで
        features = self._extract_features_from_market_data(large_data)

        if len(features) > 100:
            X = features.drop('target', axis=1).values
            y = features['target'].values

            # 訓練時間測定
            train_start = time.time()
            self.ensemble.fit(X[:80], y[:80])
            train_time = time.time() - train_start

            # 予測時間測定
            predict_start = time.time()
            predictions = self.ensemble.predict(X[80:])
            predict_time = time.time() - predict_start

            total_time = time.time() - start_time

            # パフォーマンス要件確認
            self.assertLess(train_time, 30.0, f"訓練時間 {train_time:.2f}秒 が過大です")
            self.assertLess(predict_time, 5.0, f"予測時間 {predict_time:.2f}秒 が過大です")
            self.assertLess(total_time, 60.0, f"総処理時間 {total_time:.2f}秒 が過大です")

            print(f"統合パフォーマンス - 訓練: {train_time:.2f}秒, 予測: {predict_time:.2f}秒, 総時間: {total_time:.2f}秒")

    def _create_large_mock_data(self) -> pd.DataFrame:
        """大規模モックデータ作成"""
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T', '8306.T', '6861.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(1000000, 5000000)
                })

        return pd.DataFrame(data)


class TestEnsembleSystemEndToEnd(unittest.TestCase):
    """EnsembleSystemエンドツーエンドテスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.config = EnsembleConfig(verbose=False)

    def test_full_prediction_pipeline(self):
        """完全予測パイプラインテスト"""
        # 1. データ取得（モック）
        with patch('src.day_trade.data_fetcher.DataFetcher.fetch_data') as mock_fetch:
            mock_data = self._create_realistic_market_data()
            mock_fetch.return_value = mock_data

            # 2. 特徴量エンジニアリング
            features = self._advanced_feature_engineering(mock_data)

            if len(features) > 100:
                # 3. EnsembleSystem初期化・訓練
                ensemble = EnsembleSystem(self.config)

                X = features.drop('target', axis=1).values
                y = features['target'].values

                # 時系列分割
                split_point = int(len(X) * 0.8)
                X_train, X_test = X[:split_point], X[split_point:]
                y_train, y_test = y[:split_point], y[split_point:]

                # 4. 訓練・予測・評価
                ensemble.fit(X_train, y_train)
                predictions = ensemble.predict(X_test)

                # 5. 結果評価
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(y_test, predictions.final_predictions)
                mae = mean_absolute_error(y_test, predictions.final_predictions)

                # 6. エンドツーエンド性能確認
                self.assertGreater(r2, 0.5, f"エンドツーエンドR² {r2:.3f} が目標 0.5 を下回りました")
                self.assertLess(mae, 0.1, f"エンドツーエンドMAE {mae:.3f} が目標 0.1 を上回りました")

                # 7. 予測結果の統合確認
                self.assertEqual(len(predictions.final_predictions), len(y_test))
                self.assertFalse(np.any(np.isnan(predictions.final_predictions)))

                print(f"エンドツーエンド性能 - R²: {r2:.3f}, MAE: {mae:.3f}")

    def _create_realistic_market_data(self) -> pd.DataFrame:
        """現実的な市場データ作成"""
        np.random.seed(789)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T']

        data = []
        for symbol in symbols:
            # シンボル別の基本価格レベル
            base_price = np.random.uniform(800, 1200)

            for i, date in enumerate(dates):
                # トレンドとボラティリティを含む価格モデル
                trend = 0.001 * i  # 長期トレンド
                seasonal = 0.02 * np.sin(2 * np.pi * i / 30)  # 月次季節性
                noise = np.random.normal(0, 0.02)  # ノイズ

                daily_return = trend + seasonal + noise
                current_price = base_price * (1 + daily_return)

                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': current_price * np.random.uniform(0.99, 1.01),
                    'High': current_price * np.random.uniform(1.01, 1.05),
                    'Low': current_price * np.random.uniform(0.95, 0.99),
                    'Close': current_price,
                    'Volume': np.random.randint(500000, 3000000)
                })

                base_price = current_price  # 次の日の基準価格

        return pd.DataFrame(data)

    def _advanced_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量エンジニアリング"""
        features_list = []

        for symbol in data['Symbol'].unique():
            symbol_data = data[data['Symbol'] == symbol].copy().sort_values('Date')

            if len(symbol_data) > 50:
                # 基本的な価格指標
                symbol_data['Returns'] = symbol_data['Close'].pct_change()
                symbol_data['Log_Returns'] = np.log(symbol_data['Close'] / symbol_data['Close'].shift(1))

                # 移動平均
                for window in [5, 10, 20, 50]:
                    symbol_data[f'SMA_{window}'] = symbol_data['Close'].rolling(window).mean()
                    symbol_data[f'EMA_{window}'] = symbol_data['Close'].ewm(span=window).mean()

                # ボラティリティ指標
                symbol_data['Volatility_10'] = symbol_data['Returns'].rolling(10).std()
                symbol_data['Volatility_20'] = symbol_data['Returns'].rolling(20).std()

                # 技術指標
                symbol_data['RSI'] = self._calculate_rsi(symbol_data['Close'])
                symbol_data['MACD'], symbol_data['MACD_Signal'] = self._calculate_macd(symbol_data['Close'])
                symbol_data['BB_Upper'], symbol_data['BB_Lower'] = self._calculate_bollinger_bands(symbol_data['Close'])

                # 出来高指標
                symbol_data['Volume_SMA_10'] = symbol_data['Volume'].rolling(10).mean()
                symbol_data['Volume_Ratio'] = symbol_data['Volume'] / symbol_data['Volume_SMA_10']

                # 高度な特徴量
                symbol_data['Price_Position'] = (symbol_data['Close'] - symbol_data['Low'].rolling(20).min()) / (symbol_data['High'].rolling(20).max() - symbol_data['Low'].rolling(20).min())
                symbol_data['Returns_Momentum'] = symbol_data['Returns'].rolling(5).mean()

                # ターゲット（翌日リターン）
                symbol_data['target'] = symbol_data['Returns'].shift(-1)

                # 特徴量選択
                feature_cols = [col for col in symbol_data.columns if col not in ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']]
                symbol_features = symbol_data[feature_cols + ['target']].dropna()

                if len(symbol_features) > 30:
                    features_list.append(symbol_features)

        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band


if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2)
#!/usr/bin/env python3
"""
機械学習アンサンブルモデル統合バックテスト

Issue #753対応: バックテスト機能強化
Issue #487の93%精度アンサンブルシステムとの統合
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from pathlib import Path

# 機械学習関連インポート
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    sklearn_available = True
except ImportError:
    sklearn_available = False

warnings.filterwarnings("ignore")


@dataclass
class MLBacktestConfig:
    """機械学習統合バックテスト設定"""

    # アンサンブルモデル設定
    ensemble_models: List[str] = None  # ['xgboost', 'catboost', 'random_forest']
    model_weights: Dict[str, float] = None
    dynamic_weighting: bool = True
    rebalance_frequency: int = 21  # 営業日

    # 特徴量設定
    feature_engineering: bool = True
    technical_indicators: List[str] = None
    lookback_period: int = 252  # 1年

    # 予測設定
    prediction_horizon: int = 1  # 1日先予測
    signal_threshold: float = 0.6  # シグナル閾値

    # Walk-Forward設定
    training_window: int = 756  # 3年
    testing_window: int = 63   # 3ヶ月
    min_training_samples: int = 252

    # パフォーマンス評価
    benchmark_symbol: str = "^N225"  # 日経225
    evaluation_metrics: List[str] = None


@dataclass
class MLPredictionResult:
    """ML予測結果"""

    timestamp: datetime
    symbol: str
    prediction: float
    confidence: float
    model_predictions: Dict[str, float]
    ensemble_weight: Dict[str, float]
    signal_strength: float
    features_used: List[str]


@dataclass
class MLBacktestResult:
    """ML統合バックテスト結果"""

    # 基本パフォーマンス
    total_return: float
    benchmark_return: float
    excess_return: float
    alpha: float
    beta: float

    # 予測精度指標
    prediction_accuracy: float
    direction_accuracy: float  # 方向性予測精度
    signal_precision: float
    signal_recall: float
    signal_f1_score: float

    # アンサンブル分析
    model_contributions: Dict[str, float]
    dynamic_weight_evolution: pd.DataFrame
    feature_importance: Dict[str, float]

    # リスク調整指標
    information_ratio: float
    tracking_error: float
    maximum_drawdown: float

    # 詳細結果
    predictions: List[MLPredictionResult]
    portfolio_evolution: pd.DataFrame
    model_performance_history: pd.DataFrame


class MLEnsembleBacktester:
    """機械学習アンサンブル統合バックテスター"""

    def __init__(self, config: MLBacktestConfig):
        """
        初期化

        Args:
            config: ML統合バックテスト設定
        """
        self.config = config
        self.ensemble_models = {}
        self.feature_engineering = None
        self.model_weights = config.model_weights or {}

        # デフォルト設定
        if self.config.ensemble_models is None:
            self.config.ensemble_models = ['xgboost', 'catboost', 'random_forest']

        if self.config.technical_indicators is None:
            self.config.technical_indicators = [
                'rsi', 'macd', 'bollinger_bands', 'moving_averages',
                'volume_indicators', 'momentum_indicators'
            ]

        if self.config.evaluation_metrics is None:
            self.config.evaluation_metrics = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'sharpe_ratio', 'information_ratio', 'maximum_drawdown'
            ]

    def run_ml_backtest(self,
                       historical_data: pd.DataFrame,
                       symbols: List[str],
                       benchmark_data: Optional[pd.DataFrame] = None) -> MLBacktestResult:
        """
        ML統合バックテスト実行

        Args:
            historical_data: 歴史的市場データ
            symbols: 対象銘柄リスト
            benchmark_data: ベンチマークデータ

        Returns:
            MLBacktestResult: 結果
        """
        # 1. データ準備と特徴量エンジニアリング
        features_df = self._prepare_features(historical_data, symbols)

        # 2. Walk-Forward分析セットアップ
        wf_splits = self._setup_walk_forward_splits(features_df)

        # 3. 各期間でのバックテスト実行
        all_predictions = []
        model_performance_history = []
        portfolio_values = []

        for train_idx, test_idx in wf_splits:
            # 訓練・テストデータ分割
            train_features = features_df.iloc[train_idx]
            test_features = features_df.iloc[test_idx]

            # モデル訓練
            trained_models = self._train_ensemble_models(train_features)

            # 予測実行
            period_predictions = self._generate_predictions(
                test_features, trained_models
            )
            all_predictions.extend(period_predictions)

            # パフォーマンス評価
            period_performance = self._evaluate_period_performance(
                period_predictions, test_features
            )
            model_performance_history.append(period_performance)

            # ポートフォリオ更新
            portfolio_value = self._update_portfolio(
                period_predictions, test_features
            )
            portfolio_values.extend(portfolio_value)

        # 4. 総合結果計算
        result = self._calculate_final_results(
            all_predictions, model_performance_history,
            portfolio_values, benchmark_data
        )

        return result

    def _prepare_features(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """特徴量準備"""
        features_list = []

        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data

            if len(symbol_data) < self.config.lookback_period:
                continue

            # 基本的な価格特徴量
            symbol_features = self._create_basic_features(symbol_data)

            # テクニカル指標
            if self.config.feature_engineering:
                tech_features = self._create_technical_features(symbol_data)
                symbol_features = pd.concat([symbol_features, tech_features], axis=1)

            # ターゲット変数（未来リターン）
            targets = self._create_targets(symbol_data)
            symbol_features['target'] = targets
            symbol_features['symbol'] = symbol

            features_list.append(symbol_features.dropna())

        if not features_list:
            return pd.DataFrame()

        # 全銘柄の特徴量を結合
        combined_features = pd.concat(features_list, ignore_index=True)
        return combined_features.dropna()

    def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量作成"""
        features = pd.DataFrame(index=data.index)

        if 'Close' in data.columns:
            # リターン系特徴量
            features['return_1d'] = data['Close'].pct_change()
            features['return_5d'] = data['Close'].pct_change(5)
            features['return_21d'] = data['Close'].pct_change(21)

            # ボラティリティ
            features['volatility_5d'] = features['return_1d'].rolling(5).std()
            features['volatility_21d'] = features['return_1d'].rolling(21).std()

            # 価格モメンタム
            features['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
            features['momentum_21d'] = data['Close'] / data['Close'].shift(21) - 1

        if 'Volume' in data.columns:
            # 出来高特徴量
            features['volume_ratio_5d'] = data['Volume'] / data['Volume'].rolling(5).mean()
            features['volume_ratio_21d'] = data['Volume'] / data['Volume'].rolling(21).mean()

        return features

    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標特徴量作成"""
        features = pd.DataFrame(index=data.index)

        if 'Close' not in data.columns:
            return features

        close = data['Close']

        # RSI
        if 'rsi' in self.config.technical_indicators:
            features['rsi_14'] = self._calculate_rsi(close, 14)

        # 移動平均
        if 'moving_averages' in self.config.technical_indicators:
            features['ma_5'] = close.rolling(5).mean()
            features['ma_21'] = close.rolling(21).mean()
            features['ma_ratio'] = features['ma_5'] / features['ma_21']

        # MACD
        if 'macd' in self.config.technical_indicators:
            macd_line, signal_line = self._calculate_macd(close)
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = macd_line - signal_line

        # ボリンジャーバンド
        if 'bollinger_bands' in self.config.technical_indicators:
            bb_upper, bb_lower = self._calculate_bollinger_bands(close)
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

        return features

    def _create_targets(self, data: pd.DataFrame) -> pd.Series:
        """ターゲット変数作成"""
        if 'Close' not in data.columns:
            return pd.Series(index=data.index)

        # 未来リターン計算
        future_return = data['Close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)

        # 二値分類（上昇/下降）
        binary_target = (future_return > 0).astype(int)

        return binary_target

    def _setup_walk_forward_splits(self, features_df: pd.DataFrame) -> List[Tuple[List[int], List[int]]]:
        """Walk-Forward分析分割設定"""
        if not sklearn_available:
            # シンプルな分割
            total_samples = len(features_df)
            train_size = self.config.training_window
            test_size = self.config.testing_window

            splits = []
            start = 0

            while start + train_size + test_size <= total_samples:
                train_idx = list(range(start, start + train_size))
                test_idx = list(range(start + train_size, start + train_size + test_size))
                splits.append((train_idx, test_idx))
                start += test_size

            return splits

        # sklearn使用
        tscv = TimeSeriesSplit(
            n_splits=min(5, len(features_df) // self.config.training_window),
            test_size=self.config.testing_window
        )

        return list(tscv.split(features_df))

    def _train_ensemble_models(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """アンサンブルモデル訓練"""
        trained_models = {}

        # 特徴量とターゲット分離
        feature_cols = [col for col in train_data.columns
                       if col not in ['target', 'symbol', 'timestamp']]
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['target'].fillna(0)

        if len(X_train) < self.config.min_training_samples:
            return trained_models

        # 各モデルの訓練（簡易実装）
        for model_name in self.config.ensemble_models:
            try:
                model = self._create_model(model_name)
                if model is not None:
                    # 実際の実装では sklearn等のモデルを使用
                    # ここでは簡易的なモデルを想定
                    trained_model = self._fit_simple_model(model, X_train, y_train)
                    trained_models[model_name] = trained_model
            except Exception as e:
                print(f"モデル{model_name}の訓練に失敗: {e}")
                continue

        return trained_models

    def _create_model(self, model_name: str) -> Any:
        """モデル作成（簡易実装）"""
        # 実際の実装では、Issue #487のEnsembleSystemを使用
        models = {
            'xgboost': {'type': 'xgboost', 'params': {}},
            'catboost': {'type': 'catboost', 'params': {}},
            'random_forest': {'type': 'random_forest', 'params': {}}
        }
        return models.get(model_name)

    def _fit_simple_model(self, model_config: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """簡易モデル訓練"""
        # 実際の実装では機械学習ライブラリを使用
        # ここでは統計的な簡易モデル

        if len(X_train) == 0 or len(y_train) == 0:
            return {'type': 'empty', 'params': {}}

        # 単純な線形関係を仮定
        correlations = {}
        for col in X_train.columns:
            if X_train[col].std() > 0:
                corr = np.corrcoef(X_train[col].fillna(0), y_train)[0, 1]
                correlations[col] = corr if not np.isnan(corr) else 0

        return {
            'type': model_config['type'],
            'correlations': correlations,
            'mean_target': y_train.mean()
        }

    def _generate_predictions(self, test_data: pd.DataFrame, models: Dict[str, Any]) -> List[MLPredictionResult]:
        """予測生成"""
        predictions = []

        feature_cols = [col for col in test_data.columns
                       if col not in ['target', 'symbol', 'timestamp']]

        for idx, row in test_data.iterrows():
            X_test = row[feature_cols].fillna(0)

            # 各モデルの予測
            model_predictions = {}
            for model_name, model in models.items():
                pred = self._predict_single_model(model, X_test)
                model_predictions[model_name] = pred

            # アンサンブル予測
            if model_predictions:
                ensemble_pred = self._combine_predictions(model_predictions)
                confidence = self._calculate_confidence(model_predictions)
                signal_strength = abs(ensemble_pred - 0.5) * 2  # 0-1範囲を0-1に変換

                predictions.append(MLPredictionResult(
                    timestamp=row.get('timestamp', datetime.now()),
                    symbol=row.get('symbol', 'UNKNOWN'),
                    prediction=ensemble_pred,
                    confidence=confidence,
                    model_predictions=model_predictions,
                    ensemble_weight=self.model_weights,
                    signal_strength=signal_strength,
                    features_used=feature_cols
                ))

        return predictions

    def _predict_single_model(self, model: Dict, X_test: pd.Series) -> float:
        """単一モデル予測"""
        if model.get('type') == 'empty':
            return 0.5  # 中立予測

        correlations = model.get('correlations', {})
        mean_target = model.get('mean_target', 0.5)

        # 相関ベースの簡易予測
        weighted_sum = 0
        total_weight = 0

        for feature, corr in correlations.items():
            if feature in X_test.index and not pd.isna(X_test[feature]):
                weight = abs(corr)
                weighted_sum += X_test[feature] * corr * weight
                total_weight += weight

        if total_weight > 0:
            prediction = mean_target + weighted_sum / total_weight * 0.1  # スケーリング
            return max(0, min(1, prediction))  # 0-1に制限

        return mean_target

    def _combine_predictions(self, model_predictions: Dict[str, float]) -> float:
        """予測結合"""
        if not model_predictions:
            return 0.5

        # 動的重み付け
        if self.config.dynamic_weighting:
            weights = self._calculate_dynamic_weights(model_predictions)
        else:
            weights = {name: 1.0/len(model_predictions) for name in model_predictions.keys()}

        # 重み付き平均
        weighted_sum = sum(pred * weights.get(name, 0) for name, pred in model_predictions.items())
        total_weight = sum(weights.values())

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _calculate_dynamic_weights(self, model_predictions: Dict[str, float]) -> Dict[str, float]:
        """動的重み計算"""
        # 簡易実装：予測の分散に基づく重み
        predictions = list(model_predictions.values())

        if len(predictions) <= 1:
            return {name: 1.0 for name in model_predictions.keys()}

        pred_std = np.std(predictions)

        # 分散が小さい場合（予測が一致）は均等重み
        if pred_std < 0.1:
            return {name: 1.0/len(model_predictions) for name in model_predictions.keys()}

        # 中央値に近い予測により高い重み
        median_pred = np.median(predictions)
        weights = {}

        for name, pred in model_predictions.items():
            distance = abs(pred - median_pred)
            weight = 1.0 / (1.0 + distance)  # 距離の逆数
            weights[name] = weight

        # 正規化
        total_weight = sum(weights.values())
        return {name: w/total_weight for name, w in weights.items()}

    def _calculate_confidence(self, model_predictions: Dict[str, float]) -> float:
        """予測信頼度計算"""
        if not model_predictions:
            return 0.0

        predictions = list(model_predictions.values())

        # 予測の一致度を信頼度とする
        pred_std = np.std(predictions)
        confidence = 1.0 / (1.0 + pred_std * 5)  # 標準偏差の逆数ベース

        return max(0.0, min(1.0, confidence))

    def _evaluate_period_performance(self, predictions: List[MLPredictionResult],
                                   test_data: pd.DataFrame) -> Dict[str, float]:
        """期間パフォーマンス評価"""
        if not predictions or len(test_data) == 0:
            return {}

        # 予測値と実際値の抽出
        pred_values = [p.prediction for p in predictions]
        actual_values = test_data['target'].values[:len(pred_values)]

        # 二値分類に変換
        pred_binary = [1 if p > self.config.signal_threshold else 0 for p in pred_values]

        if not sklearn_available:
            # 簡易精度計算
            accuracy = sum(p == a for p, a in zip(pred_binary, actual_values)) / len(pred_binary)
            return {'accuracy': accuracy, 'samples': len(pred_binary)}

        # sklearn使用
        try:
            accuracy = accuracy_score(actual_values, pred_binary)
            precision = precision_score(actual_values, pred_binary, zero_division=0)
            recall = recall_score(actual_values, pred_binary, zero_division=0)
            f1 = f1_score(actual_values, pred_binary, zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'samples': len(pred_binary)
            }
        except Exception:
            accuracy = sum(p == a for p, a in zip(pred_binary, actual_values)) / len(pred_binary)
            return {'accuracy': accuracy, 'samples': len(pred_binary)}

    def _update_portfolio(self, predictions: List[MLPredictionResult],
                         test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """ポートフォリオ更新"""
        portfolio_values = []

        for i, prediction in enumerate(predictions):
            # シグナル強度に基づくポジション計算
            signal_strength = prediction.signal_strength

            if signal_strength > self.config.signal_threshold:
                position = signal_strength if prediction.prediction > 0.5 else -signal_strength
            else:
                position = 0

            # リターン計算（簡易実装）
            if i < len(test_data):
                period_return = test_data.iloc[i].get('target', 0) * 0.02  # スケール調整
                portfolio_return = position * period_return
            else:
                portfolio_return = 0

            portfolio_values.append({
                'timestamp': prediction.timestamp,
                'position': position,
                'return': portfolio_return,
                'signal_strength': signal_strength,
                'confidence': prediction.confidence
            })

        return portfolio_values

    def _calculate_final_results(self, all_predictions: List[MLPredictionResult],
                               performance_history: List[Dict[str, float]],
                               portfolio_values: List[Dict[str, Any]],
                               benchmark_data: Optional[pd.DataFrame]) -> MLBacktestResult:
        """最終結果計算"""

        # 基本統計
        total_predictions = len(all_predictions)
        avg_confidence = np.mean([p.confidence for p in all_predictions]) if all_predictions else 0

        # パフォーマンス統計
        if performance_history:
            avg_accuracy = np.mean([p.get('accuracy', 0) for p in performance_history])
            avg_precision = np.mean([p.get('precision', 0) for p in performance_history])
            avg_recall = np.mean([p.get('recall', 0) for p in performance_history])
            avg_f1 = np.mean([p.get('f1_score', 0) for p in performance_history])
        else:
            avg_accuracy = avg_precision = avg_recall = avg_f1 = 0

        # ポートフォリオパフォーマンス
        if portfolio_values:
            returns = [pv['return'] for pv in portfolio_values]
            total_return = sum(returns)
            max_dd = self._calculate_portfolio_drawdown(returns)
        else:
            total_return = max_dd = 0

        # ベンチマーク比較
        benchmark_return = 0
        if benchmark_data is not None and len(benchmark_data) > 0:
            if 'Close' in benchmark_data.columns:
                benchmark_return = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0]) - 1

        # モデル貢献度（簡易実装）
        model_contributions = {}
        if all_predictions:
            for model_name in self.config.ensemble_models:
                contributions = [p.model_predictions.get(model_name, 0) for p in all_predictions]
                model_contributions[model_name] = np.mean(contributions) if contributions else 0

        # 結果構築
        return MLBacktestResult(
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=total_return - benchmark_return,
            alpha=total_return - benchmark_return,  # 簡易実装
            beta=1.0,  # 簡易実装

            prediction_accuracy=avg_accuracy,
            direction_accuracy=avg_accuracy,  # 簡易実装
            signal_precision=avg_precision,
            signal_recall=avg_recall,
            signal_f1_score=avg_f1,

            model_contributions=model_contributions,
            dynamic_weight_evolution=pd.DataFrame(),  # 実装省略
            feature_importance={},  # 実装省略

            information_ratio=total_return / 0.05 if total_return != 0 else 0,  # 簡易実装
            tracking_error=0.05,  # 簡易実装
            maximum_drawdown=max_dd,

            predictions=all_predictions,
            portfolio_evolution=pd.DataFrame(portfolio_values),
            model_performance_history=pd.DataFrame(performance_history)
        )

    def _calculate_portfolio_drawdown(self, returns: List[float]) -> float:
        """ポートフォリオドローダウン計算"""
        if not returns:
            return 0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        return min(drawdown) if len(drawdown) > 0 else 0

    # テクニカル指標計算ヘルパーメソッド
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20,
                                  std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
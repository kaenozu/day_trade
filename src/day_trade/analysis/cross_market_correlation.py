#!/usr/bin/env python3
"""
Global Trading Engine - Cross Market Correlation Analysis
クロスマーケット相関分析エンジン

Forex、Crypto、Stock間の動的相関・因果関係分析
"""

import asyncio
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from ..data.forex_data_collector import ForexDataCollector
from ..data.crypto_data_collector import CryptoDataCollector
from ..models.database import get_session
from ..models.global_models import CrossMarketCorrelation, MarketType

logger = get_context_logger(__name__)

@dataclass
class CorrelationResult:
    """相関分析結果"""
    asset1: str
    asset2: str
    market1: str
    market2: str

    # 相関係数
    pearson_correlation: float
    spearman_correlation: float
    kendall_correlation: float

    # 時系列分析
    rolling_correlation_1h: List[float]
    rolling_correlation_4h: List[float]
    rolling_correlation_1d: List[float]

    # 統計的有意性
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int

    # グレンジャー因果性
    granger_causality_1to2: Optional[float] = None
    granger_causality_2to1: Optional[float] = None

    # 動的相関
    dynamic_correlation: Optional[List[float]] = None

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class MarketRegimeChange:
    """市場レジーム変化検出結果"""
    asset: str
    market: str

    # レジーム変化点
    change_points: List[datetime]
    regime_labels: List[str]  # "low_vol", "high_vol", "trending", "ranging"

    # レジーム特性
    regime_correlations: Dict[str, Dict[str, float]]
    volatility_regimes: Dict[str, float]

    # 信頼度
    detection_confidence: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class DynamicCorrelationModel(nn.Module):
    """動的相関予測ニューラルネットワーク"""

    def __init__(self, input_size: int = 10, hidden_size: int = 64, sequence_length: int = 24):
        super().__init__()
        self.sequence_length = sequence_length

        # 特徴抽出
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2)
        )

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 相関予測ヘッド
        self.correlation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # -1 to 1
        )

        # 信頼度ヘッド
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 0 to 1
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_size]
        """
        batch_size, seq_len, _ = x.shape

        # 特徴抽出
        features = self.feature_extractor(x)  # [batch_size, seq_len, hidden_size//2]

        # LSTM処理
        lstm_out, _ = self.lstm(features)  # [batch_size, seq_len, hidden_size]

        # 最終状態
        final_state = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # 予測
        correlation = self.correlation_head(final_state)
        confidence = self.confidence_head(final_state)

        return correlation.squeeze(), confidence.squeeze()

class CrossMarketCorrelationEngine:
    """クロスマーケット相関分析エンジン"""

    def __init__(self, forex_collector: Optional[ForexDataCollector] = None,
                 crypto_collector: Optional[CryptoDataCollector] = None):
        self.forex_collector = forex_collector
        self.crypto_collector = crypto_collector

        # データキャッシュ
        self.price_data_cache: Dict[str, pd.DataFrame] = {}
        self.correlation_cache: Dict[str, CorrelationResult] = {}

        # 動的相関モデル
        self.dynamic_model = DynamicCorrelationModel()
        self.model_trained = False

        # 分析設定
        self.correlation_windows = {
            '1h': 60,    # 1時間 = 60分
            '4h': 240,   # 4時間 = 240分
            '1d': 1440,  # 1日 = 1440分
            '1w': 10080  # 1週間 = 10080分
        }

        # 監視対象ペア
        self.asset_pairs = [
            # Forex - Crypto
            ("EUR/USD", "BTCUSDT"),
            ("USD/JPY", "ETHUSDT"),
            ("GBP/USD", "BTCUSDT"),
            ("AUD/USD", "ETHUSDT"),

            # Crypto - Crypto
            ("BTCUSDT", "ETHUSDT"),
            ("BTCUSDT", "BNBUSDT"),
            ("ETHUSDT", "ADAUSDT"),

            # Forex - Forex
            ("EUR/USD", "GBP/USD"),
            ("USD/JPY", "AUD/USD"),
            ("EUR/GBP", "GBP/JPY")
        ]

        logger.info(f"Cross Market Correlation Engine initialized with {len(self.asset_pairs)} pairs")

    async def analyze_all_correlations(self) -> List[CorrelationResult]:
        """全ペア相関分析実行"""
        results = []

        logger.info("Starting comprehensive cross-market correlation analysis...")

        # 並列実行でパフォーマンス向上
        tasks = []
        for asset1, asset2 in self.asset_pairs:
            tasks.append(self._analyze_pair_correlation(asset1, asset2))

        pair_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(pair_results):
            if isinstance(result, CorrelationResult):
                results.append(result)
                # データベース保存
                try:
                    await self._save_correlation_result(result)
                except Exception as e:
                    logger.error(f"Failed to save correlation result: {e}")
            elif isinstance(result, Exception):
                asset1, asset2 = self.asset_pairs[i]
                logger.error(f"Correlation analysis failed for {asset1}-{asset2}: {result}")
            else:
                asset1, asset2 = self.asset_pairs[i]
                logger.warning(f"No correlation result for {asset1}-{asset2}")

        logger.info(f"Completed correlation analysis for {len(results)} pairs")
        return results

    async def _analyze_pair_correlation(self, asset1: str, asset2: str) -> Optional[CorrelationResult]:
        """ペア相関分析"""
        try:
            # データ取得
            data1 = await self._get_asset_data(asset1)
            data2 = await self._get_asset_data(asset2)

            if data1 is None or data2 is None or len(data1) < 100 or len(data2) < 100:
                logger.warning(f"Insufficient data for {asset1}-{asset2} correlation")
                return None

            # データ同期・リサンプリング
            aligned_data = self._align_price_data(data1, data2)
            if aligned_data is None:
                return None

            prices1, prices2 = aligned_data

            # 基本相関計算
            pearson_corr, pearson_p = pearsonr(prices1, prices2)
            spearman_corr, _ = spearmanr(prices1, prices2)
            kendall_corr, _ = stats.kendalltau(prices1, prices2)

            # 信頼区間計算
            n = len(prices1)
            confidence_interval = self._calculate_confidence_interval(pearson_corr, n)

            # ローリング相関計算
            rolling_correlations = self._calculate_rolling_correlations(prices1, prices2)

            # グレンジャー因果性テスト
            granger_1to2, granger_2to1 = self._granger_causality_test(prices1, prices2)

            # 動的相関予測
            dynamic_correlation = self._predict_dynamic_correlation(prices1, prices2)

            # 市場タイプ判定
            market1 = self._determine_market_type(asset1)
            market2 = self._determine_market_type(asset2)

            result = CorrelationResult(
                asset1=asset1,
                asset2=asset2,
                market1=market1,
                market2=market2,
                pearson_correlation=float(pearson_corr),
                spearman_correlation=float(spearman_corr),
                kendall_correlation=float(kendall_corr),
                rolling_correlation_1h=rolling_correlations.get('1h', []),
                rolling_correlation_4h=rolling_correlations.get('4h', []),
                rolling_correlation_1d=rolling_correlations.get('1d', []),
                p_value=float(pearson_p),
                confidence_interval=confidence_interval,
                sample_size=n,
                granger_causality_1to2=granger_1to2,
                granger_causality_2to1=granger_2to1,
                dynamic_correlation=dynamic_correlation
            )

            # キャッシュ更新
            cache_key = f"{asset1}_{asset2}"
            self.correlation_cache[cache_key] = result

            logger.debug(f"Correlation {asset1}-{asset2}: {pearson_corr:.4f} (p={pearson_p:.4f})")

            return result

        except Exception as e:
            logger.error(f"Correlation analysis error for {asset1}-{asset2}: {e}")
            return None

    async def _get_asset_data(self, asset: str) -> Optional[pd.Series]:
        """資産データ取得"""
        try:
            # キャッシュ確認
            if asset in self.price_data_cache:
                cached_data = self.price_data_cache[asset]
                # キャッシュが新しい場合は使用（5分以内）
                if datetime.now() - cached_data.index[-1] < timedelta(minutes=5):
                    return cached_data['price']

            # 市場タイプ判定してデータ取得
            if '/' in asset:  # Forex
                data = await self._get_forex_data(asset)
            else:  # Crypto
                data = await self._get_crypto_data(asset)

            if data is not None and len(data) > 0:
                self.price_data_cache[asset] = data
                return data['price']

            return None

        except Exception as e:
            logger.error(f"Asset data fetch error for {asset}: {e}")
            return None

    async def _get_forex_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Forexデータ取得"""
        if not self.forex_collector:
            # 模擬データ生成（実装用）
            return self._generate_mock_forex_data(pair)

        tick = self.forex_collector.get_latest_tick(pair)
        if tick:
            return pd.DataFrame({
                'price': [tick.bid_price],
                'timestamp': [tick.timestamp]
            }).set_index('timestamp')

        return None

    async def _get_crypto_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Cryptoデータ取得"""
        if not self.crypto_collector:
            # 模擬データ生成（実装用）
            return self._generate_mock_crypto_data(symbol)

        tick = await self.crypto_collector.get_market_data(symbol)
        if tick:
            return pd.DataFrame({
                'price': [tick.price],
                'timestamp': [tick.timestamp]
            }).set_index('timestamp')

        return None

    def _generate_mock_forex_data(self, pair: str, length: int = 1000) -> pd.DataFrame:
        """模擬Forexデータ生成"""
        np.random.seed(hash(pair) % 1000)  # 一貫性のためのシード

        # 基準価格設定
        base_prices = {
            "EUR/USD": 1.0800, "GBP/USD": 1.2500, "USD/JPY": 110.0,
            "AUD/USD": 0.7500, "EUR/GBP": 0.8600, "GBP/JPY": 137.5
        }

        base_price = base_prices.get(pair, 1.0000)

        # ランダムウォーク生成
        returns = np.random.normal(0, 0.0001, length)  # 0.01% 標準偏差
        prices = base_price * np.exp(np.cumsum(returns))

        timestamps = pd.date_range(
            start=datetime.now() - timedelta(minutes=length),
            periods=length,
            freq='1min'
        )

        return pd.DataFrame({
            'price': prices,
            'timestamp': timestamps
        }).set_index('timestamp')

    def _generate_mock_crypto_data(self, symbol: str, length: int = 1000) -> pd.DataFrame:
        """模擬Cryptoデータ生成"""
        np.random.seed(hash(symbol) % 1000)

        # 基準価格設定
        base_prices = {
            "BTCUSDT": 45000, "ETHUSDT": 3000, "BNBUSDT": 400,
            "ADAUSDT": 1.20, "DOTUSDT": 25.0, "LINKUSDT": 20.0
        }

        base_price = base_prices.get(symbol, 100.0)

        # より高いボラティリティ
        returns = np.random.normal(0, 0.001, length)  # 0.1% 標準偏差
        prices = base_price * np.exp(np.cumsum(returns))

        timestamps = pd.date_range(
            start=datetime.now() - timedelta(minutes=length),
            periods=length,
            freq='1min'
        )

        return pd.DataFrame({
            'price': prices,
            'timestamp': timestamps
        }).set_index('timestamp')

    def _align_price_data(self, data1: pd.Series, data2: pd.Series) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """価格データ同期"""
        try:
            # 共通時間範囲取得
            common_index = data1.index.intersection(data2.index)

            if len(common_index) < 50:  # 最小サンプル数チェック
                return None

            aligned_data1 = data1.loc[common_index].values
            aligned_data2 = data2.loc[common_index].values

            # NaN除去
            valid_mask = ~(np.isnan(aligned_data1) | np.isnan(aligned_data2))

            return aligned_data1[valid_mask], aligned_data2[valid_mask]

        except Exception as e:
            logger.error(f"Data alignment error: {e}")
            return None

    def _calculate_confidence_interval(self, correlation: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """相関係数の信頼区間計算"""
        try:
            if abs(correlation) >= 0.999:  # 完全相関の場合
                return (correlation, correlation)

            # Fisher変換
            z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            z_se = 1 / np.sqrt(n - 3)

            # 臨界値
            alpha = 1 - confidence
            z_crit = stats.norm.ppf(1 - alpha/2)

            # 信頼区間（Fisher変換空間）
            z_lower = z - z_crit * z_se
            z_upper = z + z_crit * z_se

            # 逆変換
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

            return (float(r_lower), float(r_upper))

        except Exception as e:
            logger.warning(f"Confidence interval calculation error: {e}")
            return (correlation - 0.1, correlation + 0.1)  # フォールバック

    def _calculate_rolling_correlations(self, prices1: np.ndarray, prices2: np.ndarray) -> Dict[str, List[float]]:
        """ローリング相関計算"""
        rolling_corrs = {}

        for window_name, window_size in self.correlation_windows.items():
            if len(prices1) < window_size:
                rolling_corrs[window_name] = []
                continue

            corrs = []
            for i in range(window_size, len(prices1)):
                window_data1 = prices1[i-window_size:i]
                window_data2 = prices2[i-window_size:i]

                try:
                    corr, _ = pearsonr(window_data1, window_data2)
                    if not np.isnan(corr):
                        corrs.append(float(corr))
                except:
                    continue

            rolling_corrs[window_name] = corrs

        return rolling_corrs

    def _granger_causality_test(self, prices1: np.ndarray, prices2: np.ndarray, max_lags: int = 5) -> Tuple[Optional[float], Optional[float]]:
        """グレンジャー因果性テスト"""
        try:
            # データ準備
            df = pd.DataFrame({
                'asset1': prices1,
                'asset2': prices2
            })

            # 差分取得（定常化）
            df_diff = df.diff().dropna()

            if len(df_diff) < max_lags * 3:  # 十分なデータがない場合
                return None, None

            # asset1 -> asset2 の因果性
            try:
                result_1to2 = grangercausalitytests(df_diff[['asset2', 'asset1']], max_lags, verbose=False)
                p_value_1to2 = min([result_1to2[i][0]['ssr_ftest'][1] for i in range(1, max_lags + 1)])
            except:
                p_value_1to2 = None

            # asset2 -> asset1 の因果性
            try:
                result_2to1 = grangercausalitytests(df_diff[['asset1', 'asset2']], max_lags, verbose=False)
                p_value_2to1 = min([result_2to1[i][0]['ssr_ftest'][1] for i in range(1, max_lags + 1)])
            except:
                p_value_2to1 = None

            return p_value_1to2, p_value_2to1

        except Exception as e:
            logger.debug(f"Granger causality test error: {e}")
            return None, None

    def _predict_dynamic_correlation(self, prices1: np.ndarray, prices2: np.ndarray) -> Optional[List[float]]:
        """動的相関予測"""
        try:
            if not self.model_trained:
                return None

            # 特徴量作成
            features = self._create_correlation_features(prices1, prices2)

            if features is None or len(features) < self.dynamic_model.sequence_length:
                return None

            # 予測実行
            with torch.no_grad():
                input_tensor = torch.tensor(features[-self.dynamic_model.sequence_length:], dtype=torch.float32).unsqueeze(0)
                correlation_pred, confidence = self.dynamic_model(input_tensor)

                return [float(correlation_pred), float(confidence)]

        except Exception as e:
            logger.debug(f"Dynamic correlation prediction error: {e}")
            return None

    def _create_correlation_features(self, prices1: np.ndarray, prices2: np.ndarray) -> Optional[np.ndarray]:
        """相関予測用特徴量作成"""
        try:
            # リターン計算
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))

            if len(returns1) < 50:
                return None

            features = []
            window = 24  # 24期間窓

            for i in range(window, len(returns1)):
                window_returns1 = returns1[i-window:i]
                window_returns2 = returns2[i-window:i]

                # 統計特徴量
                corr = np.corrcoef(window_returns1, window_returns2)[0, 1]
                vol1 = np.std(window_returns1)
                vol2 = np.std(window_returns2)
                mean_ret1 = np.mean(window_returns1)
                mean_ret2 = np.mean(window_returns2)

                # 追加特徴量
                skew1 = stats.skew(window_returns1)
                skew2 = stats.skew(window_returns2)
                kurt1 = stats.kurtosis(window_returns1)
                kurt2 = stats.kurtosis(window_returns2)

                # 価格モメンタム
                momentum1 = (prices1[i] - prices1[i-window]) / prices1[i-window]
                momentum2 = (prices2[i] - prices2[i-window]) / prices2[i-window]

                feature_vector = [
                    corr if not np.isnan(corr) else 0,
                    vol1, vol2, mean_ret1, mean_ret2,
                    skew1, skew2, kurt1, kurt2,
                    momentum1, momentum2
                ]

                features.append(feature_vector)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return None

    def _determine_market_type(self, asset: str) -> str:
        """市場タイプ判定"""
        if '/' in asset:
            return 'forex'
        elif asset.endswith('USDT') or asset.endswith('USD'):
            return 'crypto'
        else:
            return 'stock'  # デフォルト

    async def _save_correlation_result(self, result: CorrelationResult):
        """相関結果をデータベース保存"""
        try:
            session = get_session()

            # 市場タイプ変換
            market1_type = MarketType.FOREX if result.market1 == 'forex' else MarketType.CRYPTO
            market2_type = MarketType.FOREX if result.market2 == 'forex' else MarketType.CRYPTO

            correlation_record = CrossMarketCorrelation(
                asset1_symbol=result.asset1,
                asset1_market=market1_type,
                asset2_symbol=result.asset2,
                asset2_market=market2_type,
                correlation_1h=result.rolling_correlation_1h[-1] if result.rolling_correlation_1h else None,
                correlation_4h=result.rolling_correlation_4h[-1] if result.rolling_correlation_4h else None,
                correlation_1d=result.pearson_correlation,
                sample_size=result.sample_size,
                p_value=result.p_value,
                last_updated=result.timestamp
            )

            session.add(correlation_record)
            session.commit()
            session.close()

        except Exception as e:
            logger.error(f"Failed to save correlation result: {e}")

    def get_correlation_matrix(self) -> pd.DataFrame:
        """相関マトリクス取得"""
        try:
            assets = set()
            correlations = {}

            for result in self.correlation_cache.values():
                assets.add(result.asset1)
                assets.add(result.asset2)
                correlations[(result.asset1, result.asset2)] = result.pearson_correlation
                correlations[(result.asset2, result.asset1)] = result.pearson_correlation  # 対称

            assets = sorted(list(assets))
            matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)

            # 対角線は1.0
            for asset in assets:
                matrix.loc[asset, asset] = 1.0

            # 相関値設定
            for (asset1, asset2), corr in correlations.items():
                matrix.loc[asset1, asset2] = corr

            return matrix.fillna(0.0)

        except Exception as e:
            logger.error(f"Correlation matrix generation error: {e}")
            return pd.DataFrame()

    def detect_regime_changes(self, asset: str, lookback_days: int = 30) -> Optional[MarketRegimeChange]:
        """市場レジーム変化検出"""
        try:
            # 実装省略（複雑なアルゴリズムのため）
            # 隠れマルコフモデルやVARモデルを使用した実装が可能

            # 模擬結果返却
            return MarketRegimeChange(
                asset=asset,
                market=self._determine_market_type(asset),
                change_points=[datetime.now() - timedelta(hours=12)],
                regime_labels=["high_vol"],
                regime_correlations={"high_vol": {"correlation": 0.75}},
                volatility_regimes={"high_vol": 0.025},
                detection_confidence=0.85
            )

        except Exception as e:
            logger.error(f"Regime change detection error: {e}")
            return None

def create_correlation_engine(forex_collector: Optional[ForexDataCollector] = None,
                            crypto_collector: Optional[CryptoDataCollector] = None) -> CrossMarketCorrelationEngine:
    """相関分析エンジン作成"""
    return CrossMarketCorrelationEngine(forex_collector, crypto_collector)

async def main():
    """テスト実行"""
    engine = create_correlation_engine()

    print("Cross Market Correlation Analysis Test")
    print("=" * 50)

    # 相関分析実行
    results = await engine.analyze_all_correlations()

    print(f"\nAnalyzed {len(results)} asset pairs:")

    for result in results[:5]:  # 最初の5件表示
        print(f"\n{result.asset1} vs {result.asset2}:")
        print(f"  Pearson Correlation: {result.pearson_correlation:.4f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Sample Size: {result.sample_size}")
        if result.granger_causality_1to2:
            print(f"  Granger {result.asset1}->{result.asset2}: p={result.granger_causality_1to2:.4f}")
        if result.granger_causality_2to1:
            print(f"  Granger {result.asset2}->{result.asset1}: p={result.granger_causality_2to1:.4f}")

    # 相関マトリクス表示
    matrix = engine.get_correlation_matrix()
    if not matrix.empty:
        print(f"\nCorrelation Matrix Shape: {matrix.shape}")
        print("Sample correlations:")
        print(matrix.iloc[:3, :3])

if __name__ == "__main__":
    asyncio.run(main())

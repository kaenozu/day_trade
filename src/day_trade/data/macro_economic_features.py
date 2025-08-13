#!/usr/bin/env python3
"""
マクロ経済指標特徴量エンジン

予測精度向上のため、外部経済指標との相関を特徴量として追加
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MacroEconomicFeatures:
    """マクロ経済指標特徴量生成クラス"""

    def __init__(self):
        """初期化"""
        self.macro_symbols = {
            'nikkei': '^N225',      # 日経平均
            'topix': '^TPX',        # TOPIX
            'usdjpy': 'USDJPY=X',   # USD/JPY
            'sp500': '^GSPC',       # S&P500
            'nasdaq': '^IXIC',      # NASDAQ
            'jgb10': '^TNX',        # 10年米国債（日本国債の代替）
            'vix': '^VIX',          # VIX（恐怖指数）
            'dxy': 'DX-Y.NYB',      # ドルインデックス
            'oil': 'CL=F',          # 原油先物
            'gold': 'GC=F',         # 金先物
        }
        self.macro_cache = {}
        
        # Issue #555対応: センチメント倍数の外部化
        self.sentiment_multipliers = {
            'vix': 5.0,      # VIX倍数（恐怖指数の感度）
            'equity': 2.0,   # 株式市場倍数（S&P500等の感度）
            'fx': 1.0,       # 為替倍数（USD/JPY等の感度）
            'commodity': 1.5 # 商品倍数（金・原油等の感度）
        }

        # Issue #554対応: センチメント計算期間の外部化
        self.vix_sentiment_period = "1mo"
        self.equity_sentiment_period = "3mo"
        self.fx_sentiment_period = "3mo"

        # センチメントスコアの乗数 (Issue #555対応)
        self.vix_sentiment_multiplier = 5
        self.equity_sentiment_multiplier = 2
        self.fx_sentiment_multiplier = 1

    def get_macro_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """マクロ経済指標データ取得"""
        try:
            cache_key = f"{symbol}_{period}"

            # キャッシュチェック
            if cache_key in self.macro_cache:
                return self.macro_cache[cache_key]

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                logger.warning(f"マクロデータ取得失敗: {symbol}")
                return None

            # キャッシュ保存
            self.macro_cache[cache_key] = data
            logger.info(f"マクロデータ取得成功: {symbol} ({len(data)} 日分)")

            return data

        except Exception as e:
            logger.error(f"マクロデータ取得エラー {symbol}: {e}")
            return None

    def add_macro_features(self, stock_data: pd.DataFrame, stock_symbol: str) -> pd.DataFrame:
        """
        株価データにマクロ経済特徴量を追加

        Args:
            stock_data: 株価データ
            stock_symbol: 銘柄コード

        Returns:
            マクロ特徴量が追加された株価データ
        """
        result = stock_data.copy()

        try:
            # 日経平均との相関
            nikkei_data = self.get_macro_data(self.macro_symbols['nikkei'])
            if nikkei_data is not None:
                result = self._add_correlation_features(
                    result, nikkei_data, "Nikkei", stock_symbol
                )

            # USD/JPYとの相関
            usdjpy_data = self.get_macro_data(self.macro_symbols['usdjpy'])
            if usdjpy_data is not None:
                result = self._add_correlation_features(
                    result, usdjpy_data, "USDJPY", stock_symbol
                )

            # S&P500との相関（グローバル市場センチメント）
            sp500_data = self.get_macro_data(self.macro_symbols['sp500'])
            if sp500_data is not None:
                result = self._add_correlation_features(
                    result, sp500_data, "SP500", stock_symbol
                )

            # VIX（恐怖指数）
            vix_data = self.get_macro_data(self.macro_symbols['vix'])
            if vix_data is not None:
                result = self._add_vix_features(result, vix_data)

            logger.info(f"マクロ経済特徴量追加完了: {stock_symbol}")

        except Exception as e:
            logger.error(f"マクロ特徴量追加エラー {stock_symbol}: {e}")

        return result

    def _add_correlation_features(
        self,
        stock_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        macro_name: str,
        stock_symbol: str
    ) -> pd.DataFrame:
        """相関特徴量を追加"""
        try:
            # インデックスを日付に統一
            stock_data_indexed = stock_data.copy()
            macro_data_indexed = macro_data.copy()

            # 共通日付のみに限定
            common_dates = stock_data_indexed.index.intersection(macro_data_indexed.index)

            if len(common_dates) < 10:
                logger.warning(f"{macro_name}との共通日付が不足: {len(common_dates)} 日")
                return stock_data

            # 株価とマクロ指標の日次リターン
            stock_close_col = "終値" if "終値" in stock_data.columns else "Close"
            macro_close_col = "終値" if "終値" in macro_data.columns else "Close"

            stock_returns = stock_data_indexed[stock_close_col].pct_change()
            macro_returns = macro_data_indexed[macro_close_col].pct_change()

            # 相関係数（ローリング）
            for window in [20, 60]:
                correlation = stock_returns.rolling(window).corr(macro_returns)
                stock_data_indexed[f"{macro_name}_correlation_{window}"] = correlation

            # 相対パフォーマンス
            stock_normalized = stock_data_indexed[stock_close_col] / stock_data_indexed[stock_close_col].iloc[0]
            macro_normalized = macro_data_indexed[macro_close_col] / macro_data_indexed[macro_close_col].iloc[0]
            stock_data_indexed[f"{macro_name}_relative_performance"] = stock_normalized / macro_normalized

            # ベータ値（回帰係数）
            for window in [60, 120]:
                def calculate_beta(stock_ret, macro_ret):
                    if len(stock_ret.dropna()) < 10:
                        return np.nan
                    covariance = np.cov(stock_ret.dropna(), macro_ret.dropna())[0, 1]
                    variance = np.var(macro_ret.dropna())
                    return covariance / variance if variance != 0 else np.nan

                beta = stock_returns.rolling(window).apply(
                    lambda x: calculate_beta(x, macro_returns[x.index])
                )
                stock_data_indexed[f"{macro_name}_beta_{window}"] = beta

            logger.debug(f"{macro_name}相関特徴量追加完了")
            return stock_data_indexed

        except Exception as e:
            logger.error(f"{macro_name}相関特徴量追加エラー: {e}")
            return stock_data

    def _add_vix_features(self, stock_data: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
        """VIX（恐怖指数）特徴量を追加"""
        try:
            # VIXレベルによる市場センチメント
            vix_close = vix_data["終値"] if "終値" in vix_data.columns else vix_data["Close"]

            # VIX水準での市場分類
            stock_data["VIX_level"] = np.select(
                [vix_close < 20, vix_close < 30, vix_close >= 30],
                ["low_fear", "moderate_fear", "high_fear"],
                default="unknown"
            )

            # VIX変化率
            vix_change = vix_close.pct_change()
            stock_data["VIX_change"] = vix_change
            stock_data["VIX_volatility"] = vix_change.rolling(20).std()

            logger.debug("VIX特徴量追加完了")

        except Exception as e:
            logger.error(f"VIX特徴量追加エラー: {e}")

        return stock_data

    def get_macro_sentiment_score(self) -> Dict[str, float]:
        """マクロ経済センチメントスコア算出"""
        try:
            sentiment_scores = {}

            # VIXベースセンチメント
            vix_data = self.get_macro_data(self.macro_symbols['vix'], period=self.vix_sentiment_period)
            if vix_data is not None:
                current_vix = vix_data["Close"].iloc[-1]
                vix_avg = vix_data["Close"].mean()
                sentiment_scores["vix_sentiment"] = max(0, min(100, 100 - (current_vix - vix_avg) * self.sentiment_multipliers['vix']))

            # 株式市場センチメント（S&P500）
            sp500_data = self.get_macro_data(self.macro_symbols['sp500'], period=self.equity_sentiment_period)
            if sp500_data is not None:
                sp500_returns = sp500_data["Close"].pct_change(20).iloc[-1] * 100
                sentiment_scores["equity_sentiment"] = max(0, min(100, 50 + sp500_returns * self.sentiment_multipliers['equity']))

            # 為替センチメント（ドル強弱）
            usdjpy_data = self.get_macro_data(self.macro_symbols['usdjpy'], period=self.fx_sentiment_period)
            if usdjpy_data is not None:
                usdjpy_change = usdjpy_data["Close"].pct_change(20).iloc[-1] * 100
                sentiment_scores["fx_sentiment"] = max(0, min(100, 50 + usdjpy_change * self.sentiment_multipliers['fx']))

            # 総合センチメント
            if sentiment_scores:
                sentiment_scores["overall_sentiment"] = np.mean(list(sentiment_scores.values()))

            logger.info(f"マクロセンチメントスコア算出: {sentiment_scores}")
            return sentiment_scores

        except Exception as e:
            logger.error(f"センチメントスコア算出エラー: {e}")
            return {}


# 便利関数
def add_macro_features_to_stock(stock_data: pd.DataFrame, stock_symbol: str) -> pd.DataFrame:
    """株価データにマクロ経済特徴量を追加（便利関数）"""
    macro_engine = MacroEconomicFeatures()
    return macro_engine.add_macro_features(stock_data, stock_symbol)


if __name__ == "__main__":
    # テスト実行
    print("=== マクロ経済特徴量エンジン テスト ===")

    macro_engine = MacroEconomicFeatures()

    # センチメントスコアテスト
    sentiment = macro_engine.get_macro_sentiment_score()
    print(f"現在のマクロセンチメント: {sentiment}")

    # テスト用の模擬データ
    test_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    test_stock_data = pd.DataFrame({
        '終値': np.random.randn(len(test_dates)).cumsum() + 1000,
        '出来高': np.random.randint(1000, 10000, len(test_dates))
    }, index=test_dates)

    # マクロ特徴量追加テスト
    enhanced_data = macro_engine.add_macro_features(test_stock_data, "TEST")
    print(f"追加された特徴量数: {len(enhanced_data.columns) - len(test_stock_data.columns)}")
    print(f"新しい特徴量: {[col for col in enhanced_data.columns if col not in test_stock_data.columns]}")
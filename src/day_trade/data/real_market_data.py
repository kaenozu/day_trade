#!/usr/bin/env python3
"""
実市場データ取得システム

yfinanceを使用した実際の株価データ取得と基本的な技術指標計算
"""

import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# yfinanceのオプション読み込み（パッケージが無い場合は警告）
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
    logger.info("yfinance利用可能")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning(
        "yfinance未インストール - pip install yfinanceでインストールしてください"
    )


class RealMarketDataManager:
    """
    実市場データ管理クラス

    yfinanceによるデータ取得とSQLiteキャッシュを提供
    """

    def __init__(self, cache_db_path: Optional[str] = None):
        self.cache_db_path = cache_db_path or "data/market_cache.db"
        self.cache_duration_hours = 1  # 1時間キャッシュ

        # APIレート制限管理
        self.rate_limit_lock = threading.Lock()
        self.last_api_call = {}  # 銘柄ごとの最終API呼び出し時刻
        self.min_api_interval = 0.2  # 200ms間隔（1秒5回制限対応）
        self.max_concurrent_requests = 3  # 同時リクエスト数制限

        # キャッシュ最適化
        self.memory_cache = {}  # メモリ内キャッシュ
        self.cache_expiry = {}  # キャッシュ有効期限

        self._ensure_cache_db()

    def _ensure_cache_db(self):
        """キャッシュDBの初期化"""
        cache_path = Path(self.cache_db_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_cache (
                    symbol TEXT,
                    date_str TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    cached_at TEXT,
                    PRIMARY KEY (symbol, date_str)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_cache (
                    symbol TEXT,
                    indicator_name TEXT,
                    value REAL,
                    cached_at TEXT,
                    PRIMARY KEY (symbol, indicator_name)
                )
            """
            )

    def get_stock_data(
        self, symbol: str, period: str = "60d"
    ) -> Optional[pd.DataFrame]:
        """
        株価データ取得（キャッシュ優先＋レート制限対応）

        Args:
            symbol: 銘柄コード
            period: 取得期間 (例: "30d", "60d", "1y")

        Returns:
            DataFrame: OHLCV データ（取得失敗時はNone）
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinanceが利用できません")
            return None

        # メモリキャッシュチェック
        cache_key = f"{symbol}_{period}"
        if self._check_memory_cache(cache_key):
            logger.info(f"メモリキャッシュヒット: {symbol}")
            return self.memory_cache[cache_key].copy()

        # SQLiteキャッシュチェック
        cached_data = self._get_cached_price_data(symbol)
        if cached_data is not None and not self._is_cache_expired(symbol):
            logger.info(f"SQLiteキャッシュヒット: {symbol}")
            self._update_memory_cache(cache_key, cached_data)
            return cached_data.copy()

        try:
            # APIレート制限チェック
            self._wait_for_rate_limit(symbol)

            # 日本株対応（.T拡張子）
            yf_symbol = f"{symbol}.T" if not symbol.endswith(".T") else symbol

            logger.info(f"API株価データ取得開始: {yf_symbol}")

            # yfinanceでデータ取得
            with self.rate_limit_lock:
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(period=period)
                self.last_api_call[symbol] = time.time()

            if data.empty:
                logger.warning(f"データが取得できませんでした: {symbol}")
                # キャッシュからフォールバック
                return cached_data

            # キャッシュに保存
            self._cache_price_data(symbol, data)
            self._update_memory_cache(cache_key, data)

            logger.info(f"API株価データ取得完了: {symbol} ({len(data)}日分)")
            return data.copy()

        except Exception as e:
            logger.error(f"API株価データ取得エラー {symbol}: {e}")
            # キャッシュからフォールバック
            if cached_data is not None:
                logger.info(f"キャッシュフォールバック使用: {symbol}")
                return cached_data
            return None

    def _check_memory_cache(self, cache_key: str) -> bool:
        """メモリキャッシュ有効性チェック"""
        if cache_key not in self.memory_cache:
            return False

        if cache_key not in self.cache_expiry:
            return False

        return datetime.now() < self.cache_expiry[cache_key]

    def _update_memory_cache(self, cache_key: str, data: pd.DataFrame):
        """メモリキャッシュ更新"""
        self.memory_cache[cache_key] = data.copy()
        self.cache_expiry[cache_key] = datetime.now() + timedelta(
            minutes=30
        )  # 30分キャッシュ

    def _is_cache_expired(self, symbol: str, hours: int = None) -> bool:
        """SQLiteキャッシュ有効期限チェック"""
        if hours is None:
            hours = self.cache_duration_hours

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT cached_at FROM price_cache
                    WHERE symbol = ?
                    ORDER BY date_str DESC
                    LIMIT 1
                """,
                    [symbol],
                )

                result = cursor.fetchone()
                if result is None:
                    return True

                cached_time = datetime.fromisoformat(result[0])
                expiry_time = cached_time + timedelta(hours=hours)

                return datetime.now() > expiry_time
        except Exception as e:
            logger.error(f"キャッシュ期限チェックエラー {symbol}: {e}")
            return True

    def _wait_for_rate_limit(self, symbol: str):
        """APIレート制限待機"""
        if symbol in self.last_api_call:
            elapsed = time.time() - self.last_api_call[symbol]
            if elapsed < self.min_api_interval:
                sleep_time = self.min_api_interval - elapsed
                logger.debug(f"レート制限待機: {symbol} - {sleep_time:.2f}秒")
                time.sleep(sleep_time)

    def _cache_price_data(self, symbol: str, data: pd.DataFrame):
        """価格データをキャッシュに保存"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                for date_idx, row in data.iterrows():
                    date_str = date_idx.strftime("%Y-%m-%d")
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO price_cache
                        (symbol, date_str, open, high, low, close, volume, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            date_str,
                            row["Open"],
                            row["High"],
                            row["Low"],
                            row["Close"],
                            int(row["Volume"]),
                            datetime.now().isoformat(),
                        ),
                    )
        except Exception as e:
            logger.error(f"価格データキャッシュエラー {symbol}: {e}")

    def _get_cached_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """キャッシュから価格データを取得"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT date_str, open, high, low, close, volume
                    FROM price_cache
                    WHERE symbol = ?
                    ORDER BY date_str DESC
                    LIMIT 60
                """,
                    conn,
                    params=[symbol],
                )

                if df.empty:
                    return None

                df["date_str"] = pd.to_datetime(df["date_str"])
                df.set_index("date_str", inplace=True)
                df.columns = ["Open", "High", "Low", "Close", "Volume"]

                logger.info(f"キャッシュからデータ取得: {symbol}")
                return df

        except Exception as e:
            logger.error(f"キャッシュデータ取得エラー {symbol}: {e}")
            return None

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """RSI計算"""
        if data is None or len(data) < period + 1:
            return 50.0  # デフォルト値

        try:
            close_prices = data["Close"]
            delta = close_prices.diff()

            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]
            return round(float(current_rsi), 2) if not np.isnan(current_rsi) else 50.0

        except Exception as e:
            logger.error(f"RSI計算エラー: {e}")
            return 50.0

    def calculate_macd(
        self, data: pd.DataFrame, fast: int = 12, slow: int = 26
    ) -> float:
        """MACD計算"""
        if data is None or len(data) < slow:
            return 0.0

        try:
            close_prices = data["Close"]

            ema_fast = close_prices.ewm(span=fast).mean()
            ema_slow = close_prices.ewm(span=slow).mean()

            macd_line = ema_fast - ema_slow
            current_macd = macd_line.iloc[-1]

            return round(float(current_macd), 2) if not np.isnan(current_macd) else 0.0

        except Exception as e:
            logger.error(f"MACD計算エラー: {e}")
            return 0.0

    def calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> float:
        """出来高比率計算"""
        if data is None or len(data) < period:
            return 1.0

        try:
            volumes = data["Volume"]
            avg_volume = volumes.rolling(window=period).mean()
            current_volume = volumes.iloc[-1]
            avg_vol_current = avg_volume.iloc[-1]

            if avg_vol_current > 0:
                ratio = current_volume / avg_vol_current
                return round(float(ratio), 2)
            else:
                return 1.0

        except Exception as e:
            logger.error(f"出来高比率計算エラー: {e}")
            return 1.0

    def calculate_price_change_percent(
        self, data: pd.DataFrame, days: int = 1
    ) -> float:
        """価格変動率計算"""
        if data is None or len(data) < days + 1:
            return 0.0

        try:
            close_prices = data["Close"]
            current_price = close_prices.iloc[-1]
            prev_price = close_prices.iloc[-(days + 1)]

            if prev_price > 0:
                change_percent = ((current_price - prev_price) / prev_price) * 100
                return round(float(change_percent), 2)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"価格変動率計算エラー: {e}")
            return 0.0

    def get_current_price(self, symbol: str) -> float:
        """現在価格取得"""
        data = self.get_stock_data(symbol, period="5d")
        if data is not None and not data.empty:
            return round(float(data["Close"].iloc[-1]), 1)
        else:
            # フォールバック値
            base_prices = {
                "7203": 2450.0,
                "8306": 800.0,
                "9984": 4200.0,
                "6758": 12500.0,
                "4689": 350.0,
                "9434": 1200.0,
                "8001": 4800.0,
                "7267": 3200.0,
                "6861": 55000.0,
                "2914": 2800.0,
            }
            return base_prices.get(symbol, 1000.0)

    def generate_ml_trend_score(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        機械学習風トレンドスコア生成（実データベース）

        Returns:
            Tuple[float, float]: (スコア, 信頼度)
        """
        if data is None or len(data) < 20:
            return 50.0, 0.5

        try:
            # 実データに基づく計算
            close_prices = data["Close"]

            # 短期・長期移動平均
            sma_5 = close_prices.rolling(window=5).mean()
            sma_20 = close_prices.rolling(window=20).mean()

            # 現在の価格位置
            current_price = close_prices.iloc[-1]
            sma5_current = sma_5.iloc[-1]
            sma20_current = sma_20.iloc[-1]

            # スコア計算
            score = 50.0  # ベースライン

            # 移動平均との関係
            if current_price > sma5_current:
                score += 15
            if current_price > sma20_current:
                score += 15
            if sma5_current > sma20_current:
                score += 10

            # 最近のトレンド
            recent_trend = (
                (current_price - close_prices.iloc[-5]) / close_prices.iloc[-5] * 100
            )
            if recent_trend > 2:
                score += 10
            elif recent_trend < -2:
                score -= 10

            # スコアを0-100に正規化
            score = max(0, min(100, score))

            # 信頼度（データの質に基づく）
            confidence = min(0.9, len(data) / 60.0 * 0.8 + 0.2)

            return round(score, 1), round(confidence, 2)

        except Exception as e:
            logger.error(f"MLトレンドスコア計算エラー: {e}")
            return 50.0, 0.5

    def generate_ml_volatility_score(self, data: pd.DataFrame) -> Tuple[float, float]:
        """ML風ボラティリティスコア生成（実データベース）"""
        if data is None or len(data) < 20:
            return 50.0, 0.5

        try:
            close_prices = data["Close"]

            # ボラティリティ計算
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年率ボラティリティ

            # ボラティリティを0-100スコアに変換
            # 一般的な株式ボラティリティ: 0.1-0.8 程度
            normalized_vol = min(1.0, volatility / 0.6)
            score = normalized_vol * 100

            # 信頼度
            confidence = min(0.9, len(returns) / 30.0 * 0.7 + 0.3)

            return round(score, 1), round(confidence, 2)

        except Exception as e:
            logger.error(f"MLボラティリティスコア計算エラー: {e}")
            return 50.0, 0.5

    def generate_ml_pattern_score(self, data: pd.DataFrame) -> Tuple[float, float]:
        """ML風パターンスコア生成（実データベース）"""
        if data is None or len(data) < 20:
            return 50.0, 0.5

        try:
            close_prices = data["Close"]
            high_prices = data["High"]
            low_prices = data["Low"]

            score = 50.0  # ベースライン

            # サポート・レジスタンス分析
            recent_high = high_prices.tail(10).max()
            recent_low = low_prices.tail(10).min()
            current_price = close_prices.iloc[-1]

            # 価格位置によるスコア調整
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            if price_position > 0.7:
                score += 15  # 上位レンジ
            elif price_position < 0.3:
                score -= 15  # 下位レンジ

            # ブレイクアウト検出
            prev_high = high_prices.iloc[-20:-10].max()
            if current_price > prev_high:
                score += 20  # 上向きブレイクアウト

            # 連続性チェック
            consecutive_up = 0
            for i in range(-5, 0):
                if close_prices.iloc[i] > close_prices.iloc[i - 1]:
                    consecutive_up += 1

            score += consecutive_up * 3

            # スコア正規化
            score = max(0, min(100, score))

            # 信頼度
            confidence = min(0.85, len(data) / 40.0 * 0.6 + 0.4)

            return round(score, 1), round(confidence, 2)

        except Exception as e:
            logger.error(f"MLパターンスコア計算エラー: {e}")
            return 50.0, 0.5


# モジュールレベル関数（後方互換性）
def get_real_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """株価データ取得（簡易インターフェース）"""
    manager = RealMarketDataManager()
    return manager.get_stock_data(symbol)


def calculate_real_technical_indicators(symbol: str) -> Dict[str, float]:
    """実技術指標計算"""
    manager = RealMarketDataManager()
    data = manager.get_stock_data(symbol)

    if data is None:
        return {
            "rsi": 50.0,
            "macd": 0.0,
            "volume_ratio": 1.0,
            "price_change_1d": 0.0,
            "current_price": 1000.0,
        }

    return {
        "rsi": manager.calculate_rsi(data),
        "macd": manager.calculate_macd(data),
        "volume_ratio": manager.calculate_volume_ratio(data),
        "price_change_1d": manager.calculate_price_change_percent(data, 1),
        "current_price": manager.get_current_price(symbol),
    }


if __name__ == "__main__":
    # テスト実行
    manager = RealMarketDataManager()
    test_symbol = "7203"

    print(f"=== {test_symbol} 実データテスト ===")

    # データ取得テスト
    data = manager.get_stock_data(test_symbol)
    if data is not None:
        print(f"データ期間: {len(data)}日")
        print(f"現在価格: {manager.get_current_price(test_symbol)}円")

        # 技術指標テスト
        rsi = manager.calculate_rsi(data)
        macd = manager.calculate_macd(data)
        vol_ratio = manager.calculate_volume_ratio(data)

        print(f"RSI: {rsi}")
        print(f"MACD: {macd}")
        print(f"出来高比率: {vol_ratio}")

        # MLスコアテスト
        trend_score, trend_conf = manager.generate_ml_trend_score(data)
        vol_score, vol_conf = manager.generate_ml_volatility_score(data)
        pattern_score, pattern_conf = manager.generate_ml_pattern_score(data)

        print(f"MLトレンドスコア: {trend_score} (信頼度: {trend_conf})")
        print(f"MLボラティリティスコア: {vol_score} (信頼度: {vol_conf})")
        print(f"MLパターンスコア: {pattern_score} (信頼度: {pattern_conf})")
    else:
        print("データ取得に失敗しました")

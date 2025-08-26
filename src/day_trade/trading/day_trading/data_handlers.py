#!/usr/bin/env python3
"""
データ取得・処理ハンドラー

市場データ取得、銘柄管理、特徴量準備などのデータ処理機能を提供します。
"""

import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.day_trade.data.fetchers.yfinance_fetcher import YFinanceFetcher


logger = logging.getLogger(__name__)


class SymbolManager:
    """銘柄管理クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.daytrading_symbols = self._load_dynamic_symbols()

    def _load_dynamic_symbols(self) -> dict:
        """
        改善版動的銘柄取得（Issue #849対応）

        1. 設定ファイルからのフォールバック銘柄読み込み
        2. 動的銘柄選択システム
        3. 銘柄名辞書統合
        4. 段階的フォールバック機能
        """
        try:
            # 1. 設定ファイルからフォールバック銘柄を読み込み
            fallback_symbols = self._load_fallback_symbols_from_config()

            # 2. 動的銘柄選択を試行
            dynamic_symbols = self._try_dynamic_symbol_selection()

            # 3. 銘柄名解決を実行
            final_symbols = self._resolve_symbol_names(dynamic_symbols, fallback_symbols)

            logger.info(f"銘柄取得完了: {len(final_symbols)}銘柄（動的: {len(dynamic_symbols)}, フォールバック: {len(fallback_symbols)}）")
            return final_symbols

        except Exception as e:
            logger.error(f"銘柄取得プロセス失敗: {e}")
            # 最終フォールバック: 設定ファイルの銘柄のみ使用
            return self._load_fallback_symbols_from_config()

    def _load_fallback_symbols_from_config(self) -> dict:
        """設定ファイルからフォールバック銘柄を読み込み"""
        try:
            symbol_mapping = self.config.get("symbol_mapping", {})
            fallback_symbols = symbol_mapping.get("fallback_symbols", {})
            custom_symbols = symbol_mapping.get("custom_symbols", {})

            # フォールバックとカスタム銘柄を統合
            combined_symbols = {}
            combined_symbols.update(fallback_symbols)
            if isinstance(custom_symbols, dict):
                combined_symbols.update(custom_symbols)

            logger.info(f"設定ファイルから{len(combined_symbols)}銘柄を読み込み")
            return combined_symbols

        except Exception as e:
            logger.warning(f"設定ファイル銘柄読み込み失敗: {e}")
            # 最小限のデフォルト銘柄
            return {
                "7203": "トヨタ自動車",
                "9984": "ソフトバンクグループ",
                "8306": "三菱UFJフィナンシャル・グループ"
            }

    def _try_dynamic_symbol_selection(self) -> list:
        """動的銘柄選択を試行"""
        try:
            # 設定でdynamic selectionが有効な場合のみ実行
            symbol_selection_config = self.config.get("symbol_selection", {})
            if not symbol_selection_config.get("enable_dynamic_selection", True):
                logger.info("動的銘柄選択は設定で無効化されています")
                return []

            from src.day_trade.data.symbol_selector import DynamicSymbolSelector
            selector = DynamicSymbolSelector()

            # 設定に基づく銘柄数制限
            max_symbols = symbol_selection_config.get("max_symbols", 20)
            symbols = selector.get_liquid_symbols(limit=max_symbols)

            logger.info(f"動的銘柄選択成功: {len(symbols)}銘柄")
            return symbols

        except Exception as e:
            logger.warning(f"動的銘柄選択失敗: {e}")
            return []

    def _resolve_symbol_names(self, dynamic_symbols: list, fallback_symbols: dict) -> dict:
        """銘柄名解決（動的+フォールバック統合）"""
        symbol_dict = {}

        # 1. 動的取得銘柄の名前解決
        for symbol in dynamic_symbols:
            name = self._resolve_single_symbol_name(symbol)
            if name:
                symbol_dict[symbol] = name

        # 2. フォールバック銘柄を追加（重複は動的銘柄を優先）
        for symbol, name in fallback_symbols.items():
            if symbol not in symbol_dict:
                symbol_dict[symbol] = name

        return symbol_dict

    def _resolve_single_symbol_name(self, symbol: str) -> str:
        """単一銘柄の名前解決"""
        # 1. 銘柄名辞書から取得（最優先）
        try:
            from src.day_trade.data.symbol_names import get_symbol_name
            name = get_symbol_name(symbol)
            if name:
                logger.debug(f"Symbol name from dict: {symbol} -> {name}")
                return name
        except Exception as e:
            logger.debug(f"Symbol name dict lookup failed for {symbol}: {e}")

        # 2. 設定ファイルからの取得
        fallback_symbols = self.config.get("symbol_mapping", {}).get("fallback_symbols", {})
        if symbol in fallback_symbols:
            name = fallback_symbols[symbol]
            logger.debug(f"Symbol name from config: {symbol} -> {name}")
            return name

        # 3. フォールバック: 銘柄コードベースの名前
        fallback_name = f"銘柄{symbol}"
        logger.debug(f"Symbol name fallback: {symbol} -> {fallback_name}")
        return fallback_name

    def get_symbols(self) -> dict:
        """銘柄一覧を取得"""
        return self.daytrading_symbols


class MarketDataHandler:
    """市場データ処理クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # データフェッチャー初期化
        try:
            from src.day_trade.data.stock_fetcher import StockFetcher
            self.data_fetcher = StockFetcher()
        except ImportError:
            # フォールバック用の簡易データフェッチャー
            self.data_fetcher = YFinanceFetcher()

    def fetch_market_data(self, symbol: str, date_str: str) -> Dict[str, Any]:
        """
        改善版市場データ取得メソッド（Issue #849対応）

        1. 実データ取得の試行
        2. 段階的フォールバック機能
        3. 改善されたエラーハンドリング
        4. モックデータ通知システム
        """
        try:
            if self.data_fetcher is None:
                logger.warning(f"データフェッチャーが初期化されていません - モックデータ使用: {symbol}")
                return self._generate_intelligent_mock_data(symbol, date_str)

            # YFinanceFetcherで実データ取得を試行
            date_obj = datetime.strptime(date_str, '%Y%m%d')

            # データ取得の複数回試行（ネットワーク問題対応）
            for attempt in range(3):
                try:
                    historical_data = self.data_fetcher.get_historical_data(
                        code=symbol, period="5d", interval="1d"
                    )

                    if historical_data is not None and not historical_data.empty:
                        return self._process_real_market_data(historical_data, symbol)

                except Exception as e:
                    logger.debug(f"データ取得試行{attempt + 1}失敗 {symbol}: {e}")
                    if attempt < 2:  # 最後の試行でない場合は少し待つ
                        import time
                        time.sleep(0.5)

            # 実データ取得失敗 - モックデータフォールバック
            logger.info(f"実データ取得失敗のためモックデータ使用: {symbol}")
            return self._generate_intelligent_mock_data(symbol, date_str)

        except Exception as e:
            logger.error(f"市場データ取得中にエラー発生 {symbol}: {e}")
            return self._generate_intelligent_mock_data(symbol, date_str)

    def _process_real_market_data(self, historical_data, symbol: str) -> Dict[str, Any]:
        """実データの処理"""
        try:
            latest_data = historical_data.iloc[-1]
            prev_close = historical_data.iloc[-2]['Close'] if len(historical_data) >= 2 else latest_data['Close']

            return {
                "Open": float(latest_data.get("Open", 0)),
                "High": float(latest_data.get("High", 0)),
                "Low": float(latest_data.get("Low", 0)),
                "Close": float(latest_data.get("Close", 0)),
                "Volume": int(latest_data.get("Volume", 0)),
                "PrevClose": float(prev_close),
                "DateTime": latest_data.name,
                "DataSource": "REAL"
            }
        except Exception as e:
            logger.warning(f"実データ処理エラー {symbol}: {e}")
            return self._generate_intelligent_mock_data(symbol, "fallback")

    def _generate_intelligent_mock_data(self, symbol: str, date_str: str) -> Dict[str, Any]:
        """改善されたモックデータ生成"""
        # 設定に基づく通知制御
        mock_notification = self.config.get("data_fallback", {}).get("mock_data_notification", False)
        if mock_notification:
            logger.info(f"モックデータを生成中: {symbol}")

        # より現実的な価格レンジ（日本株価帯を考慮）
        seed_base = hash(symbol + date_str + "market_data") % 100000
        np.random.seed(seed_base)

        # 銘柄に応じた価格帯設定
        if len(symbol) == 4 and symbol.isdigit():
            # 日本株（4桁コード）
            price_base = int(symbol) * 0.5 + 1000  # 銘柄コードに基づく基準価格
        else:
            price_base = 2000  # その他の銘柄

        price_base = min(max(price_base, 100), 10000)  # 100円〜10,000円の範囲

        # より現実的な価格変動
        daily_volatility = np.random.uniform(0.01, 0.05)  # 1-5%の日中変動

        prev_close = price_base * (1 + np.random.uniform(-0.02, 0.02))
        gap = np.random.uniform(-0.01, 0.01)  # オーバーナイトギャップ
        open_price = prev_close * (1 + gap)

        high_price = open_price * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low_price = open_price * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        close_price = open_price + (high_price - low_price) * np.random.uniform(-0.5, 0.5)

        # 出来高も現実的に
        volume_base = 1_000_000 if price_base > 1000 else 5_000_000
        volume = int(volume_base * np.random.uniform(0.5, 3.0))

        return {
            "Open": round(open_price, 0),
            "High": round(high_price, 0),
            "Low": round(low_price, 0),
            "Close": round(close_price, 0),
            "Volume": volume,
            "PrevClose": round(prev_close, 0),
            "DateTime": datetime.now(),
            "DataSource": "MOCK"
        }


class FeaturePreparator:
    """特徴量準備クラス"""

    def prepare_features_for_prediction(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        予測用特徴量準備メソッド
        RandomForestモデルが必要とする形式にデータを変換する。
        ここでは、簡易的なテクニカル指標と価格・出来高特徴量を生成。
        """
        # 必要なデータポイントが存在するか確認
        if not all(k in market_data for k in ["Open", "High", "Low", "Close", "Volume", "PrevClose"]):
            # データ不足の場合、ゼロ配列を返すか、エラーハンドリングを行う
            # ここでは暫定的に、num_featuresに合うようにゼロ埋めされた配列を返す
            num_features = 10  # モデルが期待する特徴量の数
            return np.zeros((1, num_features))

        open_p = market_data["Open"]
        high_p = market_data["High"]
        low_p = market_data["Low"]
        close_p = market_data["Close"]
        volume = market_data["Volume"]
        prev_close = market_data["PrevClose"]

        # 簡易的な特徴量エンジニアリング
        # 価格特徴量
        price_change = close_p - prev_close
        daily_range = high_p - low_p

        # 出来高特徴量 (ここでは単純な出来高を特徴量として使う)

        # テクニカル指標 (ここでは簡易的なRSIとMACDをモックとして生成)
        # 実際には src/day_trade/analysis/signals.py などからインポートして利用
        # RSIを模倣
        rsi_mock = 50 + (price_change / max(1, daily_range)) * 10
        rsi_mock = np.clip(rsi_mock, 0, 100) # RSIは0-100の範囲

        # MACDを模倣
        macd_mock = (close_p - prev_close) * 10
        macd_mock = np.clip(macd_mock, -100, 100) # 適当な範囲

        # モデルが期待する特徴量の順序に合わせて配列を作成
        # RandomForestModelは (n_samples, n_features) 形式を期待
        # 例: [price_change, daily_range, volume, rsi_mock, macd_mock, ...]
        # 現状は5つの特徴量
        features_array = np.array([
            price_change,
            daily_range,
            volume,
            rsi_mock,
            macd_mock,
            open_p, high_p, low_p, close_p, prev_close # 他の価格情報も特徴量として含める
        ]).reshape(1, -1) # 1サンプル、n_features

        # RandomForestModelの_hyperparameter_optimizationでnum_features=10としていたので、ここで合わせる
        # 実際には、config/ml.json の features に基づいて、適切な特徴量エンジニアリングパイプラインを構築する必要がある

        # もし特徴量の数が足りない場合、ゼロ埋めするなどの対応が必要
        num_expected_features = 10 # RandomForestModelが期待する特徴量の数
        if features_array.shape[1] < num_expected_features:
            padding = np.zeros((1, num_expected_features - features_array.shape[1]))
            features_array = np.hstack((features_array, padding))
        elif features_array.shape[1] > num_expected_features:
            features_array = features_array[:, :num_expected_features] # 多すぎる場合は切り詰める

        return features_array
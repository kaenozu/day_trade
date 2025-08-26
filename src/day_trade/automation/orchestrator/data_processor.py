"""
データ処理・取得
高度バッチデータ収集とデータ前処理を担当
"""

from typing import Dict, List

from ...utils.logging_config import get_context_logger

# データ処理関連のインポート（CI環境対応）
try:
    from ...data.batch_data_fetcher import DataRequest, DataResponse
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    # CI環境では軽量ダミークラス使用
    DataRequest = None
    DataResponse = None
    DATA_FETCHER_AVAILABLE = False

logger = get_context_logger(__name__)


class DataProcessor:
    """データ処理・取得クラス"""

    def __init__(self, core, config, stock_helper):
        """
        初期化

        Args:
            core: オーケストレーターコア
            config: 設定オブジェクト
            stock_helper: 銘柄ヘルパー
        """
        self.core = core
        self.config = config
        self.stock_helper = stock_helper

    def execute_batch_data_collection(
        self, symbols: List[str]
    ) -> Dict[str, DataResponse]:
        """
        高度バッチデータ収集

        Args:
            symbols: 分析対象銘柄リスト

        Returns:
            シンボル別データレスポンス辞書
        """
        if not self.core.batch_fetcher or not DATA_FETCHER_AVAILABLE:
            logger.warning("バッチフェッチャーが利用できません")
            return {}

        # 銘柄名を含む詳細情報を表示
        symbol_names = [
            self.stock_helper.format_stock_display(symbol, include_code=False)
            for symbol in symbols[:5]
        ]
        if len(symbols) > 5:
            symbol_names.append(f"他{len(symbols)-5}銘柄")
        logger.info(f"バッチデータ収集開始: {len(symbols)} 銘柄 ({', '.join(symbol_names)})")

        try:
            # データリクエスト作成
            requests = [
                DataRequest(
                    symbol=symbol,
                    period="1y",  # より長期間のデータ
                    preprocessing=True,
                    features=[
                        "trend_strength",
                        "momentum",
                        "price_channel",
                        "gap_analysis",
                    ],
                    priority=5 if symbol in ["7203", "8306"] else 3,
                    cache_ttl=3600,
                )
                for symbol in symbols
            ]

            # バッチ実行
            return self.core.batch_fetcher.fetch_batch(requests, use_parallel=True)

        except Exception as e:
            logger.error(f"バッチデータ収集エラー: {e}")
            return {}

    def validate_data_quality(self, data_response: DataResponse) -> Dict[str, any]:
        """
        データ品質検証

        Args:
            data_response: データレスポンス

        Returns:
            品質評価結果
        """
        if not data_response or not data_response.success:
            return {
                "valid": False,
                "score": 0.0,
                "issues": ["データ取得失敗"],
            }

        issues = []
        score = 100.0

        try:
            data = data_response.data

            # 基本データ存在チェック
            if data is None or len(data) == 0:
                issues.append("データが空です")
                score -= 50.0

            # データサイズチェック
            if len(data) < 50:
                issues.append("データポイントが不足しています")
                score -= 20.0

            # 欠損値チェック
            if hasattr(data, 'isnull'):
                null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                if null_ratio > 0.1:  # 10%以上の欠損
                    issues.append(f"欠損値が多すぎます ({null_ratio:.1%})")
                    score -= 30.0 * null_ratio

            # 価格データ異常値チェック
            if "終値" in data.columns:
                price_data = data["終値"]
                price_change = price_data.pct_change()
                extreme_changes = price_change[abs(price_change) > 0.5]  # 50%以上の変動
                
                if len(extreme_changes) > len(price_data) * 0.01:  # 1%以上の極端な変動
                    issues.append("価格データに異常値が含まれています")
                    score -= 15.0

            return {
                "valid": score > 50.0,
                "score": max(0.0, score),
                "issues": issues,
            }

        except Exception as e:
            return {
                "valid": False,
                "score": 0.0,
                "issues": [f"データ品質チェックエラー: {e}"],
            }

    def preprocess_market_data(self, data_response: DataResponse) -> Dict[str, any]:
        """
        マーケットデータ前処理

        Args:
            data_response: データレスポンス

        Returns:
            前処理済みデータと統計情報
        """
        if not data_response or not data_response.success:
            return {
                "processed_data": None,
                "preprocessing_stats": {
                    "success": False,
                    "error": "データが無効です",
                }
            }

        try:
            data = data_response.data.copy()
            stats = {
                "success": True,
                "original_size": len(data),
                "operations": [],
            }

            # 欠損値処理
            if data.isnull().sum().sum() > 0:
                # 前向き埋め -> 後向き埋め
                data = data.fillna(method='ffill').fillna(method='bfill')
                stats["operations"].append("欠損値補完")

            # 異常値処理（IQRベース）
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    stats["operations"].append(f"{col}の異常値クリッピング ({outliers}件)")

            # データ正規化（オプション）
            if self.config.enable_data_normalization:
                numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_columns:
                    if col not in ['日付', 'Date']:  # 日付列は除外
                        mean_val = data[col].mean()
                        std_val = data[col].std()
                        if std_val > 0:
                            data[col] = (data[col] - mean_val) / std_val
                            stats["operations"].append(f"{col}の標準化")

            stats["processed_size"] = len(data)
            stats["operations_count"] = len(stats["operations"])

            return {
                "processed_data": data,
                "preprocessing_stats": stats,
            }

        except Exception as e:
            logger.error(f"データ前処理エラー: {e}")
            return {
                "processed_data": None,
                "preprocessing_stats": {
                    "success": False,
                    "error": str(e),
                }
            }

    def get_data_summary(self, data_response: DataResponse) -> Dict[str, any]:
        """
        データサマリー取得

        Args:
            data_response: データレスポンス

        Returns:
            データサマリー情報
        """
        if not data_response or not data_response.success:
            return {
                "available": False,
                "error": "データが無効です",
            }

        try:
            data = data_response.data
            
            summary = {
                "available": True,
                "data_shape": data.shape,
                "columns": list(data.columns),
                "date_range": {
                    "start": str(data.index[0]) if hasattr(data, 'index') else None,
                    "end": str(data.index[-1]) if hasattr(data, 'index') else None,
                },
                "data_quality_score": data_response.data_quality_score,
                "basic_stats": {},
            }

            # 基本統計情報
            if "終値" in data.columns:
                price_data = data["終値"]
                summary["basic_stats"]["price"] = {
                    "current": float(price_data.iloc[-1]),
                    "min": float(price_data.min()),
                    "max": float(price_data.max()),
                    "mean": float(price_data.mean()),
                    "std": float(price_data.std()),
                }

            if "出来高" in data.columns:
                volume_data = data["出来高"]
                summary["basic_stats"]["volume"] = {
                    "current": float(volume_data.iloc[-1]),
                    "mean": float(volume_data.mean()),
                    "max": float(volume_data.max()),
                }

            return summary

        except Exception as e:
            logger.error(f"データサマリー作成エラー: {e}")
            return {
                "available": False,
                "error": str(e),
            }
#!/usr/bin/env python3
"""
リアルタイム予測API
Phase F: 次世代機能拡張フェーズ

深層学習モデルを使用したリアルタイム株価予測API
"""

from typing import Dict, List, Optional, Any
import time
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from ..ml import DeepLearningModelManager, DeepLearningConfig
from ..core.optimization_strategy import OptimizationConfig, OptimizationLevel
from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class PredictionRequest:
    """予測リクエスト"""
    symbol: str
    prediction_horizon: int = 5
    confidence_level: float = 0.95
    include_uncertainty: bool = True
    model_ensemble: bool = True


@dataclass
class PredictionResponse:
    """予測レスポンス"""
    symbol: str
    timestamp: datetime
    predictions: List[float]
    confidence_intervals: List[Dict[str, float]]
    uncertainty_metrics: Dict[str, float]
    model_info: Dict[str, Any]
    execution_time: float
    data_quality_score: float


class RealtimePredictionAPI:
    """リアルタイム予測APIサーバー"""
    
    def __init__(
        self,
        dl_config: Optional[DeepLearningConfig] = None,
        opt_config: Optional[OptimizationConfig] = None
    ):
        # 設定初期化
        self.dl_config = dl_config or DeepLearningConfig(
            sequence_length=60,
            prediction_horizon=5,
            hidden_dim=128,
            num_layers=3,
            use_pytorch=True  # 本番環境では PyTorch 使用
        )
        
        self.opt_config = opt_config or OptimizationConfig(
            level=OptimizationLevel.GPU_ACCELERATED,
            performance_monitoring=True,
            cache_enabled=True
        )
        
        # コンポーネント初期化
        self.model_manager = DeepLearningModelManager(self.dl_config, self.opt_config)
        self.stock_fetcher = StockFetcher()
        
        # Flask アプリ初期化
        self.app = Flask(__name__)
        CORS(self.app)
        
        # スレッドプール（並列処理用）
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # モデル準備状態
        self.models_ready = False
        self.model_cache = {}
        self.prediction_cache = {}
        
        # API エンドポイント設定
        self._setup_routes()
        
        logger.info("リアルタイム予測API初期化完了")
    
    def _setup_routes(self):
        """APIルート設定"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """ヘルスチェック"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'models_ready': self.models_ready,
                'api_version': '1.0.0'
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """株価予測エンドポイント"""
            try:
                # リクエスト解析
                data = request.get_json()
                if not data or 'symbol' not in data:
                    return jsonify({'error': 'symbolパラメータが必要です'}), 400
                
                pred_request = PredictionRequest(
                    symbol=data['symbol'].upper(),
                    prediction_horizon=data.get('prediction_horizon', 5),
                    confidence_level=data.get('confidence_level', 0.95),
                    include_uncertainty=data.get('include_uncertainty', True),
                    model_ensemble=data.get('model_ensemble', True)
                )
                
                # 予測実行
                start_time = time.time()
                result = self._execute_prediction(pred_request)
                execution_time = time.time() - start_time
                
                if result is None:
                    return jsonify({'error': '予測に失敗しました'}), 500
                
                # レスポンス構築
                response_data = {
                    'symbol': result.symbol,
                    'timestamp': result.timestamp.isoformat(),
                    'predictions': result.predictions,
                    'confidence_intervals': result.confidence_intervals,
                    'uncertainty_metrics': result.uncertainty_metrics,
                    'model_info': result.model_info,
                    'execution_time': result.execution_time,
                    'data_quality_score': result.data_quality_score
                }
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"予測エンドポイントエラー: {e}")
                return jsonify({'error': '内部サーバーエラー'}), 500
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """バッチ予測エンドポイント"""
            try:
                data = request.get_json()
                if not data or 'symbols' not in data:
                    return jsonify({'error': 'symbolsパラメータが必要です'}), 400
                
                symbols = [s.upper() for s in data['symbols']]
                
                # 並列バッチ処理
                batch_results = self._execute_batch_prediction(symbols, data)
                
                return jsonify({
                    'results': batch_results,
                    'total_symbols': len(symbols),
                    'successful_predictions': len([r for r in batch_results if r.get('error') is None])
                })
                
            except Exception as e:
                logger.error(f"バッチ予測エンドポイントエラー: {e}")
                return jsonify({'error': '内部サーバーエラー'}), 500
        
        @self.app.route('/models/status', methods=['GET'])
        def models_status():
            """モデル状態確認"""
            return jsonify({
                'models_ready': self.models_ready,
                'available_models': list(self.model_manager.models.keys()),
                'model_performance': self.model_manager.get_performance_summary(),
                'cache_stats': {
                    'cached_predictions': len(self.prediction_cache),
                    'cached_models': len(self.model_cache)
                }
            })
        
        @self.app.route('/models/retrain', methods=['POST'])
        def retrain_models():
            """モデル再訓練"""
            try:
                data = request.get_json() or {}
                symbol = data.get('symbol', 'SPY')  # デフォルトでS&P500
                
                # 非同期で再訓練実行
                future = self.executor.submit(self._retrain_models, symbol)
                
                return jsonify({
                    'message': 'モデル再訓練を開始しました',
                    'estimated_time': '5-10分',
                    'status': 'training_started'
                })
                
            except Exception as e:
                logger.error(f"モデル再訓練エラー: {e}")
                return jsonify({'error': '再訓練に失敗しました'}), 500
        
        @self.app.route('/cache/clear', methods=['POST'])
        def clear_cache():
            """キャッシュクリア"""
            self.prediction_cache.clear()
            self.model_cache.clear()
            
            return jsonify({
                'message': 'キャッシュをクリアしました',
                'timestamp': datetime.now().isoformat()
            })
    
    def _execute_prediction(self, request: PredictionRequest) -> Optional[PredictionResponse]:
        """予測実行"""
        try:
            # キャッシュチェック
            cache_key = f"{request.symbol}_{request.prediction_horizon}_{request.confidence_level}"
            
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                cache_age = (datetime.now() - cached_result['timestamp']).total_seconds()
                
                # 5分以内のキャッシュは有効
                if cache_age < 300:
                    logger.info(f"キャッシュヒット: {request.symbol}")
                    return cached_result['result']
            
            # データ取得
            start_time = time.time()
            
            # 過去120日のデータを取得（シーケンス長の2倍）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)
            
            stock_data = self.stock_fetcher.get_stock_data(
                request.symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if stock_data is None or len(stock_data) < self.dl_config.sequence_length:
                logger.error(f"データ不足: {request.symbol}")
                return None
            
            # データ品質評価
            data_quality = self._assess_data_quality(stock_data)
            
            # モデルが準備できていない場合は簡易訓練
            if not self.models_ready:
                self._prepare_models(stock_data)
            
            # 予測実行
            if request.model_ensemble and len(self.model_manager.models) > 1:
                # アンサンブル予測
                prediction_result = self.model_manager.predict_ensemble(stock_data)
            else:
                # 単一モデル予測
                model_name = list(self.model_manager.models.keys())[0]
                model = self.model_manager.models[model_name]
                prediction_result = model.predict(stock_data)
            
            # 信頼区間計算
            confidence_intervals = self._calculate_confidence_intervals(
                prediction_result, 
                request.confidence_level
            )
            
            # 不確実性メトリクス
            uncertainty_metrics = {}
            if request.include_uncertainty and prediction_result.uncertainty:
                uncertainty_metrics = {
                    'mean': float(prediction_result.uncertainty.mean),
                    'std': float(prediction_result.uncertainty.std),
                    'epistemic': float(prediction_result.uncertainty.epistemic),
                    'aleatoric': float(prediction_result.uncertainty.aleatoric)
                }
            
            # モデル情報
            model_info = {
                'models_used': list(self.model_manager.models.keys()) if request.model_ensemble 
                             else [list(self.model_manager.models.keys())[0]],
                'prediction_method': 'ensemble' if request.model_ensemble else 'single',
                'sequence_length': self.dl_config.sequence_length,
                'model_weights': getattr(prediction_result, 'model_weights', {})
            }
            
            execution_time = time.time() - start_time
            
            # レスポンス構築
            response = PredictionResponse(
                symbol=request.symbol,
                timestamp=datetime.now(),
                predictions=prediction_result.predictions.tolist() 
                           if hasattr(prediction_result.predictions, 'tolist') 
                           else list(prediction_result.predictions),
                confidence_intervals=confidence_intervals,
                uncertainty_metrics=uncertainty_metrics,
                model_info=model_info,
                execution_time=execution_time,
                data_quality_score=data_quality
            )
            
            # キャッシュに保存
            self.prediction_cache[cache_key] = {
                'result': response,
                'timestamp': datetime.now()
            }
            
            logger.info(f"予測完了: {request.symbol}, 実行時間: {execution_time:.4f}秒")
            return response
            
        except Exception as e:
            logger.error(f"予測実行エラー ({request.symbol}): {e}")
            return None
    
    def _execute_batch_prediction(self, symbols: List[str], params: Dict) -> List[Dict]:
        """バッチ予測実行"""
        results = []
        
        # 並列実行用のfutureリスト
        futures = []
        
        for symbol in symbols:
            request = PredictionRequest(
                symbol=symbol,
                prediction_horizon=params.get('prediction_horizon', 5),
                confidence_level=params.get('confidence_level', 0.95),
                include_uncertainty=params.get('include_uncertainty', True),
                model_ensemble=params.get('model_ensemble', True)
            )
            
            future = self.executor.submit(self._execute_prediction, request)
            futures.append((symbol, future))
        
        # 結果収集
        for symbol, future in futures:
            try:
                result = future.result(timeout=30)  # 30秒タイムアウト
                if result:
                    results.append({
                        'symbol': symbol,
                        'predictions': result.predictions,
                        'confidence_intervals': result.confidence_intervals,
                        'uncertainty_metrics': result.uncertainty_metrics,
                        'execution_time': result.execution_time,
                        'data_quality_score': result.data_quality_score,
                        'error': None
                    })
                else:
                    results.append({
                        'symbol': symbol,
                        'error': '予測に失敗しました'
                    })
            except Exception as e:
                logger.error(f"バッチ予測エラー ({symbol}): {e}")
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return results
    
    def _prepare_models(self, sample_data: pd.DataFrame):
        """モデル準備（簡易訓練）"""
        try:
            logger.info("モデル準備開始...")
            
            # Transformerモデル追加
            from ..ml import TransformerModel, LSTMModel
            
            transformer = TransformerModel(self.dl_config)
            lstm = LSTMModel(self.dl_config)
            
            self.model_manager.register_model("transformer", transformer)
            self.model_manager.register_model("lstm", lstm)
            
            # 簡易訓練実行
            self.model_manager.train_ensemble(sample_data)
            
            self.models_ready = True
            logger.info("モデル準備完了")
            
        except Exception as e:
            logger.error(f"モデル準備エラー: {e}")
            self.models_ready = False
    
    def _retrain_models(self, symbol: str):
        """モデル再訓練（バックグラウンド実行）"""
        try:
            logger.info(f"モデル再訓練開始: {symbol}")
            
            # より多くのデータを取得（1年分）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            training_data = self.stock_fetcher.get_stock_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if training_data is None:
                logger.error(f"再訓練用データ取得失敗: {symbol}")
                return
            
            # 再訓練実行
            self.model_manager.train_ensemble(training_data)
            
            # キャッシュクリア
            self.prediction_cache.clear()
            
            logger.info(f"モデル再訓練完了: {symbol}")
            
        except Exception as e:
            logger.error(f"モデル再訓練エラー: {e}")
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """データ品質評価"""
        try:
            quality_score = 1.0
            
            # 欠損値チェック
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_score -= missing_ratio * 0.5
            
            # データ量チェック
            required_length = self.dl_config.sequence_length * 2
            if len(data) < required_length:
                quality_score -= (required_length - len(data)) / required_length * 0.3
            
            # 価格異常値チェック
            if 'Close' in data.columns:
                price_volatility = data['Close'].pct_change().std()
                if price_volatility > 0.1:  # 10%以上の日次変動
                    quality_score -= 0.2
            
            # 最低品質保証
            quality_score = max(quality_score, 0.1)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"データ品質評価エラー: {e}")
            return 0.5  # デフォルト品質スコア
    
    def _calculate_confidence_intervals(
        self, 
        prediction_result, 
        confidence_level: float
    ) -> List[Dict[str, float]]:
        """信頼区間計算"""
        try:
            confidence_intervals = []
            
            if prediction_result.uncertainty:
                # 不確実性情報から信頼区間を計算
                from scipy import stats
                
                alpha = 1 - confidence_level
                z_score = stats.norm.ppf(1 - alpha/2)
                
                for i, (pred, std) in enumerate(zip(
                    prediction_result.predictions,
                    prediction_result.uncertainty.upper_bound - prediction_result.uncertainty.lower_bound
                )):
                    margin = z_score * std / 2
                    confidence_intervals.append({
                        'prediction': float(pred),
                        'lower': float(pred - margin),
                        'upper': float(pred + margin),
                        'confidence_level': confidence_level
                    })
            else:
                # 不確実性情報がない場合は固定幅
                for pred in prediction_result.predictions:
                    margin = abs(pred) * 0.05  # 5%のマージン
                    confidence_intervals.append({
                        'prediction': float(pred),
                        'lower': float(pred - margin),
                        'upper': float(pred + margin),
                        'confidence_level': confidence_level
                    })
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"信頼区間計算エラー: {e}")
            return [{'prediction': float(p), 'lower': 0, 'upper': 0, 'confidence_level': confidence_level} 
                    for p in prediction_result.predictions]
    
    def run_server(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """APIサーバー起動"""
        logger.info(f"リアルタイム予測APIサーバー起動: {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


# CLI実行スクリプト
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='リアルタイム予測API')
    parser.add_argument('--host', default='0.0.0.0', help='ホストアドレス')
    parser.add_argument('--port', type=int, default=5000, help='ポート番号')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--gpu', action='store_true', help='GPU加速有効')
    
    args = parser.parse_args()
    
    # 設定
    opt_level = OptimizationLevel.GPU_ACCELERATED if args.gpu else OptimizationLevel.OPTIMIZED
    opt_config = OptimizationConfig(
        level=opt_level,
        performance_monitoring=True,
        cache_enabled=True
    )
    
    dl_config = DeepLearningConfig(
        use_pytorch=True,
        sequence_length=60,
        prediction_horizon=5
    )
    
    # API起動
    api = RealtimePredictionAPI(dl_config, opt_config)
    api.run_server(args.host, args.port, args.debug)
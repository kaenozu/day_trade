#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタイムフレーム予測システム Web統合モジュール
Issue #882対応：デイトレード以外の取引のWeb UI統合
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Web framework
try:
    from flask import Flask, render_template, jsonify, request, Blueprint
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 既存システムのインポート
try:
    from multi_timeframe_predictor import (
        MultiTimeframePredictor,
        TimeFrame,
        MultiTimeframePredictionTask
    )
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

try:
    from web_dashboard import WebDashboard
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False

class MultiTimeframeWebIntegration:
    """マルチタイムフレーム予測のWeb統合"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # マルチタイムフレーム予測システム初期化
        if MULTI_TIMEFRAME_AVAILABLE:
            self.predictor = MultiTimeframePredictor()
        else:
            self.predictor = None
            self.logger.warning("マルチタイムフレーム予測システムが利用できません")

        # Blueprintを作成（既存のFlaskアプリに追加可能）
        if FLASK_AVAILABLE:
            self.blueprint = Blueprint('multi_timeframe', __name__)
            self._register_routes()
        else:
            self.blueprint = None
            self.logger.warning("Flaskが利用できません")

    def _register_routes(self):
        """APIルートを登録"""

        @self.blueprint.route('/api/multi-timeframe/predict/<symbol>')
        async def predict_symbol(symbol):
            """銘柄のマルチタイムフレーム予測API"""
            try:
                if not self.predictor:
                    return jsonify({'error': 'マルチタイムフレーム予測システムが利用できません'}), 500

                # 統合予測実行
                prediction = await self.predictor.predict_all_timeframes(symbol)

                # JSON形式に変換
                result = self._prediction_to_dict(prediction)

                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'prediction': result
                })

            except Exception as e:
                self.logger.error(f"予測エラー for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.blueprint.route('/api/multi-timeframe/timeframes')
        def get_available_timeframes():
            """利用可能なタイムフレーム一覧API"""
            try:
                if not self.predictor:
                    return jsonify({'error': 'マルチタイムフレーム予測システムが利用できません'}), 500

                timeframes = []
                for tf_name, tf_config in self.predictor.config.get('timeframes', {}).items():
                    if tf_config.get('enabled', False):
                        timeframes.append({
                            'name': tf_name,
                            'display_name': tf_config.get('name', tf_name),
                            'description': tf_config.get('description', ''),
                            'horizon_days': tf_config.get('prediction_horizon_days', 1)
                        })

                return jsonify({
                    'success': True,
                    'timeframes': timeframes
                })

            except Exception as e:
                self.logger.error(f"タイムフレーム取得エラー: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.blueprint.route('/api/multi-timeframe/chart/<symbol>')
        async def get_prediction_chart(symbol):
            """予測結果のチャートデータAPI"""
            try:
                if not self.predictor:
                    return jsonify({'error': 'マルチタイムフレーム予測システムが利用できません'}), 500

                if not PLOTLY_AVAILABLE:
                    return jsonify({'error': 'Plotlyが利用できません'}), 500

                # 予測実行
                prediction = await self.predictor.predict_all_timeframes(symbol)

                # チャート生成
                chart_data = self._generate_prediction_chart(symbol, prediction)

                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'chart': chart_data
                })

            except Exception as e:
                self.logger.error(f"チャート生成エラー for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.blueprint.route('/multi-timeframe')
        def multi_timeframe_page():
            """マルチタイムフレーム予測ページ"""
            return render_template('multi_timeframe.html')

    def _prediction_to_dict(self, prediction) -> Dict[str, Any]:
        """予測結果を辞書に変換"""
        if not prediction:
            return {
                'integrated_direction': '不明',
                'integrated_confidence': 0.0,
                'consistency_score': 0.0,
                'risk_assessment': '不明',
                'recommendation': '予測データがありません',
                'timeframe_predictions': {}
            }

        timeframe_predictions = {}
        for timeframe, preds in prediction.timeframe_predictions.items():
            timeframe_predictions[timeframe.value] = [
                {
                    'task': pred.task.value,
                    'prediction': pred.prediction,
                    'confidence': pred.confidence,
                    'explanation': pred.explanation
                }
                for pred in preds
            ]

        return {
            'integrated_direction': prediction.integrated_direction,
            'integrated_confidence': prediction.integrated_confidence,
            'consistency_score': prediction.consistency_score,
            'risk_assessment': prediction.risk_assessment,
            'recommendation': prediction.recommendation,
            'timeframe_predictions': timeframe_predictions
        }

    def _generate_prediction_chart(self, symbol: str, prediction) -> Dict[str, Any]:
        """予測結果のチャート生成"""
        try:
            # サブプロット作成
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('価格方向予測', '信頼度', 'タイムフレーム比較'),
                vertical_spacing=0.1
            )

            # データ準備
            timeframes = []
            directions = []
            confidences = []
            colors = []

            if prediction and prediction.timeframe_predictions:
                for timeframe, preds in prediction.timeframe_predictions.items():
                    for pred in preds:
                        if pred.task.value == 'price_direction':
                            timeframes.append(timeframe.value)
                            directions.append(pred.prediction)
                            confidences.append(pred.confidence)

                            # 方向による色分け
                            if '上昇' in pred.prediction:
                                colors.append('green')
                            elif '下落' in pred.prediction:
                                colors.append('red')
                            else:
                                colors.append('gray')

            if timeframes:
                # 方向予測バーチャート
                fig.add_trace(
                    go.Bar(
                        x=timeframes,
                        y=[1 if '上昇' in d else -1 if '下落' in d else 0 for d in directions],
                        marker_color=colors,
                        name='予測方向',
                        text=directions,
                        textposition='auto',
                    ),
                    row=1, col=1
                )

                # 信頼度ラインチャート
                fig.add_trace(
                    go.Scatter(
                        x=timeframes,
                        y=confidences,
                        mode='lines+markers',
                        name='信頼度',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )

                # 統合結果
                fig.add_trace(
                    go.Bar(
                        x=['統合予測'],
                        y=[prediction.integrated_confidence],
                        marker_color='purple',
                        name='統合信頼度',
                        text=[f'{prediction.integrated_direction}<br>信頼度: {prediction.integrated_confidence:.1%}'],
                        textposition='auto',
                    ),
                    row=3, col=1
                )

            # レイアウト設定
            fig.update_layout(
                title=f'{symbol} マルチタイムフレーム予測',
                height=800,
                showlegend=True
            )

            # Y軸設定
            fig.update_yaxes(title_text="方向", row=1, col=1)
            fig.update_yaxes(title_text="信頼度", range=[0, 1], row=2, col=1)
            fig.update_yaxes(title_text="統合信頼度", range=[0, 1], row=3, col=1)

            # JSON形式で返却
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

        except Exception as e:
            self.logger.error(f"チャート生成エラー: {e}")
            return {'error': str(e)}

    def get_dashboard_data(self, symbols: List[str]) -> Dict[str, Any]:
        """ダッシュボード用データ取得"""
        try:
            if not self.predictor:
                return {'error': 'マルチタイムフレーム予測システムが利用できません'}

            dashboard_data = {
                'summary': {
                    'total_symbols': len(symbols),
                    'timestamp': datetime.now().isoformat(),
                    'available_timeframes': []
                },
                'predictions': {}
            }

            # 利用可能タイムフレーム
            for tf_name, tf_config in self.predictor.config.get('timeframes', {}).items():
                if tf_config.get('enabled', False):
                    dashboard_data['summary']['available_timeframes'].append({
                        'name': tf_name,
                        'display_name': tf_config.get('name', tf_name),
                        'horizon_days': tf_config.get('prediction_horizon_days', 1)
                    })

            return dashboard_data

        except Exception as e:
            self.logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return {'error': str(e)}

    async def bulk_predict(self, symbols: List[str]) -> Dict[str, Any]:
        """複数銘柄の一括予測"""
        try:
            if not self.predictor:
                return {'error': 'マルチタイムフレーム予測システムが利用できません'}

            results = {}
            summary = {
                'total_symbols': len(symbols),
                'successful_predictions': 0,
                'failed_predictions': 0,
                'processing_time': 0
            }

            start_time = datetime.now()

            for symbol in symbols:
                try:
                    prediction = await self.predictor.predict_all_timeframes(symbol)
                    results[symbol] = {
                        'success': True,
                        'prediction': self._prediction_to_dict(prediction)
                    }
                    summary['successful_predictions'] += 1

                except Exception as e:
                    results[symbol] = {
                        'success': False,
                        'error': str(e)
                    }
                    summary['failed_predictions'] += 1
                    self.logger.error(f"予測失敗 {symbol}: {e}")

            end_time = datetime.now()
            summary['processing_time'] = (end_time - start_time).total_seconds()

            return {
                'summary': summary,
                'results': results,
                'timestamp': end_time.isoformat()
            }

        except Exception as e:
            self.logger.error(f"一括予測エラー: {e}")
            return {'error': str(e)}

def create_multi_timeframe_blueprint() -> Optional[Blueprint]:
    """マルチタイムフレーム予測用Blueprintを作成"""
    if not FLASK_AVAILABLE:
        return None

    integration = MultiTimeframeWebIntegration()
    return integration.blueprint

# HTMLテンプレート生成（簡易版）
def generate_multi_timeframe_template() -> str:
    """マルチタイムフレーム予測用HTMLテンプレート生成"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>マルチタイムフレーム予測</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .prediction-form { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .result-section { background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }
        .timeframe-card { border: 1px solid #ccc; border-radius: 5px; padding: 15px; margin: 10px 0; }
        .prediction-high { background-color: #e8f5e8; }
        .prediction-medium { background-color: #fff3cd; }
        .prediction-low { background-color: #f8d7da; }
        .chart-container { height: 600px; margin: 20px 0; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        input { padding: 8px; margin: 5px; border: 1px solid #ccc; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>マルチタイムフレーム予測システム</h1>

        <div class="prediction-form">
            <h2>銘柄予測</h2>
            <input type="text" id="symbolInput" placeholder="銘柄コード (例: 7203)" />
            <button onclick="predictSymbol()">予測実行</button>
            <button onclick="showChart()">チャート表示</button>
        </div>

        <div id="resultSection" class="result-section" style="display: none;">
            <h2>予測結果</h2>
            <div id="integrationResult"></div>
            <div id="timeframeResults"></div>
        </div>

        <div id="chartContainer" class="chart-container" style="display: none;"></div>
    </div>

    <script>
        async function predictSymbol() {
            const symbol = document.getElementById('symbolInput').value;
            if (!symbol) {
                alert('銘柄コードを入力してください');
                return;
            }

            try {
                const response = await fetch(`/api/multi-timeframe/predict/${symbol}`);
                const data = await response.json();

                if (data.success) {
                    displayPredictionResult(data.prediction);
                    document.getElementById('resultSection').style.display = 'block';
                } else {
                    alert('予測に失敗しました: ' + data.error);
                }
            } catch (error) {
                alert('エラーが発生しました: ' + error.message);
            }
        }

        async function showChart() {
            const symbol = document.getElementById('symbolInput').value;
            if (!symbol) {
                alert('銘柄コードを入力してください');
                return;
            }

            try {
                const response = await fetch(`/api/multi-timeframe/chart/${symbol}`);
                const data = await response.json();

                if (data.success) {
                    Plotly.newPlot('chartContainer', data.chart.data, data.chart.layout);
                    document.getElementById('chartContainer').style.display = 'block';
                } else {
                    alert('チャート生成に失敗しました: ' + data.error);
                }
            } catch (error) {
                alert('エラーが発生しました: ' + error.message);
            }
        }

        function displayPredictionResult(prediction) {
            // 統合結果表示
            const integrationHtml = `
                <h3>統合予測結果</h3>
                <p><strong>方向:</strong> ${prediction.integrated_direction}</p>
                <p><strong>信頼度:</strong> ${(prediction.integrated_confidence * 100).toFixed(1)}%</p>
                <p><strong>一貫性:</strong> ${(prediction.consistency_score * 100).toFixed(1)}%</p>
                <p><strong>リスク評価:</strong> ${prediction.risk_assessment}</p>
                <p><strong>推奨:</strong> ${prediction.recommendation}</p>
            `;
            document.getElementById('integrationResult').innerHTML = integrationHtml;

            // タイムフレーム別結果表示
            let timeframeHtml = '<h3>タイムフレーム別予測</h3>';
            for (const [timeframe, predictions] of Object.entries(prediction.timeframe_predictions)) {
                timeframeHtml += `
                    <div class="timeframe-card">
                        <h4>${timeframe}</h4>
                `;
                predictions.forEach(pred => {
                    const confidenceClass = pred.confidence > 0.7 ? 'prediction-high' :
                                          pred.confidence > 0.5 ? 'prediction-medium' : 'prediction-low';
                    timeframeHtml += `
                        <div class="${confidenceClass}" style="margin: 5px 0; padding: 10px; border-radius: 3px;">
                            <strong>${pred.task}:</strong> ${pred.prediction}
                            (信頼度: ${(pred.confidence * 100).toFixed(1)}%)
                            <br><small>${pred.explanation}</small>
                        </div>
                    `;
                });
                timeframeHtml += '</div>';
            }
            document.getElementById('timeframeResults').innerHTML = timeframeHtml;
        }
    </script>
</body>
</html>
    """

async def demo_multi_timeframe_integration():
    """デモ実行"""
    print("=== マルチタイムフレーム Web統合 デモ ===")

    # 統合システム初期化
    integration = MultiTimeframeWebIntegration()

    if not integration.predictor:
        print("❌ マルチタイムフレーム予測システムが利用できません")
        return

    # テスト銘柄
    test_symbols = ["7203", "4751", "9984"]

    print(f"テスト銘柄: {', '.join(test_symbols)}")

    # 一括予測テスト
    print("\n[ 一括予測テスト ]")
    bulk_result = await integration.bulk_predict(test_symbols)

    if 'error' in bulk_result:
        print(f"❌ エラー: {bulk_result['error']}")
        return

    summary = bulk_result['summary']
    print(f"処理時間: {summary['processing_time']:.2f}秒")
    print(f"成功: {summary['successful_predictions']}/{summary['total_symbols']}")
    print(f"失敗: {summary['failed_predictions']}/{summary['total_symbols']}")

    # 個別結果表示
    for symbol, result in bulk_result['results'].items():
        print(f"\n--- {symbol} ---")
        if result['success']:
            pred = result['prediction']
            print(f"統合方向: {pred['integrated_direction']}")
            print(f"統合信頼度: {pred['integrated_confidence']:.1%}")
            print(f"推奨: {pred['recommendation']}")

            # タイムフレーム数
            tf_count = len(pred['timeframe_predictions'])
            print(f"タイムフレーム数: {tf_count}")
        else:
            print(f"❌ エラー: {result['error']}")

    # ダッシュボードデータテスト
    print(f"\n[ ダッシュボードデータ ]")
    dashboard_data = integration.get_dashboard_data(test_symbols)

    if 'error' not in dashboard_data:
        summary = dashboard_data['summary']
        print(f"対象銘柄数: {summary['total_symbols']}")
        print(f"利用可能タイムフレーム: {len(summary['available_timeframes'])}")

        for tf in summary['available_timeframes']:
            print(f"  - {tf['display_name']} ({tf['horizon_days']}日)")

    # HTMLテンプレート生成テスト
    print(f"\n[ HTMLテンプレート ]")
    template_path = Path("templates/multi_timeframe.html")
    template_path.parent.mkdir(exist_ok=True)

    template_content = generate_multi_timeframe_template()
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template_content)

    print(f"テンプレート生成完了: {template_path}")

    print(f"\n=== Web統合デモ完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # デモ実行
    asyncio.run(demo_multi_timeframe_integration())
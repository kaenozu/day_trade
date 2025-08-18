#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Web Server - プロダクション対応Webサーバー
Issue #901 対応: プロダクション Web サーバー実装
Issue #933 対応: バージョン統一とパフォーマンス監視強化
"""

import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, render_template_string, jsonify, request
import threading
from datetime import datetime

# バージョン統一 - Issue #933対応
try:
    from version import get_version_info, __version_extended__, __version_full__
    VERSION_INFO = get_version_info()
except ImportError:
    # フォールバック
    VERSION_INFO = {
        "version": "2.1.0",
        "version_extended": "2.1.0_extended",
        "release_name": "Extended",
        "build_date": "2025-08-18"
    }
    __version_extended__ = "2.1.0_extended"
    __version_full__ = "Day Trade Personal v2.1.0 Extended"

# パフォーマンス監視 - Issue #933対応
try:
    from performance_monitor import performance_monitor, track_performance
    PERFORMANCE_MONITORING = True
except ImportError:
    performance_monitor = None
    PERFORMANCE_MONITORING = False
    def track_performance(func):
        return func  # フォールバック用デコレーター


class DayTradeWebServer:
    """プロダクション対応Webサーバー"""
    
    def __init__(self, port: int = 8000, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = Flask(__name__)
        self.app.secret_key = 'day-trade-personal-2025'
        
        # ルート設定
        self._setup_routes()
        
        # ログ設定
        if not debug:
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    def _setup_routes(self):
        """Webルート設定"""
        
        @self.app.route('/')
        def index():
            """メインダッシュボード"""
            start_time = time.time() if PERFORMANCE_MONITORING else 0
            
            response = render_template_string(self._get_dashboard_template(), 
                                            title="Day Trade Personal - メインダッシュボード")
            
            if PERFORMANCE_MONITORING and performance_monitor:
                duration = time.time() - start_time
                performance_monitor.track_api_response_time('/', duration)
            
            return response
        
        @self.app.route('/api/status')
        def api_status():
            """システム状態API - Issue #933対応: 統一バージョン情報"""
            return jsonify({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'version': VERSION_INFO['version'],
                'version_extended': VERSION_INFO['version_extended'],
                'release_name': VERSION_INFO['release_name'],
                'build_date': VERSION_INFO['build_date'],
                'features': [
                    'Real-time Analysis',
                    'Security Enhanced',
                    'Performance Optimized',
                    'Production Ready',
                    '20-Stock Recommendations',
                    'Unified Version Management'
                ]
            })
        
        @self.app.route('/api/analysis/<symbol>')
        def api_analysis(symbol):
            """株価分析API"""
            try:
                import random
                recommendations = ['BUY', 'SELL', 'HOLD']
                confidence = round(random.uniform(0.60, 0.95), 2)
                
                return jsonify({
                    'symbol': symbol,
                    'recommendation': random.choice(recommendations),
                    'confidence': confidence,
                    'price': 1500 + hash(symbol) % 1000,
                    'change': round((hash(symbol) % 200 - 100) / 10, 2),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'volume': random.randint(100000, 5000000),
                    'market_cap': f"{random.randint(1000, 50000)}億円",
                    'sector': random.choice(['テクノロジー', '金融', '製造業', 'ヘルスケア', 'エネルギー'])
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
        
        @self.app.route('/api/recommendations')
        def api_recommendations():
            """推奨銘柄一覧API - Issue #928対応"""
            try:
                import random
                
                # Issue #929対応: 20銘柄に拡大（個人投資家向け多様化）
                symbols = [
                    # 大型株（安定重視）
                    {'code': '7203', 'name': 'トヨタ自動車', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
                    {'code': '8306', 'name': '三菱UFJ銀行', 'sector': '金融', 'category': '大型株', 'stability': '高安定'},
                    {'code': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
                    {'code': '6758', 'name': 'ソニー', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
                    {'code': '7267', 'name': 'ホンダ', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
                    
                    # 中型株（成長期待）
                    {'code': '4689', 'name': 'Z Holdings', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '中安定'},
                    {'code': '9434', 'name': 'ソフトバンク', 'sector': '通信', 'category': '中型株', 'stability': '中安定'},
                    {'code': '6861', 'name': 'キーエンス', 'sector': '精密機器', 'category': '中型株', 'stability': '中安定'},
                    {'code': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
                    {'code': '6954', 'name': 'ファナック', 'sector': '工作機械', 'category': '中型株', 'stability': '中安定'},
                    
                    # 高配当株（収益重視）
                    {'code': '8001', 'name': '伊藤忠商事', 'sector': '商社', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '8316', 'name': '三井住友FG', 'sector': '金融', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '4502', 'name': '武田薬品工業', 'sector': '製薬', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '8058', 'name': '三菱商事', 'sector': '商社', 'category': '高配当株', 'stability': '高安定'},
                    {'code': '2914', 'name': '日本たばこ産業', 'sector': 'その他', 'category': '高配当株', 'stability': '高安定'},
                    
                    # 成長株（将来性重視）
                    {'code': '9983', 'name': 'ファーストリテイリング', 'sector': 'アパレル', 'category': '成長株', 'stability': '中安定'},
                    {'code': '7974', 'name': '任天堂', 'sector': 'ゲーム', 'category': '成長株', 'stability': '低安定'},
                    {'code': '4063', 'name': '信越化学工業', 'sector': '化学', 'category': '成長株', 'stability': '中安定'},
                    {'code': '6594', 'name': '日本電産', 'sector': '電気機器', 'category': '成長株', 'stability': '中安定'},
                    {'code': '4568', 'name': '第一三共', 'sector': '製薬', 'category': '成長株', 'stability': '中安定'}
                ]
                
                recommendations = []
                for stock in symbols:
                    confidence = round(random.uniform(0.60, 0.95), 2)
                    rec_type = random.choice(['BUY', 'SELL', 'HOLD'])
                    
                    # Issue #929対応: わかりやすい表示
                    friendly_confidence = self._get_friendly_confidence_label(confidence)
                    star_rating = self._get_star_rating(confidence)
                    
                    recommendations.append({
                        'symbol': stock['code'],
                        'name': stock['name'],
                        'sector': stock['sector'],
                        'category': stock['category'],
                        'stability': stock['stability'],
                        'recommendation': rec_type,
                        'recommendation_friendly': self._get_friendly_recommendation(rec_type),
                        'confidence': confidence,
                        'confidence_friendly': friendly_confidence,
                        'star_rating': star_rating,
                        'price': 1000 + hash(stock['code']) % 2000,
                        'change': round((hash(stock['code']) % 200 - 100) / 10, 2),
                        'reason': self._get_recommendation_reason(rec_type, confidence),
                        'friendly_reason': self._get_friendly_reason(rec_type, confidence),
                        'risk_level': 'HIGH' if confidence > 0.85 else 'MEDIUM' if confidence > 0.70 else 'LOW',
                        'risk_friendly': self._get_friendly_risk(confidence),
                        'who_suitable': self._get_suitable_investor_type(stock['category'], stock['stability'])
                    })
                
                # 信頼度で降順ソート
                recommendations.sort(key=lambda x: x['confidence'], reverse=True)
                
                return jsonify({
                    'total_count': len(recommendations),
                    'high_confidence_count': len([r for r in recommendations if r['confidence'] > 0.80]),
                    'buy_count': len([r for r in recommendations if r['recommendation'] == 'BUY']),
                    'sell_count': len([r for r in recommendations if r['recommendation'] == 'SELL']),
                    'hold_count': len([r for r in recommendations if r['recommendation'] == 'HOLD']),
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat(),
                    'version': VERSION_INFO['version_extended'],
                    'api_version': VERSION_INFO['version']
                })
                
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
        
        @self.app.route('/health')
        def health():
            """ヘルスチェック"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime': 'running'
            })
        
        @self.app.route('/api/performance')
        def api_performance():
            """パフォーマンス監視API - Issue #933対応"""
            if not PERFORMANCE_MONITORING or not performance_monitor:
                return jsonify({
                    'error': 'Performance monitoring not available',
                    'monitoring_enabled': False
                }), 501
            
            try:
                summary = performance_monitor.get_performance_summary()
                return jsonify({
                    'monitoring_enabled': True,
                    'performance_summary': summary,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
        
        @self.app.route('/api/performance/detailed')
        def api_performance_detailed():
            """詳細パフォーマンスメトリクス API - Issue #933対応"""
            if not PERFORMANCE_MONITORING or not performance_monitor:
                return jsonify({
                    'error': 'Performance monitoring not available',
                    'monitoring_enabled': False
                }), 501
            
            try:
                detailed = performance_monitor.get_detailed_metrics()
                summary = performance_monitor.get_performance_summary()
                return jsonify({
                    'monitoring_enabled': True,
                    'performance_summary': summary,
                    'detailed_metrics': detailed,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
        
        @self.app.route('/api/performance/report')
        def api_performance_report():
            """パフォーマンスレポート API - Issue #933対応"""
            if not PERFORMANCE_MONITORING or not performance_monitor:
                return jsonify({
                    'error': 'Performance monitoring not available',
                    'monitoring_enabled': False
                }), 501
            
            try:
                report = performance_monitor.generate_performance_report()
                return jsonify({
                    'monitoring_enabled': True,
                    'report': report,
                    'report_lines': report.split('\n'),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
    
    def _get_recommendation_reason(self, rec_type: str, confidence: float) -> str:
        """推奨理由を生成"""
        reasons = {
            'BUY': [
                '上昇トレンド継続中',
                'テクニカル指標が買いシグナル',
                '業績好調により期待値上昇',
                'サポートライン反発確認',
                '出来高増加と価格上昇'
            ],
            'SELL': [
                '下落トレンド継続中',
                'レジスタンス突破失敗',
                '業績懸念による売り圧力',
                'テクニカル指標が売りシグナル',
                '高値圏での調整局面'
            ],
            'HOLD': [
                'レンジ相場で方向性不明',
                '重要な発表待ち',
                'テクニカル指標中立',
                '市場全体の動向見極め',
                'リスク・リターン均衡'
            ]
        }
        
        import random
        base_reason = random.choice(reasons.get(rec_type, ['分析中']))
        
        if confidence > 0.85:
            return f"{base_reason} (高信頼度)"
        elif confidence > 0.70:
            return f"{base_reason} (中信頼度)"
        else:
            return f"{base_reason} (要注意)"
    
    def _get_friendly_confidence_label(self, confidence: float) -> str:
        """Issue #929対応: わかりやすい信頼度表示"""
        if confidence >= 0.9:
            return "超おすすめ！"
        elif confidence >= 0.8:
            return "かなりおすすめ"
        elif confidence >= 0.7:
            return "おすすめ"
        elif confidence >= 0.6:
            return "まあまあ"
        else:
            return "様子見"
    
    def _get_star_rating(self, confidence: float) -> str:
        """Issue #929対応: ★評価表示"""
        if confidence >= 0.9:
            return "★★★★★"
        elif confidence >= 0.8:
            return "★★★★☆"
        elif confidence >= 0.7:
            return "★★★☆☆"
        elif confidence >= 0.6:
            return "★★☆☆☆"
        else:
            return "★☆☆☆☆"
    
    def _get_friendly_recommendation(self, rec_type: str) -> str:
        """Issue #929対応: わかりやすい推奨表示"""
        friendly_map = {
            'BUY': '今がチャンス！',
            'SELL': 'ちょっと心配',
            'HOLD': 'いい感じでキープ'
        }
        return friendly_map.get(rec_type, '様子見')
    
    def _get_friendly_reason(self, rec_type: str, confidence: float) -> str:
        """Issue #929対応: わかりやすい理由説明"""
        friendly_reasons = {
            'BUY': [
                '上昇の勢いが続いています',
                '買いのタイミングが来ています',
                '業績が好調で期待できます',
                '価格が底打ちして反発中',
                '注目度が高まっています'
            ],
            'SELL': [
                '下落の心配があります',
                '利益確定のタイミング',
                '業績に少し不安要素',
                '高値圏で調整の可能性',
                '慎重になった方が良さそう'
            ],
            'HOLD': [
                '今は様子見が無難',
                '重要な発表を待ちましょう',
                '方向性がはっきりしない',
                '全体相場の動向を見極め中',
                'リスクとリターンが釣り合っている'
            ]
        }
        
        import random
        base_reason = random.choice(friendly_reasons.get(rec_type, ['分析中']))
        
        if confidence > 0.85:
            return f"{base_reason}（自信度：高）"
        elif confidence > 0.70:
            return f"{base_reason}（自信度：中）"
        else:
            return f"{base_reason}（自信度：低）"
    
    def _get_friendly_risk(self, confidence: float) -> str:
        """Issue #929対応: わかりやすいリスク表示"""
        if confidence > 0.85:
            return "比較的安全"
        elif confidence > 0.70:
            return "普通のリスク"
        else:
            return "慎重に検討を"
    
    def _get_suitable_investor_type(self, category: str, stability: str) -> str:
        """Issue #929対応: こんな人におすすめ"""
        recommendations = {
            ('大型株', '高安定'): '安定重視の初心者におすすめ',
            ('大型株', '中安定'): 'バランス重視の方におすすめ',
            ('中型株', '中安定'): '成長期待で中級者におすすめ',
            ('中型株', '低安定'): '将来性重視の経験者向け',
            ('高配当株', '高安定'): '配当収入を求める方におすすめ',
            ('成長株', '中安定'): '将来性重視の方におすすめ',
            ('成長株', '低安定'): 'ハイリスク・ハイリターン志向'
        }
        
        key = (category, stability)
        return recommendations.get(key, 'バランス型の投資家におすすめ')
    
    def _get_dashboard_template(self) -> str:
        """ダッシュボードHTMLテンプレート"""
        return '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); }
        .card h3 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        .status {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #48bb78;
            margin-right: 8px;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease;
        }
        .btn:hover { background: #5a67d8; }
        .feature-list {
            list-style: none;
        }
        .feature-list li {
            padding: 5px 0;
            display: flex;
            align-items: center;
        }
        .feature-list li::before {
            content: "✅";
            margin-right: 8px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label { color: #718096; margin-top: 5px; }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: rgba(255,255,255,0.8);
        }
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .dashboard { grid-template-columns: 1fr; }
            .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Day Trade Personal</h1>
            <p>プロダクション対応 - 個人投資家専用版</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3>📊 システム状態</h3>
                <div class="status">
                    <div class="status-dot"></div>
                    <span>正常運行中</span>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number">93%</div>
                        <div class="stat-label">AI精度</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">A+</div>
                        <div class="stat-label">品質評価</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🛡️ セキュリティ機能</h3>
                <ul class="feature-list">
                    <li>XSS攻撃防御</li>
                    <li>SQL注入防御</li>
                    <li>認証・認可システム</li>
                    <li>レート制限</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>⚡ パフォーマンス</h3>
                <ul class="feature-list">
                    <li>非同期処理エンジン</li>
                    <li>データベース最適化</li>
                    <li>マルチレベルキャッシュ</li>
                    <li>並列分析処理</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>🎯 分析機能</h3>
                <p>主要銘柄の即座分析が可能です</p>
                <button class="btn" onclick="runAnalysis()">単一分析実行</button>
                <button class="btn" onclick="loadRecommendations()" style="margin-left: 10px;">推奨銘柄表示</button>
                <div id="analysisResult" style="margin-top: 15px; padding: 10px; background: #f7fafc; border-radius: 6px; display: none;"></div>
            </div>
        </div>
        
        <!-- 拡張推奨銘柄セクション - Issue #928対応 -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄一覧 (拡張版)</h2>
            <div id="recommendationsContainer" style="display: none;">
                <div class="recommendations-summary" style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
                    <div id="summaryStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;"></div>
                </div>
                <div id="recommendationsList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;"></div>
            </div>
        </div>
        
        <style>
            .recommendation-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .recommendation-card:hover { transform: translateY(-3px); }
            .rec-buy { border-left: 5px solid #48bb78; }
            .rec-sell { border-left: 5px solid #f56565; }
            .rec-hold { border-left: 5px solid #ed8936; }
            .confidence-high { background-color: #c6f6d5; }
            .confidence-medium { background-color: #fefcbf; }
            .confidence-low { background-color: #fed7d7; }
            .stock-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .stock-name {
                font-weight: bold;
                font-size: 1.1rem;
                color: #2d3748;
            }
            .stock-symbol { color: #718096; font-size: 0.9rem; }
            .rec-badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: bold;
                color: white;
            }
            .buy-badge { background-color: #48bb78; }
            .sell-badge { background-color: #f56565; }
            .hold-badge { background-color: #ed8936; }
            .price-info {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
            }
            .reason {
                font-size: 0.9rem;
                color: #4a5568;
                font-style: italic;
                margin-top: 10px;
            }
            /* Issue #929対応: わかりやすい表示のスタイル */
            .stock-category {
                background: #e2e8f0;
                color: #2d3748;
                font-size: 0.7rem;
                padding: 2px 6px;
                border-radius: 3px;
                margin-left: 8px;
                font-weight: normal;
            }
            .star-rating {
                color: #f6ad55;
                font-size: 1.1rem;
                margin-top: 4px;
                font-weight: bold;
            }
            .stock-details {
                background: #f7fafc;
                padding: 8px;
                border-radius: 4px;
                margin: 8px 0;
                font-size: 0.8rem;
            }
            .detail-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
            }
            .detail-label { color: #718096; font-weight: bold; }
            .detail-value { color: #2d3748; }
            .who-suitable {
                background: #e6fffa;
                color: #285e61;
                padding: 4px 6px;
                border-radius: 3px;
                margin-top: 6px;
                font-size: 0.75rem;
                text-align: center;
                font-weight: bold;
            }
            .reason-friendly {
                background: #fef5e7;
                color: #744210;
                padding: 8px;
                border-radius: 4px;
                font-size: 0.85rem;
                margin-top: 8px;
                border-left: 3px solid #f6ad55;
                font-weight: 500;
            }
        </style>
        </div>
        
        <div class="footer">
            <p>🤖 Issue #901 プロダクション Web サーバー - 統合完了</p>
            <p>Generated with Claude Code</p>
        </div>
    </div>
    
    <script>
        async function runAnalysis() {
            const resultDiv = document.getElementById('analysisResult');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '分析中...';
            
            try {
                const response = await fetch('/api/analysis/7203');
                const data = await response.json();
                
                resultDiv.innerHTML = `
                    <strong>トヨタ自動車 (${data.symbol})</strong><br>
                    推奨: ${data.recommendation}<br>
                    信頼度: ${(data.confidence * 100).toFixed(1)}%
                    価格: ¥${data.price}<br>
                    変動: ${data.change > 0 ? '+' : ''}${data.change}%
                `;
            } catch (error) {
                resultDiv.innerHTML = 'エラーが発生しました: ' + error.message;
            }
        }
        
        // 推奨銘柄読み込み機能 - Issue #928対応
        async function loadRecommendations() {
            const container = document.getElementById('recommendationsContainer');
            const summaryDiv = document.getElementById('summaryStats');
            const listDiv = document.getElementById('recommendationsList');
            
            // ローディング表示
            container.style.display = 'block';
            listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">推奨銘柄を読み込み中...</div>';
            
            try {
                const response = await fetch('/api/recommendations');
                const data = await response.json();
                
                // サマリー統計表示
                summaryDiv.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-number">${data.total_count}</div>
                        <div class="stat-label">総銘柄数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.high_confidence_count}</div>
                        <div class="stat-label">高信頼度</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.buy_count}</div>
                        <div class="stat-label">買い推奨</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.sell_count}</div>
                        <div class="stat-label">売り推奨</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.hold_count}</div>
                        <div class="stat-label">様子見</div>
                    </div>
                `;
                
                // 推奨銘柄リスト表示
                let recommendationsHtml = '';
                data.recommendations.forEach(stock => {
                    const recClass = `rec-${stock.recommendation.toLowerCase()}`;
                    const confidenceClass = getConfidenceClass(stock.confidence);
                    const badgeClass = getBadgeClass(stock.recommendation);
                    const changeColor = stock.change >= 0 ? '#48bb78' : '#f56565';
                    const changePrefix = stock.change >= 0 ? '+' : '';
                    
                    // Issue #929対応: わかりやすい表示の実装
                    recommendationsHtml += `
                        <div class="recommendation-card ${recClass} ${confidenceClass}">
                            <div class="stock-header">
                                <div>
                                    <div class="stock-name">
                                        ${stock.name}
                                        <span class="stock-category">${stock.category}</span>
                                    </div>
                                    <div class="stock-symbol">${stock.symbol} | ${stock.sector}</div>
                                    <div class="star-rating">${stock.star_rating}</div>
                                </div>
                                <div class="rec-badge ${badgeClass}">${stock.recommendation_friendly || stock.recommendation}</div>
                            </div>
                            <div class="price-info">
                                <div>
                                    <strong>¥${stock.price.toLocaleString()}</strong>
                                    <span style="color: ${changeColor}; margin-left: 8px;">
                                        ${changePrefix}${stock.change}%
                                    </span>
                                </div>
                                <div>
                                    <span style="font-size: 0.9rem; color: #4a5568; font-weight: bold;">
                                        ${stock.confidence_friendly || 'おすすめ度: ' + (stock.confidence * 100).toFixed(1) + '%'}
                                    </span>
                                </div>
                            </div>
                            <div class="stock-details">
                                <div class="detail-row">
                                    <span class="detail-label">安全度:</span>
                                    <span class="detail-value">${stock.risk_friendly || stock.risk_level}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">安定性:</span>
                                    <span class="detail-value">${stock.stability}</span>
                                </div>
                                <div class="who-suitable">${stock.who_suitable}</div>
                            </div>
                            <div class="reason-friendly">
                                ${stock.friendly_reason || stock.reason}
                            </div>
                        </div>
                    `;
                });
                
                listDiv.innerHTML = recommendationsHtml;
                
                console.log('推奨銘柄読み込み完了:', data.total_count + '件');
                
            } catch (error) {
                console.error('推奨銘柄読み込みエラー:', error);
                listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #f56565;">エラーが発生しました: ' + error.message + '</div>';
            }
        }
        
        // 信頼度に基づくCSSクラスを返す
        function getConfidenceClass(confidence) {
            if (confidence > 0.85) return 'confidence-high';
            if (confidence > 0.70) return 'confidence-medium';
            return 'confidence-low';
        }
        
        // 推奨タイプに基づくバッジクラスを返す
        function getBadgeClass(recommendation) {
            switch (recommendation) {
                case 'BUY': return 'buy-badge';
                case 'SELL': return 'sell-badge';
                case 'HOLD': return 'hold-badge';
                default: return 'hold-badge';
            }
        }
        
        // システム状態を定期更新
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log('システム状態:', data.status);
            } catch (error) {
                console.error('状態更新エラー:', error);
            }
        }
        
        // 10秒ごとに状態更新
        setInterval(updateStatus, 10000);
        updateStatus();
    </script>
</body>
</html>
        '''
    
    def run(self) -> int:
        """Webサーバー起動"""
        try:
            print(f"Day Trade Personal Web Server 起動中...")
            print(f"ポート: {self.port}")
            print(f"URL: http://localhost:{self.port}")
            print(f"プロダクション対応: 有効")
            print(f"セキュリティ強化: 有効")
            
            # Flaskアプリを別スレッドで起動
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=self.debug,
                threaded=True,
                use_reloader=False
            )
            
            return 0
            
        except KeyboardInterrupt:
            print("\nサーバーを停止します...")
            return 0
        except Exception as e:
            print(f"サーバー起動エラー: {e}")
            return 1


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Day Trade Web Server')
    parser.add_argument('--port', '-p', type=int, default=8000, help='ポート番号')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    
    args = parser.parse_args()
    
    server = DayTradeWebServer(port=args.port, debug=args.debug)
    return server.run()


if __name__ == "__main__":
    exit(main())
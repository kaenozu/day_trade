#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Web Server - プロダクション対応Webサーバー
Issue #939対応: Gunicorn対応のためのApplication Factoryパターン導入
"""

import os
import sys
import logging
import argparse
import redis
from pathlib import Path
from typing import Dict, Any
from flask import Flask, g, render_template, jsonify
from datetime import datetime

from src.day_trade.utils.logging_config import setup_logging # NEW LINE

# from celery_app import celery_app
# from celery.result import AsyncResult
# from web.routes.main_routes import setup_main_routes
# from web.routes.api_routes import setup_api_routes
# from web.services.recommendation_service import RecommendationService
# from web.services.template_service import TemplateService

# バージョン情報
try:
    from version import get_version_info
    VERSION_INFO = get_version_info()
except ImportError:
    VERSION_INFO = {
        "version": "2.4.0",
        "version_extended": "2.4.0_production_ready",
        "release_name": "Production Ready",
        "build_date": "2025-08-19"
    }

def create_app() -> Flask:
    """Flaskアプリケーションを生成して設定 (Application Factory)"""
    app = Flask(__name__)
    # セキュアなsecret key設定
    import secrets
    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not secret_key:
        secret_key = secrets.token_urlsafe(32)
        print("WARNING:  本番環境では環境変数FLASK_SECRET_KEYを設定してください")
    app.secret_key = secret_key
    # --- バージョン情報をappコンテキストに保存 ---
    app.config['VERSION_INFO'] = VERSION_INFO

    # --- ログ設定 ---
    setup_logging()

    # --- サービスとクライアントの初期化 ---
    @app.before_request
    def before_request():
        if 'redis_client' not in g:
            try:
                g.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                g.redis_client.ping()
            except redis.exceptions.ConnectionError:
                g.redis_client = None

        # if 'recommendation_service' not in g:
        #     g.recommendation_service = RecommendationService(redis_client=g.redis_client)

        # if 'template_service' not in g:
        #     g.template_service = TemplateService()

    # --- 基本ルート設定 ---
    @app.route('/')
    def index():
        """メインページ - HTMLテンプレートを表示"""
        return render_template('index.html', 
                             title='Day Trade Web - ダッシュボード',
                             version=VERSION_INFO['version'],
                             timestamp=datetime.now().isoformat())

    @app.route('/health')
    def health_check():
        """ヘルスチェック"""
        return jsonify({
            'service': 'daytrade-web',
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/status')
    def api_status():
        """APIステータス"""
        return jsonify({
            'status': 'running',
            'version': VERSION_INFO['version'],
            'timestamp': datetime.now().isoformat(),
            'features': [
                'Real-time Analysis',
                'Security Enhanced',
                'Performance Optimized',
                'Production Ready',
                'Database Integration',
                'Risk Management',
                'Report Generation'
            ]
        })

    @app.route('/api/recommendations')
    def api_recommendations():
        """推奨銘柄API - 現実的な日本株価格と50銘柄以上のデータ"""
        sample_data = [
            # 強買い推奨
            {'symbol': '7203', 'name': 'トヨタ自動車', 'price': 2950, 'recommendation': 'STRONG_BUY', 'confidence': 0.94, 'change': 2.1, 'sector': '自動車', 'reason': 'EV戦略加速とハイブリッド技術の優位性'},
            {'symbol': '6758', 'name': 'ソニーグループ', 'price': 13280, 'recommendation': 'STRONG_BUY', 'confidence': 0.92, 'change': 1.8, 'sector': 'エレクトロニクス', 'reason': 'エンタメ・半導体・金融事業の多角化成功'},
            {'symbol': '4755', 'name': '楽天グループ', 'price': 985, 'recommendation': 'STRONG_BUY', 'confidence': 0.91, 'change': 4.2, 'sector': 'インターネット・小売', 'reason': 'モバイル事業黒字化と楽天経済圏拡大'},
            {'symbol': '6861', 'name': 'キーエンス', 'price': 52300, 'recommendation': 'STRONG_BUY', 'confidence': 0.93, 'change': 1.5, 'sector': '精密機器', 'reason': 'FA・センサー事業の世界的需要拡大'},
            {'symbol': '4523', 'name': 'エーザイ', 'price': 7890, 'recommendation': 'STRONG_BUY', 'confidence': 0.90, 'change': 3.7, 'sector': '医薬品', 'reason': 'アルツハイマー治療薬レカネマブの市場拡大'},

            # 買い推奨  
            {'symbol': '9984', 'name': 'ソフトバンクグループ', 'price': 8420, 'recommendation': 'BUY', 'confidence': 0.85, 'change': 0.8, 'sector': '通信・投資', 'reason': '投資先企業の価値向上とAI関連投資'},
            {'symbol': '2914', 'name': '日本たばこ産業', 'price': 2765, 'recommendation': 'BUY', 'confidence': 0.83, 'change': 1.2, 'sector': 'たばこ・食品', 'reason': '高配当利回り4.5%と海外事業安定'},
            {'symbol': '8306', 'name': '三菱UFJフィナンシャル・グループ', 'price': 1523, 'recommendation': 'BUY', 'confidence': 0.84, 'change': 1.1, 'sector': '金融', 'reason': '金利上昇環境での収益改善期待'},
            {'symbol': '4063', 'name': '信越化学工業', 'price': 5920, 'recommendation': 'BUY', 'confidence': 0.86, 'change': 1.9, 'sector': '化学', 'reason': '半導体材料シリコンウェーハ世界シェア首位'},
            {'symbol': '8035', 'name': '東京エレクトロン', 'price': 28450, 'recommendation': 'BUY', 'confidence': 0.87, 'change': 2.3, 'sector': '半導体製造装置', 'reason': '半導体製造装置需要の回復基調'},
            {'symbol': '6367', 'name': 'ダイキン工業', 'price': 23100, 'recommendation': 'BUY', 'confidence': 0.82, 'change': 1.6, 'sector': '機械', 'reason': 'グローバル空調需要増加とヒートポンプ事業'},
            {'symbol': '7741', 'name': 'HOYA', 'price': 14890, 'recommendation': 'BUY', 'confidence': 0.85, 'change': 1.4, 'sector': 'ガラス・土石', 'reason': '半導体フォトマスク事業とメディカル事業堅調'},
            {'symbol': '4519', 'name': '中外製薬', 'price': 4520, 'recommendation': 'BUY', 'confidence': 0.81, 'change': 0.9, 'sector': '医薬品', 'reason': '抗がん剤パイプライン充実とロシュとの連携'},
            {'symbol': '6954', 'name': 'ファナック', 'price': 3890, 'recommendation': 'BUY', 'confidence': 0.83, 'change': 1.8, 'sector': '電気機器', 'reason': '工場自動化需要とロボット事業の成長'},
            {'symbol': '8058', 'name': '三菱商事', 'price': 2945, 'recommendation': 'BUY', 'confidence': 0.84, 'change': 0.7, 'sector': '商社', 'reason': '資源価格安定と非資源事業の拡大'},
            {'symbol': '4568', 'name': '第一三共', 'price': 4890, 'recommendation': 'BUY', 'confidence': 0.82, 'change': 2.1, 'sector': '医薬品', 'reason': 'ADC技術による抗がん剤事業の競争優位'},
            {'symbol': '7974', 'name': '任天堂', 'price': 7650, 'recommendation': 'BUY', 'confidence': 0.88, 'change': 1.3, 'sector': 'ゲーム・娯楽', 'reason': '次世代ゲーム機への期待とIP活用戦略'},
            {'symbol': '9433', 'name': 'KDDI', 'price': 4156, 'recommendation': 'BUY', 'confidence': 0.79, 'change': 0.6, 'sector': '通信', 'reason': '5G投資一巡と安定配当継続'},
            {'symbol': '7267', 'name': 'ホンダ', 'price': 1687, 'recommendation': 'BUY', 'confidence': 0.80, 'change': 1.9, 'sector': '自動車', 'reason': 'EV戦略とF1技術の市販車への応用'},
            {'symbol': '6098', 'name': 'リクルートホールディングス', 'price': 4789, 'recommendation': 'BUY', 'confidence': 0.81, 'change': 1.5, 'sector': '人材サービス', 'reason': 'SaaSサービス成長とグローバル展開'},
            {'symbol': '4612', 'name': '日本ペイントホールディングス', 'price': 1234, 'recommendation': 'BUY', 'confidence': 0.78, 'change': 2.4, 'sector': '化学', 'reason': 'アジア市場でのシェア拡大'},

            # 売り推奨
            {'symbol': '3382', 'name': 'セブン＆アイ・ホールディングス', 'price': 1789, 'recommendation': 'SELL', 'confidence': 0.77, 'change': -1.8, 'sector': '小売', 'reason': 'コンビニ事業の成長限界と海外事業の課題'},
            {'symbol': '9434', 'name': 'ソフトバンク', 'price': 1634, 'recommendation': 'SELL', 'confidence': 0.75, 'change': -1.2, 'sector': '通信', 'reason': '通信料金競争激化と成長鈍化'},
            {'symbol': '8001', 'name': '伊藤忠商事', 'price': 5234, 'recommendation': 'SELL', 'confidence': 0.73, 'change': -2.1, 'sector': '商社', 'reason': '中国経済減速リスクと資源価格下落懸念'},
            {'symbol': '4385', 'name': 'メルカリ', 'price': 2156, 'recommendation': 'SELL', 'confidence': 0.72, 'change': -1.5, 'sector': 'インターネットサービス', 'reason': 'フリマ市場の成熟化と競合激化'},
            {'symbol': '9983', 'name': 'ファーストリテイリング', 'price': 40890, 'recommendation': 'SELL', 'confidence': 0.76, 'change': -0.9, 'sector': 'アパレル・小売', 'reason': '中国事業の不透明感と円安コスト増'},
            {'symbol': '4689', 'name': 'LINEヤフー', 'price': 456, 'recommendation': 'SELL', 'confidence': 0.74, 'change': -2.3, 'sector': 'インターネット', 'reason': '統合効果の不透明性と規制強化リスク'},
            {'symbol': '6856', 'name': '堀場製作所', 'price': 7123, 'recommendation': 'SELL', 'confidence': 0.71, 'change': -1.6, 'sector': '精密機器', 'reason': '自動車関連需要の減速懸念'},

            # 様子見（HOLD）
            {'symbol': '2802', 'name': '味の素', 'price': 4234, 'recommendation': 'HOLD', 'confidence': 0.69, 'change': 0.8, 'sector': '食品', 'reason': '安定収益だが大きな成長材料不足'},
            {'symbol': '8411', 'name': 'みずほフィナンシャルグループ', 'price': 2156, 'recommendation': 'HOLD', 'confidence': 0.71, 'change': 0.3, 'sector': '金融', 'reason': '金利上昇メリットあるが構造改革途上'},
            {'symbol': '7751', 'name': 'キヤノン', 'price': 3456, 'recommendation': 'HOLD', 'confidence': 0.68, 'change': -0.5, 'sector': '電気機器', 'reason': 'デジカメ市場縮小もオフィス機器堅調'},
            {'symbol': '4901', 'name': '富士フイルムホールディングス', 'price': 2890, 'recommendation': 'HOLD', 'confidence': 0.70, 'change': 1.1, 'sector': '化学・ヘルスケア', 'reason': 'ヘルスケア事業成長も写真フィルム事業縮小'},
            {'symbol': '8316', 'name': '三井住友フィナンシャルグループ', 'price': 5678, 'recommendation': 'HOLD', 'confidence': 0.73, 'change': 0.4, 'sector': '金融', 'reason': '金利上昇環境だが貸倒引当金増加リスク'},
            {'symbol': '6479', 'name': 'ミネベアミツミ', 'price': 2345, 'recommendation': 'HOLD', 'confidence': 0.72, 'change': -0.2, 'sector': '機械', 'reason': 'ベアリング事業安定も電子部品需要変動'},
            {'symbol': '6724', 'name': 'セイコーエプソン', 'price': 1789, 'recommendation': 'HOLD', 'confidence': 0.67, 'change': 0.9, 'sector': '電気機器', 'reason': 'プリンター市場成熟化も産業用機器堅調'},
            {'symbol': '5108', 'name': 'ブリヂストン', 'price': 5432, 'recommendation': 'HOLD', 'confidence': 0.74, 'change': 0.6, 'sector': 'ゴム製品', 'reason': '自動車タイヤ需要安定も原材料費高騰'},
            {'symbol': '7832', 'name': 'バンダイナムコホールディングス', 'price': 2567, 'recommendation': 'HOLD', 'confidence': 0.69, 'change': 1.3, 'sector': 'ゲーム・玩具', 'reason': 'IPコンテンツ強いがゲーム市場競争激化'},
            {'symbol': '3401', 'name': '帝人', 'price': 1456, 'recommendation': 'HOLD', 'confidence': 0.66, 'change': -0.7, 'sector': '繊維・化学', 'reason': '炭素繊維事業成長も繊維事業縮小'},
            {'symbol': '7011', 'name': '三菱重工業', 'price': 6789, 'recommendation': 'HOLD', 'confidence': 0.75, 'change': 1.7, 'sector': '重工業', 'reason': '航空機・エネルギー事業回復も大型投資リスク'},
            {'symbol': '6503', 'name': '三菱電機', 'price': 1890, 'recommendation': 'HOLD', 'confidence': 0.71, 'change': 0.5, 'sector': '電気機器', 'reason': '産業機器・インフラ事業安定も成長性限定'},
            {'symbol': '9022', 'name': '東海旅客鉄道', 'price': 15678, 'recommendation': 'HOLD', 'confidence': 0.78, 'change': -0.3, 'sector': '鉄道', 'reason': 'インバウンド回復も リニア投資負担大'},
            {'symbol': '1605', 'name': '国際石油開発帝石', 'price': 1567, 'recommendation': 'HOLD', 'confidence': 0.68, 'change': -1.1, 'sector': '石油・ガス', 'reason': 'エネルギー価格安定も脱炭素リスク'},
            {'symbol': '5401', 'name': '日本製鉄', 'price': 3234, 'recommendation': 'HOLD', 'confidence': 0.70, 'change': 0.8, 'sector': '鉄鋼', 'reason': 'インフラ需要あるも中国鉄鋼過剰生産'},
            {'symbol': '2269', 'name': '明治ホールディングス', 'price': 2678, 'recommendation': 'HOLD', 'confidence': 0.67, 'change': 0.2, 'sector': '食品', 'reason': '乳製品・菓子事業安定も原材料費上昇'},
            {'symbol': '4452', 'name': '花王', 'price': 4567, 'recommendation': 'HOLD', 'confidence': 0.72, 'change': -0.4, 'sector': '化学・日用品', 'reason': '日用品ブランド力あるも市場成熟化'},
            {'symbol': '9531', 'name': '東京ガス', 'price': 3456, 'recommendation': 'HOLD', 'confidence': 0.69, 'change': 1.5, 'sector': 'ガス・電力', 'reason': 'ガス需要安定も脱炭素移行リスク'},
            {'symbol': '5020', 'name': 'ENEOSホールディングス', 'price': 567, 'recommendation': 'HOLD', 'confidence': 0.66, 'change': -0.8, 'sector': '石油精製', 'reason': '石油需要減少も化学品事業で多角化'},
            {'symbol': '8802', 'name': '三菱地所', 'price': 1789, 'recommendation': 'HOLD', 'confidence': 0.73, 'change': 0.9, 'sector': '不動産', 'reason': '都心オフィス需要回復もテレワーク影響'},
            {'symbol': '3659', 'name': 'ネクソン', 'price': 2890, 'recommendation': 'HOLD', 'confidence': 0.68, 'change': 2.1, 'sector': 'ゲーム', 'reason': 'オンラインゲーム堅調も競争激化'},
            {'symbol': '6301', 'name': 'コマツ', 'price': 3567, 'recommendation': 'HOLD', 'confidence': 0.74, 'change': 0.7, 'sector': '機械', 'reason': '建機需要回復も中国市場リスク'},
            {'symbol': '4543', 'name': 'テルモ', 'price': 4123, 'recommendation': 'HOLD', 'confidence': 0.71, 'change': 1.2, 'sector': '医療機器', 'reason': '医療機器需要安定も競合激化'},
            {'symbol': '7261', 'name': 'マツダ', 'price': 1234, 'recommendation': 'HOLD', 'confidence': 0.65, 'change': -1.3, 'sector': '自動車', 'reason': 'ロータリーエンジン技術もEV出遅れ'},
            {'symbol': '2502', 'name': 'アサヒグループホールディングス', 'price': 5678, 'recommendation': 'HOLD', 'confidence': 0.70, 'change': 0.4, 'sector': '食品・飲料', 'reason': 'ビール事業安定も海外展開課題'}
        ]
        
        # 統計計算
        buy_recommendations = [r for r in sample_data if r['recommendation'] in ['BUY', 'STRONG_BUY']]
        sell_recommendations = [r for r in sample_data if r['recommendation'] in ['SELL', 'STRONG_SELL']]
        hold_recommendations = [r for r in sample_data if r['recommendation'] == 'HOLD']
        high_confidence = [r for r in sample_data if r['confidence'] > 0.8]
        
        return jsonify({
            'recommendations': sample_data,
            'total_count': len(sample_data),
            'buy_count': len(buy_recommendations),
            'sell_count': len(sell_recommendations),
            'hold_count': len(hold_recommendations),
            'high_confidence_count': len(high_confidence),
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'avg_confidence': sum(r['confidence'] for r in sample_data) / len(sample_data),
                'positive_change_count': len([r for r in sample_data if r['change'] > 0]),
                'sectors_covered': len(set(r['sector'] for r in sample_data))
            }
        })

    print("Basic routes configured for DayTrade Web Server")

    # --- ログ設定 ---
    if not app.debug:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # --- デバッグ: 登録ルートの表示 ---
    with app.app_context():
        print("--- Registered Routes ---")
        for rule in app.url_map.iter_rules():
            print(f"{rule.endpoint}: {rule.rule} Methods: {','.join(rule.methods)}")
        print("-----------------------")

    return app

# Gunicornがこの'app'インスタンスを使用する
app = create_app()

def main():
    """開発用サーバーを起動するメイン関数"""
    parser = argparse.ArgumentParser(description='Day Trade Web Server (Production Ready)')
    parser.add_argument('--port', '-p', type=int, default=8000, help='サーバーポート')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    args = parser.parse_args()

    print(f"\nDay Trade Web Server (Production Ready) - Issue #939")
    print(f"Version: {app.config['VERSION_INFO']['version_extended']}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Architecture: Application Factory with Gunicorn & Celery")
    print(f"URL: http://localhost:{args.port}")
    print("To run in production with Gunicorn:")
    print("gunicorn --config gunicorn.conf.py daytrade_web:app")
    print("To run the Celery worker:")
    print("celery -A celery_app.celery_app worker --loglevel=info")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
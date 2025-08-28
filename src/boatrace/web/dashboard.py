"""
Boatrace Webダッシュボード

Flask ベースのWebインターフェース
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.exceptions import NotFound

from ..core.api_client import BoatraceAPIClient
from ..core.race_manager import RaceManager
from ..core.stadium_manager import StadiumManager
from ..data.database import init_database
from ..data.data_collector import DataCollector
from ..prediction.prediction_engine import PredictionEngine
from ..prediction.racer_analyzer import RacerAnalyzer
from ..prediction.race_analyzer import RaceAnalyzer
from ..betting.ticket_manager import TicketManager
from ..betting.betting_strategy import StrategyManager
from ..betting.portfolio import Portfolio

logger = logging.getLogger(__name__)


def create_app(config=None):
    """Flaskアプリケーション作成"""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # 設定
    app.config['SECRET_KEY'] = 'boatrace-prediction-system-secret-key'
    app.config['JSON_AS_ASCII'] = False
    
    if config:
        app.config.update(config)
    
    # システム初期化
    try:
        app.database = init_database()
        app.api_client = BoatraceAPIClient()
        app.race_manager = RaceManager(app.api_client)
        app.stadium_manager = StadiumManager()
        app.data_collector = DataCollector(app.api_client, app.database)
        
        # 分析・予想システム
        app.racer_analyzer = RacerAnalyzer(app.database)
        app.race_analyzer = RaceAnalyzer(app.database, app.racer_analyzer)
        app.prediction_engine = PredictionEngine(
            app.database, app.racer_analyzer, app.race_analyzer
        )
        
        # 投票管理
        app.ticket_manager = TicketManager(app.database)
        app.strategy_manager = StrategyManager()
        app.portfolio = Portfolio(Decimal('100000'), app.database)
        
        logger.info("Webアプリケーション初期化完了")
        
    except Exception as e:
        logger.error(f"アプリケーション初期化エラー: {e}")
        # 開発時のフォールバック
        app.database = None
    
    # ルート登録
    register_routes(app)
    
    return app


def register_routes(app: Flask):
    """ルート登録"""
    
    @app.route('/')
    def index():
        """メインダッシュボード"""
        try:
            # 今日のレース取得
            today_races = app.race_manager.get_today_races()
            
            # ポートフォリオ状況
            portfolio_balance = app.portfolio.get_current_balance()
            
            # 注目レース（上位3レース）
            featured_races = today_races[:3] if today_races else []
            
            # システム統計
            system_stats = {
                'total_races_today': len(today_races),
                'api_status': 'OK' if app.api_client.validate_api_connectivity() else 'ERROR',
                'portfolio_roi': portfolio_balance.get('roi', 0),
                'current_balance': portfolio_balance.get('current_balance', 0)
            }
            
            return render_template('dashboard.html',
                                 today_races=featured_races,
                                 portfolio=portfolio_balance,
                                 system_stats=system_stats)
                                 
        except Exception as e:
            logger.error(f"ダッシュボード表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/races')
    def races_list():
        """レース一覧"""
        try:
            target_date = request.args.get('date')
            stadium_filter = request.args.get('stadium', type=int)
            
            if target_date:
                race_date = datetime.strptime(target_date, '%Y-%m-%d').date()
                races = app.race_manager.get_races_by_date(race_date)
            else:
                races = app.race_manager.get_today_races()
            
            # 競技場フィルター
            if stadium_filter:
                races = [r for r in races if r.stadium_number == stadium_filter]
            
            # 競技場一覧（フィルター用）
            stadiums = app.stadium_manager.get_all_stadiums()
            
            return render_template('races.html',
                                 races=races,
                                 stadiums=stadiums,
                                 selected_stadium=stadium_filter,
                                 selected_date=target_date or date.today().isoformat())
                                 
        except Exception as e:
            logger.error(f"レース一覧表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/race/<race_id>')
    def race_detail(race_id: str):
        """レース詳細・予想"""
        try:
            # レース分析
            race_analysis = app.race_analyzer.analyze_race(race_id)
            if not race_analysis:
                raise NotFound(f"レース {race_id} が見つかりません")
            
            # 予想実行
            prediction = app.prediction_engine.predict_race(race_id)
            
            # 投票戦略推奨
            recommended_strategy = None
            betting_recommendations = []
            
            if prediction and race_analysis:
                strategy_name = app.strategy_manager.select_best_strategy(
                    prediction, {'competitiveness': race_analysis.competitiveness}
                )
                strategy = app.strategy_manager.get_strategy(strategy_name)
                recommended_strategy = strategy_name
                
                if strategy:
                    betting_recommendations = strategy.generate_bets(
                        prediction, 
                        Decimal('5000'),  # デフォルト予算
                        {'competitiveness': race_analysis.competitiveness}
                    )
            
            return render_template('race_detail.html',
                                 race_analysis=race_analysis,
                                 prediction=prediction,
                                 recommended_strategy=recommended_strategy,
                                 betting_recommendations=betting_recommendations)
                                 
        except Exception as e:
            logger.error(f"レース詳細表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/stadiums')
    def stadiums_list():
        """競技場一覧"""
        try:
            stadiums = app.stadium_manager.get_all_stadiums()
            
            # 競技場分析情報を追加
            stadium_analyses = {}
            for stadium_num in stadiums.keys():
                analysis = app.stadium_manager.get_stadium_analysis(stadium_num)
                if analysis:
                    stadium_analyses[stadium_num] = analysis
            
            return render_template('stadiums.html',
                                 stadiums=stadiums,
                                 analyses=stadium_analyses)
                                 
        except Exception as e:
            logger.error(f"競技場一覧表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/stadium/<int:stadium_number>')
    def stadium_detail(stadium_number: int):
        """競技場詳細"""
        try:
            # 競技場分析
            analysis = app.stadium_manager.get_stadium_analysis(stadium_number)
            if not analysis:
                raise NotFound(f"競技場 {stadium_number} が見つかりません")
            
            # 今日のレース
            today_races = app.race_manager.get_races_by_stadium(stadium_number)
            
            # 統計情報
            from ..data.data_processor import DataProcessor
            processor = DataProcessor(app.database)
            stadium_stats = processor.calculate_stadium_statistics(stadium_number)
            
            return render_template('stadium_detail.html',
                                 analysis=analysis,
                                 today_races=today_races,
                                 stats=stadium_stats)
                                 
        except Exception as e:
            logger.error(f"競技場詳細表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/portfolio')
    def portfolio_dashboard():
        """ポートフォリオダッシュボード"""
        try:
            # 基本収支
            balance = app.portfolio.get_current_balance()
            
            # パフォーマンス指標
            performance = app.portfolio.get_performance_metrics(30)
            
            # リスク分析
            risk_analysis = app.portfolio.get_risk_analysis()
            
            # 投票履歴（最近10件）
            recent_tickets = app.ticket_manager.get_daily_results(date.today())
            
            # 戦略別成績
            strategy_performance = app.ticket_manager.get_best_performing_strategies(30)
            
            return render_template('portfolio.html',
                                 balance=balance,
                                 performance=performance,
                                 risk_analysis=risk_analysis,
                                 recent_tickets=recent_tickets[:10],
                                 strategy_performance=strategy_performance)
                                 
        except Exception as e:
            logger.error(f"ポートフォリオ表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/betting')  
    def betting_interface():
        """投票インターフェース"""
        try:
            # 今日のレース（予想可能）
            today_races = app.race_manager.get_today_races()
            
            # 投票戦略一覧
            strategies = app.strategy_manager.get_all_strategies()
            
            # 予算推奨
            balance = app.portfolio.get_current_balance()
            suggested_budget = min(
                float(balance['current_balance']) * 0.05,  # 5%まで
                10000  # 最大1万円
            )
            
            return render_template('betting.html',
                                 races=today_races[:10],  # 上位10レース
                                 strategies=strategies,
                                 suggested_budget=suggested_budget,
                                 current_balance=balance['current_balance'])
                                 
        except Exception as e:
            logger.error(f"投票画面表示エラー: {e}")
            return render_template('error.html', error=str(e))
    
    @app.route('/api/predict/<race_id>', methods=['POST'])
    def api_predict_race(race_id: str):
        """レース予想API"""
        try:
            prediction = app.prediction_engine.predict_race(race_id)
            if not prediction:
                return jsonify({'error': 'レース予想に失敗しました'}), 400
            
            # JSON シリアライズ用に変換
            result = {
                'race_id': prediction.race_id,
                'predicted_at': prediction.predicted_at.isoformat(),
                'confidence': float(prediction.confidence),
                'win_probabilities': {
                    str(k): float(v) for k, v in prediction.win_probabilities.items()
                },
                'place_probabilities': {
                    str(k): float(v) for k, v in prediction.place_probabilities.items()
                },
                'recommended_bets': prediction.recommended_bets
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"予想API エラー: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/purchase_ticket', methods=['POST'])
    def api_purchase_ticket():
        """舟券購入API"""
        try:
            data = request.get_json()
            
            required_fields = ['race_id', 'bet_type', 'numbers', 'amount']
            if not all(field in data for field in required_fields):
                return jsonify({'error': '必須フィールドが不足しています'}), 400
            
            from ..betting.ticket_manager import BetType
            bet_type = BetType(data['bet_type'])
            
            ticket_id = app.ticket_manager.purchase_ticket(
                race_id=data['race_id'],
                bet_type=bet_type,
                numbers=data['numbers'],
                amount=Decimal(str(data['amount'])),
                strategy_name=data.get('strategy_name', 'Manual'),
                confidence=Decimal(str(data.get('confidence', 0.5)))
            )
            
            return jsonify({
                'ticket_id': ticket_id,
                'message': '舟券購入記録完了'
            })
            
        except Exception as e:
            logger.error(f"舟券購入API エラー: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data_collection', methods=['POST'])
    def api_collect_data():
        """データ収集API"""
        try:
            result = app.data_collector.collect_today_data()
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"データ収集API エラー: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        """404エラーハンドラー"""
        return render_template('error.html', 
                             error="ページが見つかりません", 
                             error_code=404), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500エラーハンドラー"""
        logger.error(f"内部エラー: {error}")
        return render_template('error.html', 
                             error="内部サーバーエラーが発生しました", 
                             error_code=500), 500


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
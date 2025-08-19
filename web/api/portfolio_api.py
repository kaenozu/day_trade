#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio API - ポートフォリオ管理API
スイングトレード用のポートフォリオ機能のREST API
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from typing import Dict, Any
from web.services.portfolio_service import PortfolioService
from web.services.recommendation_service import RecommendationService

# ブループリント作成
portfolio_api = Blueprint('portfolio_api', __name__)

# サービス初期化
portfolio_service = PortfolioService()
recommendation_service = RecommendationService()

logger = logging.getLogger(__name__)

@portfolio_api.route('/api/portfolio/positions', methods=['GET'])
def get_positions():
    """ポジション一覧取得"""
    try:
        status = request.args.get('status')  # 'OPEN', 'CLOSED', または None
        positions = portfolio_service.get_positions(status)
        
        # 現在価格を更新
        open_positions = [p for p in positions if p.status == 'OPEN']
        if open_positions:
            symbols = [p.symbol for p in open_positions]
            portfolio_service.update_current_prices(symbols)
        
        # レスポンス用にフォーマット
        formatted_positions = []
        for position in positions:
            formatted_positions.append({
                'id': position.id,
                'symbol': position.symbol,
                'name': position.name,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'entry_date': position.entry_date,
                'position_type': position.position_type,
                'strategy_type': position.strategy_type,
                'target_price': position.target_price,
                'stop_loss_price': position.stop_loss_price,
                'status': position.status,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'realized_pnl': position.realized_pnl if position.status == 'CLOSED' else 0,
                'holding_days': position.holding_days,
                'investment_amount': position.investment_amount,
                'current_value': position.total_value,
                'sector': position.sector,
                'notes': position.notes
            })
        
        return jsonify({
            'success': True,
            'positions': formatted_positions,
            'count': len(formatted_positions)
        })
        
    except Exception as e:
        logger.error(f"ポジション取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/summary', methods=['GET'])
def get_portfolio_summary():
    """ポートフォリオサマリー取得"""
    try:
        summary = portfolio_service.get_portfolio_summary()
        
        return jsonify({
            'success': True,
            'summary': {
                'total_value': summary.total_value,
                'total_cost': summary.total_cost,
                'total_pnl': summary.total_pnl,
                'total_pnl_pct': summary.total_pnl_pct,
                'cash_balance': summary.cash_balance,
                'total_assets': summary.total_assets,
                'positions_count': summary.positions_count,
                'sectors_exposure': summary.sectors_exposure,
                'top_holdings': summary.top_holdings,
                'daily_change': summary.daily_change,
                'daily_change_pct': summary.daily_change_pct,
                'last_updated': summary.last_updated
            }
        })
        
    except Exception as e:
        logger.error(f"サマリー取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/position', methods=['POST'])
def create_position():
    """新規ポジション作成"""
    try:
        data = request.get_json()
        
        # 必須フィールドの検証
        required_fields = ['symbol', 'quantity', 'investment_amount']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'必須フィールドが不足: {field}'}), 400
        
        symbol = data['symbol']
        quantity = int(data['quantity'])
        investment_amount = float(data['investment_amount'])
        
        # 推奨データから基本情報を取得
        recommendations = recommendation_service.get_recommendations()
        stock_data = next((r for r in recommendations if r['symbol'] == symbol), None)
        
        if not stock_data:
            return jsonify({'success': False, 'error': '該当する推奨銘柄が見つかりません'}), 404
        
        # ポジション作成
        position_id = portfolio_service.create_position_from_recommendation(
            symbol=symbol,
            recommendation_data=stock_data,
            investment_amount=investment_amount
        )
        
        if position_id:
            return jsonify({
                'success': True,
                'position_id': position_id,
                'message': f'ポジション作成完了: {symbol}'
            })
        else:
            return jsonify({'success': False, 'error': 'ポジション作成に失敗しました'}), 500
            
    except Exception as e:
        logger.error(f"ポジション作成エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/position/<position_id>/close', methods=['POST'])
def close_position(position_id: str):
    """ポジションクローズ"""
    try:
        data = request.get_json() or {}
        exit_price = data.get('exit_price')  # None の場合は現在価格を使用
        notes = data.get('notes', '手動クローズ')
        
        success = portfolio_service.close_position(position_id, exit_price, notes)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'ポジションクローズ完了: {position_id}'
            })
        else:
            return jsonify({'success': False, 'error': 'ポジションクローズに失敗しました'}), 500
            
    except Exception as e:
        logger.error(f"ポジションクローズエラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/dashboard', methods=['GET'])
def get_dashboard_data():
    """ダッシュボード用データ取得"""
    try:
        dashboard_data = portfolio_service.get_portfolio_dashboard_data()
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"ダッシュボードデータ取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/alerts', methods=['GET'])
def get_alerts():
    """ポートフォリオアラート取得"""
    try:
        dashboard_data = portfolio_service.get_portfolio_dashboard_data()
        alerts = dashboard_data.get('alerts', [])
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        logger.error(f"アラート取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/recommendations', methods=['GET'])
def get_position_recommendations():
    """ポジション推奨アクション取得"""
    try:
        recommendations = portfolio_service.get_position_recommendations()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"推奨アクション取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/performance', methods=['GET'])
def get_performance_analysis():
    """パフォーマンス分析取得"""
    try:
        # 基本パフォーマンス指標
        performance = portfolio_service.get_performance_metrics()
        
        # 戦略別パフォーマンス
        strategy_performance = portfolio_service.get_performance_by_strategy()
        
        # リスク分析
        risk_analysis = portfolio_service.get_risk_analysis()
        
        return jsonify({
            'success': True,
            'performance': performance,
            'strategy_performance': strategy_performance,
            'risk_analysis': risk_analysis
        })
        
    except Exception as e:
        logger.error(f"パフォーマンス分析エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/transactions', methods=['GET'])
def get_transactions():
    """取引履歴取得"""
    try:
        limit = request.args.get('limit', 50, type=int)
        transactions = portfolio_service.get_transactions(limit)
        
        # レスポンス用にフォーマット
        formatted_transactions = []
        for txn in transactions:
            formatted_transactions.append({
                'id': txn.id,
                'symbol': txn.symbol,
                'name': txn.name,
                'action': txn.action,
                'quantity': txn.quantity,
                'price': txn.price,
                'total_amount': txn.total_amount,
                'commission': txn.commission,
                'net_amount': txn.net_amount,
                'date': txn.date,
                'notes': txn.notes
            })
        
        return jsonify({
            'success': True,
            'transactions': formatted_transactions,
            'count': len(formatted_transactions)
        })
        
    except Exception as e:
        logger.error(f"取引履歴取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/position/<position_id>', methods=['GET'])
def get_position_detail(position_id: str):
    """個別ポジション詳細取得"""
    try:
        position = portfolio_service.get_position_by_id(position_id)
        
        if not position:
            return jsonify({'success': False, 'error': 'ポジションが見つかりません'}), 404
        
        return jsonify({
            'success': True,
            'position': {
                'id': position.id,
                'symbol': position.symbol,
                'name': position.name,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'entry_date': position.entry_date,
                'position_type': position.position_type,
                'strategy_type': position.strategy_type,
                'target_price': position.target_price,
                'stop_loss_price': position.stop_loss_price,
                'exit_date': position.exit_date,
                'exit_price': position.exit_price,
                'status': position.status,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'realized_pnl': position.realized_pnl,
                'holding_days': position.holding_days,
                'investment_amount': position.investment_amount,
                'current_value': position.total_value,
                'sector': position.sector,
                'category': position.category,
                'notes': position.notes,
                'last_updated': position.last_updated
            }
        })
        
    except Exception as e:
        logger.error(f"ポジション詳細取得エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@portfolio_api.route('/api/portfolio/position/<position_id>/update', methods=['PUT'])
def update_position(position_id: str):
    """ポジション更新"""
    try:
        data = request.get_json()
        position = portfolio_service.get_position_by_id(position_id)
        
        if not position:
            return jsonify({'success': False, 'error': 'ポジションが見つかりません'}), 404
        
        # 更新可能フィールド
        updateable_fields = ['target_price', 'stop_loss_price', 'notes']
        
        for field in updateable_fields:
            if field in data:
                setattr(position, field, data[field])
        
        position.last_updated = datetime.now().isoformat()
        
        # 保存
        portfolio_service._save_position(position)
        
        return jsonify({
            'success': True,
            'message': f'ポジション更新完了: {position_id}'
        })
        
    except Exception as e:
        logger.error(f"ポジション更新エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
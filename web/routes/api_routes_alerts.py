#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert API Routes Module - アラート・通知API
アラート関連のAPIエンドポイント定義
"""

from flask import Flask, jsonify, request
from datetime import datetime
from typing import Dict, Any, List

# アラートサービスのインポート
try:
    from web.services.alert_service import AlertService, AlertType, AlertPriority, AlertStatus
    ALERT_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"アラートサービス読み込みエラー: {e}")
    ALERT_SERVICE_AVAILABLE = False

def setup_alert_routes(app: Flask) -> None:
    """アラートAPIルート設定"""
    
    # アラートサービス初期化
    alert_service = AlertService() if ALERT_SERVICE_AVAILABLE else None
    
    @app.route('/api/alerts')
    def api_alerts_list():
        """アラート一覧API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            user_id = request.args.get('user_id', 'default_user')
            status = request.args.get('status')
            
            status_enum = None
            if status:
                try:
                    status_enum = AlertStatus(status.upper())
                except ValueError:
                    return jsonify({
                        'error': f'無効なステータス: {status}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            alerts = alert_service.get_alerts(user_id, status_enum)
            
            alerts_data = []
            for alert in alerts:
                alerts_data.append({
                    'id': alert.id,
                    'user_id': alert.user_id,
                    'alert_type': alert.alert_type.value,
                    'symbol': alert.symbol,
                    'condition': alert.condition,
                    'message_template': alert.message_template,
                    'priority': alert.priority.value,
                    'status': alert.status.value,
                    'created_at': alert.created_at,
                    'expires_at': alert.expires_at,
                    'last_triggered': alert.last_triggered,
                    'trigger_count': alert.trigger_count,
                    'max_triggers': alert.max_triggers,
                    'cooldown_minutes': alert.cooldown_minutes,
                    'is_email_enabled': alert.is_email_enabled,
                    'is_popup_enabled': alert.is_popup_enabled,
                    'is_sound_enabled': alert.is_sound_enabled
                })
            
            return jsonify({
                'alerts': alerts_data,
                'count': len(alerts_data),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/create', methods=['POST'])
    def api_alerts_create():
        """アラート作成API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            
            # 必須フィールドチェック
            required_fields = ['alert_type', 'symbol', 'condition', 'message_template']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            # Enum変換
            try:
                alert_type = AlertType(data['alert_type'].upper())
                priority = AlertPriority(data.get('priority', 'MEDIUM').upper())
            except ValueError as e:
                return jsonify({
                    'error': f'無効な値: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            alert_id = alert_service.create_alert(
                user_id=data.get('user_id', 'default_user'),
                alert_type=alert_type,
                symbol=data['symbol'],
                condition=data['condition'],
                message_template=data['message_template'],
                priority=priority,
                expires_hours=data.get('expires_hours')
            )
            
            if alert_id:
                return jsonify({
                    'success': True,
                    'alert_id': alert_id,
                    'message': 'アラートを作成しました',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': 'アラート作成に失敗しました',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/price', methods=['POST'])
    def api_alerts_create_price():
        """価格アラート作成API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            
            required_fields = ['symbol', 'threshold']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            alert_id = alert_service.create_price_alert(
                user_id=data.get('user_id', 'default_user'),
                symbol=data['symbol'],
                threshold=float(data['threshold']),
                above=data.get('above', True),
                expires_hours=data.get('expires_hours', 24)
            )
            
            if alert_id:
                direction = "上昇" if data.get('above', True) else "下落"
                return jsonify({
                    'success': True,
                    'alert_id': alert_id,
                    'message': f"{data['symbol']} の価格{direction}アラートを作成しました",
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': 'アラート作成に失敗しました',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/change', methods=['POST'])
    def api_alerts_create_change():
        """変動率アラート作成API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            
            required_fields = ['symbol', 'change_pct']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': f'必須フィールドが不足: {field}',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            alert_id = alert_service.create_change_alert(
                user_id=data.get('user_id', 'default_user'),
                symbol=data['symbol'],
                change_pct=float(data['change_pct']),
                positive=data.get('positive', False),
                expires_hours=data.get('expires_hours', 24)
            )
            
            if alert_id:
                direction = "上昇" if data.get('positive', False) else "下落"
                return jsonify({
                    'success': True,
                    'alert_id': alert_id,
                    'message': f"{data['symbol']} の{data['change_pct']}%{direction}アラートを作成しました",
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': 'アラート作成に失敗しました',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/<alert_id>', methods=['DELETE'])
    def api_alerts_delete(alert_id: str):
        """アラート削除API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            success = alert_service.delete_alert(alert_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'アラートを削除しました',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': 'アラートが見つかりません',
                    'timestamp': datetime.now().isoformat()
                }), 404
                
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/notifications')
    def api_notifications_list():
        """通知一覧API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            limit = int(request.args.get('limit', 50))
            unread_only = request.args.get('unread_only', 'false').lower() == 'true'
            
            notifications = alert_service.get_notifications(limit, unread_only)
            
            notifications_data = []
            for notification in notifications:
                notifications_data.append({
                    'id': notification.id,
                    'alert_id': notification.alert_id,
                    'symbol': notification.symbol,
                    'message': notification.message,
                    'priority': notification.priority.value,
                    'data': notification.data,
                    'timestamp': notification.timestamp,
                    'is_read': notification.is_read
                })
            
            return jsonify({
                'notifications': notifications_data,
                'count': len(notifications_data),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/notifications/<notification_id>/read', methods=['POST'])
    def api_notifications_mark_read(notification_id: str):
        """通知既読マークAPI"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            success = alert_service.mark_notification_read(notification_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': '通知を既読にしました',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'error': '通知が見つかりません',
                    'timestamp': datetime.now().isoformat()
                }), 404
                
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/stats')
    def api_alerts_stats():
        """アラート統計API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            stats = alert_service.get_alert_stats()
            return jsonify({
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/monitoring/start', methods=['POST'])
    def api_alerts_start_monitoring():
        """アラート監視開始API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            alert_service.start_monitoring()
            return jsonify({
                'success': True,
                'message': 'アラート監視を開始しました',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/monitoring/stop', methods=['POST'])
    def api_alerts_stop_monitoring():
        """アラート監視停止API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            alert_service.stop_monitoring()
            return jsonify({
                'success': True,
                'message': 'アラート監視を停止しました',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/api/alerts/update-prices', methods=['POST'])
    def api_alerts_update_prices():
        """アラート用価格更新API"""
        if not ALERT_SERVICE_AVAILABLE:
            return jsonify({
                'error': 'アラートサービスが利用できません',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        try:
            data = request.get_json()
            
            if 'prices' not in data:
                return jsonify({
                    'error': '価格データが不足しています',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            alert_service.update_prices(data['prices'])
            
            return jsonify({
                'success': True,
                'message': 'アラート用価格データを更新しました',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
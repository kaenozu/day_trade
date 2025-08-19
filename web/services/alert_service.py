#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert & Notification Service - アラート・通知サービス
価格アラート、ポートフォリオ変動通知、市場状況警告等の機能
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum
import threading
import time

class AlertType(Enum):
    """アラートタイプ"""
    PRICE_ABOVE = "PRICE_ABOVE"
    PRICE_BELOW = "PRICE_BELOW"
    PRICE_CHANGE = "PRICE_CHANGE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    TECHNICAL_SIGNAL = "TECHNICAL_SIGNAL"
    PORTFOLIO_LOSS = "PORTFOLIO_LOSS"
    PORTFOLIO_GAIN = "PORTFOLIO_GAIN"
    NEWS_ALERT = "NEWS_ALERT"
    MARKET_ALERT = "MARKET_ALERT"

class AlertPriority(Enum):
    """アラート優先度"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertStatus(Enum):
    """アラート状態"""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    DISABLED = "DISABLED"
    EXPIRED = "EXPIRED"

@dataclass
class Alert:
    """アラート設定"""
    id: str
    user_id: str
    alert_type: AlertType
    symbol: str
    condition: Dict[str, Any]  # 条件設定 (threshold, operator等)
    message_template: str
    priority: AlertPriority
    status: AlertStatus
    created_at: str
    expires_at: Optional[str] = None
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    max_triggers: int = 1
    cooldown_minutes: int = 60
    is_email_enabled: bool = False
    is_popup_enabled: bool = True
    is_sound_enabled: bool = True

@dataclass
class AlertNotification:
    """アラート通知"""
    id: str
    alert_id: str
    symbol: str
    message: str
    priority: AlertPriority
    data: Dict[str, Any]  # 価格、変動率等の詳細データ
    timestamp: str
    is_read: bool = False

class AlertService:
    """アラート・通知サービス"""
    
    def __init__(self, data_dir: str = "data/alerts"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.alerts_file = self.data_dir / "alerts.json"
        self.notifications_file = self.data_dir / "notifications.json"
        self.settings_file = self.data_dir / "settings.json"
        
        self.logger = logging.getLogger(__name__)
        
        # アラートチェック用スレッド
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 価格データキャッシュ
        self.price_cache = {}
        self.last_price_update = None
        
        # 初期化
        self._initialize_files()
        self.settings = self._load_settings()
        
        # 通知コールバック
        self.notification_callbacks = []
    
    def _initialize_files(self):
        """データファイルの初期化"""
        if not self.alerts_file.exists():
            self._save_alerts([])
        
        if not self.notifications_file.exists():
            self._save_notifications([])
        
        if not self.settings_file.exists():
            default_settings = {
                "monitoring_interval": 30,  # 30秒間隔
                "max_notifications": 100,
                "email_settings": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": ""
                },
                "sound_settings": {
                    "enabled": True,
                    "volume": 0.8,
                    "sound_file": ""
                }
            }
            self._save_settings(default_settings)
    
    def _load_alerts(self) -> List[Alert]:
        """アラート読み込み"""
        try:
            with open(self.alerts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                alerts = []
                for item in data:
                    # Enum型の復元
                    item['alert_type'] = AlertType(item['alert_type'])
                    item['priority'] = AlertPriority(item['priority'])
                    item['status'] = AlertStatus(item['status'])
                    alerts.append(Alert(**item))
                return alerts
        except Exception as e:
            self.logger.error(f"アラート読み込みエラー: {e}")
            return []
    
    def _save_alerts(self, alerts: List[Alert]):
        """アラート保存"""
        try:
            data = []
            for alert in alerts:
                alert_dict = asdict(alert)
                # Enum型を文字列に変換
                alert_dict['alert_type'] = alert.alert_type.value
                alert_dict['priority'] = alert.priority.value
                alert_dict['status'] = alert.status.value
                data.append(alert_dict)
            
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"アラート保存エラー: {e}")
    
    def _load_notifications(self) -> List[AlertNotification]:
        """通知履歴読み込み"""
        try:
            with open(self.notifications_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                notifications = []
                for item in data:
                    item['priority'] = AlertPriority(item['priority'])
                    notifications.append(AlertNotification(**item))
                return notifications
        except Exception as e:
            self.logger.error(f"通知履歴読み込みエラー: {e}")
            return []
    
    def _save_notifications(self, notifications: List[AlertNotification]):
        """通知履歴保存"""
        try:
            data = []
            for notification in notifications:
                notif_dict = asdict(notification)
                notif_dict['priority'] = notification.priority.value
                data.append(notif_dict)
            
            with open(self.notifications_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"通知履歴保存エラー: {e}")
    
    def _load_settings(self) -> Dict[str, Any]:
        """設定読み込み"""
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            return {}
    
    def _save_settings(self, settings: Dict[str, Any]):
        """設定保存"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"設定保存エラー: {e}")
    
    def create_alert(self, user_id: str, alert_type: AlertType, symbol: str,
                    condition: Dict[str, Any], message_template: str,
                    priority: AlertPriority = AlertPriority.MEDIUM,
                    expires_hours: Optional[int] = None) -> str:
        """アラート作成"""
        try:
            alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            
            expires_at = None
            if expires_hours:
                expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()
            
            alert = Alert(
                id=alert_id,
                user_id=user_id,
                alert_type=alert_type,
                symbol=symbol,
                condition=condition,
                message_template=message_template,
                priority=priority,
                status=AlertStatus.ACTIVE,
                created_at=datetime.now().isoformat(),
                expires_at=expires_at
            )
            
            alerts = self._load_alerts()
            alerts.append(alert)
            self._save_alerts(alerts)
            
            self.logger.info(f"アラート作成: {alert_id}")
            return alert_id
            
        except Exception as e:
            self.logger.error(f"アラート作成エラー: {e}")
            return ""
    
    def create_price_alert(self, user_id: str, symbol: str, threshold: float,
                          above: bool = True, expires_hours: int = 24) -> str:
        """価格アラート作成"""
        alert_type = AlertType.PRICE_ABOVE if above else AlertType.PRICE_BELOW
        operator = "above" if above else "below"
        
        condition = {
            "threshold": threshold,
            "operator": operator
        }
        
        message_template = f"{symbol} が ¥{{price:,.0f}} に到達しました（目標: {operator} ¥{threshold:,.0f}）"
        
        return self.create_alert(
            user_id=user_id,
            alert_type=alert_type,
            symbol=symbol,
            condition=condition,
            message_template=message_template,
            priority=AlertPriority.MEDIUM,
            expires_hours=expires_hours
        )
    
    def create_change_alert(self, user_id: str, symbol: str, change_pct: float,
                           positive: bool = False, expires_hours: int = 24) -> str:
        """変動率アラート作成"""
        condition = {
            "change_pct": change_pct,
            "positive": positive
        }
        
        direction = "上昇" if positive else "下落"
        message_template = f"{symbol} が {change_pct:+.1f}% {direction}しました（現在価格: ¥{{price:,.0f}}）"
        
        return self.create_alert(
            user_id=user_id,
            alert_type=AlertType.PRICE_CHANGE,
            symbol=symbol,
            condition=condition,
            message_template=message_template,
            priority=AlertPriority.HIGH if abs(change_pct) > 5 else AlertPriority.MEDIUM,
            expires_hours=expires_hours
        )
    
    def update_prices(self, price_data: Dict[str, Dict[str, float]]):
        """価格データ更新"""
        try:
            self.price_cache.update(price_data)
            self.last_price_update = datetime.now()
            
            # アクティブなアラートをチェック
            self._check_alerts()
            
        except Exception as e:
            self.logger.error(f"価格更新エラー: {e}")
    
    def _check_alerts(self):
        """アラート条件チェック"""
        try:
            alerts = self._load_alerts()
            triggered_alerts = []
            
            for alert in alerts:
                if alert.status != AlertStatus.ACTIVE:
                    continue
                
                # 期限チェック
                if alert.expires_at and datetime.now() > datetime.fromisoformat(alert.expires_at):
                    alert.status = AlertStatus.EXPIRED
                    continue
                
                # クールダウンチェック
                if alert.last_triggered:
                    last_trigger = datetime.fromisoformat(alert.last_triggered)
                    if (datetime.now() - last_trigger).total_seconds() < alert.cooldown_minutes * 60:
                        continue
                
                # 価格データ確認
                if alert.symbol not in self.price_cache:
                    continue
                
                price_data = self.price_cache[alert.symbol]
                
                # アラート条件チェック
                if self._evaluate_alert_condition(alert, price_data):
                    triggered_alerts.append(alert)
            
            # トリガーされたアラートを処理
            for alert in triggered_alerts:
                self._trigger_alert(alert, self.price_cache[alert.symbol])
            
            # 更新されたアラートを保存
            if triggered_alerts:
                self._save_alerts(alerts)
            
        except Exception as e:
            self.logger.error(f"アラートチェックエラー: {e}")
    
    def _evaluate_alert_condition(self, alert: Alert, price_data: Dict[str, float]) -> bool:
        """アラート条件評価"""
        try:
            current_price = price_data.get('current_price', 0)
            change_pct = price_data.get('price_change_pct', 0)
            
            if alert.alert_type == AlertType.PRICE_ABOVE:
                return current_price > alert.condition['threshold']
            
            elif alert.alert_type == AlertType.PRICE_BELOW:
                return current_price < alert.condition['threshold']
            
            elif alert.alert_type == AlertType.PRICE_CHANGE:
                target_change = alert.condition['change_pct']
                is_positive = alert.condition['positive']
                
                if is_positive:
                    return change_pct >= target_change
                else:
                    return change_pct <= -abs(target_change)
            
            elif alert.alert_type == AlertType.VOLUME_SPIKE:
                volume = price_data.get('volume', 0)
                avg_volume = alert.condition.get('avg_volume', 0)
                spike_ratio = alert.condition.get('spike_ratio', 2.0)
                return volume > avg_volume * spike_ratio
            
            return False
            
        except Exception as e:
            self.logger.error(f"アラート条件評価エラー: {e}")
            return False
    
    def _trigger_alert(self, alert: Alert, price_data: Dict[str, float]):
        """アラートトリガー処理"""
        try:
            # アラート更新
            alert.last_triggered = datetime.now().isoformat()
            alert.trigger_count += 1
            
            # 最大トリガー数チェック
            if alert.trigger_count >= alert.max_triggers:
                alert.status = AlertStatus.TRIGGERED
            
            # 通知メッセージ作成
            message = alert.message_template.format(**price_data)
            
            # 通知作成
            notification = AlertNotification(
                id=f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alert.symbol}",
                alert_id=alert.id,
                symbol=alert.symbol,
                message=message,
                priority=alert.priority,
                data=price_data,
                timestamp=datetime.now().isoformat()
            )
            
            # 通知保存
            self._add_notification(notification)
            
            # 通知送信
            self._send_notification(notification, alert)
            
            self.logger.info(f"アラートトリガー: {alert.id} - {message}")
            
        except Exception as e:
            self.logger.error(f"アラートトリガーエラー: {e}")
    
    def _add_notification(self, notification: AlertNotification):
        """通知追加"""
        try:
            notifications = self._load_notifications()
            notifications.insert(0, notification)  # 最新を先頭に
            
            # 最大件数制限
            max_notifications = self.settings.get("max_notifications", 100)
            if len(notifications) > max_notifications:
                notifications = notifications[:max_notifications]
            
            self._save_notifications(notifications)
            
        except Exception as e:
            self.logger.error(f"通知追加エラー: {e}")
    
    def _send_notification(self, notification: AlertNotification, alert: Alert):
        """通知送信"""
        try:
            # コールバック実行
            for callback in self.notification_callbacks:
                try:
                    callback(notification, alert)
                except Exception as e:
                    self.logger.error(f"通知コールバックエラー: {e}")
            
            # TODO: メール送信
            if alert.is_email_enabled and self.settings.get("email_settings", {}).get("enabled", False):
                self._send_email_notification(notification)
            
            # TODO: 音声通知
            if alert.is_sound_enabled and self.settings.get("sound_settings", {}).get("enabled", False):
                self._play_sound_notification(notification)
            
        except Exception as e:
            self.logger.error(f"通知送信エラー: {e}")
    
    def _send_email_notification(self, notification: AlertNotification):
        """メール通知送信"""
        # 実装予定: SMTPを使ったメール送信
        pass
    
    def _play_sound_notification(self, notification: AlertNotification):
        """音声通知再生"""
        # 実装予定: システム音またはファイル再生
        pass
    
    def add_notification_callback(self, callback: Callable[[AlertNotification, Alert], None]):
        """通知コールバック追加"""
        self.notification_callbacks.append(callback)
    
    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("アラート監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("アラート監視停止")
    
    def _monitoring_loop(self):
        """監視ループ"""
        interval = self.settings.get("monitoring_interval", 30)
        
        while self.is_monitoring:
            try:
                # アクティブなアラートがある場合のみチェック
                alerts = self._load_alerts()
                active_alerts = [a for a in alerts if a.status == AlertStatus.ACTIVE]
                
                if active_alerts and self.price_cache:
                    self._check_alerts()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(interval)
    
    def get_alerts(self, user_id: str = None, status: AlertStatus = None) -> List[Alert]:
        """アラート取得"""
        alerts = self._load_alerts()
        
        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_notifications(self, limit: int = 50, unread_only: bool = False) -> List[AlertNotification]:
        """通知取得"""
        notifications = self._load_notifications()
        
        if unread_only:
            notifications = [n for n in notifications if not n.is_read]
        
        return notifications[:limit]
    
    def mark_notification_read(self, notification_id: str) -> bool:
        """通知既読マーク"""
        try:
            notifications = self._load_notifications()
            
            for notification in notifications:
                if notification.id == notification_id:
                    notification.is_read = True
                    self._save_notifications(notifications)
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"通知既読マークエラー: {e}")
            return False
    
    def delete_alert(self, alert_id: str) -> bool:
        """アラート削除"""
        try:
            alerts = self._load_alerts()
            original_count = len(alerts)
            
            alerts = [a for a in alerts if a.id != alert_id]
            
            if len(alerts) < original_count:
                self._save_alerts(alerts)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"アラート削除エラー: {e}")
            return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """アラート統計"""
        try:
            alerts = self._load_alerts()
            notifications = self._load_notifications()
            
            return {
                "total_alerts": len(alerts),
                "active_alerts": len([a for a in alerts if a.status == AlertStatus.ACTIVE]),
                "triggered_alerts": len([a for a in alerts if a.status == AlertStatus.TRIGGERED]),
                "total_notifications": len(notifications),
                "unread_notifications": len([n for n in notifications if not n.is_read]),
                "last_update": self.last_price_update.isoformat() if self.last_price_update else None,
                "monitoring_status": self.is_monitoring
            }
            
        except Exception as e:
            self.logger.error(f"アラート統計エラー: {e}")
            return {}
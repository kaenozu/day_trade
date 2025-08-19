#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notification System - プッシュ通知・アラートシステム
Issue #949対応: リアルタイム通知 + 複数チャネル + インテリジェントアラート
"""

import json
import os
import smtplib
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from collections import defaultdict, deque
import hashlib
import sqlite3

# WebSocket対応
try:
    import websockets
    import asyncio
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# プッシュ通知対応
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class NotificationType(Enum):
    """通知タイプ"""
    PRICE_ALERT = "PRICE_ALERT"         # 価格アラート
    AI_SIGNAL = "AI_SIGNAL"             # AI信号
    NEWS_UPDATE = "NEWS_UPDATE"         # ニュース更新
    SYSTEM_STATUS = "SYSTEM_STATUS"     # システム状況
    TRADING_OPPORTUNITY = "TRADING_OPPORTUNITY"  # 取引機会
    RISK_WARNING = "RISK_WARNING"       # リスク警告
    MAINTENANCE = "MAINTENANCE"         # メンテナンス
    CUSTOM = "CUSTOM"                   # カスタム


class Priority(Enum):
    """優先度"""
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class DeliveryChannel(Enum):
    """配信チャネル"""
    EMAIL = "EMAIL"
    WEBSOCKET = "WEBSOCKET"
    DESKTOP = "DESKTOP"
    MOBILE_PUSH = "MOBILE_PUSH"
    SLACK = "SLACK"
    DISCORD = "DISCORD"
    LINE = "LINE"


class NotificationStatus(Enum):
    """通知状況"""
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


@dataclass
class NotificationRule:
    """通知ルール"""
    rule_id: str
    name: str
    notification_type: NotificationType
    conditions: Dict[str, Any]
    channels: List[DeliveryChannel]
    priority: Priority = Priority.NORMAL
    enabled: bool = True
    cooldown_minutes: int = 30
    max_per_day: int = 100
    template: str = ""
    created_at: datetime = None


@dataclass
class NotificationContent:
    """通知内容"""
    title: str
    message: str
    data: Dict[str, Any] = None
    action_url: str = ""
    expires_at: datetime = None


@dataclass
class Notification:
    """通知"""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    priority: Priority
    content: NotificationContent
    channels: List[DeliveryChannel]
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = None
    sent_at: datetime = None
    delivered_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3


class EmailNotifier:
    """メール通知"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    async def send_notification(self, notification: Notification, recipient: str) -> bool:
        """メール送信"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient
            msg['Subject'] = notification.content.title
            
            # HTMLメール本文
            html_body = f"""
            <html>
            <body>
                <h2>{notification.content.title}</h2>
                <p>{notification.content.message}</p>
                
                {self._generate_data_table(notification.content.data)}
                
                <div style="margin-top: 20px;">
                    <p><small>優先度: {notification.priority.name}</small></p>
                    <p><small>送信時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </div>
                
                {self._generate_action_button(notification.content.action_url)}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html', 'utf-8'))
            
            # SMTP送信
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email sent to {recipient}: {notification.content.title}")
            return True
            
        except Exception as e:
            logging.error(f"Email send failed to {recipient}: {e}")
            return False
    
    def _generate_data_table(self, data: Dict[str, Any]) -> str:
        """データテーブル生成"""
        if not data:
            return ""
        
        table_rows = []
        for key, value in data.items():
            table_rows.append(f"<tr><td><strong>{key}:</strong></td><td>{value}</td></tr>")
        
        return f"""
        <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
            {''.join(table_rows)}
        </table>
        """
    
    def _generate_action_button(self, action_url: str) -> str:
        """アクションボタン生成"""
        if not action_url:
            return ""
        
        return f"""
        <div style="margin-top: 20px; text-align: center;">
            <a href="{action_url}" 
               style="background-color: #007cba; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 4px; display: inline-block;">
                詳細を確認
            </a>
        </div>
        """


class WebSocketNotifier:
    """WebSocket通知"""
    
    def __init__(self):
        self.connected_clients = set()
        self.server = None
        
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """WebSocketサーバー開始"""
        if not HAS_WEBSOCKETS:
            logging.warning("WebSocket library not available")
            return
        
        self.server = await websockets.serve(
            self.handle_client,
            host,
            port
        )
        logging.info(f"WebSocket server started on ws://{host}:{port}")
    
    async def handle_client(self, websocket, path):
        """クライアント処理"""
        self.connected_clients.add(websocket)
        logging.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            logging.info(f"WebSocket client disconnected")
    
    async def broadcast_notification(self, notification: Notification) -> int:
        """通知ブロードキャスト"""
        if not self.connected_clients:
            return 0
        
        message = json.dumps({
            'id': notification.notification_id,
            'type': notification.notification_type.value,
            'priority': notification.priority.name,
            'title': notification.content.title,
            'message': notification.content.message,
            'data': notification.content.data,
            'action_url': notification.content.action_url,
            'timestamp': notification.created_at.isoformat()
        }, ensure_ascii=False)
        
        sent_count = 0
        disconnected = set()
        
        for client in self.connected_clients:
            try:
                await client.send(message)
                sent_count += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logging.error(f"WebSocket send failed: {e}")
        
        # 切断されたクライアントを削除
        self.connected_clients -= disconnected
        
        return sent_count


class DesktopNotifier:
    """デスクトップ通知"""
    
    def __init__(self):
        self.notification_history = deque(maxlen=100)
    
    async def send_notification(self, notification: Notification) -> bool:
        """デスクトップ通知送信"""
        try:
            # Windows Toast通知（簡易版）
            import subprocess
            
            title = notification.content.title
            message = notification.content.message
            
            # PowerShell経由でToast通知
            ps_script = f"""
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
            
            $template = @"
            <toast>
                <visual>
                    <binding template="ToastGeneric">
                        <text>{title}</text>
                        <text>{message}</text>
                    </binding>
                </visual>
                <actions>
                    <action content="確認" arguments="view_details"/>
                    <action content="閉じる" arguments="dismiss"/>
                </actions>
            </toast>
            "@
            
            $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xml.LoadXml($template)
            
            $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Day Trade Personal").Show($toast)
            """
            
            process = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if process.returncode == 0:
                self.notification_history.append({
                    'title': title,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            
        except Exception as e:
            logging.error(f"Desktop notification failed: {e}")
        
        return False


class NotificationRule Engine:
    """通知ルールエンジン"""
    
    def __init__(self):
        self.rules: Dict[str, NotificationRule] = {}
        self.rule_counters: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.last_triggered: Dict[str, datetime] = {}
        
        # デフォルトルール作成
        self._create_default_rules()
    
    def _create_default_rules(self):
        """デフォルトルール作成"""
        # 高信頼度AI信号アラート
        ai_signal_rule = NotificationRule(
            rule_id="ai_high_confidence",
            name="高信頼度AI信号",
            notification_type=NotificationType.AI_SIGNAL,
            conditions={
                "confidence": {"operator": ">=", "value": 0.8},
                "signal_type": {"operator": "in", "value": ["BUY", "SELL"]}
            },
            channels=[DeliveryChannel.EMAIL, DeliveryChannel.WEBSOCKET, DeliveryChannel.DESKTOP],
            priority=Priority.HIGH,
            cooldown_minutes=15,
            template="AI信号: {symbol} - {signal_type} (信頼度: {confidence:.1%})"
        )
        self.add_rule(ai_signal_rule)
        
        # 価格変動アラート
        price_alert_rule = NotificationRule(
            rule_id="price_change",
            name="価格変動アラート",
            notification_type=NotificationType.PRICE_ALERT,
            conditions={
                "change_percent": {"operator": ">=", "value": 5.0}
            },
            channels=[DeliveryChannel.WEBSOCKET, DeliveryChannel.DESKTOP],
            priority=Priority.NORMAL,
            cooldown_minutes=5,
            template="価格変動: {symbol} {change_percent:+.1f}% ({price}円)"
        )
        self.add_rule(price_alert_rule)
        
        # システムエラーアラート
        system_error_rule = NotificationRule(
            rule_id="system_error",
            name="システムエラー",
            notification_type=NotificationType.SYSTEM_STATUS,
            conditions={
                "level": {"operator": ">=", "value": "ERROR"}
            },
            channels=[DeliveryChannel.EMAIL, DeliveryChannel.WEBSOCKET],
            priority=Priority.CRITICAL,
            cooldown_minutes=1,
            template="システムエラー: {message}"
        )
        self.add_rule(system_error_rule)
    
    def add_rule(self, rule: NotificationRule):
        """ルール追加"""
        rule.created_at = datetime.now()
        self.rules[rule.rule_id] = rule
        logging.info(f"Added notification rule: {rule.name}")
    
    def evaluate_rules(self, event_type: NotificationType, 
                      event_data: Dict[str, Any]) -> List[NotificationRule]:
        """ルール評価"""
        triggered_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if rule.notification_type != event_type:
                continue
            
            # クールダウンチェック
            if self._is_in_cooldown(rule):
                continue
            
            # 1日の上限チェック
            if self._exceeds_daily_limit(rule):
                continue
            
            # 条件評価
            if self._evaluate_conditions(rule.conditions, event_data):
                triggered_rules.append(rule)
                self.last_triggered[rule.rule_id] = datetime.now()
                self._increment_counter(rule.rule_id)
        
        return triggered_rules
    
    def _is_in_cooldown(self, rule: NotificationRule) -> bool:
        """クールダウン中かチェック"""
        if rule.rule_id not in self.last_triggered:
            return False
        
        elapsed = datetime.now() - self.last_triggered[rule.rule_id]
        return elapsed < timedelta(minutes=rule.cooldown_minutes)
    
    def _exceeds_daily_limit(self, rule: NotificationRule) -> bool:
        """1日の上限超過チェック"""
        today = datetime.now().strftime('%Y-%m-%d')
        count = self.rule_counters[rule.rule_id].get(today, 0)
        return count >= rule.max_per_day
    
    def _increment_counter(self, rule_id: str):
        """カウンター増加"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.rule_counters[rule_id][today] = \
            self.rule_counters[rule_id].get(today, 0) + 1
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], 
                           event_data: Dict[str, Any]) -> bool:
        """条件評価"""
        for field, condition in conditions.items():
            if field not in event_data:
                return False
            
            field_value = event_data[field]
            operator = condition["operator"]
            condition_value = condition["value"]
            
            if not self._evaluate_condition(field_value, operator, condition_value):
                return False
        
        return True
    
    def _evaluate_condition(self, field_value: Any, operator: str, 
                          condition_value: Any) -> bool:
        """個別条件評価"""
        if operator == "==":
            return field_value == condition_value
        elif operator == "!=":
            return field_value != condition_value
        elif operator == ">=":
            return field_value >= condition_value
        elif operator == "<=":
            return field_value <= condition_value
        elif operator == ">":
            return field_value > condition_value
        elif operator == "<":
            return field_value < condition_value
        elif operator == "in":
            return field_value in condition_value
        elif operator == "not_in":
            return field_value not in condition_value
        elif operator == "contains":
            return str(condition_value) in str(field_value)
        
        return False


class NotificationSystem:
    """通知システムメイン"""
    
    def __init__(self):
        self.rule_engine = NotificationRuleEngine()
        self.notifiers = {}
        self.notification_queue = asyncio.Queue()
        self.notification_history: List[Notification] = []
        
        # データベース初期化
        self._init_database()
        
        # 通知配信者初期化
        self._init_notifiers()
        
        # バックグラウンド処理開始
        self.running = True
        asyncio.create_task(self._process_notifications())
    
    def _init_database(self):
        """データベース初期化"""
        self.db_path = "data/notifications.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                type TEXT,
                priority INTEGER,
                title TEXT,
                message TEXT,
                data TEXT,
                channels TEXT,
                status TEXT,
                created_at DATETIME,
                sent_at DATETIME,
                delivered_at DATETIME,
                retry_count INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON notifications(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON notifications(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON notifications(status)
        """)
        
        conn.commit()
        conn.close()
    
    def _init_notifiers(self):
        """通知配信者初期化"""
        # メール通知
        try:
            self.notifiers[DeliveryChannel.EMAIL] = EmailNotifier(
                smtp_server="smtp.gmail.com",
                smtp_port=587,
                username="noreply@daytrade.com",  # 設定ファイルから読み取り
                password="app_password"           # 設定ファイルから読み取り
            )
        except Exception as e:
            logging.warning(f"Email notifier initialization failed: {e}")
        
        # WebSocket通知
        self.notifiers[DeliveryChannel.WEBSOCKET] = WebSocketNotifier()
        
        # デスクトップ通知  
        self.notifiers[DeliveryChannel.DESKTOP] = DesktopNotifier()
    
    async def trigger_notification(self, event_type: NotificationType,
                                 event_data: Dict[str, Any], user_id: str = "default"):
        """通知トリガー"""
        # ルール評価
        triggered_rules = self.rule_engine.evaluate_rules(event_type, event_data)
        
        for rule in triggered_rules:
            # 通知内容生成
            content = self._generate_content(rule, event_data)
            
            # 通知作成
            notification = Notification(
                notification_id=self._generate_notification_id(),
                user_id=user_id,
                notification_type=event_type,
                priority=rule.priority,
                content=content,
                channels=rule.channels,
                created_at=datetime.now()
            )
            
            # キューに追加
            await self.notification_queue.put(notification)
            
            # データベース保存
            self._save_notification(notification)
    
    def _generate_content(self, rule: NotificationRule, 
                         event_data: Dict[str, Any]) -> NotificationContent:
        """通知内容生成"""
        # テンプレート処理
        if rule.template:
            try:
                title = rule.template.format(**event_data)
                message = title  # 簡単な実装
            except KeyError as e:
                title = f"{rule.name}"
                message = f"データ: {event_data}"
        else:
            title = rule.name
            message = json.dumps(event_data, ensure_ascii=False, indent=2)
        
        return NotificationContent(
            title=title,
            message=message,
            data=event_data,
            action_url=event_data.get('action_url', ''),
            expires_at=datetime.now() + timedelta(hours=24)
        )
    
    def _generate_notification_id(self) -> str:
        """通知ID生成"""
        timestamp = str(int(time.time() * 1000))
        random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"notif_{timestamp}_{random_str}"
    
    async def _process_notifications(self):
        """通知処理ループ"""
        while self.running:
            try:
                # キューから通知取得（1秒タイムアウト）
                notification = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )
                
                # 通知配信
                await self._deliver_notification(notification)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Notification processing error: {e}")
    
    async def _deliver_notification(self, notification: Notification):
        """通知配信"""
        delivery_results = {}
        
        for channel in notification.channels:
            notifier = self.notifiers.get(channel)
            if not notifier:
                logging.warning(f"No notifier for channel: {channel}")
                continue
            
            try:
                if channel == DeliveryChannel.EMAIL:
                    # メール配信（受信者設定が必要）
                    result = await notifier.send_notification(
                        notification, 
                        "user@example.com"  # 実際はユーザー設定から取得
                    )
                elif channel == DeliveryChannel.WEBSOCKET:
                    # WebSocket配信
                    sent_count = await notifier.broadcast_notification(notification)
                    result = sent_count > 0
                elif channel == DeliveryChannel.DESKTOP:
                    # デスクトップ配信
                    result = await notifier.send_notification(notification)
                else:
                    result = False
                
                delivery_results[channel] = result
                
            except Exception as e:
                logging.error(f"Notification delivery failed on {channel}: {e}")
                delivery_results[channel] = False
        
        # 配信状況更新
        if any(delivery_results.values()):
            notification.status = NotificationStatus.SENT
            notification.sent_at = datetime.now()
        else:
            notification.status = NotificationStatus.FAILED
            notification.retry_count += 1
            
            # リトライ
            if notification.retry_count < notification.max_retries:
                await asyncio.sleep(60)  # 1分後にリトライ
                await self.notification_queue.put(notification)
        
        # データベース更新
        self._update_notification(notification)
        
        # 履歴に追加
        self.notification_history.append(notification)
    
    def _save_notification(self, notification: Notification):
        """通知保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO notifications 
            (id, user_id, type, priority, title, message, data, channels, status, created_at, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            notification.notification_id,
            notification.user_id,
            notification.notification_type.value,
            notification.priority.value,
            notification.content.title,
            notification.content.message,
            json.dumps(notification.content.data),
            json.dumps([ch.value for ch in notification.channels]),
            notification.status.value,
            notification.created_at,
            notification.retry_count
        ))
        
        conn.commit()
        conn.close()
    
    def _update_notification(self, notification: Notification):
        """通知更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE notifications 
            SET status = ?, sent_at = ?, delivered_at = ?, retry_count = ?
            WHERE id = ?
        """, (
            notification.status.value,
            notification.sent_at,
            notification.delivered_at,
            notification.retry_count,
            notification.notification_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """通知統計取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本統計
        cursor.execute("SELECT COUNT(*) FROM notifications")
        total_notifications = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM notifications WHERE status = 'SENT'")
        sent_notifications = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM notifications WHERE status = 'FAILED'")
        failed_notifications = cursor.fetchone()[0]
        
        # タイプ別統計
        cursor.execute("""
            SELECT type, COUNT(*) FROM notifications 
            GROUP BY type ORDER BY COUNT(*) DESC
        """)
        type_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_notifications': total_notifications,
            'sent_notifications': sent_notifications,
            'failed_notifications': failed_notifications,
            'success_rate': sent_notifications / max(1, total_notifications),
            'type_distribution': type_stats
        }


# グローバルインスタンス
notification_system = NotificationSystem()


async def send_price_alert(symbol: str, current_price: float, 
                          change_percent: float, user_id: str = "default"):
    """価格アラート送信"""
    await notification_system.trigger_notification(
        NotificationType.PRICE_ALERT,
        {
            'symbol': symbol,
            'price': current_price,
            'change_percent': change_percent,
            'action_url': f'/stocks/{symbol}'
        },
        user_id
    )


async def send_ai_signal_alert(symbol: str, signal_type: str, confidence: float,
                              reason: str, user_id: str = "default"):
    """AI信号アラート送信"""
    await notification_system.trigger_notification(
        NotificationType.AI_SIGNAL,
        {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'reason': reason,
            'action_url': f'/ai-analysis/{symbol}'
        },
        user_id
    )


async def send_system_alert(level: str, message: str, user_id: str = "default"):
    """システムアラート送信"""
    await notification_system.trigger_notification(
        NotificationType.SYSTEM_STATUS,
        {
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat()
        },
        user_id
    )


def get_notification_stats() -> Dict[str, Any]:
    """通知統計取得"""
    return notification_system.get_notification_stats()


async def test_notification_system():
    """通知システムテスト"""
    print("=== Notification System Test ===")
    
    # WebSocketサーバー開始
    if HAS_WEBSOCKETS:
        websocket_notifier = notification_system.notifiers.get(DeliveryChannel.WEBSOCKET)
        if websocket_notifier:
            await websocket_notifier.start_server()
    
    # テスト通知送信
    print("1. Testing price alert...")
    await send_price_alert("7203", 1850.5, 5.2)
    
    print("2. Testing AI signal alert...")
    await send_ai_signal_alert("8306", "BUY", 0.85, "技術指標改善")
    
    print("3. Testing system alert...")
    await send_system_alert("WARNING", "高負荷状況を検出")
    
    # 少し待機（配信処理完了のため）
    await asyncio.sleep(2)
    
    # 統計確認
    print("4. Notification statistics:")
    stats = get_notification_stats()
    print(f"Total notifications: {stats['total_notifications']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Type distribution: {stats['type_distribution']}")
    
    print("Notification system test completed!")


if __name__ == "__main__":
    import asyncio
    os.makedirs('data', exist_ok=True)
    asyncio.run(test_notification_system())
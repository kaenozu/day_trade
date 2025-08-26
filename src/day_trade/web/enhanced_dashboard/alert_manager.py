#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert Manager - アラート管理システム

アラートの監視、条件判定、通知管理を行う
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .config import DashboardConfig, AlertConfig


class AlertManager:
    """アラート管理システム"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_alerts: Dict[str, AlertConfig] = {}
        self.alert_history: List[Dict[str, Any]] = []

    async def check_alerts(
        self,
        symbol: str,
        current_data: Dict[str, Any],
        user_alerts: List[AlertConfig]
    ) -> List[Dict[str, Any]]:
        """アラートチェック"""
        triggered_alerts = []

        try:
            for alert in user_alerts:
                if not alert.enabled or alert.symbol != symbol:
                    continue

                # クールダウンチェック
                if alert.last_triggered:
                    cooldown = timedelta(minutes=5)  # 5分間のクールダウン
                    if datetime.now() - alert.last_triggered < cooldown:
                        continue

                # アラート条件チェック
                if await self._check_alert_condition(alert, current_data):
                    alert_notification = {
                        'alert_id': alert.alert_id,
                        'symbol': symbol,
                        'type': alert.alert_type,
                        'message': self._generate_alert_message(alert, current_data),
                        'timestamp': datetime.now().isoformat(),
                        'severity': self._get_alert_severity(alert, current_data)
                    }

                    triggered_alerts.append(alert_notification)
                    alert.last_triggered = datetime.now()

                    # 履歴に追加
                    self.alert_history.append(alert_notification)

        except Exception as e:
            self.logger.error(f"アラートチェックエラー: {e}")

        return triggered_alerts

    async def _check_alert_condition(self, alert: AlertConfig, data: Dict[str, Any]) -> bool:
        """アラート条件の確認"""
        try:
            if alert.alert_type == 'price_threshold':
                current_price = data.get('price', 0)
                if alert.condition == '>':
                    return current_price > alert.threshold
                elif alert.condition == '<':
                    return current_price < alert.threshold

            elif alert.alert_type == 'price_change_percent':
                change_percent = data.get('change_percent', 0)
                if alert.condition == '>':
                    return abs(change_percent) > alert.threshold

            elif alert.alert_type == 'volume_spike':
                volume = data.get('volume', 0)
                # 通常の出来高と比較（簡略化）
                return volume > alert.threshold

            elif alert.alert_type == 'price_change_absolute':
                change = abs(data.get('change', 0))
                if alert.condition == '>':
                    return change > alert.threshold

            elif alert.alert_type == 'rsi_threshold':
                rsi = data.get('rsi', 50)
                if alert.condition == '>':
                    return rsi > alert.threshold
                elif alert.condition == '<':
                    return rsi < alert.threshold

            return False

        except Exception as e:
            self.logger.error(f"アラート条件チェックエラー: {e}")
            return False

    def _generate_alert_message(self, alert: AlertConfig, data: Dict[str, Any]) -> str:
        """アラートメッセージ生成"""
        try:
            if alert.alert_type == 'price_threshold':
                return (
                    f"{alert.symbol} 価格が{alert.threshold}円を"
                    f"{alert.condition}ました (現在: {data.get('price', 0):.2f}円)"
                )
            elif alert.alert_type == 'price_change_percent':
                return (
                    f"{alert.symbol} 価格変動が{alert.threshold}%を超えました "
                    f"(変動: {data.get('change_percent', 0):.2f}%)"
                )
            elif alert.alert_type == 'volume_spike':
                return (
                    f"{alert.symbol} 出来高急増を検知しました "
                    f"(出来高: {data.get('volume', 0):,})"
                )
            elif alert.alert_type == 'price_change_absolute':
                return (
                    f"{alert.symbol} 価格変動が{alert.threshold}円を超えました "
                    f"(変動: {data.get('change', 0):.2f}円)"
                )
            elif alert.alert_type == 'rsi_threshold':
                rsi = data.get('rsi', 50)
                if alert.condition == '>':
                    status = "買われすぎ"
                else:
                    status = "売られすぎ"
                return (
                    f"{alert.symbol} RSIが{status}レベルです "
                    f"(RSI: {rsi:.1f})"
                )
            else:
                return f"{alert.symbol} アラートが発生しました"

        except Exception as e:
            self.logger.error(f"アラートメッセージ生成エラー: {e}")
            return f"{alert.symbol} アラート"

    def _get_alert_severity(self, alert: AlertConfig, data: Dict[str, Any]) -> str:
        """アラート重要度の判定"""
        try:
            if alert.alert_type == 'price_change_percent':
                change = abs(data.get('change_percent', 0))
                if change > 10:
                    return 'critical'
                elif change > 5:
                    return 'warning'
                else:
                    return 'info'

            elif alert.alert_type == 'volume_spike':
                volume = data.get('volume', 0)
                if volume > alert.threshold * 3:
                    return 'critical'
                elif volume > alert.threshold * 2:
                    return 'warning'
                else:
                    return 'info'

            elif alert.alert_type == 'rsi_threshold':
                rsi = data.get('rsi', 50)
                if rsi > 80 or rsi < 20:
                    return 'critical'
                elif rsi > 70 or rsi < 30:
                    return 'warning'
                else:
                    return 'info'

            else:
                return 'info'

        except Exception as e:
            self.logger.error(f"アラート重要度判定エラー: {e}")
            return 'info'

    def add_alert(self, alert: AlertConfig):
        """アラートの追加"""
        self.active_alerts[alert.alert_id] = alert
        self.logger.info(f"アラート追加: {alert.alert_id} ({alert.symbol})")

    def remove_alert(self, alert_id: str):
        """アラートの削除"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            self.logger.info(f"アラート削除: {alert_id}")

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """アラート履歴の取得"""
        return self.alert_history[-limit:]

    def clear_alert_history(self):
        """アラート履歴のクリア"""
        self.alert_history.clear()
        self.logger.info("アラート履歴をクリアしました")

    def get_active_alerts(self) -> Dict[str, AlertConfig]:
        """アクティブなアラートの取得"""
        return self.active_alerts.copy()

    def update_alert(self, alert_id: str, updates: Dict[str, Any]):
        """アラートの更新"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            for key, value in updates.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
            self.logger.info(f"アラート更新: {alert_id}")
        else:
            self.logger.warning(f"アラートが見つかりません: {alert_id}")

    def enable_alert(self, alert_id: str):
        """アラートの有効化"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].enabled = True
            self.logger.info(f"アラート有効化: {alert_id}")

    def disable_alert(self, alert_id: str):
        """アラートの無効化"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].enabled = False
            self.logger.info(f"アラート無効化: {alert_id}")

    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計情報の取得"""
        total_alerts = len(self.alert_history)
        if total_alerts == 0:
            return {
                'total_alerts': 0,
                'alerts_by_severity': {},
                'alerts_by_type': {},
                'alerts_by_symbol': {}
            }

        severity_counts = {}
        type_counts = {}
        symbol_counts = {}

        for alert in self.alert_history:
            severity = alert.get('severity', 'info')
            alert_type = alert.get('type', 'unknown')
            symbol = alert.get('symbol', 'unknown')

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return {
            'total_alerts': total_alerts,
            'alerts_by_severity': severity_counts,
            'alerts_by_type': type_counts,
            'alerts_by_symbol': symbol_counts
        }
#!/usr/bin/env python3
"""
Day Trade Personal - 分析履歴管理システム

個人投資家向けのシンプルな分析履歴保存・比較機能
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

class PersonalAnalysisHistory:
    """個人向け分析履歴管理"""

    def __init__(self, db_path: str = "personal_analysis_history.db"):
        self.db_path = Path(db_path)
        self.init_database()

    def init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 分析履歴テーブル作成
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_type TEXT,
                symbol_count INTEGER,
                recommendations TEXT,  -- JSON形式
                portfolio_data TEXT,   -- JSON形式（オプション）
                total_score REAL,
                buy_count INTEGER,
                performance_time REAL
            )
        ''')

        # アラート履歴テーブル作成
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT,
                symbol TEXT,
                message TEXT,
                priority INTEGER,  -- 1:高 2:中 3:低
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def save_analysis_result(self, analysis_data: Dict[str, Any]):
        """分析結果を履歴に保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # データ抽出
        recommendations = analysis_data.get('recommendations', [])
        portfolio_data = analysis_data.get('portfolio_data')
        analysis_type = analysis_data.get('analysis_type', 'basic')
        performance_time = analysis_data.get('performance_time', 0.0)

        # 統計計算
        total_score = sum(r.get('score', 0) for r in recommendations) / len(recommendations) if recommendations else 0
        buy_count = sum(1 for r in recommendations if r.get('action') in ['買い', '強い買い'])

        cursor.execute('''
            INSERT INTO analysis_history
            (analysis_type, symbol_count, recommendations, portfolio_data, total_score, buy_count, performance_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_type,
            len(recommendations),
            json.dumps(recommendations, ensure_ascii=False),
            json.dumps(portfolio_data, ensure_ascii=False) if portfolio_data else None,
            total_score,
            buy_count,
            performance_time
        ))

        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return analysis_id

    def get_recent_analyses(self, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """最近の分析結果取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = datetime.now() - timedelta(days=days)

        cursor.execute('''
            SELECT id, analysis_date, analysis_type, symbol_count,
                   total_score, buy_count, performance_time
            FROM analysis_history
            WHERE analysis_date >= ?
            ORDER BY analysis_date DESC
            LIMIT ?
        ''', (since_date, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'date': row[1],
                'type': row[2],
                'symbol_count': row[3],
                'total_score': row[4],
                'buy_count': row[5],
                'performance_time': row[6]
            })

        conn.close()
        return results

    def get_analysis_detail(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """分析詳細取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM analysis_history WHERE id = ?
        ''', (analysis_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'id': row[0],
            'date': row[1],
            'type': row[2],
            'symbol_count': row[3],
            'recommendations': json.loads(row[4]) if row[4] else [],
            'portfolio_data': json.loads(row[5]) if row[5] else None,
            'total_score': row[6],
            'buy_count': row[7],
            'performance_time': row[8]
        }

    def get_performance_trend(self, days: int = 30) -> Dict[str, Any]:
        """パフォーマンストレンド分析"""
        conn = sqlite3.connect(self.db_path)

        since_date = datetime.now() - timedelta(days=days)

        # 分析回数トレンド
        df_count = pd.read_sql('''
            SELECT DATE(analysis_date) as date, COUNT(*) as count
            FROM analysis_history
            WHERE analysis_date >= ?
            GROUP BY DATE(analysis_date)
            ORDER BY date
        ''', conn, params=(since_date,))

        # スコアトレンド
        df_score = pd.read_sql('''
            SELECT DATE(analysis_date) as date, AVG(total_score) as avg_score
            FROM analysis_history
            WHERE analysis_date >= ?
            GROUP BY DATE(analysis_date)
            ORDER BY date
        ''', conn, params=(since_date,))

        # 買い推奨率トレンド
        df_buy_rate = pd.read_sql('''
            SELECT DATE(analysis_date) as date,
                   AVG(CAST(buy_count AS REAL) / symbol_count) as buy_rate
            FROM analysis_history
            WHERE analysis_date >= ? AND symbol_count > 0
            GROUP BY DATE(analysis_date)
            ORDER BY date
        ''', conn, params=(since_date,))

        conn.close()

        return {
            'analysis_count_trend': df_count.to_dict('records'),
            'score_trend': df_score.to_dict('records'),
            'buy_rate_trend': df_buy_rate.to_dict('records')
        }

    def save_alert(self, alert_type: str, symbol: str, message: str, priority: int = 2):
        """アラート保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO alert_history (alert_type, symbol, message, priority)
            VALUES (?, ?, ?, ?)
        ''', (alert_type, symbol, message, priority))

        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return alert_id

    def get_unread_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """未読アラート取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, alert_date, alert_type, symbol, message, priority
            FROM alert_history
            WHERE acknowledged = FALSE
            ORDER BY priority ASC, alert_date DESC
            LIMIT ?
        ''', (limit,))

        alerts = []
        for row in cursor.fetchall():
            priority_text = {1: '高', 2: '中', 3: '低'}.get(row[5], '不明')
            alerts.append({
                'id': row[0],
                'date': row[1],
                'type': row[2],
                'symbol': row[3],
                'message': row[4],
                'priority': row[5],
                'priority_text': priority_text
            })

        conn.close()
        return alerts

    def acknowledge_alert(self, alert_id: int):
        """アラート既読処理"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE alert_history SET acknowledged = TRUE WHERE id = ?
        ''', (alert_id,))

        conn.commit()
        conn.close()

    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """サマリーレポート生成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = datetime.now() - timedelta(days=days)

        # 基本統計
        cursor.execute('''
            SELECT
                COUNT(*) as total_analyses,
                AVG(total_score) as avg_score,
                AVG(buy_count) as avg_buy_count,
                AVG(performance_time) as avg_time,
                MAX(total_score) as best_score,
                MIN(total_score) as worst_score
            FROM analysis_history
            WHERE analysis_date >= ?
        ''', (since_date,))

        stats = cursor.fetchone()

        # 分析タイプ別統計
        cursor.execute('''
            SELECT analysis_type, COUNT(*) as count
            FROM analysis_history
            WHERE analysis_date >= ?
            GROUP BY analysis_type
        ''', (since_date,))

        type_stats = {row[0]: row[1] for row in cursor.fetchall()}

        # アラート統計
        cursor.execute('''
            SELECT COUNT(*) as total_alerts,
                   SUM(CASE WHEN acknowledged = TRUE THEN 1 ELSE 0 END) as read_alerts
            FROM alert_history
            WHERE alert_date >= ?
        ''', (since_date,))

        alert_stats = cursor.fetchone()

        conn.close()

        return {
            'period_days': days,
            'analysis_stats': {
                'total_analyses': stats[0] or 0,
                'avg_score': round(stats[1] or 0, 1),
                'avg_buy_count': round(stats[2] or 0, 1),
                'avg_time': round(stats[3] or 0, 2),
                'best_score': round(stats[4] or 0, 1),
                'worst_score': round(stats[5] or 0, 1)
            },
            'type_breakdown': type_stats,
            'alert_stats': {
                'total_alerts': alert_stats[0] or 0,
                'read_alerts': alert_stats[1] or 0,
                'unread_alerts': (alert_stats[0] or 0) - (alert_stats[1] or 0)
            }
        }

class PersonalAlertSystem:
    """個人向けアラートシステム"""

    def __init__(self, history_manager: PersonalAnalysisHistory):
        self.history = history_manager
        self.alert_rules = {
            'high_score_alert': 85.0,      # 高スコアアラート閾値
            'low_confidence_alert': 60.0,   # 低信頼度アラート閾値
            'portfolio_change_alert': 10.0, # ポートフォリオ変化アラート閾値（%）
        }

    def check_analysis_alerts(self, recommendations: List[Dict[str, Any]]):
        """分析結果に基づくアラートチェック"""
        alerts_generated = []

        for rec in recommendations:
            symbol = rec.get('symbol', '')
            score = rec.get('score', 0)
            confidence = rec.get('confidence', 0)
            action = rec.get('action', '')

            # 高スコア銘柄アラート
            if score >= self.alert_rules['high_score_alert'] and action in ['買い', '強い買い']:
                message = f"{rec.get('name', symbol)} が高スコア ({score:.1f}点) で買い推奨です！"
                alert_id = self.history.save_alert('high_score', symbol, message, priority=1)
                alerts_generated.append(alert_id)

            # 低信頼度アラート
            if confidence <= self.alert_rules['low_confidence_alert']:
                message = f"{rec.get('name', symbol)} の分析信頼度が低い ({confidence:.0f}%) ため注意が必要です"
                alert_id = self.history.save_alert('low_confidence', symbol, message, priority=3)
                alerts_generated.append(alert_id)

        return alerts_generated

    def check_portfolio_alerts(self, current_portfolio: Dict[str, Any], previous_portfolio: Optional[Dict[str, Any]]):
        """ポートフォリオ変化アラート"""
        if not previous_portfolio:
            return []

        alerts_generated = []
        current_allocation = current_portfolio.get('recommended_allocation', {})
        previous_allocation = previous_portfolio.get('recommended_allocation', {})

        # 新規推奨銘柄
        new_symbols = set(current_allocation.keys()) - set(previous_allocation.keys())
        for symbol in new_symbols:
            alloc = current_allocation[symbol]
            message = f"新規推奨銘柄: {alloc['name']} ({alloc['allocation_percent']:.1f}%配分)"
            alert_id = self.history.save_alert('new_recommendation', symbol, message, priority=2)
            alerts_generated.append(alert_id)

        # 配分大幅変化
        for symbol in set(current_allocation.keys()) & set(previous_allocation.keys()):
            current_pct = current_allocation[symbol]['allocation_percent']
            previous_pct = previous_allocation[symbol]['allocation_percent']
            change_pct = abs(current_pct - previous_pct)

            if change_pct >= self.alert_rules['portfolio_change_alert']:
                direction = "増加" if current_pct > previous_pct else "減少"
                message = f"{current_allocation[symbol]['name']} の推奨配分が{direction} ({previous_pct:.1f}% → {current_pct:.1f}%)"
                alert_id = self.history.save_alert('allocation_change', symbol, message, priority=2)
                alerts_generated.append(alert_id)

        return alerts_generated

    def display_alerts(self):
        """アラート表示"""
        alerts = self.history.get_unread_alerts()

        if not alerts:
            print("[アラート] 新しいアラートはありません")
            return

        print(f"\n[アラート] 未読アラート {len(alerts)}件")
        print("-" * 50)

        for alert in alerts:
            priority_icon = {"高": "[!!!]", "中": "[!]", "低": "[i]"}.get(alert['priority_text'], "[?]")

            print(f"{priority_icon} {alert['type']}: {alert['symbol']}")
            print(f"    {alert['message']}")
            print(f"    日時: {alert['date']}")
            print()

    def acknowledge_all_alerts(self):
        """全アラート既読処理"""
        alerts = self.history.get_unread_alerts()
        for alert in alerts:
            self.history.acknowledge_alert(alert['id'])

        if alerts:
            print(f"[アラート] {len(alerts)}件のアラートを既読にしました")

# 使用例
def demo_history_system():
    """履歴システムデモ"""
    print("=== 分析履歴・アラートシステム デモ ===")

    # システム初期化
    history = PersonalAnalysisHistory()
    alert_system = PersonalAlertSystem(history)

    # サンプル分析結果の保存
    sample_analysis = {
        'analysis_type': 'multi_symbol',
        'recommendations': [
            {'symbol': '7203', 'name': 'トヨタ自動車', 'score': 87.5, 'confidence': 85, 'action': '買い'},
            {'symbol': '8306', 'name': '三菱UFJ', 'score': 72.3, 'confidence': 65, 'action': '検討'}
        ],
        'performance_time': 1.2
    }

    analysis_id = history.save_analysis_result(sample_analysis)
    print(f"分析結果を保存しました (ID: {analysis_id})")

    # アラートチェック
    alerts = alert_system.check_analysis_alerts(sample_analysis['recommendations'])
    print(f"アラート {len(alerts)}件を生成しました")

    # 履歴表示
    recent = history.get_recent_analyses(days=7)
    print(f"\n最近の分析履歴: {len(recent)}件")

    # アラート表示
    alert_system.display_alerts()

    # サマリーレポート
    summary = history.generate_summary_report(days=7)
    print(f"\n7日間サマリー:")
    print(f"  分析回数: {summary['analysis_stats']['total_analyses']}回")
    print(f"  平均スコア: {summary['analysis_stats']['avg_score']:.1f}点")
    print(f"  未読アラート: {summary['alert_stats']['unread_alerts']}件")

if __name__ == "__main__":
    demo_history_system()
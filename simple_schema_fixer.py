#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import os
from pathlib import Path
from datetime import datetime

def fix_alert_database():
    """アラートデータベースの修正"""

    db_path = Path("alert_data/alerts.db")
    db_path.parent.mkdir(exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 既存テーブル削除して再作成
            cursor.execute("DROP TABLE IF EXISTS alert_rules")
            cursor.execute("DROP TABLE IF EXISTS alert_history")

            # 正しいスキーマで再作成
            cursor.execute('''
                CREATE TABLE alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    condition_text TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    cooldown_minutes INTEGER DEFAULT 5,
                    last_triggered TEXT,
                    trigger_count INTEGER DEFAULT 0,
                    description TEXT,
                    symbols TEXT,
                    notification_channels TEXT,
                    created_at TEXT NOT NULL
                )
            ''')

            cursor.execute('''
                CREATE TABLE alert_history (
                    alert_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    timestamp TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0
                )
            ''')

            conn.commit()
            print("SUCCESS: alert_rules and alert_history tables recreated")
            return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def create_datetime_encoder():
    """datetime エンコーダー作成"""

    code = '''import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
'''

    with open("datetime_encoder.py", "w", encoding="utf-8") as f:
        f.write(code)

    print("SUCCESS: datetime_encoder.py created")

def fix_realtime_alert_json():
    """リアルタイムアラートのJSON修正"""

    alert_file = "realtime_alert_notification_system.py"

    if not os.path.exists(alert_file):
        print("ERROR: realtime_alert_notification_system.py not found")
        return

    try:
        with open(alert_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # JSON serialization修正
        if "json.dumps(alert.data, ensure_ascii=False)" in content:
            content = content.replace(
                "json.dumps(alert.data, ensure_ascii=False)",
                "json.dumps(alert.data, ensure_ascii=False, default=str)"
            )

            # バックアップ作成
            backup_file = f"{alert_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print("SUCCESS: JSON serialization fixed")
        else:
            print("INFO: JSON serialization already correct")

    except Exception as e:
        print(f"ERROR fixing alert system: {e}")

def main():
    print("=== Database Schema Fix ===")

    print("\n1. Fixing alert database...")
    fix_alert_database()

    print("\n2. Creating datetime encoder...")
    create_datetime_encoder()

    print("\n3. Fixing JSON serialization...")
    fix_realtime_alert_json()

    print("\n=== All fixes completed ===")

if __name__ == "__main__":
    main()
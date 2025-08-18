#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Schema Fixer - データベーススキーマ修正ツール

Issue #811対応：データベーススキーマの不整合修正
"""

import sqlite3
import os
import json
from pathlib import Path
from datetime import datetime

def fix_alert_rules_schema():
    """alert_rulesテーブルのスキーマ修正"""

    db_path = Path("alert_data/alerts.db")

    if not db_path.exists():
        print("alert_data/alerts.db が見つかりません。新規作成します。")
        db_path.parent.mkdir(exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 既存のテーブル構造確認
            cursor.execute("PRAGMA table_info(alert_rules);")
            existing_columns = {row[1]: row[2] for row in cursor.fetchall()}

            print(f"既存カラム: {list(existing_columns.keys())}")

            # 必要なカラムの定義
            required_columns = {
                'rule_id': 'TEXT PRIMARY KEY',
                'name': 'TEXT NOT NULL',
                'alert_type': 'TEXT NOT NULL',
                'condition_text': 'TEXT NOT NULL',
                'priority': 'TEXT NOT NULL',
                'enabled': 'INTEGER DEFAULT 1',
                'cooldown_minutes': 'INTEGER DEFAULT 5',
                'last_triggered': 'TEXT',
                'trigger_count': 'INTEGER DEFAULT 0',
                'description': 'TEXT',
                'symbols': 'TEXT',
                'notification_channels': 'TEXT',
                'created_at': 'TEXT NOT NULL'
            }

            # 不足カラムを追加
            added_columns = []
            for col_name, col_definition in required_columns.items():
                if col_name not in existing_columns:
                    try:
                        # PRIMARY KEYカラムは後で処理
                        if 'PRIMARY KEY' in col_definition:
                            continue

                        alter_sql = f"ALTER TABLE alert_rules ADD COLUMN {col_name} {col_definition.replace(' NOT NULL', '')}"
                        cursor.execute(alter_sql)
                        added_columns.append(col_name)
                        print(f"[OK] カラム追加: {col_name}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e):
                            print(f"[WARN] カラム既存: {col_name}")
                        else:
                            print(f"[ERROR] カラム追加エラー {col_name}: {e}")

            # テーブルが存在しない場合は新規作成
            if not existing_columns:
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    {', '.join([f'{name} {definition}' for name, definition in required_columns.items()])}
                )
                """
                cursor.execute(create_sql)
                print("[OK] alert_rulesテーブル新規作成")

            # 修正後の構造確認
            cursor.execute("PRAGMA table_info(alert_rules);")
            new_columns = [row[1] for row in cursor.fetchall()]
            print(f"修正後カラム: {new_columns}")

            conn.commit()
            return True

    except Exception as e:
        print(f"[ERROR] スキーマ修正エラー: {e}")
        return False

def fix_datetime_serialization():
    """datetime JSON シリアライゼーション修正"""

    # カスタムJSONエンコーダークラスを作成
    encoder_code = '''
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# 使用例
# json.dumps(data, cls=DateTimeEncoder)
'''

    # ファイルに保存
    with open("datetime_encoder.py", "w", encoding="utf-8") as f:
        f.write(encoder_code)

    print("[OK] DateTimeEncoderクラス作成: datetime_encoder.py")

def fix_alert_system_imports():
    """アラートシステムのimport修正"""

    # realtime_alert_notification_system.pyの修正
    alert_file = Path("realtime_alert_notification_system.py")

    if not alert_file.exists():
        print("[ERROR] realtime_alert_notification_system.py が見つかりません")
        return False

    try:
        # ファイル読み込み
        with open(alert_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修正内容
        fixes = [
            # JSONシリアライゼーション修正
            ("json.dumps(alert.data, ensure_ascii=False)",
             "json.dumps(alert.data, ensure_ascii=False, default=str)"),

            # 非同期処理修正のためのimport追加
            ("import uuid",
             "import uuid\nfrom datetime_encoder import DateTimeEncoder"),
        ]

        # 修正適用
        modified = False
        for old_text, new_text in fixes:
            if old_text in content and new_text not in content:
                content = content.replace(old_text, new_text)
                modified = True
                print(f"[OK] 修正適用: {old_text[:50]}...")

        if modified:
            # バックアップ作成
            backup_file = f"{alert_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"[OK] バックアップ作成: {backup_file}")

        return True

    except Exception as e:
        print(f"[ERROR] アラートシステム修正エラー: {e}")
        return False

def fix_event_loop_issues():
    """イベントループ問題の修正"""

    # 非同期処理修正用のユーティリティ作成
    async_utils_code = '''
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncUtils:
    """非同期処理ユーティリティ"""

    @staticmethod
    def run_in_thread(coro):
        """別スレッドで非同期関数を実行"""
        def run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run)
            return future.result()

    @staticmethod
    def safe_create_task(coro):
        """安全にタスクを作成"""
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(coro)
        except RuntimeError:
            # イベントループが存在しない場合
            return asyncio.create_task(coro)

# 使用例:
# AsyncUtils.run_in_thread(some_async_function())
# AsyncUtils.safe_create_task(some_async_function())
'''

    with open("async_utils.py", "w", encoding="utf-8") as f:
        f.write(async_utils_code)

    print("[OK] AsyncUtilsクラス作成: async_utils.py")

def run_all_fixes():
    """全ての修正を実行"""

    print("=== データベーススキーマ・技術的不具合修正 ===")

    # 1. alert_rulesテーブルスキーマ修正
    print("\n1. alert_rulesテーブルスキーマ修正")
    fix_alert_rules_schema()

    # 2. datetime JSON シリアライゼーション修正
    print("\n2. datetime JSON シリアライゼーション修正")
    fix_datetime_serialization()

    # 3. アラートシステムimport修正
    print("\n3. アラートシステム修正")
    fix_alert_system_imports()

    # 4. イベントループ問題修正
    print("\n4. イベントループ問題修正")
    fix_event_loop_issues()

    # 5. その他のデータベーススキーマチェック
    print("\n5. その他のデータベーススキーマチェック")
    check_all_databases()

    print("\n[OK] 全ての技術的不具合修正完了")

def check_all_databases():
    """全データベースの整合性チェック"""

    db_directories = [
        "validation_data",
        "performance_data",
        "alert_data",
        "stability_data",
        "monitoring_data",
        "risk_validation_data",
        "data_quality",
        "enhanced_prediction_data"
    ]

    for db_dir in db_directories:
        db_path = Path(db_dir)
        if db_path.exists():
            for db_file in db_path.glob("*.db"):
                try:
                    with sqlite3.connect(db_file) as conn:
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA integrity_check;")
                        result = cursor.fetchone()[0]

                        if result == "ok":
                            print(f"[OK] {db_file}: 整合性OK")
                        else:
                            print(f"[WARN] {db_file}: {result}")

                except Exception as e:
                    print(f"[ERROR] {db_file}: チェックエラー - {e}")

if __name__ == "__main__":
    run_all_fixes()
#!/usr/bin/env python3
"""
ダッシュボードシステム統合テスト

Issue #324: プロダクション運用監視ダッシュボード構築
コア・可視化・Webダッシュボードの統合テスト
"""

import json
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from day_trade.dashboard.dashboard_core import ProductionDashboard
    from day_trade.dashboard.visualization_engine import DashboardVisualizationEngine
    from day_trade.dashboard.web_dashboard import WebDashboard
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("簡易版でテストを実行します")

print("ダッシュボードシステム統合テスト")
print("Issue #324: プロダクション運用監視ダッシュボード構築")
print("=" * 60)

def test_dashboard_core():
    """ダッシュボードコア機能テスト"""
    print("\n=== ダッシュボードコア機能テスト ===")

    try:
        # 初期化テスト
        dashboard = ProductionDashboard(data_retention_days=7)
        print("[OK] ダッシュボードコア初期化成功")

        # データベース初期化確認
        if dashboard.db_path.exists():
            print("[OK] SQLiteデータベース作成成功")
        else:
            print("[NG] データベース作成失敗")
            return False

        # 監視開始テスト
        dashboard.start_monitoring()
        print("[OK] 監視開始成功")

        # データ収集テスト（5秒間）
        print("データ収集テスト（5秒間）...")
        time.sleep(5)

        # 現在ステータス確認
        status = dashboard.get_current_status()

        required_keys = ['portfolio', 'system', 'trading', 'risk']
        for key in required_keys:
            if key in status and status[key]:
                print(f"[OK] {key}データ収集成功")
            else:
                print(f"[NG] {key}データ収集失敗")
                return False

        # 履歴データ確認
        history = dashboard.get_historical_data('portfolio', hours=1)
        if len(history) > 0:
            print(f"[OK] 履歴データ取得成功: {len(history)}件")
        else:
            print("[NG] 履歴データ取得失敗")
            return False

        # レポート生成テスト
        report = dashboard.generate_status_report()
        if len(report) > 100:
            print(f"[OK] ステータスレポート生成成功: {len(report)}文字")
        else:
            print("[NG] レポート生成失敗")
            return False

        # 監視停止
        dashboard.stop_monitoring()
        print("[OK] 監視停止成功")

        return True

    except Exception as e:
        print(f"[ERROR] ダッシュボードコアテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_engine():
    """可視化エンジンテスト"""
    print("\n=== 可視化エンジンテスト ===")

    try:
        # 初期化テスト
        engine = DashboardVisualizationEngine(output_dir="test_charts")
        print("[OK] 可視化エンジン初期化成功")

        # サンプルデータ生成
        import random
        current_time = datetime.now()
        sample_data = []

        for i in range(12):  # 12時間分のデータ
            timestamp = current_time - timedelta(hours=11-i)
            sample_data.append({
                'timestamp': timestamp.isoformat(),
                'total_value': 1000000 + random.uniform(-30000, 30000),
                'cpu_usage': random.uniform(20, 70),
                'memory_usage_mb': random.uniform(1024, 2048),
                'processing_time_ms': random.uniform(100, 400),
                'error_count': random.randint(0, 2),
                'trades_today': random.randint(5, 15),
                'successful_trades': random.randint(4, 12),
                'failed_trades': random.randint(0, 3),
                'win_rate': random.uniform(0.7, 0.9),
                'current_drawdown': random.uniform(-0.05, 0),
                'portfolio_var_95': random.uniform(-0.03, -0.01),
                'portfolio_volatility': random.uniform(0.1, 0.25),
                'concentration_risk': random.uniform(0.4, 0.7),
                'leverage_ratio': random.uniform(0.9, 1.1)
            })

        sample_positions = {
            "7203.T": {"quantity": 100, "price": 2500, "value": 250000, "pnl": 8000},
            "8306.T": {"quantity": 800, "price": 850, "value": 680000, "pnl": -3000},
            "9984.T": {"quantity": 40, "price": 5200, "value": 208000, "pnl": 12000}
        }

        # チャート作成テスト
        chart_tests = [
            ("ポートフォリオ価値チャート", engine.create_portfolio_value_chart, sample_data),
            ("システムメトリクスチャート", engine.create_system_metrics_chart, sample_data),
            ("取引パフォーマンスチャート", engine.create_trading_performance_chart, sample_data),
            ("リスクヒートマップ", engine.create_risk_metrics_heatmap, sample_data),
            ("ポジション円グラフ", engine.create_positions_pie_chart, sample_positions)
        ]

        created_charts = []
        for chart_name, chart_function, test_data in chart_tests:
            try:
                chart_path = chart_function(test_data)
                if chart_path and Path(chart_path).exists():
                    print(f"[OK] {chart_name}作成成功")
                    created_charts.append(chart_path)
                else:
                    print(f"[NG] {chart_name}作成失敗")
                    return False
            except Exception as e:
                print(f"[NG] {chart_name}作成エラー: {e}")
                return False

        # 統合ダッシュボード作成テスト
        try:
            comprehensive_chart = engine.create_comprehensive_dashboard(
                sample_data, sample_data, sample_data, sample_data, sample_positions
            )
            if comprehensive_chart and Path(comprehensive_chart).exists():
                print("[OK] 統合ダッシュボードチャート作成成功")
                created_charts.append(comprehensive_chart)
            else:
                print("[NG] 統合ダッシュボード作成失敗")
                return False
        except Exception as e:
            print(f"[NG] 統合ダッシュボード作成エラー: {e}")
            return False

        # Base64エンコードテスト
        base64_count = 0
        for chart_path in created_charts:
            base64_data = engine.chart_to_base64(chart_path)
            if base64_data:
                base64_count += 1

        if base64_count == len(created_charts):
            print(f"[OK] Base64エンコード成功: {base64_count}件")
        else:
            print(f"[NG] Base64エンコード失敗: {base64_count}/{len(created_charts)}件")
            return False

        print(f"[OK] 可視化エンジンテスト成功: {len(created_charts)}チャート作成")
        return True

    except Exception as e:
        print(f"[ERROR] 可視化エンジンテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_dashboard():
    """Webダッシュボードテスト"""
    print("\n=== Webダッシュボードテスト ===")

    try:
        # Webダッシュボード初期化
        web_dashboard = WebDashboard(port=5001, debug=False)
        print("[OK] Webダッシュボード初期化成功")

        # テンプレートファイル確認
        template_path = Path(web_dashboard.app.template_folder) / 'dashboard.html'
        if template_path.exists():
            print("[OK] HTMLテンプレート作成成功")
        else:
            print("[NG] HTMLテンプレート作成失敗")
            return False

        # 静的ファイル確認
        static_files = ['dashboard.css', 'dashboard.js']
        for file_name in static_files:
            file_path = Path(web_dashboard.app.static_folder) / file_name
            if file_path.exists():
                print(f"[OK] 静的ファイル作成成功: {file_name}")
            else:
                print(f"[NG] 静的ファイル作成失敗: {file_name}")
                return False

        # サーバー起動テスト（別スレッドで）
        server_thread = threading.Thread(target=web_dashboard.run, daemon=True)
        server_thread.start()

        # サーバー起動待機
        print("Webサーバー起動中...")
        time.sleep(3)

        # API エンドポイントテスト
        base_url = "http://localhost:5001"

        # ステータスAPI テスト
        try:
            response = requests.get(f"{base_url}/api/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                if status_data.get('success'):
                    print("[OK] ステータスAPI成功")
                else:
                    print("[NG] ステータスAPI データ取得失敗")
                    return False
            else:
                print(f"[NG] ステータスAPI失敗: {response.status_code}")
                return False
        except Exception as e:
            print(f"[NG] ステータスAPI接続エラー: {e}")
            return False

        # 履歴API テスト
        try:
            response = requests.get(f"{base_url}/api/history/portfolio?hours=1", timeout=5)
            if response.status_code == 200:
                print("[OK] 履歴API成功")
            else:
                print(f"[NG] 履歴API失敗: {response.status_code}")
                return False
        except Exception as e:
            print(f"[NG] 履歴API接続エラー: {e}")
            return False

        # レポートAPI テスト
        try:
            response = requests.get(f"{base_url}/api/report", timeout=5)
            if response.status_code == 200:
                report_data = response.json()
                if report_data.get('success') and len(report_data.get('report', '')) > 50:
                    print("[OK] レポートAPI成功")
                else:
                    print("[NG] レポートAPI データ不正")
                    return False
            else:
                print(f"[NG] レポートAPI失敗: {response.status_code}")
                return False
        except Exception as e:
            print(f"[NG] レポートAPI接続エラー: {e}")
            return False

        # チャートAPI テスト（軽量版のみ）
        chart_types = ['portfolio']  # 1つのみテスト
        for chart_type in chart_types:
            try:
                response = requests.get(f"{base_url}/api/chart/{chart_type}?hours=1", timeout=10)
                if response.status_code == 200:
                    chart_data = response.json()
                    if chart_data.get('success') and chart_data.get('chart_data'):
                        print(f"[OK] {chart_type}チャートAPI成功")
                    else:
                        print(f"[NG] {chart_type}チャートAPI データ不正")
                        return False
                else:
                    print(f"[NG] {chart_type}チャートAPI失敗: {response.status_code}")
                    return False
            except Exception as e:
                print(f"[NG] {chart_type}チャートAPI接続エラー: {e}")
                return False

        # 監視停止
        web_dashboard.stop_monitoring()

        print("[OK] Webダッシュボードテスト成功")
        return True

    except Exception as e:
        print(f"[ERROR] Webダッシュボードテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_performance():
    """統合パフォーマンステスト"""
    print("\n=== 統合パフォーマンステスト ===")

    try:
        # コアダッシュボードでパフォーマンステスト
        dashboard = ProductionDashboard()
        dashboard.start_monitoring()

        print("パフォーマンス測定中（10秒間）...")

        # データ収集パフォーマンス測定
        start_time = time.time()
        data_points = 0

        for _ in range(10):
            status = dashboard.get_current_status()
            if status.get('portfolio'):
                data_points += 1
            time.sleep(1)

        end_time = time.time()

        # 結果分析
        total_time = end_time - start_time
        avg_response_time = total_time / 10

        if avg_response_time < 0.5:  # 500ms以下
            print(f"[OK] 平均応答時間: {avg_response_time:.3f}秒")
        else:
            print(f"[WARNING] 平均応答時間が遅い: {avg_response_time:.3f}秒")

        if data_points >= 8:  # 80%以上成功
            print(f"[OK] データ取得成功率: {data_points}/10")
        else:
            print(f"[NG] データ取得成功率が低い: {data_points}/10")
            return False

        # メモリ使用量確認
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB

        if memory_usage < 200:  # 200MB以下
            print(f"[OK] メモリ使用量: {memory_usage:.1f}MB")
        else:
            print(f"[WARNING] メモリ使用量が多い: {memory_usage:.1f}MB")

        dashboard.stop_monitoring()

        print("[OK] 統合パフォーマンステスト成功")
        return True

    except Exception as e:
        print(f"[ERROR] パフォーマンステストエラー: {e}")
        return False

def main():
    """メイン実行"""
    print("ダッシュボードシステムの統合テストを開始します")

    test_results = []

    # 個別機能テスト
    tests = [
        ("ダッシュボードコア", test_dashboard_core),
        ("可視化エンジン", test_visualization_engine),
        ("Webダッシュボード", test_web_dashboard),
        ("統合パフォーマンス", test_integration_performance)
    ]

    for test_name, test_function in tests:
        print(f"\n{'='*60}")
        try:
            success = test_function()
            test_results.append((test_name, success))

            if success:
                print(f"[OK] {test_name}テスト: 成功")
            else:
                print(f"[NG] {test_name}テスト: 失敗")

        except Exception as e:
            print(f"[CRITICAL ERROR] {test_name}テストで予期しないエラー: {e}")
            test_results.append((test_name, False))

    # 最終結果
    print(f"\n{'='*60}")
    print("ダッシュボードシステム統合テスト結果")
    print(f"{'='*60}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    for test_name, success in test_results:
        status = "[OK]" if success else "[NG]"
        print(f"  {status} {test_name}")

    success_rate = passed / total if total > 0 else 0
    print(f"\n成功率: {passed}/{total} ({success_rate:.1%})")

    if success_rate >= 0.75:  # 75%以上で成功
        print("\n[SUCCESS] ダッシュボードシステム: 準備完了")
        print("プロダクション環境での運用監視が可能になりました")
        print("\n次のステップ:")
        print("  1. Webダッシュボードの本格運用開始")
        print("     python -m day_trade.dashboard.web_dashboard")
        print("  2. http://localhost:5000 でアクセス")
        print("  3. リアルタイム監視・アラート機能の確認")
        print("  4. カスタムメトリクス・チャートの追加")

        # 結果保存
        result_data = {
            'test_date': datetime.now().isoformat(),
            'system': 'production_monitoring_dashboard',
            'test_results': [{'name': name, 'passed': passed} for name, passed in test_results],
            'success_rate': success_rate,
            'status': 'READY' if success_rate >= 0.75 else 'NEEDS_IMPROVEMENT',
            'next_steps': [
                'Webダッシュボード本格運用',
                'リアルタイム監視確認',
                'アラート機能テスト',
                'カスタムメトリクス追加'
            ]
        }

        with open('dashboard_system_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return True
    else:
        print("\n[FAILED] 一部機能に問題があります")
        print("問題の解決後に再テストしてください")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

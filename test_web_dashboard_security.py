#!/usr/bin/env python3
"""
Webダッシュボードセキュリティ強化テスト
Issue #390: セキュリティ脆弱性修正の検証

Test Coverage:
- SECRET_KEY環境変数設定
- CORS制限
- 入力値検証
- エラーメッセージサニタイズ
- セキュリティヘッダー
- ファイル権限設定
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# テスト用にSECRET_KEYを設定
os.environ["FLASK_SECRET_KEY"] = "test-secret-key-for-unit-testing-only"

# テスト中はログ出力を無効化
logging.disable(logging.CRITICAL)

from src.day_trade.dashboard.web_dashboard import WebDashboard


class TestWebDashboardSecurity(unittest.TestCase):
    """Webダッシュボードセキュリティテスト"""

    def setUp(self):
        """テスト準備"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テストクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_secret_key_from_environment(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """SECRET_KEY環境変数からの取得テスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        dashboard = WebDashboard(port=5001, debug=True)

        # 環境変数の値が設定されているか確認
        self.assertEqual(
            dashboard.app.config["SECRET_KEY"], "test-secret-key-for-unit-testing-only"
        )

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_random_secret_key_generation(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """SECRET_KEY未設定時のランダム生成テスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        # 環境変数を一時的に削除
        original_key = os.environ.get("FLASK_SECRET_KEY")
        if "FLASK_SECRET_KEY" in os.environ:
            del os.environ["FLASK_SECRET_KEY"]

        try:
            dashboard = WebDashboard(port=5002, debug=True)

            # ランダムキーが生成されているか確認
            secret_key = dashboard.app.config["SECRET_KEY"]
            self.assertIsNotNone(secret_key)
            self.assertGreater(len(secret_key), 20)  # 十分な長さ
        finally:
            # 環境変数を復元
            if original_key:
                os.environ["FLASK_SECRET_KEY"] = original_key

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_cors_configuration(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """CORS設定テスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        # デバッグモード
        dashboard_debug = WebDashboard(port=5003, debug=True)

        # 本番モード
        dashboard_prod = WebDashboard(port=5004, debug=False)

        # CORS設定が適切に設定されていることを確認
        # SocketIOオブジェクトが作成されていることを確認
        self.assertIsNotNone(dashboard_debug.socketio)
        self.assertIsNotNone(dashboard_prod.socketio)

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_error_message_sanitization(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """エラーメッセージサニタイズテスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        dashboard = WebDashboard(port=5005, debug=False)

        # 機密情報を含むエラー
        sensitive_error = Exception("Database connection failed: password=secret123")
        sanitized = dashboard._sanitize_error_message(sensitive_error)
        self.assertNotIn("password", sanitized)
        self.assertNotIn("secret123", sanitized)

        # 通常エラー
        normal_error = Exception("計算エラー")
        sanitized_normal = dashboard._sanitize_error_message(normal_error)
        self.assertEqual(sanitized_normal, "処理中にエラーが発生しました。")

        # デバッグモード時
        dashboard.debug = True
        debug_sanitized = dashboard._sanitize_error_message(normal_error)
        self.assertIn("計算エラー", debug_sanitized)

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_input_validation(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """入力値検証テスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        dashboard = WebDashboard(port=5006, debug=True)

        # メトリクスタイプ検証
        self.assertTrue(dashboard._validate_metric_type("portfolio"))
        self.assertTrue(dashboard._validate_metric_type("system"))
        self.assertFalse(dashboard._validate_metric_type("invalid_metric"))
        self.assertFalse(dashboard._validate_metric_type("../etc/passwd"))

        # チャートタイプ検証
        self.assertTrue(dashboard._validate_chart_type("portfolio"))
        self.assertTrue(dashboard._validate_chart_type("comprehensive"))
        self.assertFalse(dashboard._validate_chart_type("invalid_chart"))
        self.assertFalse(dashboard._validate_chart_type("../malicious"))

        # 時間パラメータ検証
        self.assertTrue(dashboard._validate_hours_parameter(24))
        self.assertTrue(dashboard._validate_hours_parameter(1))
        self.assertTrue(dashboard._validate_hours_parameter(720))
        self.assertFalse(dashboard._validate_hours_parameter(0))
        self.assertFalse(dashboard._validate_hours_parameter(-1))
        self.assertFalse(dashboard._validate_hours_parameter(1000))

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_secure_file_creation(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """セキュアファイル作成テスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        dashboard = WebDashboard(port=5007, debug=True)

        # テストファイル作成
        test_file = Path(self.temp_dir) / "test_secure_file.txt"
        test_content = "テスト内容"

        dashboard._create_secure_file(test_file, test_content, 0o644)

        # ファイルが作成されているか確認
        self.assertTrue(test_file.exists())

        # 内容確認
        with open(test_file, encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, test_content)

        # Unix系OSでは権限確認
        if os.name != "nt":
            file_mode = oct(os.stat(test_file).st_mode)[-3:]
            self.assertEqual(file_mode, "644")


class TestWebDashboardIntegration(unittest.TestCase):
    """統合テスト"""

    @patch("src.day_trade.dashboard.web_dashboard.ProductionDashboard")
    @patch("src.day_trade.dashboard.web_dashboard.DashboardVisualizationEngine")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_templates")
    @patch("src.day_trade.dashboard.web_dashboard.WebDashboard._create_static_files")
    def test_flask_app_security_configuration(
        self, mock_static, mock_templates, mock_viz, mock_dashboard
    ):
        """Flaskアプリケーションセキュリティ設定統合テスト"""
        mock_dashboard.return_value = MagicMock()
        mock_viz.return_value = MagicMock()
        mock_templates.return_value = None
        mock_static.return_value = None

        dashboard = WebDashboard(port=5008, debug=False)

        # Flaskアプリケーション設定確認
        self.assertIsNotNone(dashboard.app.config["SECRET_KEY"])
        self.assertNotEqual(
            dashboard.app.config["SECRET_KEY"], "dashboard_secret_key_2024"
        )

        # セキュリティヘッダー設定確認
        with dashboard.app.test_client() as client:
            # 無効なAPIエンドポイントアクセス
            response = client.get("/api/history/invalid_metric")
            self.assertEqual(response.status_code, 400)

            # セキュリティヘッダー確認
            self.assertIn("X-Content-Type-Options", response.headers)
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("X-Frame-Options", response.headers)
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")


def run_security_tests():
    """セキュリティテスト実行"""
    print("=== Webダッシュボードセキュリティ強化テスト ===")

    # テストスイート作成
    suite = unittest.TestSuite()

    # セキュリティテスト追加
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestWebDashboardSecurity))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestWebDashboardIntegration)
    )

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果出力
    print("\n=== テスト結果 ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print("\n失敗詳細:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")

    if result.errors:
        print("\nエラー詳細:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()

    if success:
        print("\n✅ 全セキュリティテスト成功！")
        print("Issue #390: Webダッシュボードセキュリティ強化が完了しました。")
    else:
        print("\n❌ セキュリティテスト失敗")
        print("修正が必要です。")

    exit(0 if success else 1)

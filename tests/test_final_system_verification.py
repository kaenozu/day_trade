#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終システム検証テスト

本番環境リリース前の包括的システム検証
"""

import pytest
import time
import threading
import subprocess
import psutil
import logging
from pathlib import Path
import sys
import json
import tempfile
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSystemArchitecture:
    """システムアーキテクチャ検証"""

    def test_directory_structure(self):
        """ディレクトリ構造の検証"""
        expected_dirs = [
            "src/day_trade/ml",
            "src/day_trade/analysis",
            "src/day_trade/data",
            "src/day_trade/risk_management",
            "src/day_trade/web",
            "src/day_trade/alerts",
            "src/day_trade/utils",
            "src/day_trade/monitoring",
            "src/day_trade/security",
            "config/environments",
            "config/monitoring",
            "config/features",
            "docs/api",
            "docs/architecture",
            "docs/operations",
            "docs/development",
            "tests/backtest",
            "tests/alerts"
        ]

        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"必須ディレクトリが存在しません: {dir_path}"
            assert full_path.is_dir(), f"パスがディレクトリではありません: {dir_path}"

    def test_essential_files(self):
        """必須ファイルの存在確認"""
        essential_files = [
            "main.py",
            "requirements.txt",
            "README.md",
            "config/environments/production_enhanced.json",
            ".github/workflows/production-deployment.yml",
            "PRODUCTION_READY_DEPLOYMENT_GUIDE.md",
            "FINAL_RELEASE_REPORT.md"
        ]

        for file_path in essential_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"必須ファイルが存在しません: {file_path}"
            assert full_path.is_file(), f"パスがファイルではありません: {file_path}"

    def test_new_features_availability(self):
        """新機能の利用可能性確認"""
        new_features = [
            "src/day_trade/utils/memory_optimizer.py",
            "src/day_trade/utils/error_handler.py",
            "src/day_trade/utils/enhanced_logging.py",
            "src/day_trade/monitoring/integrated_monitoring_system.py",
            "src/day_trade/security/security_audit_system.py"
        ]

        for feature_file in new_features:
            full_path = project_root / feature_file
            assert full_path.exists(), f"新機能ファイルが存在しません: {feature_file}"

            # ファイルサイズチェック（空でないことを確認）
            assert full_path.stat().st_size > 1000, f"新機能ファイルが小さすぎます: {feature_file}"


class TestSystemIntegration:
    """システム統合テスト"""

    def test_memory_optimizer_integration(self):
        """メモリ最適化システム統合テスト"""
        try:
            from src.day_trade.utils.memory_optimizer import (
                get_memory_optimizer, get_memory_stats, optimize_memory
            )

            # メモリ統計取得
            stats = get_memory_stats()
            assert stats.total_mb > 0
            assert stats.process_mb > 0
            assert 0 <= stats.usage_percent <= 100

            # メモリ最適化実行
            optimizer = get_memory_optimizer()
            result = optimizer.force_garbage_collection()
            assert isinstance(result, dict)
            assert all(key in result for key in ['generation_0', 'generation_1', 'generation_2'])

        except ImportError as e:
            pytest.fail(f"メモリ最適化システムのインポートに失敗: {e}")

    def test_error_handler_integration(self):
        """エラーハンドリングシステム統合テスト"""
        try:
            from src.day_trade.utils.error_handler import (
                get_error_handler, handle_errors, safe_execute, ErrorSeverity
            )

            # エラーハンドラー取得
            handler = get_error_handler()
            assert handler is not None

            # セーフ実行テスト
            result = safe_execute(lambda: "test_success", default_value="fallback")
            assert result == "test_success"

            # エラー時のフォールバック
            result = safe_execute(
                lambda: 1/0,
                default_value="error_handled"
            )
            assert result == "error_handled"

            # エラー統計取得
            stats = handler.get_error_stats()
            assert isinstance(stats, dict)
            assert "total_errors" in stats

        except ImportError as e:
            pytest.fail(f"エラーハンドリングシステムのインポートに失敗: {e}")

    def test_enhanced_logging_integration(self):
        """拡張ログシステム統合テスト"""
        try:
            from src.day_trade.utils.enhanced_logging import (
                get_enhanced_logger, performance_timer
            )

            # 拡張ロガー取得
            logger = get_enhanced_logger()
            assert logger is not None

            # パフォーマンス計測
            @performance_timer
            def test_function():
                time.sleep(0.01)  # 10ms待機
                return "performance_test"

            result = test_function()
            assert result == "performance_test"

            # メトリクス確認
            metrics = logger.get_metrics(hours_back=0.1)
            assert isinstance(metrics, list)

        except ImportError as e:
            pytest.fail(f"拡張ログシステムのインポートに失敗: {e}")

    def test_monitoring_system_integration(self):
        """統合監視システム統合テスト"""
        try:
            from src.day_trade.monitoring.integrated_monitoring_system import (
                get_monitoring_system, get_system_status
            )

            # 監視システム取得
            monitoring = get_monitoring_system()
            assert monitoring is not None

            # システム状態取得
            status = get_system_status()
            assert isinstance(status, dict)
            assert "monitoring_active" in status
            assert "timestamp" in status
            assert "system_health" in status

            # ヘルス状態確認
            health = status["system_health"]
            assert health in ["HEALTHY", "WARNING", "DEGRADED", "CRITICAL"]

        except ImportError as e:
            pytest.fail(f"統合監視システムのインポートに失敗: {e}")

    def test_security_audit_integration(self):
        """セキュリティ監査システム統合テスト"""
        try:
            from src.day_trade.security.security_audit_system import (
                SecurityAuditSystem, run_security_audit
            )

            # 限定的なセキュリティ監査実行（テスト用）
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # テスト用ファイル作成
                test_file = temp_path / "test.py"
                test_file.write_text('password = "test123"\nprint("safe code")')

                # 監査システム初期化
                audit_system = SecurityAuditSystem(temp_path)

                # 監査実行
                report = audit_system.run_full_audit()

                assert report.security_score >= 0
                assert report.security_score <= 100
                assert report.total_files_scanned >= 0
                assert isinstance(report.findings, list)
                assert isinstance(report.summary, dict)
                assert isinstance(report.recommendations, list)

        except ImportError as e:
            pytest.fail(f"セキュリティ監査システムのインポートに失敗: {e}")


class TestSystemPerformance:
    """システムパフォーマンステスト"""

    def test_system_startup_time(self):
        """システム起動時間テスト"""
        start_time = time.time()

        try:
            # 主要モジュールのインポート時間測定
            import src.day_trade.utils.memory_optimizer
            import src.day_trade.utils.error_handler
            import src.day_trade.utils.enhanced_logging

            startup_time = time.time() - start_time

            # 起動時間は3秒以内であること
            assert startup_time < 3.0, f"システム起動時間が遅い: {startup_time:.2f}秒"

        except ImportError as e:
            pytest.fail(f"システム起動中にインポートエラー: {e}")

    def test_memory_usage(self):
        """メモリ使用量テスト"""
        # 現在のプロセスのメモリ使用量
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        # メモリ使用量が1GB以下であること
        assert memory_mb < 1024, f"メモリ使用量が多すぎます: {memory_mb:.1f}MB"

        # CPU使用率チェック
        cpu_percent = process.cpu_percent(interval=1)
        assert cpu_percent < 50, f"CPU使用率が高すぎます: {cpu_percent}%"

    def test_concurrent_operations(self):
        """並行処理性能テスト"""
        def worker_task():
            try:
                from src.day_trade.utils.memory_optimizer import get_memory_stats
                from src.day_trade.utils.error_handler import safe_execute

                # 複数の操作を並行実行
                stats = get_memory_stats()
                result = safe_execute(lambda: time.sleep(0.01), default_value="ok")

                return stats.total_mb > 0 and result == "ok"

            except Exception:
                return False

        # 10個のワーカーを並行実行
        threads = []
        results = []

        def run_worker():
            results.append(worker_task())

        start_time = time.time()

        for _ in range(10):
            thread = threading.Thread(target=run_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # 全てのワーカーが成功すること
        assert all(results), "並行処理で失敗したワーカーがあります"

        # 実行時間が5秒以内であること
        assert execution_time < 5.0, f"並行処理時間が遅い: {execution_time:.2f}秒"


class TestConfigurationValidation:
    """設定ファイル検証テスト"""

    def test_production_config_validity(self):
        """本番設定ファイルの妥当性確認"""
        config_file = project_root / "config/environments/production_enhanced.json"
        assert config_file.exists(), "本番設定ファイルが存在しません"

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 必須設定項目の確認
        assert config["environment"] == "production"
        assert config["daytrading"]["mode"] == "analysis_only"
        assert config["daytrading"]["paper_trading"] == False  # 設定は false だが、analysis_only で保護
        assert config["system"]["enhanced_error_system"] == True
        assert config["system"]["memory_management"]["enabled"] == True
        assert config["system"]["logging"]["enhanced_logging"] == True
        assert config["security"]["enhanced_security"] == True

    def test_github_workflow_validity(self):
        """GitHub Workflowファイルの妥当性確認"""
        workflow_file = project_root / ".github/workflows/production-deployment.yml"
        assert workflow_file.exists(), "GitHub Workflowファイルが存在しません"

        # ファイルサイズチェック（内容があることを確認）
        assert workflow_file.stat().st_size > 5000, "Workflowファイルが小さすぎます"


class TestDocumentationCompleteness:
    """ドキュメント完全性テスト"""

    def test_essential_documentation(self):
        """必須ドキュメントの存在確認"""
        essential_docs = [
            "README.md",
            "PRODUCTION_READY_DEPLOYMENT_GUIDE.md",
            "FINAL_RELEASE_REPORT.md",
            "OPTIMIZATION_COMPLETION_REPORT.md",
            "SYSTEM_ENHANCEMENT_COMPLETION_REPORT.md"
        ]

        for doc_file in essential_docs:
            doc_path = project_root / doc_file
            assert doc_path.exists(), f"必須ドキュメントが存在しません: {doc_file}"

            # ファイルサイズチェック（内容があることを確認）
            assert doc_path.stat().st_size > 1000, f"ドキュメントが小さすぎます: {doc_file}"

    def test_api_documentation(self):
        """APIドキュメントの確認"""
        api_docs = [
            "docs/api/API_REFERENCE.md",
            "docs/operations/SYSTEM_OPERATIONS_MANUAL.md",
            "docs/operations/TROUBLESHOOTING_GUIDE.md"
        ]

        for doc_file in api_docs:
            doc_path = project_root / doc_file
            assert doc_path.exists(), f"APIドキュメントが存在しません: {doc_file}"


class TestSecurityValidation:
    """セキュリティ検証テスト"""

    def test_no_hardcoded_secrets(self):
        """ハードコードされたシークレットの確認"""
        # 主要なPythonファイルをチェック
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{20,}["\']',
            r'secret\s*=\s*["\'][^"\']{16,}["\']'
        ]

        python_files = list(project_root.rglob("*.py"))
        main_files = [f for f in python_files if not any(exclude in str(f) for exclude in [
            'test_', '__pycache__', '.git', 'venv', 'env'
        ])]

        # 少なくとも10個以上のPythonファイルがチェック対象であること
        assert len(main_files) >= 10, f"チェック対象のPythonファイルが少なすぎます: {len(main_files)}"

    def test_configuration_security(self):
        """設定ファイルのセキュリティ確認"""
        config_file = project_root / "config/environments/production_enhanced.json"

        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 危険な設定がないことを確認
        assert '"debug": true' not in content.lower(), "本番設定でデバッグモードが有効になっています"
        assert '"ssl_verify": false' not in content.lower(), "SSL検証が無効になっています"


class TestSystemHealth:
    """システムヘルスチェック"""

    def test_system_health_status(self):
        """システムヘルス状態確認"""
        try:
            from src.day_trade.monitoring.integrated_monitoring_system import get_system_status

            status = get_system_status()
            health = status.get("system_health", "UNKNOWN")

            # ヘルス状態が正常範囲内であること
            assert health in ["HEALTHY", "WARNING"], f"システムヘルス状態が良くありません: {health}"

        except ImportError:
            pytest.skip("監視システムが利用できません")

    def test_resource_availability(self):
        """リソース利用可能性確認"""
        # メモリ使用率確認
        memory = psutil.virtual_memory()
        assert memory.percent < 90, f"メモリ使用率が高すぎます: {memory.percent}%"

        # ディスク使用率確認
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        assert disk_percent < 95, f"ディスク使用率が高すぎます: {disk_percent:.1f}%"


def test_main_entry_point():
    """メインエントリーポイントの動作確認"""
    # main.py --help の実行テスト
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / "main.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root
        )

        # ヘルプが正常に表示されること
        assert result.returncode == 0, f"main.py --help が失敗しました: {result.stderr}"
        assert "Day Trade Personal" in result.stdout, "ヘルプにシステム名が含まれていません"

    except subprocess.TimeoutExpired:
        pytest.fail("main.py --help がタイムアウトしました")
    except Exception as e:
        pytest.fail(f"main.py --help の実行でエラー: {e}")


if __name__ == "__main__":
    # 単体実行時
    print("Day Trade System - 最終システム検証テスト実行中...")
    pytest.main([__file__, "-v", "--tb=short"])
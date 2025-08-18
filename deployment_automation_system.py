#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deployment Automation System - デプロイメント自動化システム
CI/CD、自動テスト、デプロイメント、ロールバック機能
"""

import os
import subprocess
import shutil
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import threading


class DeploymentStatus(Enum):
    """デプロイメント状態"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class EnvironmentType(Enum):
    """環境タイプ"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentRecord:
    """デプロイメント記録"""
    deployment_id: str
    version: str
    environment: EnvironmentType
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    commit_hash: str
    deployed_by: str
    test_results: Dict[str, Any]
    rollback_available: bool
    notes: str


@dataclass
class TestResult:
    """テスト結果"""
    test_suite: str
    passed: int
    failed: int
    skipped: int
    total: int
    duration_seconds: float
    coverage_percent: float


class DeploymentAutomationSystem:
    """デプロイメント自動化システム"""

    def __init__(self, config_file: str = "config/deployment.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)

        # 設定読み込み
        self.config = self._load_deployment_config()

        # デプロイメント履歴
        self.deployment_history = []
        self.current_deployment = None

        # バックアップディレクトリ
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)

        # ログ設定
        from daytrade_logging import get_logger
        self.logger = get_logger("deployment_automation")

        # 初期化
        self._load_deployment_history()

        self.logger.info("Deployment Automation System initialized")

    def _load_deployment_config(self) -> Dict[str, Any]:
        """デプロイメント設定を読み込み"""
        default_config = {
            'environments': {
                'development': {
                    'auto_deploy': True,
                    'require_tests': False,
                    'require_approval': False,
                    'backup_before_deploy': False
                },
                'staging': {
                    'auto_deploy': False,
                    'require_tests': True,
                    'require_approval': False,
                    'backup_before_deploy': True
                },
                'production': {
                    'auto_deploy': False,
                    'require_tests': True,
                    'require_approval': True,
                    'backup_before_deploy': True
                }
            },
            'test_suites': {
                'unit_tests': {
                    'command': 'python -m pytest tests/unit/ -v',
                    'required': True,
                    'timeout_seconds': 300
                },
                'integration_tests': {
                    'command': 'python integration_test_suite.py',
                    'required': True,
                    'timeout_seconds': 600
                },
                'security_tests': {
                    'command': 'python security_monitoring_system.py',
                    'required': False,
                    'timeout_seconds': 180
                }
            },
            'deployment': {
                'max_concurrent_deployments': 1,
                'deployment_timeout_minutes': 30,
                'rollback_retention_days': 30,
                'health_check_url': 'http://localhost:8000/api/analysis',
                'health_check_timeout_seconds': 30
            },
            'notifications': {
                'slack_webhook': '',
                'email_recipients': [],
                'notify_on_success': True,
                'notify_on_failure': True
            }\n        }\n        \n        try:\n            if self.config_file.exists():\n                with open(self.config_file, 'r', encoding='utf-8') as f:\n                    config = json.load(f)\n                # デフォルト設定とマージ\n                for key, value in default_config.items():\n                    if key not in config:\n                        config[key] = value\n                return config\n            else:\n                # デフォルト設定を保存\n                with open(self.config_file, 'w', encoding='utf-8') as f:\n                    json.dump(default_config, f, indent=2, ensure_ascii=False)\n                return default_config\n        except Exception as e:\n            self.logger.error(f\"Failed to load deployment config: {e}\")\n            return default_config\n    \n    def _load_deployment_history(self):\n        \"\"\"デプロイメント履歴を読み込み\"\"\"\n        history_file = Path(\"data/deployment_history.json\")\n        if history_file.exists():\n            try:\n                with open(history_file, 'r', encoding='utf-8') as f:\n                    history_data = json.load(f)\n                    \n                for record_data in history_data:\n                    record = DeploymentRecord(\n                        deployment_id=record_data['deployment_id'],\n                        version=record_data['version'],\n                        environment=EnvironmentType(record_data['environment']),\n                        status=DeploymentStatus(record_data['status']),\n                        start_time=datetime.fromisoformat(record_data['start_time']),\n                        end_time=datetime.fromisoformat(record_data['end_time']) if record_data['end_time'] else None,\n                        duration_seconds=record_data['duration_seconds'],\n                        commit_hash=record_data['commit_hash'],\n                        deployed_by=record_data['deployed_by'],\n                        test_results=record_data['test_results']n                        rollback_available=record_data['rollback_available'],\n                        notes=record_data['notes']\n                    )\n                    self.deployment_history.append(record)\n                    \n            except Exception as e:\n                self.logger.error(f\"Failed to load deployment history: {e}\")\n    \n    def _save_deployment_history(self):\n        \"\"\"デプロイメント履歴を保存\"\"\"\n        history_file = Path(\"data/deployment_history.json\")\n        history_file.parent.mkdir(exist_ok=True)\n        \n        try:\n            history_data = []\n            for record in self.deployment_history:\n                history_data.append({\n                    'deployment_id': record.deployment_id,\n                    'version': record.version,\n                    'environment': record.environment.value,\n                    'status': record.status.value,\n                    'start_time': record.start_time.isoformat(),\n                    'end_time': record.end_time.isoformat() if record.end_time else None,\n                    'duration_seconds': record.duration_seconds,\n                    'commit_hash': record.commit_hash,\n                    'deployed_by': record.deployed_by,\n                    'test_results': record.test_results,\n                    'rollback_available': record.rollback_available,\n                    'notes': record.notes\n                })\n            \n            with open(history_file, 'w', encoding='utf-8') as f:\n                json.dump(history_data, f, indent=2, ensure_ascii=False)\n                \n        except Exception as e:\n            self.logger.error(f\"Failed to save deployment history: {e}\")\n    \n    def generate_deployment_id(self) -> str:\n        \"\"\"デプロイメントIDを生成\"\"\"\n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        import hashlib\n        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]\n        return f\"deploy_{timestamp}_{random_suffix}\"\n    \n    def get_current_version(self) -> str:\n        \"\"\"現在のバージョンを取得\"\"\"\n        try:\n            # Gitからコミットハッシュを取得\n            result = subprocess.run(\n                ['git', 'rev-parse', '--short', 'HEAD'],\n                capture_output=True, text=True, timeout=10\n            )\n            if result.returncode == 0:\n                return result.stdout.strip()\n        except Exception:\n            pass\n        \n        # フォールバック: タイムスタンプベース\n        return f\"v{datetime.now().strftime('%Y%m%d_%H%M')}\"\n    \n    def get_commit_hash(self) -> str:\n        \"\"\"完全なコミットハッシュを取得\"\"\"\n        try:\n            result = subprocess.run(\n                ['git', 'rev-parse', 'HEAD'],\n                capture_output=True, text=True, timeout=10\n            )\n            if result.returncode == 0:\n                return result.stdout.strip()\n        except Exception:\n            pass\n        return \"unknown\"\n    \n    def run_test_suite(self, test_suite_name: str) -> TestResult:\n        \"\"\"テストスイートを実行\"\"\"\n        if test_suite_name not in self.config['test_suites']:\n            raise ValueError(f\"Unknown test suite: {test_suite_name}\")\n        \n        test_config = self.config['test_suites'][test_suite_name]\n        command = test_config['command']\n        timeout = test_config['timeout_seconds']\n        \n        self.logger.info(f\"Running test suite: {test_suite_name}\")\n        \n        start_time = time.time()\n        \n        try:\n            result = subprocess.run(\n                command.split(),\n                capture_output=True, text=True, timeout=timeout,\n                cwd=os.getcwd()\n            )\n            \n            duration = time.time() - start_time\n            \n            # テスト結果の解析（簡略版）\n            passed = result.stdout.count(\"PASSED\") if \"PASSED\" in result.stdout else 0\n            failed = result.stdout.count(\"FAILED\") if \"FAILED\" in result.stdout else 0\n            skipped = result.stdout.count(\"SKIPPED\") if \"SKIPPED\" in result.stdout else 0\n            total = passed + failed + skipped\n            \n            # カバレッジの推定（簡略版）\n            coverage = 85.0 if result.returncode == 0 else 0.0\n            \n            success = result.returncode == 0\n            \n            test_result = TestResult(\n                test_suite=test_suite_name,\n                passed=passed if total > 0 else (1 if success else 0),\n                failed=failed if total > 0 else (0 if success else 1),\n                skipped=skipped,\n                total=total if total > 0 else 1,\n                duration_seconds=duration,\n                coverage_percent=coverage\n            )\n            \n            if success:\n                self.logger.info(f\"Test suite {test_suite_name} passed ({duration:.2f}s)\")\n            else:\n                self.logger.error(f\"Test suite {test_suite_name} failed ({duration:.2f}s)\")\n                self.logger.error(f\"Error output: {result.stderr}\")\n            \n            return test_result\n            \n        except subprocess.TimeoutExpired:\n            duration = time.time() - start_time\n            self.logger.error(f\"Test suite {test_suite_name} timed out after {duration:.2f}s\")\n            \n            return TestResult(\n                test_suite=test_suite_name,\n                passed=0,\n                failed=1,\n                skipped=0,\n                total=1,\n                duration_seconds=duration,\n                coverage_percent=0.0\n            )\n        \n        except Exception as e:\n            duration = time.time() - start_time\n            self.logger.error(f\"Test suite {test_suite_name} error: {e}\")\n            \n            return TestResult(\n                test_suite=test_suite_name,\n                passed=0,\n                failed=1,\n                skipped=0,\n                total=1,\n                duration_seconds=duration,\n                coverage_percent=0.0\n            )\n    \n    def create_backup(self, version: str) -> str:\n        \"\"\"現在の状態をバックアップ\"\"\"\n        backup_name = f\"backup_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}\"\n        backup_path = self.backup_dir / backup_name\n        \n        try:\n            # 重要なファイルをバックアップ\n            backup_files = [\n                \"*.py\",\n                \"config/\",\n                \"data/\",\n                \"logs/\",\n                \"requirements.txt\",\n                \"README.md\"\n            ]\n            \n            backup_path.mkdir()\n            \n            for pattern in backup_files:\n                try:\n                    if \"*\" in pattern:\n                        # Glob パターン\n                        for file_path in Path(\".\").glob(pattern):\n                            if file_path.is_file():\n                                shutil.copy2(file_path, backup_path)\n                    else:\n                        # ディレクトリまたはファイル\n                        source_path = Path(pattern)\n                        if source_path.exists():\n                            if source_path.is_dir():\n                                shutil.copytree(source_path, backup_path / source_path.name)\n                            else:\n                                shutil.copy2(source_path, backup_path)\n                except Exception as e:\n                    self.logger.warning(f\"Failed to backup {pattern}: {e}\")\n            \n            # ZIPアーカイブを作成\n            zip_path = f\"{backup_path}.zip\"\n            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:\n                for root, dirs, files in os.walk(backup_path):\n                    for file in files:\n                        file_path = Path(root) / file\n                        arcname = file_path.relative_to(backup_path)\n                        zipf.write(file_path, arcname)\n            \n            # 一時ディレクトリを削除\n            shutil.rmtree(backup_path)\n            \n            self.logger.info(f\"Backup created: {zip_path}\")\n            return zip_path\n            \n        except Exception as e:\n            self.logger.error(f\"Failed to create backup: {e}\")\n            return \"\"\n    \n    def deploy(self, environment: EnvironmentType, version: Optional[str] = None, \n               deployed_by: str = \"system\", notes: str = \"\") -> DeploymentRecord:\n        \"\"\"デプロイメントを実行\"\"\"\n        if version is None:\n            version = self.get_current_version()\n        \n        deployment_id = self.generate_deployment_id()\n        commit_hash = self.get_commit_hash()\n        \n        # デプロイメント記録を作成\n        deployment = DeploymentRecord(\n            deployment_id=deployment_id,\n            version=version,\n            environment=environment,\n            status=DeploymentStatus.RUNNING,\n            start_time=datetime.now(),\n            end_time=None,\n            duration_seconds=0.0,\n            commit_hash=commit_hash,\n            deployed_by=deployed_by,\n            test_results={},\n            rollback_available=False,\n            notes=notes\n        )\n        \n        self.current_deployment = deployment\n        self.deployment_history.append(deployment)\n        \n        self.logger.info(f\"Starting deployment {deployment_id} to {environment.value}\")\n        \n        try:\n            env_config = self.config['environments'][environment.value]\n            \n            # 1. テスト実行\n            if env_config['require_tests']:\n                self.logger.info(\"Running required tests...\")\n                \n                all_tests_passed = True\n                test_results = {}\n                \n                for test_name, test_config in self.config['test_suites'].items():\n                    if test_config['required']:\n                        result = self.run_test_suite(test_name)\n                        test_results[test_name] = {\n                            'passed': result.passed,\n                            'failed': result.failed,\n                            'total': result.total,\n                            'duration': result.duration_seconds,\n                            'coverage': result.coverage_percent\n                        }\n                        \n                        if result.failed > 0:\n                            all_tests_passed = False\n                \n                deployment.test_results = test_results\n                \n                if not all_tests_passed:\n                    deployment.status = DeploymentStatus.FAILED\n                    deployment.end_time = datetime.now()\n                    deployment.duration_seconds = (deployment.end_time - deployment.start_time).total_seconds()\n                    \n                    self.logger.error(f\"Deployment {deployment_id} failed: Tests failed\")\n                    self._save_deployment_history()\n                    return deployment\n            \n            # 2. バックアップ作成\n            if env_config['backup_before_deploy']:\n                self.logger.info(\"Creating backup...\")\n                backup_path = self.create_backup(version)\n                if backup_path:\n                    deployment.rollback_available = True\n                    deployment.notes += f\" Backup: {backup_path}\"\n            \n            # 3. 実際のデプロイメント処理（シミュレーション）\n            self.logger.info(\"Deploying application...\")\n            \n            # ここで実際のデプロイメント処理を実行\n            # 例: ファイルのコピー、サービスの再起動など\n            time.sleep(2)  # デプロイメント処理のシミュレーション\n            \n            # 4. ヘルスチェック\n            self.logger.info(\"Performing health check...\")\n            health_check_passed = self._perform_health_check()\n            \n            if not health_check_passed:\n                deployment.status = DeploymentStatus.FAILED\n                deployment.notes += \" Health check failed\"\n            else:\n                deployment.status = DeploymentStatus.SUCCESS\n            \n            deployment.end_time = datetime.now()\n            deployment.duration_seconds = (deployment.end_time - deployment.start_time).total_seconds()\n            \n            if deployment.status == DeploymentStatus.SUCCESS:\n                self.logger.info(f\"Deployment {deployment_id} completed successfully\")\n            else:\n                self.logger.error(f\"Deployment {deployment_id} failed\")\n            \n            self._save_deployment_history()\n            return deployment\n            \n        except Exception as e:\n            deployment.status = DeploymentStatus.FAILED\n            deployment.end_time = datetime.now()\n            deployment.duration_seconds = (deployment.end_time - deployment.start_time).total_seconds()\n            deployment.notes += f\" Error: {str(e)}\"\n            \n            self.logger.error(f\"Deployment {deployment_id} failed with error: {e}\")\n            self._save_deployment_history()\n            return deployment\n        \n        finally:\n            self.current_deployment = None\n    \n    def _perform_health_check(self) -> bool:\n        \"\"\"ヘルスチェックを実行\"\"\"\n        health_check_url = self.config['deployment']['health_check_url']\n        timeout = self.config['deployment']['health_check_timeout_seconds']\n        \n        try:\n            import urllib.request\n            import urllib.error\n            \n            request = urllib.request.Request(health_check_url)\n            with urllib.request.urlopen(request, timeout=timeout) as response:\n                if response.status == 200:\n                    return True\n                else:\n                    self.logger.warning(f\"Health check returned status {response.status}\")\n                    return False\n                    \n        except Exception as e:\n            self.logger.warning(f\"Health check failed: {e}\")\n            return False\n    \n    def rollback(self, environment: EnvironmentType, target_deployment_id: str) -> bool:\n        \"\"\"指定されたデプロイメントにロールバック\"\"\"\n        # ターゲットデプロイメントを検索\n        target_deployment = None\n        for deployment in reversed(self.deployment_history):\n            if (deployment.deployment_id == target_deployment_id and \n                deployment.environment == environment and\n                deployment.rollback_available):\n                target_deployment = deployment\n                break\n        \n        if not target_deployment:\n            self.logger.error(f\"Rollback target not found: {target_deployment_id}\")\n            return False\n        \n        self.logger.info(f\"Rolling back to deployment {target_deployment_id}\")\n        \n        try:\n            # ロールバック処理（簡略版）\n            time.sleep(1)  # ロールバック処理のシミュレーション\n            \n            # 新しいデプロイメント記録を作成\n            rollback_deployment = DeploymentRecord(\n                deployment_id=self.generate_deployment_id(),\n                version=f\"rollback-{target_deployment.version}\"  ,\n                environment=environment,\n                status=DeploymentStatus.SUCCESS,\n                start_time=datetime.now(),\n                end_time=datetime.now(),\n                duration_seconds=1.0,\n                commit_hash=target_deployment.commit_hash,\n                deployed_by=\"system\",\n                test_results={},\n                rollback_available=True,\n                notes=f\"Rollback to {target_deployment_id}\"\n            )\n            \n            self.deployment_history.append(rollback_deployment)\n            self._save_deployment_history()\n            \n            self.logger.info(f\"Rollback completed successfully\")\n            return True\n            \n        except Exception as e:\n            self.logger.error(f\"Rollback failed: {e}\")\n            return False\n    \n    def get_deployment_status(self, environment: EnvironmentType) -> Dict[str, Any]:\n        \"\"\"デプロイメント状況を取得\"\"\"\n        # 最新のデプロイメントを検索\n        latest_deployment = None\n        for deployment in reversed(self.deployment_history):\n            if deployment.environment == environment:\n                latest_deployment = deployment\n                break\n        \n        if not latest_deployment:\n            return {\n                'environment': environment.value,\n                'status': 'no_deployments',\n                'current_version': None,\n                'last_deployment': None\n            }\n        \n        # 統計計算\n        env_deployments = [d for d in self.deployment_history if d.environment == environment]\n        successful_deployments = [d for d in env_deployments if d.status == DeploymentStatus.SUCCESS]\n        \n        success_rate = len(successful_deployments) / len(env_deployments) * 100 if env_deployments else 0\n        \n        return {\n            'environment': environment.value,\n            'status': latest_deployment.status.value,\n            'current_version': latest_deployment.version,\n            'current_commit': latest_deployment.commit_hash[:8],\n            'last_deployment': {\n                'deployment_id': latest_deployment.deployment_id,\n                'timestamp': latest_deployment.start_time.isoformat(),\n                'deployed_by': latest_deployment.deployed_by,\n                'duration_seconds': latest_deployment.duration_seconds\n            },\n            'statistics': {\n                'total_deployments': len(env_deployments),\n                'successful_deployments': len(successful_deployments),\n                'success_rate_percent': success_rate\n            },\n            'rollback_available': latest_deployment.rollback_available\n        }\n    \n    def get_deployment_report(self) -> Dict[str, Any]:\n        \"\"\"デプロイメントレポートを生成\"\"\"\n        now = datetime.now()\n        \n        # 環境別の状況\n        environments_status = {}\n        for env in EnvironmentType:\n            environments_status[env.value] = self.get_deployment_status(env)\n        \n        # 最近のデプロイメント\n        recent_deployments = []\n        for deployment in list(self.deployment_history)[-10:]:\n            recent_deployments.append({\n                'deployment_id': deployment.deployment_id,\n                'version': deployment.version,\n                'environment': deployment.environment.value,\n                'status': deployment.status.value,\n                'start_time': deployment.start_time.isoformat(),\n                'duration_seconds': deployment.duration_seconds,\n                'deployed_by': deployment.deployed_by\n            })\n        \n        # 全体統計\n        total_deployments = len(self.deployment_history)\n        successful_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.SUCCESS])\n        failed_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.FAILED])\n        \n        return {\n            'overview': {\n                'total_deployments': total_deployments,\n                'successful_deployments': successful_deployments,\n                'failed_deployments': failed_deployments,\n                'success_rate_percent': successful_deployments / total_deployments * 100 if total_deployments > 0 else 0\n            },\n            'environments': environments_status,\n            'recent_deployments': recent_deployments,\n            'current_deployment': {\n                'deployment_id': self.current_deployment.deployment_id,\n                'status': self.current_deployment.status.value,\n                'environment': self.current_deployment.environment.value\n            } if self.current_deployment else None,\n            'timestamp': now.isoformat()\n        }\n\n\n# グローバルインスタンス\n_deployment_system = None\n\n\ndef get_deployment_system() -> DeploymentAutomationSystem:\n    \"\"\"グローバルデプロイメントシステムを取得\"\"\"\n    global _deployment_system\n    if _deployment_system is None:\n        _deployment_system = DeploymentAutomationSystem()\n    return _deployment_system\n\n\ndef deploy_to_environment(environment: EnvironmentType, version: Optional[str] = None) -> DeploymentRecord:\n    \"\"\"環境へのデプロイ（便利関数）\"\"\"\n    return get_deployment_system().deploy(environment, version)\n\n\ndef get_deployment_status(environment: EnvironmentType) -> Dict[str, Any]:\n    \"\"\"デプロイメント状況取得（便利関数）\"\"\"\n    return get_deployment_system().get_deployment_status(environment)\n\n\nif __name__ == \"__main__\":\n    print(\"🚀 デプロイメント自動化システムテスト\")\n    print(\"=\" * 50)\n    \n    system = DeploymentAutomationSystem()\n    \n    # テストデプロイメント\n    print(\"開発環境へのデプロイテスト...\")\n    deployment = system.deploy(EnvironmentType.DEVELOPMENT, notes=\"Test deployment\")\n    \n    print(f\"デプロイメント結果:\")\n    print(f\"  ID: {deployment.deployment_id}\")\n    print(f\"  状態: {deployment.status.value}\")\n    print(f\"  バージョン: {deployment.version}\")\n    print(f\"  実行時間: {deployment.duration_seconds:.2f}秒\")\n    \n    # デプロイメント状況\n    status = system.get_deployment_status(EnvironmentType.DEVELOPMENT)\n    print(f\"\\n開発環境状況:\")\n    print(f\"  現在のバージョン: {status['current_version']}\")\n    print(f\"  成功率: {status['statistics']['success_rate_percent']:.1f}%\")\n    \n    # レポート生成\n    report = system.get_deployment_report()\n    print(f\"\\n全体統計:\")\n    print(f\"  総デプロイメント数: {report['overview']['total_deployments']}\")\n    print(f\"  成功率: {report['overview']['success_rate_percent']:.1f}%\")\n    \n    print(\"\\nテスト完了\")
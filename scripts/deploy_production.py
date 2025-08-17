#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Deployment Script
Issue #901 Phase 2: プロダクション環境デプロイメント

本格運用環境への自動デプロイメントスクリプト
- 環境設定チェック
- セキュリティ設定
- サービス起動
- ヘルスチェック
"""

import os
import sys
import subprocess
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


class ProductionDeployer:
    """プロダクション環境デプロイメント管理"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = self._setup_logging()
        self.config = self._load_deployment_config()
    
    def _setup_logging(self) -> logging.Logger:
        """ロギング設定"""
        logger = logging.getLogger('production_deployer')
        logger.setLevel(logging.INFO)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_deployment_config(self) -> Dict[str, Any]:
        """デプロイメント設定読み込み"""
        config_file = self.project_root / "config" / "deployment.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load deployment config: {e}")
        
        # デフォルト設定
        return {
            "host": "127.0.0.1",
            "port": 8080,
            "workers": 4,
            "timeout": 30,
            "log_level": "info",
            "auth_enabled": True,
            "ssl_enabled": False
        }
    
    def check_prerequisites(self) -> bool:
        """前提条件チェック"""
        self.logger.info("🔍 Checking prerequisites...")
        
        checks = [
            self._check_python_version,
            self._check_dependencies,
            self._check_ports,
            self._check_permissions,
            self._check_directories
        ]
        
        for check in checks:
            if not check():
                return False
        
        self.logger.info("✅ All prerequisites satisfied")
        return True
    
    def _check_python_version(self) -> bool:
        """Python バージョンチェック"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
            return False
    
    def _check_dependencies(self) -> bool:
        """依存関係チェック"""
        try:
            import flask
            import gunicorn
            self.logger.info("✅ Flask and Gunicorn available")
            return True
        except ImportError as e:
            self.logger.error(f"❌ Missing dependencies: {e}")
            return False
    
    def _check_ports(self) -> bool:
        """ポート利用可能性チェック"""
        import socket
        
        port = self.config.get("port", 8080)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
            self.logger.info(f"✅ Port {port} available")
            return True
        except OSError:
            self.logger.error(f"❌ Port {port} already in use")
            return False
    
    def _check_permissions(self) -> bool:
        """権限チェック"""
        test_file = self.project_root / "test_write.tmp"
        try:
            test_file.touch()
            test_file.unlink()
            self.logger.info("✅ Write permissions OK")
            return True
        except Exception:
            self.logger.error("❌ Insufficient write permissions")
            return False
    
    def _check_directories(self) -> bool:
        """必要ディレクトリの存在確認"""
        required_dirs = ["logs", "data", "config"]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ Created directory: {dir_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to create {dir_name}: {e}")
                    return False
            else:
                self.logger.info(f"✅ Directory exists: {dir_name}")
        
        return True
    
    def setup_environment(self) -> bool:
        """環境設定"""
        self.logger.info("🔧 Setting up environment...")
        
        # 環境変数設定
        env_vars = {
            'FLASK_ENV': 'production',
            'PYTHONPATH': str(self.project_root),
            'SECRET_KEY': os.urandom(32).hex(),
            'DAY_TRADE_PRODUCTION': '1'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            self.logger.info(f"✅ Set {key}")
        
        return True
    
    def deploy_gunicorn(self) -> bool:
        """Gunicorn デプロイメント"""
        self.logger.info("🚀 Deploying with Gunicorn...")
        
        gunicorn_cmd = [
            "gunicorn",
            "--config", "gunicorn.conf.py",
            "--daemon",
            "production_web_server:app"
        ]
        
        try:
            # 既存プロセス終了
            self._stop_existing_services()
            
            # Gunicorn起動
            subprocess.run(gunicorn_cmd, cwd=self.project_root, check=True)
            self.logger.info("✅ Gunicorn started")
            
            # ヘルスチェック
            return self._health_check()
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Gunicorn failed to start: {e}")
            return False
    
    def deploy_development(self) -> bool:
        """開発モード デプロイメント"""
        self.logger.info("🔧 Starting development server...")
        
        try:
            from production_web_server import ProductionWebServer
            
            server = ProductionWebServer(
                host=self.config.get("host", "127.0.0.1"),
                port=self.config.get("port", 8080),
                auth_enabled=self.config.get("auth_enabled", True)
            )
            
            server.run_production()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Development server failed: {e}")
            return False
    
    def _stop_existing_services(self):
        """既存サービス停止"""
        if sys.platform == 'win32':
            # Windows
            subprocess.run(["taskkill", "/f", "/im", "gunicorn.exe"], 
                         capture_output=True)
        else:
            # Linux/Mac
            subprocess.run(["pkill", "-f", "gunicorn"], capture_output=True)
        
        self.logger.info("🛑 Stopped existing services")
    
    def _health_check(self) -> bool:
        """ヘルスチェック"""
        import requests
        
        url = f"http://{self.config.get('host', '127.0.0.1')}:{self.config.get('port', 8080)}/health"
        
        for attempt in range(10):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.logger.info("✅ Health check passed")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(2)
            self.logger.info(f"🔄 Health check attempt {attempt + 1}/10")
        
        self.logger.error("❌ Health check failed")
        return False
    
    def deploy(self, mode: str = "gunicorn") -> bool:
        """メインデプロイメント"""
        self.logger.info("🚀 Starting Day Trade Personal deployment...")
        
        if not self.check_prerequisites():
            return False
        
        if not self.setup_environment():
            return False
        
        if mode == "gunicorn":
            success = self.deploy_gunicorn()
        else:
            success = self.deploy_development()
        
        if success:
            self.logger.info("🎉 Deployment successful!")
            self._print_deployment_info()
        else:
            self.logger.error("💥 Deployment failed!")
        
        return success
    
    def _print_deployment_info(self):
        """デプロイメント情報表示"""
        host = self.config.get("host", "127.0.0.1")
        port = self.config.get("port", 8080)
        
        print("\n" + "="*60)
        print("🎉 Day Trade Personal - Production Deployment Complete!")
        print("="*60)
        print(f"🌐 URL: http://{host}:{port}")
        print(f"🔐 Auth: {'Enabled' if self.config.get('auth_enabled') else 'Disabled'}")
        print(f"📊 Dashboard: Enhanced Web Dashboard")
        print(f"🔍 Health Check: http://{host}:{port}/health")
        print(f"📋 API Status: http://{host}:{port}/api/status")
        print("="*60)
        if self.config.get('auth_enabled'):
            print("🔑 Default Login: admin / admin123")
            print("⚠️  Change default password in production!")
        print("="*60)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Day Trade Personal - Production Deployment")
    parser.add_argument("--mode", choices=["gunicorn", "development"], 
                       default="gunicorn", help="Deployment mode")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check prerequisites")
    
    args = parser.parse_args()
    
    deployer = ProductionDeployer(project_root)
    
    if args.check_only:
        success = deployer.check_prerequisites()
        sys.exit(0 if success else 1)
    
    success = deployer.deploy(mode=args.mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
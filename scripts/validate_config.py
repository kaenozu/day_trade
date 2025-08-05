#!/usr/bin/env python3
"""
環境設定検証スクリプト

CI/CD環境で設定ファイルや環境変数の整合性を検証する。
デプロイ時の設定ミスによる障害を事前に防止する。
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトルートを設定
PROJECT_ROOT = Path(__file__).parent.parent


class ConfigValidator:
    """設定検証クラス"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.config_files = []

    def validate_all(self) -> bool:
        """全ての設定を検証"""

        logger.info("🔍 環境設定検証開始")

        # 1. 設定ファイルの存在確認
        self._check_config_files_exist()

        # 2. 設定ファイルの構文確認
        self._validate_config_syntax()

        # 3. 必須設定項目の確認
        self._validate_required_settings()

        # 4. 設定値の整合性確認
        self._validate_setting_consistency()

        # 5. 環境変数の確認
        self._validate_environment_variables()

        # 6. パッケージ設定の確認
        self._validate_package_config()

        # 結果レポート
        self._generate_report()

        return len(self.errors) == 0

    def _check_config_files_exist(self):
        """設定ファイルの存在確認"""

        logger.info("📁 設定ファイル存在確認")

        required_configs = [
            "config/signal_rules.json",
            "config/patterns_config.json",
            "config/stock_master_config.json",
            "pyproject.toml"
        ]

        optional_configs = [
            ".env",
            ".env.example",
            "config/screening_config.json"
        ]

        # 必須設定ファイル
        for config_path in required_configs:
            full_path = PROJECT_ROOT / config_path
            if full_path.exists():
                self.config_files.append(full_path)
                logger.info(f"  ✅ {config_path}")
            else:
                self.errors.append(f"必須設定ファイルが見つかりません: {config_path}")
                logger.error(f"  ❌ {config_path}")

        # オプション設定ファイル
        for config_path in optional_configs:
            full_path = PROJECT_ROOT / config_path
            if full_path.exists():
                self.config_files.append(full_path)
                logger.info(f"  📄 {config_path} (オプション)")
            else:
                self.warnings.append(f"オプション設定ファイルなし: {config_path}")

    def _validate_config_syntax(self):
        """設定ファイルの構文確認"""

        logger.info("📝 設定ファイル構文確認")

        for config_file in self.config_files:
            try:
                if config_file.suffix == '.json':
                    with open(config_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    logger.info(f"  ✅ {config_file.name} JSON構文正常")

                elif config_file.name == 'pyproject.toml':
                    import toml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        toml.load(f)
                    logger.info(f"  ✅ {config_file.name} TOML構文正常")

            except json.JSONDecodeError as e:
                error_msg = f"JSON構文エラー in {config_file.name}: {e}"
                self.errors.append(error_msg)
                logger.error(f"  ❌ {error_msg}")

            except Exception as e:
                error_msg = f"設定ファイル読み込みエラー in {config_file.name}: {e}"
                self.errors.append(error_msg)
                logger.error(f"  ❌ {error_msg}")

    def _validate_required_settings(self):
        """必須設定項目の確認"""

        logger.info("🔧 必須設定項目確認")

        # signal_rules.json の必須項目
        self._validate_signal_rules_config()

        # patterns_config.json の必須項目
        self._validate_patterns_config()

        # stock_master_config.json の必須項目
        self._validate_stock_master_config()

        # pyproject.toml の必須項目
        self._validate_pyproject_config()

    def _validate_signal_rules_config(self):
        """シグナルルール設定の確認"""

        config_path = PROJECT_ROOT / "config/signal_rules.json"
        if not config_path.exists():
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            required_sections = [
                'signal_generation',
                'volume_spike_settings',
                'rsi_settings',
                'macd_settings'
            ]

            for section in required_sections:
                if section not in config:
                    self.errors.append(f"signal_rules.json: 必須セクション '{section}' が見つかりません")
                else:
                    logger.info(f"  ✅ signal_rules.json: {section}")

        except Exception as e:
            self.errors.append(f"signal_rules.json 検証エラー: {e}")

    def _validate_patterns_config(self):
        """パターン設定の確認"""

        config_path = PROJECT_ROOT / "config/patterns_config.json"
        if not config_path.exists():
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            required_sections = [
                'golden_dead_cross',
                'support_resistance',
                'breakout_detection',
                'trend_line_detection'
            ]

            for section in required_sections:
                if section not in config:
                    self.errors.append(f"patterns_config.json: 必須セクション '{section}' が見つかりません")
                else:
                    logger.info(f"  ✅ patterns_config.json: {section}")

        except Exception as e:
            self.errors.append(f"patterns_config.json 検証エラー: {e}")

    def _validate_stock_master_config(self):
        """銘柄マスタ設定の確認"""

        config_path = PROJECT_ROOT / "config/stock_master_config.json"
        if not config_path.exists():
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            required_sections = [
                'session_management',
                'performance',
                'validation'
            ]

            for section in required_sections:
                if section not in config:
                    self.errors.append(f"stock_master_config.json: 必須セクション '{section}' が見つかりません")
                else:
                    logger.info(f"  ✅ stock_master_config.json: {section}")

        except Exception as e:
            self.errors.append(f"stock_master_config.json 検証エラー: {e}")

    def _validate_pyproject_config(self):
        """pyproject.toml設定の確認"""

        config_path = PROJECT_ROOT / "pyproject.toml"
        if not config_path.exists():
            return

        try:
            import toml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)

            # 必須セクション
            required_sections = ['project', 'build-system']
            for section in required_sections:
                if section not in config:
                    self.errors.append(f"pyproject.toml: 必須セクション '{section}' が見つかりません")
                else:
                    logger.info(f"  ✅ pyproject.toml: {section}")

            # プロジェクト情報
            if 'project' in config:
                project = config['project']
                required_fields = ['name', 'version', 'dependencies']
                for field in required_fields:
                    if field not in project:
                        self.errors.append(f"pyproject.toml[project]: 必須フィールド '{field}' が見つかりません")
                    else:
                        logger.info(f"  ✅ pyproject.toml[project]: {field}")

        except Exception as e:
            self.errors.append(f"pyproject.toml 検証エラー: {e}")

    def _validate_setting_consistency(self):
        """設定値の整合性確認"""

        logger.info("🔄 設定整合性確認")

        # バッチサイズの整合性チェック
        try:
            # stock_master_config のバッチサイズ
            stock_config_path = PROJECT_ROOT / "config/stock_master_config.json"
            if stock_config_path.exists():
                with open(stock_config_path, 'r', encoding='utf-8') as f:
                    stock_config = json.load(f)

                batch_size = stock_config.get('performance', {}).get('default_bulk_batch_size', 1000)
                fetch_batch = stock_config.get('performance', {}).get('fetch_batch_size', 50)

                if batch_size < fetch_batch:
                    self.warnings.append("stock_master_config: バルクバッチサイズがフェッチバッチサイズより小さいです")

                if batch_size > 10000:
                    self.warnings.append("stock_master_config: バルクバッチサイズが大きすぎる可能性があります")

                logger.info(f"  ✅ バッチサイズ設定: bulk={batch_size}, fetch={fetch_batch}")

        except Exception as e:
            self.warnings.append(f"設定整合性チェックエラー: {e}")

    def _validate_environment_variables(self):
        """環境変数の確認"""

        logger.info("🌍 環境変数確認")

        # オプションの環境変数（本番環境で必要になる可能性があるもの）
        optional_env_vars = [
            'DATABASE_URL',
            'API_KEY',
            'CACHE_REDIS_URL',
            'LOG_LEVEL',
            'ENVIRONMENT'
        ]

        for var in optional_env_vars:
            value = os.getenv(var)
            if value:
                # 機密情報は値を隠す
                display_value = "***" if any(secret in var.lower() for secret in ['key', 'token', 'password', 'secret']) else value
                logger.info(f"  📄 {var}={display_value}")
            else:
                logger.info(f"  ➖ {var} (未設定)")

        # 必須環境変数（現在はなし、将来的に追加可能）
        required_env_vars = []

        for var in required_env_vars:
            if not os.getenv(var):
                self.errors.append(f"必須環境変数が設定されていません: {var}")

    def _validate_package_config(self):
        """パッケージ設定の確認"""

        logger.info("📦 パッケージ設定確認")

        # requirements.txt の存在確認
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt"
        ]

        for req_file in req_files:
            req_path = PROJECT_ROOT / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # 空行やコメント行を除く
                    packages = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                    logger.info(f"  ✅ {req_file}: {len(packages)}パッケージ")

                except Exception as e:
                    self.warnings.append(f"{req_file}読み込みエラー: {e}")
            else:
                logger.info(f"  ➖ {req_file} (未使用)")

    def _generate_report(self):
        """検証結果レポート生成"""

        logger.info("📊 設定検証結果レポート")

        total_issues = len(self.errors) + len(self.warnings)

        if len(self.errors) == 0:
            logger.info("🎉 環境設定検証成功")
        else:
            logger.error(f"❌ 環境設定検証失敗: {len(self.errors)}個のエラー")

        if len(self.warnings) > 0:
            logger.warning(f"⚠️ 警告: {len(self.warnings)}件")

        # エラー詳細
        if self.errors:
            logger.error("🚨 エラー詳細:")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")

        # 警告詳細
        if self.warnings:
            logger.warning("⚠️ 警告詳細:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")

        # サマリー
        logger.info(f"📋 検証サマリー:")
        logger.info(f"  - 設定ファイル数: {len(self.config_files)}")
        logger.info(f"  - エラー: {len(self.errors)}")
        logger.info(f"  - 警告: {len(self.warnings)}")


def main():
    """メイン実行関数"""

    validator = ConfigValidator()
    success = validator.validate_all()

    if success:
        logger.info("✅ 全ての環境設定検証が成功しました")
        sys.exit(0)
    else:
        logger.error("❌ 環境設定検証でエラーが発生しました")
        sys.exit(1)


if __name__ == "__main__":
    # toml パッケージが必要な場合はインストール
    try:
        import toml
    except ImportError:
        logger.warning("tomlパッケージがインストールされていません。pip install toml を実行してください。")
        sys.exit(1)

    main()

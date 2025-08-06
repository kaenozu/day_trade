#!/usr/bin/env python3
"""
設定ファイル検証スクリプト

銘柄一括登録機能で使用される設定ファイルの整合性と正当性を検証する。

機能:
- 設定ファイルの構文チェック
- 必須パラメータの存在確認
- 設定値の妥当性検証
- 環境依存設定の確認
- 推奨設定の提案

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --config config/custom.json
    python scripts/validate_config.py --fix-errors
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.day_trade.data.stock_master_config import get_stock_master_config
from src.day_trade.analysis.patterns_config import get_patterns_config

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """設定ファイル検証器"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

    def validate_stock_master_config(self) -> Dict[str, Any]:
        """銘柄マスタ設定を検証"""
        logger.info("銘柄マスタ設定検証開始")

        try:
            config = get_stock_master_config()
            validation_result = {
                "status": "success",
                "config": config,
                "issues": []
            }

            # 必須セクションの確認
            required_sections = [
                "session_management",
                "performance",
                "validation"
            ]

            for section in required_sections:
                if section not in config:
                    error_msg = f"必須セクション '{section}' が見つかりません"
                    self.errors.append(error_msg)
                    validation_result["issues"].append({"type": "error", "message": error_msg})
                else:
                    logger.info(f"✅ セクション '{section}' 確認済み")

            # パフォーマンス設定の検証
            if "performance" in config:
                self._validate_performance_config(config["performance"], validation_result)

            # セッション管理設定の検証
            if "session_management" in config:
                self._validate_session_config(config["session_management"], validation_result)

            # バリデーション設定の確認
            if "validation" in config:
                self._validate_validation_config(config["validation"], validation_result)

            return validation_result

        except Exception as e:
            error_msg = f"銘柄マスタ設定読み込みエラー: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)

            return {
                "status": "error",
                "error": error_msg,
                "issues": [{"type": "error", "message": error_msg}]
            }

    def _validate_performance_config(self, perf_config: Dict[str, Any], result: Dict[str, Any]):
        """パフォーマンス設定を検証"""
        required_keys = [
            "default_bulk_batch_size",
            "fetch_batch_size",
            "max_concurrent_requests"
        ]

        for key in required_keys:
            if key not in perf_config:
                error_msg = f"performance.{key} が設定されていません"
                self.errors.append(error_msg)
                result["issues"].append({"type": "error", "message": error_msg})
            else:
                value = perf_config[key]

                # 値の範囲チェック
                if key == "default_bulk_batch_size":
                    if not isinstance(value, int) or value <= 0 or value > 10000:
                        warning_msg = f"bulk_batch_size ({value}) は1-10000の範囲で設定することを推奨します"
                        self.warnings.append(warning_msg)
                        result["issues"].append({"type": "warning", "message": warning_msg})

                elif key == "fetch_batch_size":
                    if not isinstance(value, int) or value <= 0 or value > 1000:
                        warning_msg = f"fetch_batch_size ({value}) は1-1000の範囲で設定することを推奨します"
                        self.warnings.append(warning_msg)
                        result["issues"].append({"type": "warning", "message": warning_msg})

                elif key == "max_concurrent_requests":
                    if not isinstance(value, int) or value <= 0 or value > 50:
                        warning_msg = f"max_concurrent_requests ({value}) は1-50の範囲で設定することを推奨します"
                        self.warnings.append(warning_msg)
                        result["issues"].append({"type": "warning", "message": warning_msg})

    def _validate_session_config(self, session_config: Dict[str, Any], result: Dict[str, Any]):
        """セッション設定を検証"""
        recommended_keys = [
            "request_timeout",
            "connection_timeout",
            "retry_count"
        ]

        for key in recommended_keys:
            if key not in session_config:
                suggestion_msg = f"session_management.{key} の設定を推奨します"
                self.suggestions.append(suggestion_msg)
                result["issues"].append({"type": "suggestion", "message": suggestion_msg})
            else:
                value = session_config[key]

                # タイムアウト値の妥当性チェック
                if "timeout" in key:
                    if not isinstance(value, (int, float)) or value <= 0 or value > 300:
                        warning_msg = f"{key} ({value}) は1-300秒の範囲で設定することを推奨します"
                        self.warnings.append(warning_msg)
                        result["issues"].append({"type": "warning", "message": warning_msg})

                elif key == "retry_count":
                    if not isinstance(value, int) or value < 0 or value > 10:
                        warning_msg = f"retry_count ({value}) は0-10の範囲で設定することを推奨します"
                        self.warnings.append(warning_msg)
                        result["issues"].append({"type": "warning", "message": warning_msg})

    def _validate_validation_config(self, validation_config: Dict[str, Any], result: Dict[str, Any]):
        """バリデーション設定を検証"""
        recommended_settings = {
            "validate_symbol_format": True,
            "validate_company_name": True,
            "skip_invalid_records": True
        }

        for key, recommended_value in recommended_settings.items():
            if key not in validation_config:
                suggestion_msg = f"validation.{key} の設定を推奨します（推奨値: {recommended_value}）"
                self.suggestions.append(suggestion_msg)
                result["issues"].append({"type": "suggestion", "message": suggestion_msg})
            else:
                value = validation_config[key]
                if not isinstance(value, bool):
                    warning_msg = f"validation.{key} はboolean値である必要があります（現在の値: {value}）"
                    self.warnings.append(warning_msg)
                    result["issues"].append({"type": "warning", "message": warning_msg})

    def validate_patterns_config(self) -> Dict[str, Any]:
        """パターン設定を検証"""
        logger.info("パターン設定検証開始")

        try:
            config = get_patterns_config()
            validation_result = {
                "status": "success",
                "config": config,
                "issues": []
            }

            # 必須パターン設定の確認
            required_patterns = [
                "golden_dead_cross",
                "support_resistance",
                "breakout_detection"
            ]

            for pattern in required_patterns:
                if pattern not in config:
                    error_msg = f"必須パターン設定 '{pattern}' が見つかりません"
                    self.errors.append(error_msg)
                    validation_result["issues"].append({"type": "error", "message": error_msg})
                else:
                    logger.info(f"✅ パターン設定 '{pattern}' 確認済み")

                    # パターン固有の検証
                    self._validate_pattern_specific_config(
                        pattern, config[pattern], validation_result
                    )

            return validation_result

        except Exception as e:
            error_msg = f"パターン設定読み込みエラー: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)

            return {
                "status": "error",
                "error": error_msg,
                "issues": [{"type": "error", "message": error_msg}]
            }

    def _validate_pattern_specific_config(
        self, pattern_name: str, pattern_config: Dict[str, Any], result: Dict[str, Any]
    ):
        """パターン固有の設定検証"""

        if pattern_name == "golden_dead_cross":
            required_params = ["fast_period", "slow_period", "confidence_threshold"]

            for param in required_params:
                if param not in pattern_config:
                    error_msg = f"{pattern_name}.{param} が設定されていません"
                    self.errors.append(error_msg)
                    result["issues"].append({"type": "error", "message": error_msg})
                else:
                    value = pattern_config[param]

                    if param in ["fast_period", "slow_period"]:
                        if not isinstance(value, int) or value <= 0 or value > 200:
                            warning_msg = f"{pattern_name}.{param} ({value}) は1-200の範囲が推奨されます"
                            self.warnings.append(warning_msg)
                            result["issues"].append({"type": "warning", "message": warning_msg})

                    elif param == "confidence_threshold":
                        if not isinstance(value, (int, float)) or value < 0 or value > 100:
                            warning_msg = f"{pattern_name}.{param} ({value}) は0-100の範囲が推奨されます"
                            self.warnings.append(warning_msg)
                            result["issues"].append({"type": "warning", "message": warning_msg})

        elif pattern_name == "support_resistance":
            if "window_size" in pattern_config:
                window_size = pattern_config["window_size"]
                if not isinstance(window_size, int) or window_size <= 0 or window_size > 100:
                    warning_msg = f"{pattern_name}.window_size ({window_size}) は1-100の範囲が推奨されます"
                    self.warnings.append(warning_msg)
                    result["issues"].append({"type": "warning", "message": warning_msg})

    def validate_environment_settings(self) -> Dict[str, Any]:
        """環境設定を検証"""
        logger.info("環境設定検証開始")

        validation_result = {
            "status": "success",
            "environment_variables": {},
            "issues": []
        }

        # 重要な環境変数をチェック
        important_env_vars = [
            "DATABASE_URL",
            "LOG_LEVEL",
            "CACHE_ENABLED",
            "MAX_WORKERS"
        ]

        for env_var in important_env_vars:
            value = os.getenv(env_var)
            validation_result["environment_variables"][env_var] = value

            if value is None:
                suggestion_msg = f"環境変数 {env_var} が設定されていません（オプション）"
                self.suggestions.append(suggestion_msg)
                validation_result["issues"].append({"type": "suggestion", "message": suggestion_msg})
            else:
                logger.info(f"✅ 環境変数 {env_var}={value}")

                # 値の妥当性チェック
                if env_var == "LOG_LEVEL":
                    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    if value.upper() not in valid_levels:
                        warning_msg = f"LOG_LEVEL ({value}) は {valid_levels} のいずれかが推奨されます"
                        self.warnings.append(warning_msg)
                        validation_result["issues"].append({"type": "warning", "message": warning_msg})

                elif env_var == "MAX_WORKERS":
                    try:
                        workers = int(value)
                        if workers <= 0 or workers > 50:
                            warning_msg = f"MAX_WORKERS ({workers}) は1-50の範囲が推奨されます"
                            self.warnings.append(warning_msg)
                            validation_result["issues"].append({"type": "warning", "message": warning_msg})
                    except ValueError:
                        error_msg = f"MAX_WORKERS ({value}) は整数である必要があります"
                        self.errors.append(error_msg)
                        validation_result["issues"].append({"type": "error", "message": error_msg})

        return validation_result

    def validate_file_permissions(self) -> Dict[str, Any]:
        """ファイル権限を検証"""
        logger.info("ファイル権限検証開始")

        validation_result = {
            "status": "success",
            "permissions": {},
            "issues": []
        }

        # 重要なディレクトリ/ファイルの権限をチェック
        paths_to_check = [
            PROJECT_ROOT / "config",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "logs",
            PROJECT_ROOT / "scripts"
        ]

        for path in paths_to_check:
            if not path.exists():
                warning_msg = f"パス {path} が存在しません"
                self.warnings.append(warning_msg)
                validation_result["issues"].append({"type": "warning", "message": warning_msg})
                validation_result["permissions"][str(path)] = "存在しない"
            else:
                # 読み込み権限チェック
                readable = os.access(path, os.R_OK)
                writable = os.access(path, os.W_OK)

                permissions = []
                if readable:
                    permissions.append("R")
                if writable:
                    permissions.append("W")

                permission_str = "".join(permissions) if permissions else "なし"
                validation_result["permissions"][str(path)] = permission_str

                logger.info(f"✅ {path}: {permission_str}")

                # ディレクトリの場合、書き込み権限が必要
                if path.is_dir() and not writable:
                    warning_msg = f"ディレクトリ {path} に書き込み権限がありません"
                    self.warnings.append(warning_msg)
                    validation_result["issues"].append({"type": "warning", "message": warning_msg})

        return validation_result

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括的な検証レポートを生成"""
        logger.info("包括的検証レポート生成開始")

        report = {
            "validation_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now()),
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "total_suggestions": len(self.suggestions)
            },
            "validations": {}
        }

        # 各検証を実行
        validations = [
            ("stock_master_config", self.validate_stock_master_config),
            ("patterns_config", self.validate_patterns_config),
            ("environment_settings", self.validate_environment_settings),
            ("file_permissions", self.validate_file_permissions)
        ]

        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                report["validations"][validation_name] = result

            except Exception as e:
                error_msg = f"{validation_name} 検証中にエラー: {e}"
                logger.error(error_msg)
                report["validations"][validation_name] = {
                    "status": "error",
                    "error": error_msg
                }

        # 全体的な問題点をまとめ
        report["issues_summary"] = {
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }

        # 総合評価
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                report["overall_status"] = "excellent"
                report["overall_message"] = "設定は完璧です"
            elif len(self.warnings) <= 2:
                report["overall_status"] = "good"
                report["overall_message"] = "設定は良好です（軽微な警告があります）"
            else:
                report["overall_status"] = "fair"
                report["overall_message"] = "設定は使用可能ですが、改善の余地があります"
        else:
            report["overall_status"] = "poor"
            report["overall_message"] = "設定にエラーがあります。修正が必要です"

        return report

    def fix_common_issues(self) -> List[str]:
        """一般的な問題を自動修正"""
        logger.info("一般的な問題の自動修正開始")

        fixed_issues = []

        # ディレクトリの作成
        directories_to_create = [
            PROJECT_ROOT / "config",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "logs"
        ]

        for directory in directories_to_create:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    fixed_msg = f"ディレクトリを作成しました: {directory}"
                    logger.info(fixed_msg)
                    fixed_issues.append(fixed_msg)

                except Exception as e:
                    logger.error(f"ディレクトリ作成失敗 {directory}: {e}")

        # デフォルト設定ファイルの作成
        default_configs = [
            {
                "path": PROJECT_ROOT / "config" / "stock_master_config.json",
                "content": {
                    "session_management": {
                        "request_timeout": 30,
                        "connection_timeout": 10,
                        "retry_count": 3
                    },
                    "performance": {
                        "default_bulk_batch_size": 100,
                        "fetch_batch_size": 50,
                        "max_concurrent_requests": 5
                    },
                    "validation": {
                        "validate_symbol_format": True,
                        "validate_company_name": True,
                        "skip_invalid_records": True
                    }
                }
            },
            {
                "path": PROJECT_ROOT / "config" / "patterns_config.json",
                "content": {
                    "golden_dead_cross": {
                        "fast_period": 5,
                        "slow_period": 25,
                        "confidence_threshold": 70
                    },
                    "support_resistance": {
                        "window_size": 20,
                        "strength_threshold": 3
                    },
                    "breakout_detection": {
                        "volume_factor": 1.5,
                        "price_change_threshold": 2.0
                    }
                }
            }
        ]

        for config_info in default_configs:
            config_path = config_info["path"]
            config_content = config_info["content"]

            if not config_path.exists():
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, ensure_ascii=False, indent=2)

                    fixed_msg = f"デフォルト設定ファイルを作成しました: {config_path}"
                    logger.info(fixed_msg)
                    fixed_issues.append(fixed_msg)

                except Exception as e:
                    logger.error(f"設定ファイル作成失敗 {config_path}: {e}")

        return fixed_issues


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="設定ファイル検証スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/validate_config.py
    python scripts/validate_config.py --fix-errors
    python scripts/validate_config.py --output-report validation_report.json
        """
    )

    parser.add_argument(
        '--fix-errors',
        action='store_true',
        help='一般的な問題を自動修正'
    )

    parser.add_argument(
        '--output-report',
        type=str,
        help='検証レポートをJSONファイルに出力'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細なログを出力'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        logger.info("=== 設定ファイル検証開始 ===")

        validator = ConfigValidator()

        # 自動修正
        if args.fix_errors:
            fixed_issues = validator.fix_common_issues()
            if fixed_issues:
                logger.info("自動修正結果:")
                for issue in fixed_issues:
                    logger.info(f"  ✅ {issue}")
            else:
                logger.info("自動修正できる問題はありませんでした")

        # 包括的検証
        report = validator.generate_comprehensive_report()

        # 結果表示
        logger.info("=== 検証結果サマリー ===")
        logger.info(f"総合評価: {report['overall_status'].upper()}")
        logger.info(f"メッセージ: {report['overall_message']}")
        logger.info(f"エラー: {report['summary']['total_errors']}件")
        logger.info(f"警告: {report['summary']['total_warnings']}件")
        logger.info(f"提案: {report['summary']['total_suggestions']}件")

        # 詳細な問題点表示
        issues_summary = report['issues_summary']

        if issues_summary['errors']:
            logger.error("🚨 エラー:")
            for error in issues_summary['errors']:
                logger.error(f"  - {error}")

        if issues_summary['warnings']:
            logger.warning("⚠️ 警告:")
            for warning in issues_summary['warnings']:
                logger.warning(f"  - {warning}")

        if issues_summary['suggestions']:
            logger.info("💡 改善提案:")
            for suggestion in issues_summary['suggestions']:
                logger.info(f"  - {suggestion}")

        # レポートファイル出力
        if args.output_report:
            with open(args.output_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"詳細レポートを出力しました: {args.output_report}")

        logger.info("=== 設定ファイル検証完了 ===")

        # 終了コード（エラーがある場合は1）
        return 1 if report['summary']['total_errors'] > 0 else 0

    except KeyboardInterrupt:
        logger.info("検証が中断されました")
        return 1

    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # datetime が必要な場合
    from datetime import datetime

    exit(main())

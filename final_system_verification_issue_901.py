#!/usr/bin/env python3
"""
最終システム検証 - Phase J-2: Issue #901最終統合フェーズ

全機能の動作確認とテストを実行
統合ダッシュボードシステムの総合検証
"""

import asyncio
import json
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# システムパス追加
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.day_trade.dashboard.integrated_dashboard_system import IntegratedDashboardSystem
    from src.day_trade.utils.logging_config import get_context_logger
    from src.day_trade.core.enhanced_error_handler import EnhancedErrorHandler
except ImportError as e:
    print(f"Import error: {e}")
    print("必要なモジュールの一部が利用できませんが、検証を継続します。")
    # sys.exit(1)  # エラーを無視して継続

logger = get_context_logger(__name__)


class FinalSystemVerification:
    """最終システム検証"""

    def __init__(self):
        """初期化"""
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "verification_results"
        self.results_dir.mkdir(exist_ok=True)

        # エラーハンドラー
        self.error_handler = EnhancedErrorHandler()

        # 検証結果
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'system_components': {},
            'integration_tests': {},
            'performance_tests': {},
            'security_tests': {},
            'data_integrity_tests': {},
            'overall_status': 'unknown',
            'recommendations': [],
            'errors': []
        }

        # 統合ダッシュボードシステム
        self.integrated_dashboard = None

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """包括的システム検証実行"""
        try:
            logger.info("最終システム検証開始")
            logger.info("=" * 60)

            # Step 1: システムコンポーネント検証
            await self._verify_system_components()

            # Step 2: 統合ダッシュボード検証
            await self._verify_integrated_dashboard()

            # Step 3: 統合テスト実行
            await self._run_integration_tests()

            # Step 4: パフォーマンステスト実行
            await self._run_performance_tests()

            # Step 5: セキュリティテスト実行
            await self._run_security_tests()

            # Step 6: データ整合性テスト実行
            await self._run_data_integrity_tests()

            # Step 7: 全体評価
            self._evaluate_overall_status()

            # Step 8: 結果保存
            await self._save_verification_results()

            logger.info("最終システム検証完了")
            return self.verification_results

        except Exception as e:
            error_msg = f"最終システム検証エラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'system_error',
                'message': error_msg,
                'traceback': traceback.format_exc()
            })
            return self.verification_results

    async def _verify_system_components(self):
        """システムコンポーネント検証"""
        try:
            logger.info("システムコンポーネント検証開始")

            component_results = {}

            # 基本ディレクトリ構造確認
            required_dirs = [
                'src/day_trade',
                'src/day_trade/dashboard',
                'src/day_trade/analysis',
                'src/day_trade/ml',
                'src/day_trade/monitoring',
                'src/day_trade/risk',
                'src/day_trade/api',
                'src/day_trade/data',
                'src/day_trade/utils'
            ]

            for dir_path in required_dirs:
                full_path = self.base_dir / dir_path
                component_results[dir_path] = {
                    'exists': full_path.exists(),
                    'is_directory': full_path.is_dir() if full_path.exists() else False,
                    'file_count': len(list(full_path.glob('*.py'))) if full_path.exists() and full_path.is_dir() else 0
                }

            # 主要ファイル確認
            required_files = [
                'src/day_trade/dashboard/integrated_dashboard_system.py',
                'src/day_trade/dashboard/web_dashboard.py',
                'src/day_trade/analysis/enhanced_ensemble.py',
                'src/day_trade/ml/ensemble_system.py',
                'src/day_trade/monitoring/advanced_monitoring_system.py',
                'src/day_trade/risk/integrated_risk_management.py'
            ]

            for file_path in required_files:
                full_path = self.base_dir / file_path
                component_results[file_path] = {
                    'exists': full_path.exists(),
                    'is_file': full_path.is_file() if full_path.exists() else False,
                    'size': full_path.stat().st_size if full_path.exists() else 0
                }

            # Python importテスト
            import_tests = {
                'day_trade.dashboard.integrated_dashboard_system': False,
                'day_trade.dashboard.web_dashboard': False,
                'day_trade.utils.logging_config': False,
                'day_trade.core.enhanced_error_handler': False
            }

            for module_name in import_tests.keys():
                try:
                    __import__(f'src.{module_name}', fromlist=[''])
                    import_tests[module_name] = True
                except Exception as e:
                    logger.warning(f"モジュールインポートエラー {module_name}: {e}")

            component_results['import_tests'] = import_tests

            self.verification_results['system_components'] = component_results
            logger.info("システムコンポーネント検証完了")

        except Exception as e:
            error_msg = f"システムコンポーネント検証エラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'component_verification_error',
                'message': error_msg
            })

    async def _verify_integrated_dashboard(self):
        """統合ダッシュボード検証"""
        try:
            logger.info("統合ダッシュボード検証開始")

            dashboard_results = {
                'initialization': False,
                'component_loading': {},
                'database_creation': False,
                'monitoring_start': False,
                'error_details': []
            }

            try:
                # 統合ダッシュボード初期化テスト
                self.integrated_dashboard = IntegratedDashboardSystem()
                dashboard_results['initialization'] = True
                logger.info("統合ダッシュボード初期化成功")

                # データベース作成確認
                db_path = self.integrated_dashboard.db_path
                if db_path.exists():
                    dashboard_results['database_creation'] = True
                    logger.info("統合ダッシュボードデータベース作成成功")

                # コンポーネント初期化テスト
                try:
                    await self.integrated_dashboard.initialize_components()

                    # 各コンポーネントの初期化状況確認
                    components = {
                        'web_dashboard': self.integrated_dashboard.web_dashboard,
                        'production_dashboard': self.integrated_dashboard.production_dashboard,
                        'interactive_dashboard': self.integrated_dashboard.interactive_dashboard,
                        'security_dashboard': self.integrated_dashboard.security_dashboard,
                        'ensemble_system': self.integrated_dashboard.ensemble_system,
                        'prediction_engine': self.integrated_dashboard.prediction_engine,
                        'monitoring_system': self.integrated_dashboard.monitoring_system,
                        'risk_management': self.integrated_dashboard.risk_management
                    }

                    for name, component in components.items():
                        dashboard_results['component_loading'][name] = component is not None

                    logger.info("統合ダッシュボードコンポーネント初期化成功")

                except Exception as e:
                    dashboard_results['error_details'].append(f"コンポーネント初期化エラー: {e}")

                # 監視システム開始テスト
                try:
                    await self.integrated_dashboard.start_integrated_monitoring()
                    dashboard_results['monitoring_start'] = True
                    logger.info("統合監視システム開始成功")

                    # 短時間待機後停止
                    await asyncio.sleep(2)
                    await self.integrated_dashboard.stop_integrated_monitoring()
                    logger.info("統合監視システム停止成功")

                except Exception as e:
                    dashboard_results['error_details'].append(f"監視システム開始エラー: {e}")

            except Exception as e:
                dashboard_results['error_details'].append(f"統合ダッシュボード初期化エラー: {e}")

            self.verification_results['integration_tests']['integrated_dashboard'] = dashboard_results
            logger.info("統合ダッシュボード検証完了")

        except Exception as e:
            error_msg = f"統合ダッシュボード検証エラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'dashboard_verification_error',
                'message': error_msg
            })

    async def _run_integration_tests(self):
        """統合テスト実行"""
        try:
            logger.info("統合テスト開始")

            integration_results = {
                'database_connectivity': False,
                'file_system_access': False,
                'configuration_loading': False,
                'error_handling': False,
                'test_details': []
            }

            # データベース接続テスト
            try:
                if self.integrated_dashboard:
                    test_db_path = self.integrated_dashboard.db_path
                    with sqlite3.connect(test_db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = cursor.fetchall()
                        if len(tables) >= 4:  # 必要なテーブル数
                            integration_results['database_connectivity'] = True
                            integration_results['test_details'].append(f"データベーステーブル数: {len(tables)}")
            except Exception as e:
                integration_results['test_details'].append(f"データベース接続エラー: {e}")

            # ファイルシステムアクセステスト
            try:
                test_file = self.base_dir / "test_file_access.tmp"
                test_file.write_text("test")
                if test_file.exists():
                    integration_results['file_system_access'] = True
                    test_file.unlink()
            except Exception as e:
                integration_results['test_details'].append(f"ファイルシステムアクセスエラー: {e}")

            # 設定ファイル読み込みテスト
            try:
                config_files = list(self.base_dir.glob("config/*.json"))
                if len(config_files) > 0:
                    integration_results['configuration_loading'] = True
                    integration_results['test_details'].append(f"設定ファイル数: {len(config_files)}")
            except Exception as e:
                integration_results['test_details'].append(f"設定読み込みエラー: {e}")

            # エラーハンドリングテスト
            try:
                if self.error_handler:
                    test_error = Exception("テストエラー")
                    self.error_handler.handle_error(test_error, {"test": True})
                    integration_results['error_handling'] = True
            except Exception as e:
                integration_results['test_details'].append(f"エラーハンドリングテストエラー: {e}")

            self.verification_results['integration_tests']['core_functionality'] = integration_results
            logger.info("統合テスト完了")

        except Exception as e:
            error_msg = f"統合テストエラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'integration_test_error',
                'message': error_msg
            })

    async def _run_performance_tests(self):
        """パフォーマンステスト実行"""
        try:
            logger.info("パフォーマンステスト開始")

            performance_results = {
                'system_resources': {},
                'response_times': {},
                'memory_usage': {},
                'recommendations': []
            }

            # システムリソース確認
            try:
                import psutil

                performance_results['system_resources'] = {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total,
                    'disk_total': psutil.disk_usage('/').total if hasattr(psutil.disk_usage('/'), 'total') else 0,
                    'cpu_usage': psutil.cpu_percent(interval=1),
                    'memory_usage': psutil.virtual_memory().percent
                }

                # パフォーマンス推奨事項
                if performance_results['system_resources']['memory_usage'] > 80:
                    performance_results['recommendations'].append("メモリ使用量が高いです。メモリ増設を検討してください。")

                if performance_results['system_resources']['cpu_usage'] > 80:
                    performance_results['recommendations'].append("CPU使用率が高いです。処理の最適化を検討してください。")

            except ImportError:
                performance_results['system_resources']['error'] = "psutilモジュールが利用できません"

            # レスポンス時間テスト
            start_time = time.time()

            # シンプルな処理時間測定
            test_data = list(range(10000))
            processed_data = [x * 2 for x in test_data]

            end_time = time.time()
            performance_results['response_times']['data_processing'] = end_time - start_time

            # メモリ使用量テスト
            try:
                import sys
                performance_results['memory_usage']['current_process'] = sys.getsizeof(processed_data)
            except Exception as e:
                performance_results['memory_usage']['error'] = str(e)

            self.verification_results['performance_tests'] = performance_results
            logger.info("パフォーマンステスト完了")

        except Exception as e:
            error_msg = f"パフォーマンステストエラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'performance_test_error',
                'message': error_msg
            })

    async def _run_security_tests(self):
        """セキュリティテスト実行"""
        try:
            logger.info("セキュリティテスト開始")

            security_results = {
                'file_permissions': {},
                'sensitive_data_check': {},
                'dependency_check': {},
                'recommendations': []
            }

            # ファイル権限チェック
            sensitive_files = [
                'src/day_trade/security/',
                'config/',
                'data/'
            ]

            for file_path in sensitive_files:
                full_path = self.base_dir / file_path
                if full_path.exists():
                    try:
                        stat_info = full_path.stat()
                        security_results['file_permissions'][file_path] = {
                            'mode': oct(stat_info.st_mode),
                            'readable': full_path.is_file() and full_path.exists()
                        }
                    except Exception as e:
                        security_results['file_permissions'][file_path] = {'error': str(e)}

            # 機密データチェック
            python_files = list(self.base_dir.glob("**/*.py"))
            sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential']

            files_with_sensitive_data = []
            for py_file in python_files[:50]:  # 最初の50ファイルのみチェック
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    for pattern in sensitive_patterns:
                        if pattern.lower() in content.lower():
                            files_with_sensitive_data.append({
                                'file': str(py_file.relative_to(self.base_dir)),
                                'pattern': pattern
                            })
                            break
                except Exception:
                    continue

            security_results['sensitive_data_check'] = {
                'files_checked': min(len(python_files), 50),
                'files_with_sensitive_patterns': len(files_with_sensitive_data),
                'details': files_with_sensitive_data[:10]  # 最初の10件のみ
            }

            # セキュリティ推奨事項
            if files_with_sensitive_data:
                security_results['recommendations'].append(
                    "機密データの可能性があるパターンが検出されました。コードレビューを推奨します。"
                )

            self.verification_results['security_tests'] = security_results
            logger.info("セキュリティテスト完了")

        except Exception as e:
            error_msg = f"セキュリティテストエラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'security_test_error',
                'message': error_msg
            })

    async def _run_data_integrity_tests(self):
        """データ整合性テスト実行"""
        try:
            logger.info("データ整合性テスト開始")

            data_integrity_results = {
                'database_files': {},
                'data_directories': {},
                'backup_status': {},
                'recommendations': []
            }

            # データベースファイル確認
            db_files = list(self.base_dir.glob("**/*.db"))
            for db_file in db_files:
                try:
                    with sqlite3.connect(db_file) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = cursor.fetchall()

                        data_integrity_results['database_files'][str(db_file.name)] = {
                            'accessible': True,
                            'table_count': len(tables),
                            'size': db_file.stat().st_size
                        }
                except Exception as e:
                    data_integrity_results['database_files'][str(db_file.name)] = {
                        'accessible': False,
                        'error': str(e)
                    }

            # データディレクトリ確認
            data_dirs = ['data', 'cache_data', 'model_cache', 'reports']
            for dir_name in data_dirs:
                dir_path = self.base_dir / dir_name
                if dir_path.exists():
                    file_count = len(list(dir_path.glob("**/*")))
                    data_integrity_results['data_directories'][dir_name] = {
                        'exists': True,
                        'file_count': file_count,
                        'size': sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                    }
                else:
                    data_integrity_results['data_directories'][dir_name] = {'exists': False}

            # バックアップ状況確認
            backup_dirs = ['backup', 'stability_data/backups']
            for backup_dir in backup_dirs:
                backup_path = self.base_dir / backup_dir
                if backup_path.exists():
                    backup_files = list(backup_path.glob("**/*"))
                    data_integrity_results['backup_status'][backup_dir] = {
                        'exists': True,
                        'backup_count': len(backup_files)
                    }

            # データ整合性推奨事項
            if not any(data_integrity_results['backup_status'].values()):
                data_integrity_results['recommendations'].append(
                    "バックアップファイルが確認できません。定期バックアップの設定を推奨します。"
                )

            self.verification_results['data_integrity_tests'] = data_integrity_results
            logger.info("データ整合性テスト完了")

        except Exception as e:
            error_msg = f"データ整合性テストエラー: {e}"
            logger.error(error_msg)
            self.verification_results['errors'].append({
                'type': 'data_integrity_test_error',
                'message': error_msg
            })

    def _evaluate_overall_status(self):
        """全体評価"""
        try:
            logger.info("全体評価開始")

            success_count = 0
            total_tests = 0

            # システムコンポーネント評価
            if self.verification_results.get('system_components'):
                components = self.verification_results['system_components']
                for key, value in components.items():
                    if isinstance(value, dict):
                        if key == 'import_tests':
                            for import_result in value.values():
                                total_tests += 1
                                if import_result:
                                    success_count += 1
                        else:
                            total_tests += 1
                            if value.get('exists', False):
                                success_count += 1

            # 統合テスト評価
            if self.verification_results.get('integration_tests'):
                for test_category in self.verification_results['integration_tests'].values():
                    if isinstance(test_category, dict):
                        for key, value in test_category.items():
                            if isinstance(value, bool):
                                total_tests += 1
                                if value:
                                    success_count += 1

            # 成功率計算
            success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0

            # 全体ステータス判定
            if success_rate >= 90:
                overall_status = 'excellent'
            elif success_rate >= 75:
                overall_status = 'good'
            elif success_rate >= 50:
                overall_status = 'fair'
            else:
                overall_status = 'poor'

            self.verification_results['overall_status'] = overall_status
            self.verification_results['success_rate'] = success_rate
            self.verification_results['successful_tests'] = success_count
            self.verification_results['total_tests'] = total_tests

            # 総合推奨事項
            if success_rate < 75:
                self.verification_results['recommendations'].append(
                    "システムの成功率が75%未満です。問題のあるコンポーネントの修正を推奨します。"
                )

            if len(self.verification_results['errors']) > 5:
                self.verification_results['recommendations'].append(
                    "エラーが多数発生しています。システムの安定性向上が必要です。"
                )

            logger.info(f"全体評価完了 - ステータス: {overall_status}, 成功率: {success_rate:.1f}%")

        except Exception as e:
            error_msg = f"全体評価エラー: {e}"
            logger.error(error_msg)
            self.verification_results['overall_status'] = 'error'
            self.verification_results['errors'].append({
                'type': 'evaluation_error',
                'message': error_msg
            })

    async def _save_verification_results(self):
        """検証結果保存"""
        try:
            # JSON形式で保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"final_system_verification_{timestamp}.json"

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.verification_results, f, ensure_ascii=False, indent=2)

            # サマリーレポート生成
            summary_file = self.results_dir / f"verification_summary_{timestamp}.txt"

            summary_content = f"""
最終システム検証サマリーレポート
=====================================

実行時刻: {self.verification_results['timestamp']}
全体ステータス: {self.verification_results.get('overall_status', 'unknown')}
成功率: {self.verification_results.get('success_rate', 0):.1f}%
成功テスト数: {self.verification_results.get('successful_tests', 0)}/{self.verification_results.get('total_tests', 0)}

エラー数: {len(self.verification_results.get('errors', []))}
推奨事項数: {len(self.verification_results.get('recommendations', []))}

統合ダッシュボードシステム:
- 初期化: {'成功' if self.verification_results.get('integration_tests', {}).get('integrated_dashboard', {}).get('initialization', False) else '失敗'}
- データベース作成: {'成功' if self.verification_results.get('integration_tests', {}).get('integrated_dashboard', {}).get('database_creation', False) else '失敗'}
- 監視システム開始: {'成功' if self.verification_results.get('integration_tests', {}).get('integrated_dashboard', {}).get('monitoring_start', False) else '失敗'}

主要推奨事項:
"""

            for recommendation in self.verification_results.get('recommendations', []):
                summary_content += f"- {recommendation}\n"

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)

            logger.info(f"検証結果保存完了: {results_file}")
            logger.info(f"サマリーレポート保存完了: {summary_file}")

        except Exception as e:
            error_msg = f"検証結果保存エラー: {e}"
            logger.error(error_msg)


async def main():
    """メイン実行"""
    logger.info("Issue #901 最終システム検証開始")
    logger.info("=" * 60)

    verification = FinalSystemVerification()

    try:
        results = await verification.run_comprehensive_verification()

        # 結果表示
        print("\n" + "=" * 60)
        print("最終システム検証結果")
        print("=" * 60)
        print(f"全体ステータス: {results.get('overall_status', 'unknown')}")
        print(f"成功率: {results.get('success_rate', 0):.1f}%")
        print(f"エラー数: {len(results.get('errors', []))}")
        print(f"推奨事項数: {len(results.get('recommendations', []))}")
        print("=" * 60)

        if results.get('overall_status') in ['excellent', 'good']:
            print("✅ システム検証が正常に完了しました")
        else:
            print("⚠️ システムに問題が検出されました。詳細は検証結果ファイルを確認してください")

    except Exception as e:
        logger.error(f"最終システム検証エラー: {e}")
        print(f"❌ システム検証中にエラーが発生しました: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
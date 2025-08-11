#!/usr/bin/env python3
"""
セキュリティ統合管理システム
Issue #419: セキュリティ強化 - 統合セキュリティ管理システム

脆弱性管理、データ保護、アクセス制御、設定管理、テストフレームワークを
統合したセキュリティ管理システム
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .access_control import AccessControlManager, create_access_control_manager
    from .data_protection import DataProtectionManager, create_data_protection_manager
    from .security_config import SecurityConfigManager, create_security_config_manager
    from .security_test_framework import (
        SecurityTestFramework,
        create_security_test_framework,
    )
    from .vulnerability_manager import (
        VulnerabilityManager,
        create_vulnerability_manager,
    )
except ImportError:
    # テスト環境やスタンドアロン実行時の対応
    VulnerabilityManager = None
    DataProtectionManager = None
    AccessControlManager = None
    SecurityConfigManager = None
    SecurityTestFramework = None

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class SecurityManager:
    """
    セキュリティ統合管理システム

    全てのセキュリティコンポーネントを統合管理し、
    包括的なセキュリティ運用を提供
    """

    def __init__(self, base_path: str = "security", security_level: str = "high"):
        """
        初期化

        Args:
            base_path: セキュリティデータベースパス
            security_level: セキュリティレベル (low, medium, high, critical)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # セキュリティレベル設定
        from .security_config import SecurityLevel

        security_level_map = {
            "low": SecurityLevel.LOW,
            "medium": SecurityLevel.MEDIUM,
            "high": SecurityLevel.HIGH,
            "critical": SecurityLevel.CRITICAL,
        }
        self.security_level = security_level_map.get(security_level, SecurityLevel.HIGH)

        # セキュリティコンポーネント初期化
        self.config_manager: Optional[SecurityConfigManager] = None
        self.vulnerability_manager: Optional[VulnerabilityManager] = None
        self.data_protection_manager: Optional[DataProtectionManager] = None
        self.access_control_manager: Optional[AccessControlManager] = None
        self.test_framework: Optional[SecurityTestFramework] = None

        # 初期化実行
        self._initialize_components()

        logger.info(f"SecurityManager初期化完了 (レベル: {security_level})")

    def _initialize_components(self):
        """セキュリティコンポーネント初期化"""
        try:
            # 設定管理システム
            if SecurityConfigManager:
                self.config_manager = create_security_config_manager(
                    config_path=str(self.base_path / "config"),
                    security_level=self.security_level,
                )

            # 脆弱性管理システム
            if VulnerabilityManager:
                self.vulnerability_manager = create_vulnerability_manager(
                    storage_path=str(self.base_path / "vulnerabilities")
                )

            # データ保護システム
            if DataProtectionManager:
                self.data_protection_manager = create_data_protection_manager(
                    storage_path=str(self.base_path / "data_protection")
                )

            # アクセス制御システム
            if AccessControlManager:
                self.access_control_manager = create_access_control_manager(
                    storage_path=str(self.base_path / "access_control")
                )

            # セキュリティテストフレームワーク
            if SecurityTestFramework:
                self.test_framework = create_security_test_framework(
                    output_path=str(self.base_path / "test_results")
                )

            logger.info("セキュリティコンポーネント初期化完了")

        except Exception as e:
            logger.error(f"セキュリティコンポーネント初期化エラー: {e}")

    async def run_comprehensive_security_assessment(self) -> Dict[str, Any]:
        """包括的セキュリティ評価実行"""
        logger.info("包括的セキュリティ評価開始")
        start_time = datetime.utcnow()

        assessment_results = {
            "assessment_id": f"security-assessment-{int(start_time.timestamp())}",
            "started_at": start_time.isoformat(),
            "security_level": self.security_level.value,
            "components_status": {},
            "vulnerability_scan": None,
            "security_test_results": None,
            "configuration_analysis": None,
            "access_control_audit": None,
            "data_protection_status": None,
            "overall_security_score": 0.0,
            "recommendations": [],
            "executive_summary": "",
        }

        # 1. コンポーネント状態確認
        assessment_results["components_status"] = self._check_components_status()

        # 2. 脆弱性スキャン実行
        if self.vulnerability_manager:
            try:
                logger.info("脆弱性スキャン実行中...")
                scan_results = await self.vulnerability_manager.run_comprehensive_scan(
                    "."
                )
                vulnerability_report = (
                    await self.vulnerability_manager.generate_security_report()
                )
                assessment_results["vulnerability_scan"] = vulnerability_report
            except Exception as e:
                logger.error(f"脆弱性スキャンエラー: {e}")
                assessment_results["vulnerability_scan"] = {"error": str(e)}

        # 3. セキュリティテスト実行
        if self.test_framework:
            try:
                logger.info("セキュリティテスト実行中...")
                test_context = self._prepare_test_context()
                test_results = await self.test_framework.run_all_tests(**test_context)
                assessment_results["security_test_results"] = test_results
            except Exception as e:
                logger.error(f"セキュリティテストエラー: {e}")
                assessment_results["security_test_results"] = {"error": str(e)}

        # 4. 設定分析
        if self.config_manager:
            try:
                logger.info("設定分析実行中...")
                config_report = self.config_manager.get_security_report()
                assessment_results["configuration_analysis"] = config_report
            except Exception as e:
                logger.error(f"設定分析エラー: {e}")
                assessment_results["configuration_analysis"] = {"error": str(e)}

        # 5. アクセス制御監査
        if self.access_control_manager:
            try:
                logger.info("アクセス制御監査実行中...")
                access_report = self.access_control_manager.get_security_report()
                assessment_results["access_control_audit"] = access_report
            except Exception as e:
                logger.error(f"アクセス制御監査エラー: {e}")
                assessment_results["access_control_audit"] = {"error": str(e)}

        # 6. データ保護状態確認
        if self.data_protection_manager:
            try:
                logger.info("データ保護状態確認中...")
                data_protection_report = (
                    self.data_protection_manager.get_security_report()
                )
                assessment_results["data_protection_status"] = data_protection_report
            except Exception as e:
                logger.error(f"データ保護状態確認エラー: {e}")
                assessment_results["data_protection_status"] = {"error": str(e)}

        # 7. 統合分析とスコア計算
        end_time = datetime.utcnow()
        assessment_results["completed_at"] = end_time.isoformat()
        assessment_results["duration_seconds"] = (end_time - start_time).total_seconds()

        # 総合セキュリティスコア計算
        assessment_results[
            "overall_security_score"
        ] = self._calculate_overall_security_score(assessment_results)

        # 推奨事項生成
        assessment_results[
            "recommendations"
        ] = self._generate_comprehensive_recommendations(assessment_results)

        # エグゼクティブサマリー生成
        assessment_results["executive_summary"] = self._generate_executive_summary(
            assessment_results
        )

        # 結果保存
        await self._save_assessment_results(assessment_results)

        logger.info(
            f"包括的セキュリティ評価完了: スコア {assessment_results['overall_security_score']:.1f}/100"
        )

        return assessment_results

    def _check_components_status(self) -> Dict[str, Any]:
        """セキュリティコンポーネント状態確認"""
        status = {
            "config_manager": {
                "available": self.config_manager is not None,
                "status": "active" if self.config_manager else "unavailable",
            },
            "vulnerability_manager": {
                "available": self.vulnerability_manager is not None,
                "status": "active" if self.vulnerability_manager else "unavailable",
            },
            "data_protection_manager": {
                "available": self.data_protection_manager is not None,
                "status": "active" if self.data_protection_manager else "unavailable",
            },
            "access_control_manager": {
                "available": self.access_control_manager is not None,
                "status": "active" if self.access_control_manager else "unavailable",
            },
            "test_framework": {
                "available": self.test_framework is not None,
                "status": "active" if self.test_framework else "unavailable",
            },
        }

        # 追加ステータス情報
        if self.vulnerability_manager:
            try:
                summary = self.vulnerability_manager.get_vulnerability_summary()
                status["vulnerability_manager"]["vulnerabilities_count"] = summary[
                    "total_vulnerabilities"
                ]
                status["vulnerability_manager"]["critical_open"] = summary[
                    "critical_open"
                ]
            except Exception:
                pass

        if self.access_control_manager:
            try:
                status["access_control_manager"]["users_count"] = len(
                    self.access_control_manager.users
                )
                status["access_control_manager"]["active_sessions"] = len(
                    [
                        s
                        for s in self.access_control_manager.sessions.values()
                        if s.is_valid()
                    ]
                )
            except Exception:
                pass

        return status

    def _prepare_test_context(self) -> Dict[str, Any]:
        """セキュリティテスト用コンテキスト準備"""
        context = {
            "password_policy": None,
            "session_manager": None,
            "input_validators": None,
            "data_protection_manager": None,
            "security_config": None,
            "target_host": "localhost",
            "target_ports": [80, 443, 8080],
        }

        # 利用可能なコンポーネントを設定
        if self.config_manager:
            context["password_policy"] = self.config_manager.password_policy
            context["security_config"] = self.config_manager

        if self.access_control_manager:
            context["session_manager"] = self.access_control_manager

        if self.data_protection_manager:
            context["data_protection_manager"] = self.data_protection_manager

        return context

    def _calculate_overall_security_score(self, assessment: Dict[str, Any]) -> float:
        """総合セキュリティスコア計算"""
        scores = []
        weights = []

        # セキュリティテストスコア (重み: 30%)
        if (
            assessment.get("security_test_results")
            and "security_score" in assessment["security_test_results"]
        ):
            scores.append(assessment["security_test_results"]["security_score"])
            weights.append(30)

        # 脆弱性管理スコア (重み: 25%)
        if assessment.get("vulnerability_scan"):
            vuln_data = assessment["vulnerability_scan"]
            if "summary" in vuln_data:
                summary = vuln_data["summary"]
                # 脆弱性数に基づくスコア計算
                total_vulns = summary.get("total_vulnerabilities", 0)
                critical_vulns = summary.get("critical_open", 0)

                vuln_score = 100
                if total_vulns > 0:
                    vuln_score -= min(total_vulns * 5, 50)  # 一般脆弱性による減点
                if critical_vulns > 0:
                    vuln_score -= critical_vulns * 20  # 重大脆弱性による大幅減点

                scores.append(max(vuln_score, 0))
                weights.append(25)

        # 設定管理スコア (重み: 20%)
        if assessment.get("configuration_analysis"):
            config_data = assessment["configuration_analysis"]
            if "configuration_valid" in config_data:
                config_score = 100 if config_data["configuration_valid"] else 60
                validation_issues = len(config_data.get("validation_issues", []))
                config_score -= validation_issues * 10

                scores.append(max(config_score, 0))
                weights.append(20)

        # アクセス制御スコア (重み: 15%)
        if assessment.get("access_control_audit"):
            access_data = assessment["access_control_audit"]
            user_stats = access_data.get("user_statistics", {})

            access_score = 100
            # MFA有効化率によるスコア調整
            total_users = user_stats.get("total_users", 1)
            mfa_users = user_stats.get("mfa_enabled_users", 0)
            mfa_ratio = mfa_users / total_users if total_users > 0 else 0
            access_score = access_score * (0.5 + 0.5 * mfa_ratio)  # 50-100%

            # ロックされたユーザーによる減点
            locked_users = user_stats.get("locked_users", 0)
            access_score -= locked_users * 5

            scores.append(max(access_score, 0))
            weights.append(15)

        # データ保護スコア (重み: 10%)
        if assessment.get("data_protection_status"):
            data_data = assessment["data_protection_status"]
            if "encryption_status" in data_data:
                encryption_status = data_data["encryption_status"]
                data_score = (
                    100
                    if encryption_status.get("crypto_library_available", False)
                    else 50
                )

                scores.append(data_score)
                weights.append(10)

        # 加重平均計算
        if scores and weights:
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight

        return 0.0

    def _generate_comprehensive_recommendations(
        self, assessment: Dict[str, Any]
    ) -> List[str]:
        """包括的推奨事項生成"""
        recommendations = []

        # 総合スコアによる全体的推奨事項
        overall_score = assessment["overall_security_score"]
        if overall_score < 50:
            recommendations.append(
                "🚨 セキュリティ状況が危険です。直ちに包括的なセキュリティ改善が必要です。"
            )
        elif overall_score < 70:
            recommendations.append(
                "⚠️ セキュリティ改善が必要です。優先的に対応してください。"
            )
        elif overall_score < 90:
            recommendations.append(
                "🟡 セキュリティは概ね良好ですが、いくつかの改善点があります。"
            )
        else:
            recommendations.append(
                "✅ 優秀なセキュリティ実装です。現在の状態を維持してください。"
            )

        # 各コンポーネントからの推奨事項統合
        if assessment.get("vulnerability_scan", {}).get("recommendations"):
            recommendations.extend(assessment["vulnerability_scan"]["recommendations"])

        if assessment.get("security_test_results", {}).get("recommendations"):
            recommendations.extend(
                assessment["security_test_results"]["recommendations"]
            )

        if assessment.get("configuration_analysis", {}).get("recommendations"):
            recommendations.extend(
                assessment["configuration_analysis"]["recommendations"]
            )

        if assessment.get("access_control_audit", {}).get("recommendations"):
            recommendations.extend(
                assessment["access_control_audit"]["recommendations"]
            )

        if assessment.get("data_protection_status", {}).get("recommendations"):
            recommendations.extend(
                assessment["data_protection_status"]["recommendations"]
            )

        # 重複除去
        unique_recommendations = list(dict.fromkeys(recommendations))

        return unique_recommendations[:10]  # 上位10件に制限

    def _generate_executive_summary(self, assessment: Dict[str, Any]) -> str:
        """エグゼクティブサマリー生成"""
        overall_score = assessment["overall_security_score"]
        duration = assessment["duration_seconds"]

        summary = f"""Issue #419 セキュリティ強化 - 包括的セキュリティ評価結果

実行日時: {assessment['started_at']}
評価時間: {duration:.1f}秒
セキュリティレベル: {assessment['security_level'].upper()}
総合セキュリティスコア: {overall_score:.1f}/100

== コンポーネント状況 ==
"""

        # コンポーネント状況サマリー
        components = assessment.get("components_status", {})
        active_components = sum(1 for comp in components.values() if comp["available"])
        total_components = len(components)

        summary += (
            f"アクティブコンポーネント: {active_components}/{total_components}\n\n"
        )

        # 主要結果サマリー
        if assessment.get("vulnerability_scan"):
            vuln_summary = assessment["vulnerability_scan"].get("summary", {})
            total_vulns = vuln_summary.get("total_vulnerabilities", 0)
            critical_vulns = vuln_summary.get("critical_open", 0)
            summary += f"脆弱性: 総数{total_vulns}件 (重大{critical_vulns}件)\n"

        if assessment.get("security_test_results"):
            test_stats = assessment["security_test_results"].get("statistics", {})
            total_tests = test_stats.get("total_tests", 0)
            failed_tests = test_stats.get("failed", 0)
            summary += f"セキュリティテスト: {failed_tests}/{total_tests}件失敗\n"

        if assessment.get("access_control_audit"):
            user_stats = assessment["access_control_audit"].get("user_statistics", {})
            total_users = user_stats.get("total_users", 0)
            mfa_users = user_stats.get("mfa_enabled_users", 0)
            summary += f"ユーザー管理: 総数{total_users}人 (MFA有効{mfa_users}人)\n"

        summary += "\n== 総合評価 ==\n"

        # 総合評価
        if overall_score >= 90:
            summary += "🟢 優秀 - セキュリティ実装は業界最高水準です。\n"
            summary += (
                "現在の高いセキュリティ水準を維持し、定期的な評価を継続してください。"
            )
        elif overall_score >= 70:
            summary += "🟡 良好 - セキュリティは適切ですが改善の余地があります。\n"
            summary += "検出された問題を順次解決し、より高いセキュリティレベルを目指してください。"
        elif overall_score >= 50:
            summary += "🟠 要改善 - 重要なセキュリティ問題があります。\n"
            summary += (
                "高リスクの問題を優先的に解決し、包括的なセキュリティ強化が必要です。"
            )
        else:
            summary += "🔴 危険 - 深刻なセキュリティ脆弱性が存在します。\n"
            summary += "直ちに緊急対応を実施し、システム全体のセキュリティを根本的に見直してください。"

        return summary

    async def _save_assessment_results(self, assessment: Dict[str, Any]):
        """評価結果保存"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON詳細結果
        json_file = self.base_path / f"comprehensive_assessment_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)

        # テキストサマリー
        summary_file = self.base_path / f"security_summary_{timestamp}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(assessment["executive_summary"])
            f.write("\n\n== 推奨事項 ==\n")
            for i, rec in enumerate(assessment["recommendations"], 1):
                f.write(f"{i}. {rec}\n")

        logger.info(f"包括的セキュリティ評価結果保存完了: {json_file}")

    async def get_security_dashboard(self) -> Dict[str, Any]:
        """セキュリティダッシュボード情報取得"""
        logger.info("セキュリティダッシュボード情報生成中...")

        dashboard = {
            "generated_at": datetime.utcnow().isoformat(),
            "security_level": self.security_level.value,
            "components": {},
            "alerts": [],
            "metrics": {},
            "recent_activities": [],
        }

        # コンポーネント状況
        dashboard["components"] = self._check_components_status()

        # 脆弱性アラート
        if self.vulnerability_manager:
            try:
                summary = self.vulnerability_manager.get_vulnerability_summary()
                if summary["critical_open"] > 0:
                    dashboard["alerts"].append(
                        {
                            "type": "critical_vulnerability",
                            "message": f"{summary['critical_open']}件の重大な脆弱性が未解決です",
                            "severity": "critical",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                dashboard["metrics"]["vulnerabilities"] = summary
            except Exception as e:
                logger.error(f"脆弱性情報取得エラー: {e}")

        # アクセス制御アラート
        if self.access_control_manager:
            try:
                access_report = self.access_control_manager.get_security_report()
                user_stats = access_report.get("user_statistics", {})

                if user_stats.get("locked_users", 0) > 0:
                    dashboard["alerts"].append(
                        {
                            "type": "locked_accounts",
                            "message": f"{user_stats['locked_users']}個のアカウントがロックされています",
                            "severity": "high",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                dashboard["metrics"]["access_control"] = user_stats
            except Exception as e:
                logger.error(f"アクセス制御情報取得エラー: {e}")

        # データ保護状況
        if self.data_protection_manager:
            try:
                data_report = self.data_protection_manager.get_security_report()
                dashboard["metrics"]["data_protection"] = {
                    "key_count": data_report.get("key_management", {}).get(
                        "total_keys", 0
                    ),
                    "rotation_needed": data_report.get("key_management", {}).get(
                        "rotation_needed", 0
                    ),
                }

                if data_report.get("key_management", {}).get("rotation_needed", 0) > 0:
                    dashboard["alerts"].append(
                        {
                            "type": "key_rotation",
                            "message": f"{data_report['key_management']['rotation_needed']}個のキーでローテーションが必要です",
                            "severity": "medium",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
            except Exception as e:
                logger.error(f"データ保護情報取得エラー: {e}")

        return dashboard

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("SecurityManager クリーンアップ開始")

        # 各コンポーネントのクリーンアップ
        if self.vulnerability_manager and hasattr(
            self.vulnerability_manager, "cleanup"
        ):
            await self.vulnerability_manager.cleanup()

        if self.config_manager:
            self.config_manager.save_configuration()

        logger.info("SecurityManager クリーンアップ完了")


# Factory function
def create_security_manager(
    base_path: str = "security", security_level: str = "high"
) -> SecurityManager:
    """SecurityManagerファクトリ関数"""
    return SecurityManager(base_path=base_path, security_level=security_level)


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #419 セキュリティ統合管理システムテスト ===")

        try:
            # セキュリティ統合管理システム初期化
            security_manager = create_security_manager(
                base_path="test_security", security_level="high"
            )

            print("\n1. セキュリティマネージャー初期化完了")
            print(f"セキュリティレベル: {security_manager.security_level.value}")

            # コンポーネント状況確認
            components_status = security_manager._check_components_status()
            print("\n2. セキュリティコンポーネント状況:")
            for component, status in components_status.items():
                print(
                    f"  {component}: {'✅' if status['available'] else '❌'} ({status['status']})"
                )

            # セキュリティダッシュボード
            print("\n3. セキュリティダッシュボード生成中...")
            dashboard = await security_manager.get_security_dashboard()

            print(f"生成時刻: {dashboard['generated_at']}")
            print(f"アラート数: {len(dashboard['alerts'])}")

            if dashboard["alerts"]:
                print("アクティブアラート:")
                for alert in dashboard["alerts"]:
                    print(f"  🚨 {alert['severity'].upper()}: {alert['message']}")
            else:
                print("アクティブアラートなし")

            # 包括的セキュリティ評価実行
            print("\n4. 包括的セキュリティ評価実行中...")
            print("(注意: この処理には時間がかかる場合があります)")

            assessment = await security_manager.run_comprehensive_security_assessment()

            print("\n=== セキュリティ評価結果 ===")
            print(f"評価ID: {assessment['assessment_id']}")
            print(f"実行時間: {assessment['duration_seconds']:.2f}秒")
            print(
                f"総合セキュリティスコア: {assessment['overall_security_score']:.1f}/100"
            )

            print("\nコンポーネント実行状況:")
            for component, status in assessment["components_status"].items():
                result = "✅" if status["available"] else "❌"
                print(f"  {component}: {result}")

            print("\n主要推奨事項:")
            for i, rec in enumerate(assessment["recommendations"][:5], 1):
                print(f"  {i}. {rec}")

            print("\nエグゼクティブサマリー:")
            print(assessment["executive_summary"])

            # クリーンアップ
            await security_manager.cleanup()

        except Exception as e:
            print(f"テスト実行エラー: {e}")
            import traceback

            traceback.print_exc()

        print("\n=== Issue #419 セキュリティ強化プロジェクト完了 ===")
        print("実装されたセキュリティコンポーネント:")
        print("✅ 脆弱性管理システム (vulnerability_manager.py)")
        print("✅ データ保護・暗号化システム (data_protection.py)")
        print("✅ アクセス制御・認証システム (access_control.py)")
        print("✅ セキュリティ設定管理 (security_config.py)")
        print("✅ セキュリティテストフレームワーク (security_test_framework.py)")
        print("✅ 統合セキュリティ管理システム (security_manager.py)")
        print(
            "✅ CI/CD セキュリティスキャンワークフロー (.github/workflows/security-scan.yml)"
        )

    asyncio.run(main())

#!/usr/bin/env python3
"""
セキュリティ管制センター - メインコントローラー
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from .compliance_monitor import ComplianceMonitor
from .dashboard import SecurityDashboard
from .database_manager import DatabaseManager
from .enums import SecurityLevel, ThreatCategory
from .incident_response import IncidentResponseOrchestrator
from .models import SecurityThreat
from .security_scanner import SecurityScanner
from .threat_intelligence import ThreatIntelligenceEngine

try:
    from ..access_control_audit_system import get_access_control_auditor
    from ..dependency_vulnerability_manager import get_vulnerability_manager
    from ..enhanced_data_protection import get_data_protection_manager
    from ..sast_dast_security_testing import get_security_test_coordinator
    from ..secure_coding_enforcer import get_secure_coding_enforcer

    SECURITY_COMPONENTS_AVAILABLE = True
except ImportError:
    SECURITY_COMPONENTS_AVAILABLE = False
    logging.warning("一部セキュリティコンポーネントが利用できません")


class ComprehensiveSecurityControlCenter:
    """包括的セキュリティ管制センター"""

    def __init__(self, db_path: str = "security_control_center.db"):
        self.logger = logging.getLogger(__name__)

        # データベース管理
        self.db_manager = DatabaseManager(db_path)

        # コンポーネント初期化
        self.threat_intelligence = ThreatIntelligenceEngine()
        self.incident_response = IncidentResponseOrchestrator()
        self.compliance_monitor = ComplianceMonitor()
        self.security_scanner = SecurityScanner(self.threat_intelligence)
        self.dashboard = SecurityDashboard()

        # セキュリティコンポーネント統合
        if SECURITY_COMPONENTS_AVAILABLE:
            try:
                self.vulnerability_manager = get_vulnerability_manager()
                self.coding_enforcer = get_secure_coding_enforcer()
                self.data_protection = get_data_protection_manager()
                self.access_auditor = get_access_control_auditor()
                self.security_tester = get_security_test_coordinator()
            except Exception as e:
                self.logger.warning(
                    f"一部セキュリティコンポーネントの初期化に失敗: {e}"
                )
                self.vulnerability_manager = None
                self.coding_enforcer = None
                self.data_protection = None
                self.access_auditor = None
                self.security_tester = None
        else:
            self.vulnerability_manager = None
            self.coding_enforcer = None
            self.data_protection = None
            self.access_auditor = None
            self.security_tester = None

        # インメモリストレージ
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.incidents = {}

        self.logger.info("包括的セキュリティ管制センター初期化完了")

    async def run_comprehensive_security_scan(self) -> Dict[str, Any]:
        """包括的セキュリティスキャン"""
        scan_results = await self.security_scanner.run_comprehensive_scan(
            self.vulnerability_manager, self.coding_enforcer
        )

        # スキャン結果から検出された脅威を登録
        for threat in scan_results.get("threats", []):
            await self.register_threat(threat)
            scan_results["incidents_created"] = []

        # コンプライアンスチェック結果を統合
        try:
            compliance_results = await self.compliance_monitor.check_compliance(
                "PCI_DSS"
            )
            scan_results["compliance_status"] = compliance_results

            # コンプライアンス違反を脅威として登録
            for violation in compliance_results.get("violations", []):
                threat = SecurityThreat(
                    id=f"compliance_{int(time.time())}_{hash(violation['requirement']) % 10000}",
                    title=f"コンプライアンス違反: {violation['requirement']}",
                    description=violation["description"],
                    category=ThreatCategory.COMPLIANCE,
                    severity=SecurityLevel(violation.get("severity", "medium")),
                    source="compliance_monitor",
                    indicators=violation,
                )
                await self.register_threat(threat)
                scan_results["threats_detected"].append(threat.id)

        except Exception as e:
            self.logger.error(f"コンプライアンスチェックエラー: {e}")

        # 推奨事項生成
        scan_results["recommendations"] = (
            await self.dashboard._generate_security_recommendations(
                self.active_threats
            )
        )

        return scan_results

    async def register_threat(self, threat: SecurityThreat) -> str:
        """脅威を登録"""
        try:
            self.active_threats[threat.id] = threat

            # データベースに保存
            await self.db_manager.save_threat(threat)

            # 高リスク脅威の場合、自動的にインシデントを作成
            if threat.severity in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]:
                incident = await self.incident_response.handle_threat(threat)
                await self.register_incident(incident)

            self.logger.info(f"脅威登録: {threat.id} ({threat.severity.value})")
            return threat.id

        except Exception as e:
            self.logger.error(f"脅威登録エラー: {e}")
            raise

    async def register_incident(self, incident) -> str:
        """インシデントを登録"""
        try:
            self.incidents[incident.id] = incident

            # データベースに保存
            await self.db_manager.save_incident(incident)

            self.logger.info(f"インシデント登録: {incident.id}")
            return incident.id

        except Exception as e:
            self.logger.error(f"インシデント登録エラー: {e}")
            raise

    async def get_security_dashboard(self) -> Dict[str, Any]:
        """セキュリティダッシュボードデータ取得"""
        component_status = await self.dashboard.get_component_status(
            self.vulnerability_manager,
            self.coding_enforcer,
            self.data_protection,
            self.access_auditor,
            self.security_tester,
        )

        return await self.dashboard.get_dashboard_data(
            self.active_threats, self.incidents, component_status
        )

    async def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # 期限切れの脅威を削除
            current_time = datetime.now(timezone.utc)
            expired_threat_ids = await self.db_manager.cleanup_expired_threats(
                current_time
            )

            for threat_id in expired_threat_ids:
                if threat_id in self.active_threats:
                    del self.active_threats[threat_id]

            self.logger.info(f"期限切れ脅威を削除: {len(expired_threat_ids)}件")

        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")


# グローバルインスタンス
_security_control_center = None


def get_security_control_center() -> ComprehensiveSecurityControlCenter:
    """グローバルセキュリティ管制センターを取得"""
    global _security_control_center
    if _security_control_center is None:
        _security_control_center = ComprehensiveSecurityControlCenter()
    return _security_control_center


if __name__ == "__main__":
    # テスト実行
    async def test_security_control_center():
        print("=== 包括的セキュリティ管制センターテスト ===")

        try:
            # セキュリティ管制センター初期化
            control_center = ComprehensiveSecurityControlCenter()

            print("\n1. セキュリティ管制センター初期化完了")

            # 包括的セキュリティスキャン実行
            print("\n2. 包括的セキュリティスキャン実行中...")
            scan_results = await control_center.run_comprehensive_security_scan()

            print("   スキャン結果:")
            print(f"   - スキャンID: {scan_results['scan_id']}")
            print(f"   - 成功: {scan_results['scan_successful']}")
            print(f"   - スキャン時間: {scan_results.get('scan_duration', 0):.2f}秒")
            print(
                f"   - スキャンコンポーネント: {len(scan_results['components_scanned'])}"
            )
            print(f"   - 検出脅威数: {len(scan_results['threats_detected'])}")

            # セキュリティダッシュボード取得
            print("\n3. セキュリティダッシュボード取得...")
            dashboard = await control_center.get_security_dashboard()

            if "error" not in dashboard:
                print(f"   システム: {dashboard['overview']['system_name']}")
                print(f"   ステータス: {dashboard['overview']['status']}")

                threat_summary = dashboard["threat_summary"]
                print("   脅威サマリー:")
                print(f"   - 総脅威数: {threat_summary['total']}")

                metrics = dashboard["security_metrics"]
                print("   セキュリティメトリクス:")
                print(f"   - セキュリティスコア: {metrics.security_score:.1f}/100")
            else:
                print(f"   エラー: {dashboard['error']}")

            # テスト脅威登録
            print("\n4. テスト脅威登録...")
            test_threat = SecurityThreat(
                id="test_threat_001",
                title="テスト脅威",
                description="セキュリティ管制センターのテスト用脅威",
                category=ThreatCategory.VULNERABILITY,
                severity=SecurityLevel.MEDIUM,
                source="test_system",
                indicators={"test": True},
            )

            threat_id = await control_center.register_threat(test_threat)
            print(f"   登録された脅威ID: {threat_id}")

            # クリーンアップ
            print("\n5. クリーンアップ実行...")
            await control_center.cleanup()

            print("\n[成功] 包括的セキュリティ管制センターテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_security_control_center())
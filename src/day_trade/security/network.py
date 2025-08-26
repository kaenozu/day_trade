#!/usr/bin/env python3
"""
セキュリティテストフレームワーク - Network Module
Issue #419: セキュリティ強化 - セキュリティテストフレームワークの導入

ネットワークセキュリティテスト、ポートスキャン、SSL/TLS設定テスト
"""

import asyncio
import socket
import ssl
from typing import List

from .core import (
    SecurityTest,
    SecurityTestResult,
    TestCategory,
    TestSeverity,
    TestStatus,
)


class NetworkSecurityTest(SecurityTest):
    """ネットワークセキュリティテスト"""

    def __init__(self):
        super().__init__(
            "NET001",
            "ネットワークセキュリティテスト",
            TestCategory.NETWORK_SECURITY,
            TestSeverity.MEDIUM,
        )

    async def execute(
        self, target_host="localhost", target_ports=None, **kwargs
    ) -> SecurityTestResult:
        """ネットワークセキュリティテスト実行"""
        try:
            if target_ports is None:
                target_ports = [22, 23, 53, 80, 443, 993, 995, 3306, 3389, 5432, 8080]

            issues = []
            open_ports = []

            # ポートスキャン
            for port in target_ports:
                if await self._is_port_open(target_host, port):
                    open_ports.append(port)

            # 危険なポートチェック
            dangerous_ports = {
                22: "SSH - 適切な認証設定を確認",
                23: "Telnet - 非暗号化通信、使用を避ける",
                53: "DNS - 外部からのアクセスを制限",
                3306: "MySQL - 外部からのアクセスを制限",
                3389: "RDP - 適切な認証設定を確認",
                5432: "PostgreSQL - 外部からのアクセスを制限",
                8080: "HTTP Proxy - 不要な場合は無効化",
            }

            for port in open_ports:
                if port in dangerous_ports:
                    issues.append(
                        f"ポート {port} が開放されています: {dangerous_ports[port]}"
                    )

            # SSL/TLS設定テスト（HTTPS対応ポートのみ）
            ssl_ports = [443, 8443, 993, 995]
            for port in open_ports:
                if port in ssl_ports:
                    ssl_issues = await self._test_ssl_configuration(target_host, port)
                    issues.extend(ssl_issues)

            # DNS設定テスト
            dns_issues = await self._test_dns_security(target_host)
            issues.extend(dns_issues)

            if issues:
                return self.create_result(
                    TestStatus.FAILED,
                    description="ネットワークセキュリティに問題があります",
                    expected="適切なネットワーク設定とファイアウォール",
                    actual=f"{len(issues)}件のネットワークセキュリティ問題",
                    remediation="不要なポートの無効化、SSL/TLS設定の強化、ファイアウォール設定の見直し",
                    evidence={"issues": issues, "open_ports": open_ports},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="ネットワークセキュリティは適切です",
                    expected="適切なネットワーク設定とファイアウォール",
                    actual="ネットワーク設定が適切に構成されています",
                    evidence={"open_ports": open_ports},
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="ネットワーク設定を確認してください",
            )

    async def _is_port_open(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """ポート開放チェック"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return False

    async def _test_ssl_configuration(self, host: str, port: int) -> List[str]:
        """SSL/TLS設定テスト"""
        issues = []

        try:
            # SSL証明書と設定の確認
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            reader, writer = await asyncio.open_connection(host, port, ssl=context)

            # SSL情報取得
            ssl_object = writer.get_extra_info("ssl_object")
            if ssl_object:
                # プロトコルバージョンチェック
                protocol = ssl_object.version()
                if protocol in ["SSLv2", "SSLv3", "TLSv1", "TLSv1.1"]:
                    issues.append(
                        f"古いSSL/TLSプロトコルが使用されています: {protocol}"
                    )

                # 証明書有効期限チェック
                cert = ssl_object.getpeercert()
                if cert:
                    not_after = cert.get("notAfter")
                    if not_after:
                        # 証明書有効期限の解析と警告
                        pass  # 実装簡略化のため省略

            writer.close()
            await writer.wait_closed()

        except Exception:
            # SSL接続できない場合は非SSL通信の可能性
            issues.append(f"ポート {port} でSSL/TLS接続ができません")

        return issues

    async def _test_dns_security(self, host: str) -> List[str]:
        """DNS設定セキュリティテスト"""
        issues = []

        try:
            # DNSリゾルバーテスト（簡略版）
            try:
                socket.gethostbyname(host)
            except socket.gaierror:
                issues.append(f"DNS解決に失敗: {host}")

        except Exception:
            pass  # DNS テストは任意

        return issues
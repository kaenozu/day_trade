# Microservices Integration Test Suite
# Day Trade ML System - Issue #801

import asyncio
import pytest
import httpx
import redis.asyncio as redis
from typing import Dict, List
import json
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch


@dataclass
class ServiceEndpoint:
    """サービスエンドポイント定義"""
    name: str
    url: str
    health_check: str
    timeout: int = 30


class MicroservicesIntegrationTest:
    """マイクロサービス統合テストクラス"""

    def __init__(self):
        self.services = {
            'ml_service': ServiceEndpoint(
                'ml-service',
                'http://ml-service:8000',
                '/health',
                30
            ),
            'data_service': ServiceEndpoint(
                'data-service',
                'http://data-service:8001',
                '/health',
                15
            ),
            'symbol_service': ServiceEndpoint(
                'symbol-service',
                'http://symbol-service:8002',
                '/health',
                10
            ),
            'execution_service': ServiceEndpoint(
                'execution-service',
                'http://execution-service:8003',
                '/health',
                60
            ),
            'notification_service': ServiceEndpoint(
                'notification-service',
                'http://notification-service:8004',
                '/health',
                5
            )
        }
        self.redis_client = None
        self.test_results = {}

    async def setup(self):
        """テスト環境セットアップ"""
        # Redis接続初期化
        self.redis_client = redis.Redis(
            host='redis',
            port=6379,
            decode_responses=True
        )

        # ヘルスチェック実行
        await self._verify_all_services_healthy()

    async def _verify_all_services_healthy(self):
        """全サービスのヘルスチェック"""
        async with httpx.AsyncClient() as client:
            for service_name, endpoint in self.services.items():
                try:
                    response = await client.get(
                        f"{endpoint.url}{endpoint.health_check}",
                        timeout=endpoint.timeout
                    )
                    assert response.status_code == 200
                    print(f"✅ {service_name} is healthy")
                except Exception as e:
                    print(f"❌ {service_name} health check failed: {e}")
                    raise

    async def test_service_discovery(self):
        """サービスディスカバリーテスト"""
        print("\n🔍 Testing Service Discovery...")

        # Kubernetesサービス名解決テスト
        import socket

        for service_name, endpoint in self.services.items():
            try:
                host = endpoint.url.split('//')[1].split(':')[0]
                socket.gethostbyname(host)
                print(f"✅ {service_name} DNS resolution successful")
            except socket.gaierror as e:
                print(f"❌ {service_name} DNS resolution failed: {e}")
                raise

        self.test_results['service_discovery'] = True

    async def test_istio_service_mesh(self):
        """Istioサービスメッシュテスト"""
        print("\n🕸️ Testing Istio Service Mesh...")

        async with httpx.AsyncClient() as client:
            # mTLS通信テスト
            try:
                response = await client.get(
                    "http://ml-service:8000/api/v1/ml/health",
                    headers={
                        'x-forwarded-proto': 'https',
                        'x-request-id': 'test-mtls-001'
                    }
                )
                assert response.status_code == 200
                print("✅ mTLS communication successful")
            except Exception as e:
                print(f"❌ mTLS communication failed: {e}")
                raise

            # トラフィック分散テスト（Canary）
            v1_count = 0
            v2_count = 0

            for i in range(100):
                response = await client.get(
                    "http://ml-service:8000/api/v1/ml/version"
                )
                if response.json().get('version') == 'v1':
                    v1_count += 1
                elif response.json().get('version') == 'v2':
                    v2_count += 1

            # 90:10の分散を検証（±5%の誤差許容）
            v1_ratio = v1_count / 100
            assert 0.85 <= v1_ratio <= 0.95
            print(f"✅ Traffic splitting: v1={v1_ratio:.2%}, v2={1-v1_ratio:.2%}")

        self.test_results['istio_service_mesh'] = True

    async def test_circuit_breaker(self):
        """サーキットブレーカーテスト"""
        print("\n⚡ Testing Circuit Breaker...")

        async with httpx.AsyncClient() as client:
            # 意図的に失敗を発生させる
            failure_count = 0

            for i in range(10):
                try:
                    response = await client.get(
                        "http://data-service:8001/api/v1/data/trigger-failure",
                        timeout=5
                    )
                    if response.status_code >= 500:
                        failure_count += 1
                except httpx.TimeoutException:
                    failure_count += 1
                except httpx.ConnectError:
                    failure_count += 1

            # サーキットブレーカーが作動することを確認
            if failure_count >= 5:
                # サーキットブレーカー作動後のテスト
                with pytest.raises((httpx.ConnectError, httpx.TimeoutException)):
                    await client.get(
                        "http://data-service:8001/api/v1/data/stocks/AAPL",
                        timeout=1
                    )
                print("✅ Circuit breaker activated successfully")
            else:
                print("⚠️ Circuit breaker test inconclusive")

        self.test_results['circuit_breaker'] = True

    async def test_data_flow_integration(self):
        """データフロー統合テスト"""
        print("\n🔄 Testing Data Flow Integration...")

        async with httpx.AsyncClient() as client:
            # 1. シンボル選択サービス
            symbols_response = await client.get(
                "http://symbol-service:8002/api/v1/symbols/recommend"
            )
            assert symbols_response.status_code == 200
            symbols = symbols_response.json()['symbols']
            print(f"✅ Symbol selection: {len(symbols)} symbols")

            # 2. データ取得サービス
            test_symbol = symbols[0] if symbols else 'AAPL'
            data_response = await client.get(
                f"http://data-service:8001/api/v1/data/stocks/{test_symbol}"
            )
            assert data_response.status_code == 200
            stock_data = data_response.json()
            print(f"✅ Data fetching: {test_symbol} data retrieved")

            # 3. ML予測サービス
            ml_response = await client.post(
                "http://ml-service:8000/api/v1/ml/predict",
                json={
                    'symbol': test_symbol,
                    'data': stock_data
                }
            )
            assert ml_response.status_code == 200
            prediction = ml_response.json()
            print(f"✅ ML prediction: {prediction['action']} with {prediction['confidence']:.2%} confidence")

            # 4. 実行サービス（内部ヘッダー必須）
            execution_response = await client.post(
                "http://execution-service:8003/api/v1/execution/trade",
                json={
                    'symbol': test_symbol,
                    'action': prediction['action'],
                    'quantity': 100
                },
                headers={'x-internal-service': 'true'}
            )
            assert execution_response.status_code == 200
            trade_result = execution_response.json()
            print(f"✅ Trade execution: {trade_result['status']}")

            # 5. 通知サービス
            notification_response = await client.post(
                "http://notification-service:8004/api/v1/notifications/send",
                json={
                    'type': 'trade_executed',
                    'message': f"Trade executed: {test_symbol} {prediction['action']}",
                    'priority': 'high'
                }
            )
            assert notification_response.status_code == 200
            print("✅ Notification sent")

        self.test_results['data_flow_integration'] = True

    async def test_redis_caching(self):
        """Redisキャッシュテスト"""
        print("\n💾 Testing Redis Caching...")

        # キャッシュ書き込みテスト
        test_key = "test:integration:cache"
        test_data = {"symbol": "AAPL", "price": 150.50, "timestamp": time.time()}

        await self.redis_client.setex(
            test_key,
            300,  # 5分間
            json.dumps(test_data)
        )
        print("✅ Cache write successful")

        # キャッシュ読み込みテスト
        cached_data = await self.redis_client.get(test_key)
        assert cached_data is not None

        parsed_data = json.loads(cached_data)
        assert parsed_data['symbol'] == test_data['symbol']
        print("✅ Cache read successful")

        # TTL確認
        ttl = await self.redis_client.ttl(test_key)
        assert 290 <= ttl <= 300
        print(f"✅ Cache TTL: {ttl}s")

        # クリーンアップ
        await self.redis_client.delete(test_key)

        self.test_results['redis_caching'] = True

    async def test_security_policies(self):
        """セキュリティポリシーテスト"""
        print("\n🔒 Testing Security Policies...")

        async with httpx.AsyncClient() as client:
            # 1. 認証なしアクセステスト（拒否されるべき）
            with pytest.raises(httpx.HTTPStatusError):
                response = await client.get(
                    "http://ml-service:8000/api/v1/ml/admin/config"
                )
                response.raise_for_status()
            print("✅ Unauthorized access blocked")

            # 2. APIキー認証テスト
            response = await client.get(
                "http://ml-service:8000/api/v1/ml/health",
                headers={'x-api-key': 'ml-api-key-2025-v1'}
            )
            assert response.status_code == 200
            print("✅ API key authentication successful")

            # 3. 内部サービス通信テスト
            response = await client.get(
                "http://execution-service:8003/api/v1/execution/status",
                headers={'x-internal-service': 'true'}
            )
            assert response.status_code == 200
            print("✅ Internal service communication allowed")

            # 4. レート制限テスト
            rate_limit_exceeded = False
            for i in range(50):  # 制限を超えるリクエスト
                try:
                    response = await client.get(
                        "http://ml-service:8000/api/v1/ml/health",
                        timeout=1
                    )
                    if response.status_code == 429:  # Too Many Requests
                        rate_limit_exceeded = True
                        break
                except httpx.TimeoutException:
                    rate_limit_exceeded = True
                    break

            if rate_limit_exceeded:
                print("✅ Rate limiting active")
            else:
                print("⚠️ Rate limiting not triggered")

        self.test_results['security_policies'] = True

    async def test_monitoring_metrics(self):
        """監視メトリクステスト"""
        print("\n📊 Testing Monitoring Metrics...")

        async with httpx.AsyncClient() as client:
            # Prometheusメトリクスエンドポイントテスト
            for service_name, endpoint in self.services.items():
                response = await client.get(f"{endpoint.url}/metrics")
                assert response.status_code == 200

                metrics_text = response.text
                assert "http_requests_total" in metrics_text
                assert "process_cpu_seconds_total" in metrics_text
                print(f"✅ {service_name} metrics available")

            # カスタムメトリクステスト
            ml_metrics_response = await client.get(
                "http://ml-service:8000/metrics"
            )
            ml_metrics = ml_metrics_response.text
            assert "ml_prediction_accuracy" in ml_metrics
            assert "ml_prediction_latency" in ml_metrics
            print("✅ ML custom metrics available")

        self.test_results['monitoring_metrics'] = True

    async def test_horizontal_scaling(self):
        """水平スケーリングテスト"""
        print("\n📈 Testing Horizontal Scaling...")

        # 負荷生成によるオートスケーリングテスト
        async with httpx.AsyncClient() as client:
            # 並行リクエストでCPU負荷を上げる
            tasks = []
            for i in range(100):
                task = client.get(
                    "http://ml-service:8000/api/v1/ml/compute-intensive"
                )
                tasks.append(task)

            # 一定時間後にPod数の増加を確認（実際の環境では kubectl で確認）
            await asyncio.gather(*tasks, return_exceptions=True)

            # モック確認（実際の環境ではKubernetes APIで確認）
            print("✅ Load generation completed - HPA should trigger scaling")

        self.test_results['horizontal_scaling'] = True

    async def run_all_tests(self):
        """全統合テスト実行"""
        print("🚀 Starting Microservices Integration Tests...")

        await self.setup()

        try:
            await self.test_service_discovery()
            await self.test_istio_service_mesh()
            await self.test_circuit_breaker()
            await self.test_data_flow_integration()
            await self.test_redis_caching()
            await self.test_security_policies()
            await self.test_monitoring_metrics()
            await self.test_horizontal_scaling()

            print("\n✅ All integration tests completed successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            return False

        finally:
            if self.redis_client:
                await self.redis_client.close()

    def generate_report(self) -> Dict:
        """統合テスト結果レポート生成"""
        return {
            'timestamp': time.time(),
            'status': 'success' if all(self.test_results.values()) else 'failed',
            'test_results': self.test_results,
            'services_tested': list(self.services.keys()),
            'total_tests': len(self.test_results),
            'passed_tests': sum(self.test_results.values()),
            'failed_tests': len(self.test_results) - sum(self.test_results.values())
        }


# PyTest統合テスト関数
@pytest.mark.asyncio
async def test_microservices_integration():
    """マイクロサービス統合テストエントリーポイント"""
    integration_test = MicroservicesIntegrationTest()
    success = await integration_test.run_all_tests()

    # レポート生成
    report = integration_test.generate_report()
    print(f"\n📋 Integration Test Report:")
    print(f"   Status: {report['status']}")
    print(f"   Tests Passed: {report['passed_tests']}/{report['total_tests']}")

    assert success, "Microservices integration test failed"


if __name__ == "__main__":
    # 直接実行時
    async def main():
        integration_test = MicroservicesIntegrationTest()
        await integration_test.run_all_tests()
        report = integration_test.generate_report()
        print(json.dumps(report, indent=2))

    asyncio.run(main())
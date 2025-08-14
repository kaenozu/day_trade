#!/bin/bash
# Integration Test Execution Script
# Day Trade ML System - Issue #801

set -e

echo "🚀 Day Trade ML System - Microservices Integration Test"
echo "=================================================="

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 関数定義
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 環境変数チェック
check_environment() {
    log_info "Checking environment variables..."

    if [ -z "$SLACK_WEBHOOK_URL" ]; then
        log_warning "SLACK_WEBHOOK_URL not set - notifications will be disabled"
        export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/dummy"
    fi

    log_success "Environment check completed"
}

# Docker環境チェック
check_docker() {
    log_info "Checking Docker environment..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    log_success "Docker environment check completed"
}

# 古いコンテナとネットワークのクリーンアップ
cleanup() {
    log_info "Cleaning up previous test environment..."

    cd "$(dirname "$0")/../deployment"

    # 古いコンテナ停止と削除
    docker-compose -f docker-compose.integration.yml down --volumes --remove-orphans 2>/dev/null || true

    # 古いイメージクリーンアップ（開発用イメージのみ）
    docker image prune -f --filter label=service 2>/dev/null || true

    log_success "Cleanup completed"
}

# サービスビルドと起動
start_services() {
    log_info "Building and starting microservices..."

    cd "$(dirname "$0")/../deployment"

    # サービスビルド
    log_info "Building service images..."
    docker-compose -f docker-compose.integration.yml build --parallel

    # サービス起動（バックグラウンド）
    log_info "Starting services..."
    docker-compose -f docker-compose.integration.yml up -d \
        redis \
        ml-service \
        data-service \
        symbol-service \
        execution-service \
        notification-service \
        prometheus \
        grafana \
        jaeger

    log_success "Services started"
}

# ヘルスチェック
wait_for_services() {
    log_info "Waiting for services to be healthy..."

    cd "$(dirname "$0")/../deployment"

    # サービスヘルスチェック
    services=("redis" "ml-service" "data-service" "symbol-service" "execution-service" "notification-service")

    for service in "${services[@]}"; do
        log_info "Checking $service health..."

        timeout=300  # 5分
        counter=0

        while [ $counter -lt $timeout ]; do
            if docker-compose -f docker-compose.integration.yml ps $service | grep -q "healthy\|Up"; then
                log_success "$service is healthy"
                break
            fi

            if [ $counter -ge $timeout ]; then
                log_error "$service failed to become healthy within $timeout seconds"
                docker-compose -f docker-compose.integration.yml logs $service
                exit 1
            fi

            sleep 2
            counter=$((counter + 2))
        done
    done

    log_success "All services are healthy"
}

# 統合テスト実行
run_integration_tests() {
    log_info "Running integration tests..."

    cd "$(dirname "$0")/../deployment"

    # テスト結果ディレクトリ作成
    mkdir -p ./test_results

    # 統合テスト実行
    if docker-compose -f docker-compose.integration.yml run --rm integration-tests; then
        log_success "Integration tests passed!"
        return 0
    else
        log_error "Integration tests failed!"

        # 失敗時のサービスログ出力
        log_info "Collecting service logs for debugging..."
        for service in ml-service data-service symbol-service execution-service notification-service; do
            echo "=== $service logs ===" >> ./test_results/service_logs.txt
            docker-compose -f docker-compose.integration.yml logs $service >> ./test_results/service_logs.txt 2>&1
            echo "" >> ./test_results/service_logs.txt
        done

        return 1
    fi
}

# テスト結果レポート生成
generate_report() {
    log_info "Generating test report..."

    cd "$(dirname "$0")/../deployment"

    # テスト結果ファイル確認
    if [ -f "./test_results/integration_test_results.xml" ]; then
        log_info "Test results available at: ./test_results/integration_test_results.xml"
    fi

    # サービスステータス確認
    echo "=== Service Status ===" > ./test_results/service_status.txt
    docker-compose -f docker-compose.integration.yml ps >> ./test_results/service_status.txt

    # システムリソース使用状況
    echo "=== Resource Usage ===" >> ./test_results/resource_usage.txt
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" >> ./test_results/resource_usage.txt

    log_success "Test report generated in ./test_results/"
}

# 最終クリーンアップ
final_cleanup() {
    log_info "Performing final cleanup..."

    cd "$(dirname "$0")/../deployment"

    # サービス停止
    docker-compose -f docker-compose.integration.yml down --volumes 2>/dev/null || true

    log_success "Final cleanup completed"
}

# メイン実行フロー
main() {
    echo "Starting integration test execution..."

    # 前処理
    check_environment
    check_docker
    cleanup

    # サービス起動
    start_services
    wait_for_services

    # テスト実行
    test_result=0
    if ! run_integration_tests; then
        test_result=1
    fi

    # 後処理
    generate_report

    # クリーンアップ（オプション）
    if [ "${KEEP_SERVICES:-false}" != "true" ]; then
        final_cleanup
    else
        log_info "Services kept running (KEEP_SERVICES=true)"
        log_info "Access Grafana at: http://localhost:3000 (admin/admin123)"
        log_info "Access Jaeger at: http://localhost:16686"
        log_info "Access Prometheus at: http://localhost:9090"
    fi

    # 最終結果
    if [ $test_result -eq 0 ]; then
        log_success "🎉 Integration tests completed successfully!"
        exit 0
    else
        log_error "❌ Integration tests failed!"
        exit 1
    fi
}

# スクリプト実行
main "$@"
#!/bin/bash
# Day Trade Microservices Kubernetes Deployment Script
# Issue #418: Microservices Architecture

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE_PRODUCTION="trading-production"
NAMESPACE_MONITORING="trading-monitoring"
KUBECTL_CMD="kubectl"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v $KUBECTL_CMD &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    if ! $KUBECTL_CMD cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    print_success "kubectl is available and connected to cluster"
}

# Function to create namespaces
create_namespaces() {
    print_status "Creating namespaces..."
    $KUBECTL_CMD apply -f config/namespaces.yaml
    print_success "Namespaces created"
}

# Function to create secrets
create_secrets() {
    print_status "Creating secrets..."
    print_warning "Please update the secret values in config/secrets.yaml before production deployment"
    $KUBECTL_CMD apply -f config/secrets.yaml
    print_success "Secrets created"
}

# Function to deploy database services (PostgreSQL, Redis, TimescaleDB)
deploy_databases() {
    print_status "Deploying database services..."

    # Check if database manifests exist
    if [ ! -d "databases" ]; then
        print_warning "Database manifests not found. Please deploy PostgreSQL, Redis, and TimescaleDB manually."
        print_warning "Required services:"
        print_warning "  - PostgreSQL (for general data)"
        print_warning "  - TimescaleDB (for time-series market data)"
        print_warning "  - Redis (for caching and messaging)"
        return
    fi

    $KUBECTL_CMD apply -f databases/
    print_success "Database services deployed"
}

# Function to deploy core trading services
deploy_trading_services() {
    print_status "Deploying core trading services..."

    # Deploy services in dependency order
    local services=(
        "market-data-service.yaml"
        "analysis-service.yaml"
        "trading-engine-service.yaml"
        "hft-service.yaml"
    )

    for service in "${services[@]}"; do
        if [ -f "services/$service" ]; then
            print_status "Deploying $service..."
            $KUBECTL_CMD apply -f "services/$service"
            # Wait for deployment to be ready
            service_name=$(basename "$service" .yaml)
            $KUBECTL_CMD -n $NAMESPACE_PRODUCTION rollout status deployment/$service_name --timeout=300s
            print_success "$service deployed successfully"
        else
            print_error "Service file not found: services/$service"
        fi
    done
}

# Function to deploy API Gateway
deploy_api_gateway() {
    print_status "Deploying API Gateway (Kong)..."
    $KUBECTL_CMD apply -f ingress/api-gateway.yaml
    $KUBECTL_CMD -n $NAMESPACE_PRODUCTION rollout status deployment/kong-gateway --timeout=300s
    print_success "API Gateway deployed"
}

# Function to deploy monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack..."

    # Deploy Prometheus
    $KUBECTL_CMD apply -f monitoring/prometheus.yaml
    $KUBECTL_CMD -n $NAMESPACE_MONITORING rollout status deployment/prometheus --timeout=300s

    # Deploy Grafana
    $KUBECTL_CMD apply -f monitoring/grafana.yaml
    $KUBECTL_CMD -n $NAMESPACE_MONITORING rollout status deployment/grafana --timeout=300s

    print_success "Monitoring stack deployed"
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."

    # Check all pods are running
    print_status "Checking pod status in $NAMESPACE_PRODUCTION..."
    $KUBECTL_CMD -n $NAMESPACE_PRODUCTION get pods

    print_status "Checking pod status in $NAMESPACE_MONITORING..."
    $KUBECTL_CMD -n $NAMESPACE_MONITORING get pods

    # Check services
    print_status "Checking services..."
    $KUBECTL_CMD -n $NAMESPACE_PRODUCTION get services

    # Check ingress
    print_status "Checking ingress resources..."
    $KUBECTL_CMD -n $NAMESPACE_PRODUCTION get ingress

    print_success "Deployment verification completed"
}

# Function to show access information
show_access_info() {
    print_status "Getting access information..."

    # Get external IP of API Gateway
    GATEWAY_IP=$($KUBECTL_CMD -n $NAMESPACE_PRODUCTION get service kong-gateway-proxy -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

    if [ "$GATEWAY_IP" = "pending" ] || [ -z "$GATEWAY_IP" ]; then
        GATEWAY_IP=$($KUBECTL_CMD -n $NAMESPACE_PRODUCTION get service kong-gateway-proxy -o jsonpath='{.spec.clusterIP}')
        print_warning "LoadBalancer IP is pending. Using ClusterIP: $GATEWAY_IP"
    fi

    # Get Grafana access information
    GRAFANA_IP=$($KUBECTL_CMD -n $NAMESPACE_MONITORING get service grafana -o jsonpath='{.spec.clusterIP}')

    print_success "Deployment completed successfully!"
    echo
    echo "Access Information:"
    echo "==================="
    echo "API Gateway (Kong):     http://$GATEWAY_IP"
    echo "API Endpoints:"
    echo "  Market Data:          http://$GATEWAY_IP/api/v1/market"
    echo "  Trading:              http://$GATEWAY_IP/api/v1/trading"
    echo "  HFT:                  http://$GATEWAY_IP/api/v1/hft"
    echo "  Analysis:             http://$GATEWAY_IP/api/v1/analysis"
    echo
    echo "Monitoring:"
    echo "  Prometheus:           http://$($KUBECTL_CMD -n $NAMESPACE_MONITORING get service prometheus -o jsonpath='{.spec.clusterIP}'):9090"
    echo "  Grafana:              http://$GRAFANA_IP:3000 (admin/CHANGE_ME_GRAFANA_PASSWORD)"
    echo
    echo "To access services from outside the cluster, consider setting up port forwarding:"
    echo "  kubectl -n $NAMESPACE_MONITORING port-forward svc/grafana 3000:3000"
    echo "  kubectl -n $NAMESPACE_PRODUCTION port-forward svc/kong-gateway-proxy 8080:80"
}

# Function to cleanup deployment
cleanup() {
    if [ "${1:-}" = "cleanup" ]; then
        print_warning "This will delete all trading system resources. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_status "Cleaning up deployment..."

            $KUBECTL_CMD delete namespace $NAMESPACE_PRODUCTION --ignore-not-found=true
            $KUBECTL_CMD delete namespace $NAMESPACE_MONITORING --ignore-not-found=true

            print_success "Cleanup completed"
        else
            print_status "Cleanup cancelled"
        fi
        exit 0
    fi
}

# Function to show usage
usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  deploy      Deploy the complete trading system (default)"
    echo "  services    Deploy only trading services"
    echo "  monitoring  Deploy only monitoring stack"
    echo "  verify      Verify existing deployment"
    echo "  cleanup     Remove all trading system resources"
    echo "  help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0                 # Deploy everything"
    echo "  $0 services        # Deploy only trading services"
    echo "  $0 cleanup         # Remove all resources"
}

# Main deployment function
main() {
    case "${1:-deploy}" in
        "help"|"-h"|"--help")
            usage
            exit 0
            ;;
        "cleanup")
            cleanup "$1"
            ;;
        "verify")
            check_kubectl
            verify_deployment
            show_access_info
            ;;
        "services")
            check_kubectl
            create_namespaces
            create_secrets
            deploy_databases
            deploy_trading_services
            deploy_api_gateway
            verify_deployment
            show_access_info
            ;;
        "monitoring")
            check_kubectl
            create_namespaces
            deploy_monitoring
            verify_deployment
            ;;
        "deploy"|"")
            print_status "Starting Day Trade Microservices deployment..."
            echo

            check_kubectl
            create_namespaces
            create_secrets
            deploy_databases
            deploy_trading_services
            deploy_api_gateway
            deploy_monitoring
            verify_deployment
            show_access_info
            ;;
        *)
            print_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"

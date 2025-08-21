#!/bin/bash
# Day Trade Personal - Deployment Script

set -e

echo "🚀 Starting day-trade-personal deployment..."

# 環境変数確認
if [ -z "$ENVIRONMENT" ]; then
    ENVIRONMENT="production"
fi

echo "Environment: $ENVIRONMENT"
echo "Cloud Provider: VERCEL"

# Docker関連
if command -v docker &> /dev/null; then
    echo "📦 Building Docker image..."
    docker build -t day-trade-personal:1.0.0 .

    echo "🧪 Running tests..."
    docker run --rm day-trade-personal:1.0.0 python -m pytest

    echo "🔍 Security scan..."
    docker run --rm day-trade-personal:1.0.0 python security_assessment.py
fi

# クラウドプロバイダ別デプロイ
case "VERCEL" in
    "HEROKU")
        echo "🌐 Deploying to Heroku..."
        heroku container:push web --app day-trade-personal
        heroku container:release web --app day-trade-personal
        ;;
    "RAILWAY")
        echo "🚄 Deploying to Railway..."
        railway up
        ;;
    "VERCEL")
        echo "▲ Deploying to Vercel..."
        vercel --prod
        ;;
    *)
        echo "⚠️ Manual deployment required for VERCEL"
        ;;
esac

echo "✅ Deployment completed!"
echo "🌍 Application URL: https://daytrade.vercel.example.com"

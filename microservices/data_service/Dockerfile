# Data Management Service Dockerfile
# Day Trade ML System - Microservices Architecture

FROM python:3.11-slim

# メタデータ
LABEL maintainer="Day Trade ML Team"
LABEL service="data-service"
LABEL version="1.0.0"
LABEL description="Data Management Service with Real-time Data Fetching"

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SERVICE_NAME=data-service
ENV SERVICE_PORT=8001

# システム依存関係インストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ設定
WORKDIR /app

# Python依存関係コピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルコピー
COPY . .

# 非rootユーザー作成
RUN groupadd -r dataservice && useradd -r -g dataservice dataservice
RUN chown -R dataservice:dataservice /app
USER dataservice

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ポート公開
EXPOSE 8001

# エントリポイント
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
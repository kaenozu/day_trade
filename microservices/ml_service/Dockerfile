# ML Prediction Service Dockerfile
# Day Trade ML System - Microservices Architecture

FROM python:3.11-slim

# メタデータ
LABEL maintainer="Day Trade ML Team"
LABEL service="ml-service"
LABEL version="1.0.0"
LABEL description="ML Prediction Service with 93% Accuracy EnsembleSystem"

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SERVICE_NAME=ml-service
ENV SERVICE_PORT=8000

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
RUN groupadd -r mlservice && useradd -r -g mlservice mlservice
RUN chown -R mlservice:mlservice /app
USER mlservice

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ポート公開
EXPOSE 8000

# エントリポイント
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
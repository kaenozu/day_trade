# Integration Test Container
# Day Trade ML System - Issue #801

FROM python:3.11-slim

# メタデータ
LABEL maintainer="Day Trade ML Team"
LABEL service="integration-tests"
LABEL version="1.0.0"
LABEL description="Microservices Integration Test Suite"

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# システム依存関係インストール
RUN apt-get update && apt-get install -y \
    curl \
    dnsutils \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ設定
WORKDIR /app

# Python依存関係インストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# テストファイルコピー
COPY . .

# 結果ディレクトリ作成
RUN mkdir -p /app/test_results

# 非rootユーザー作成
RUN groupadd -r tester && useradd -r -g tester tester
RUN chown -R tester:tester /app
USER tester

# エントリポイント
CMD ["python", "-m", "pytest", "test_microservices_integration.py", "-v", "--tb=short", "--junitxml=/app/test_results/integration_test_results.xml"]
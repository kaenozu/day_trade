# Issue #800 Phase 1: MLサービス用Dockerfile
# EnsembleSystem (93%精度) + モデル推論API

FROM python:3.11-slim

# 作業ディレクトリ設定
WORKDIR /app

# システム依存関係インストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係コピー・インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードコピー
COPY . .

# MLモデル・データディレクトリ作成
RUN mkdir -p /app/models /app/data /app/logs

# 非rootユーザー作成・設定
RUN adduser --disabled-password --gecos '' mluser && \
    chown -R mluser:mluser /app
USER mluser

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ポート公開
EXPOSE 8000

# エントリーポイント設定
ENTRYPOINT ["./entrypoint.sh"]

# デフォルトコマンド
CMD ["python", "-m", "src.day_trade.api.ml_service"]
# Issue #800 Phase 1: データサービス用Dockerfile
# DataFetcher + SmartSymbolSelector + リアルタイムデータパイプライン

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

# データ・ログディレクトリ作成
RUN mkdir -p /app/data /app/logs /app/cache

# 非rootユーザー作成・設定
RUN adduser --disabled-password --gecos '' datauser && \
    chown -R datauser:datauser /app
USER datauser

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ポート公開
EXPOSE 8001

# エントリーポイント設定
ENTRYPOINT ["./entrypoint.sh"]

# デフォルトコマンド
CMD ["python", "-m", "src.day_trade.api.data_service"]
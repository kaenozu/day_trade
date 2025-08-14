# Issue #800 Phase 1: スケジューラサービス用Dockerfile
# ExecutionScheduler + タスク管理・監視 + 自動化ワークフロー

FROM python:3.11-slim

# 作業ディレクトリ設定
WORKDIR /app

# システム依存関係インストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係コピー・インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードコピー
COPY . .

# スケジューラ・ログディレクトリ作成
RUN mkdir -p /app/tasks /app/logs /app/schedules

# 非rootユーザー作成・設定
RUN adduser --disabled-password --gecos '' scheduler && \
    chown -R scheduler:scheduler /app
USER scheduler

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# ポート公開
EXPOSE 8002

# エントリーポイント設定
ENTRYPOINT ["./entrypoint.sh"]

# デフォルトコマンド
CMD ["python", "-m", "src.day_trade.api.scheduler_service"]
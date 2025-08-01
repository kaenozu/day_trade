# マルチステージビルドで効率化
FROM python:3.11-slim as builder

# システム依存関係を最小限に
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pythonの最適化
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 依存関係のインストール（キャッシュ効率化）
WORKDIR /app
COPY pyproject.toml requirements.txt* ./
RUN pip install --user --no-cache-dir -r requirements.txt || \
    pip install --user --no-cache-dir -e .

# プロダクション用の軽量イメージ
FROM python:3.11-slim as production

# 非rootユーザーでセキュリティ向上
RUN groupadd -r appuser && useradd -r -g appuser appuser

# システム依存関係（実行時のみ）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python環境設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# ユーザー切り替えとファイルコピー
USER appuser
WORKDIR /app

# ビルド済み依存関係をコピー
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# アプリケーションコードをコピー
COPY --chown=appuser:appuser . .

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import day_trade; print('OK')" || exit 1

# デフォルトコマンド
CMD ["python", "-m", "day_trade"]

# 開発用イメージ
FROM production as development

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# 開発用依存関係
RUN pip install --user --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-mock \
    pre-commit \
    ruff \
    mypy

# 開発用設定
ENV ENVIRONMENT=development

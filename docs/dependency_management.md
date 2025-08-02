# 依存関係管理ガイド

## 概要

Day Tradeプロジェクトでは、モダンなPythonパッケージ管理手法を採用しています。

## 管理手法

### 主要ファイル

- **pyproject.toml**: プロジェクトの設定と依存関係の定義
- **requirements.txt**: 本番用依存関係（pip-toolsで生成）
- **requirements-dev.txt**: 開発用依存関係（legacy）

### 推奨ワークフロー

#### 1. 開発環境のセットアップ

```bash
# プロジェクトをクローン後
make install-dev

# または手動で
pip install -e ".[dev]"
```

#### 2. 依存関係の追加

新しい依存関係は`pyproject.toml`に追加してください：

```toml
# 本番用依存関係
dependencies = [
    "new-package>=1.0.0",
]

# 開発用依存関係
[project.optional-dependencies]
dev = [
    "new-dev-package>=2.0.0",
]
```

#### 3. 依存関係の確認

```bash
# 古いパッケージをチェック
make deps-check

# または直接実行
python scripts/dependency_manager.py check
```

#### 4. 依存関係の更新

```bash
# 更新可能なパッケージをドライラン表示
make deps-update

# 実際に更新
python scripts/dependency_manager.py update
```

#### 5. セキュリティチェック

```bash
# セキュリティ脆弱性チェック
safety check

# 依存関係レポート生成
make deps-report
```

## ツール

### dependency_manager.py

プロジェクト専用の依存関係管理ツールです。

```bash
# 使用可能コマンド
python scripts/dependency_manager.py --help

# レポート生成
python scripts/dependency_manager.py report

# 古いパッケージチェック
python scripts/dependency_manager.py check

# パッケージ更新
python scripts/dependency_manager.py update

# requirements.txt同期
python scripts/dependency_manager.py sync
```

### Makeコマンド

```bash
# 依存関係チェック
make deps-check

# 依存関係更新
make deps-update

# 依存関係レポート
make deps-report

# requirements.txt同期
make deps-sync
```

## ベストプラクティス

### 1. バージョン制約

適切なバージョン制約を使用してください：

```toml
dependencies = [
    "pandas>=2.0.0,<3.0.0",  # メジャーバージョン固定
    "numpy>=1.24.0",         # 最小バージョン指定
    "click~=8.1.0",          # 互換バージョン
]
```

### 2. 定期的な更新

- 月1回程度の頻度で依存関係を確認
- セキュリティアップデートは即座に適用
- メジャーバージョンアップは慎重に検討

### 3. 環境分離

```bash
# 本番用のみ
pip install -e .

# 開発用含む
pip install -e ".[dev]"
```

### 4. CI/CD統合

```yaml
# .github/workflows/dependencies.yml
- name: Check dependencies
  run: |
    make deps-check
    safety check
```

## トラブルシューティング

### よくある問題

#### 1. パッケージインストールエラー

```bash
# pip更新
pip install --upgrade pip

# キャッシュクリア
pip cache purge

# 強制再インストール
pip install --force-reinstall package_name
```

#### 2. 依存関係の競合

```bash
# 依存関係ツリー表示
pip show --files package_name

# 競合チェック
pip check
```

#### 3. セキュリティ脆弱性

```bash
# 脆弱性チェック
safety check

# 特定パッケージ更新
pip install --upgrade vulnerable_package
```

## 参考情報

- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [pip-tools Documentation](https://pip-tools.readthedocs.io/)
- [Safety Documentation](https://pyup.io/safety/)

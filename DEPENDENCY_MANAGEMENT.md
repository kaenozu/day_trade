# 依存関係管理ガイド

このプロジェクトでは、依存関係を以下のファイルで管理しています。

## ファイル構成

### 本番用依存関係

- **`requirements.txt`**: 本番環境で必要な依存関係のみ
- **`pyproject.toml`**: モダンなPython依存関係管理 (本番用は`dependencies`セクション)

### 開発用依存関係

- **`requirements-dev.txt`**: 開発用ライブラリ（テスト、リンター、フォーマッタなど）
- **`pyproject.toml`**: 開発用依存関係 (`project.optional-dependencies.dev`セクション)

## 使用方法

### 本番環境の依存関係インストール

```bash
# requirements.txtを使用
pip install -r requirements.txt

# または pyproject.toml を使用
pip install -e .
```

### 開発環境の依存関係インストール

```bash
# requirements-dev.txt を使用
pip install -r requirements.txt -r requirements-dev.txt

# または pyproject.toml を使用（推奨）
pip install -e .[dev]
```

## 依存関係の追加・更新

### 本番用依存関係を追加する場合

1. `requirements.txt` に追加
2. `pyproject.toml` の `dependencies` セクションに同じバージョンで追加

### 開発用依存関係を追加する場合

1. `requirements-dev.txt` に追加
2. `pyproject.toml` の `project.optional-dependencies.dev` セクションに同じバージョンで追加

### バージョン管理方針

- **最小バージョン指定**: `>=x.y.z` 形式を使用
- **一貫性確保**: `requirements.txt` と `pyproject.toml` のバージョンを同期
- **セキュリティ考慮**: 定期的に依存関係の脆弱性をチェック

## 品質管理

### セキュリティチェック

```bash
# 脆弱性チェック
safety check

# セキュリティ監査
bandit -r src/
```

### 依存関係の最新化

```bash
# 現在のバージョン確認
pip list --outdated

# requirements.txt の更新（pip-tools使用時）
pip-compile --upgrade requirements.in
pip-compile --upgrade requirements-dev.in
```

## CI/CD統合

GitHub Actions では `pip install -e .[dev]` を使用して開発用依存関係を含めてインストールしています：

```yaml
- name: Install dependencies (cached)
  run: |
    python -m pip install --upgrade pip
    pip install -e .[dev]
```

## ベストプラクティス

1. **本番と開発の分離**: 本番環境には不要な開発用ツールをインストールしない
2. **バージョン固定**: 重要な依存関係は適切にバージョン範囲を指定
3. **定期更新**: セキュリティパッチと機能更新を定期的に確認
4. **テスト**: 依存関係更新後は必ずテストを実行
5. **ドキュメント更新**: 重要な依存関係変更時はドキュメントも更新

## トラブルシューティング

### 依存関係の競合が発生した場合

```bash
# 競合の確認
pip check

# 仮想環境の再構築
pip freeze > backup-requirements.txt
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]
```

### インストールエラーが発生した場合

1. Python バージョンの確認 (>=3.8 が必要)
2. pip の最新化: `pip install --upgrade pip`
3. システム依存ライブラリの確認
4. 仮想環境の使用を確認

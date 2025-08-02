# テスト実行とカバレッジガイド

このドキュメントでは、day_tradeプロジェクトのテスト実行とコードカバレッジの計測について説明します。

## テスト環境のセットアップ

### 開発環境の依存関係インストール

```bash
# pyproject.tomlを使用（推奨）
pip install -e .[dev]

# または requirements ファイルを使用
pip install -r requirements.txt -r requirements-dev.txt
```

## テスト実行

### 基本的なテスト実行

```bash
# 全テスト実行
pytest

# 詳細な出力でテスト実行
pytest -v

# 特定のディレクトリのテスト実行
pytest tests/core/
```

### カバレッジ付きテスト実行

```bash
# 基本的なカバレッジレポート
pytest --cov=src/day_trade

# 詳細なカバレッジレポート（不足行表示付き）
pytest --cov=src/day_trade --cov-report=term-missing

# HTMLレポート生成
pytest --cov=src/day_trade --cov-report=html

# 複数形式のレポート生成
pytest --cov=src/day_trade \
       --cov-report=xml \
       --cov-report=html \
       --cov-report=term-missing \
       --cov-report=json
```

## カバレッジレポートの種類

### 1. ターミナルレポート（term-missing）

```bash
pytest --cov=src/day_trade --cov-report=term-missing
```

- コマンドライン上でカバレッジ結果を表示
- カバーされていない行番号も表示
- 素早い確認に最適

### 2. HTMLレポート

```bash
pytest --cov=src/day_trade --cov-report=html
```

- `htmlcov/` ディレクトリに詳細なHTMLレポートを生成
- ブラウザで `htmlcov/index.html` を開いて確認
- ファイルごとの詳細なカバレッジ情報
- カバーされていない行をハイライト表示

### 3. XMLレポート（CI用）

```bash
pytest --cov=src/day_trade --cov-report=xml
```

- `coverage.xml` ファイルを生成
- CI/CDパイプラインやCodecovとの連携に使用

### 4. JSONレポート

```bash
pytest --cov=src/day_trade --cov-report=json
```

- `coverage.json` ファイルを生成
- プログラムからの解析や他ツールとの連携に使用

## カバレッジ設定

### pyproject.toml設定

カバレッジの設定は `pyproject.toml` で管理されています：

```toml
[tool.coverage.run]
source = ["src"]
branch = true  # ブランチカバレッジも計測

[tool.coverage.report]
show_missing = true      # 不足行を表示
precision = 2           # 精度設定
fail_under = 80         # 最小カバレッジしきい値（80%）

[tool.coverage.html]
directory = "htmlcov"    # HTMLレポート出力先

[tool.coverage.xml]
output = "coverage.xml"  # XMLレポート出力先
```

### 除外設定

以下のファイル・行は自動的にカバレッジ計測から除外されます：

- テストファイル（`*/tests/*`、`*/test_*.py`）
- デモファイル（`*/demo_*.py`、`*/example_*.py`）
- デバッグ用コード（`if __name__ == "__main__":`）
- 抽象メソッド（`@abstractmethod`）
- 型チェック用コード（`if TYPE_CHECKING:`）

## カバレッジのベストプラクティス

### 1. カバレッジ目標

- **最小目標**: 80%（プロジェクト設定）
- **推奨目標**: 90%以上
- **重要機能**: 95%以上

### 2. カバレッジの質

カバレッジ率だけでなく、以下の点も重要です：

- **ブランチカバレッジ**: 条件分岐の全パターンをテスト
- **エッジケース**: 境界値や異常系のテスト
- **統合テスト**: 複数コンポーネント間の連携テスト

### 3. カバレッジレポートの活用

```bash
# 1. 開発中の確認
pytest --cov=src/day_trade --cov-report=term-missing

# 2. 詳細分析
pytest --cov=src/day_trade --cov-report=html
# → htmlcov/index.html をブラウザで開く

# 3. CI用レポート生成
pytest --cov=src/day_trade --cov-report=xml --cov-fail-under=80
```

## CI/CDでのカバレッジ

### GitHub Actions

プロジェクトのCIでは自動的に以下が実行されます：

1. **カバレッジ計測**: 全ユニットテスト実行時
2. **レポート生成**: XML、HTML、JSON形式
3. **Codecov連携**: オンラインでのカバレッジ可視化
4. **アーティファクト保存**: 30日間保存

### カバレッジしきい値チェック

```bash
# 80%未満の場合、テストが失敗
pytest --cov=src/day_trade --cov-fail-under=80
```

## トラブルシューティング

### よくある問題

1. **カバレッジが低い場合**
   ```bash
   # どの行がカバーされていないか確認
   pytest --cov=src/day_trade --cov-report=term-missing
   ```

2. **ブランチカバレッジが不足**
   ```bash
   # HTMLレポートで詳細確認
   pytest --cov=src/day_trade --cov-report=html
   # htmlcov/index.html で各ファイルの分岐状況を確認
   ```

3. **特定ファイルを除外したい**
   ```python
   # コード内で除外指定
   def debug_function():  # pragma: no cover
       print("This won't be counted in coverage")
   ```

## 参考リンク

- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Codecov](https://codecov.io/) - オンラインカバレッジレポート

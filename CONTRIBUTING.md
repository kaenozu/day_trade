# 開発・貢献ガイド (CONTRIBUTING.md)

day_tradeプロジェクトへの貢献をありがとうございます！このドキュメントでは、効率的で一貫性のある開発を行うためのガイドラインを説明します。

## 目次

- [開発環境のセットアップ](#開発環境のセットアップ)
- [開発ワークフロー](#開発ワークフロー)
- [コーディング規約](#コーディング規約)
- [テストガイドライン](#テストガイドライン)
- [プルリクエストのガイドライン](#プルリクエストのガイドライン)
- [イシューの報告](#イシューの報告)

## 開発環境のセットアップ

### 必要なソフトウェア

- Python 3.8以上
- Git
- pip

### セットアップ手順

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/kaenozu/day_trade.git
   cd day_trade
   ```

2. **仮想環境の作成（推奨）**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **依存関係のインストール**
   ```bash
   # 推奨方法：pyproject.tomlを使用
   pip install -e .[dev]

   # または、requirementsファイルを使用
   pip install -r requirements.txt -r requirements-dev.txt
   ```

4. **Pre-commitフックのセットアップ**
   ```bash
   pre-commit install
   ```

5. **動作確認**
   ```bash
   # テストの実行
   pytest

   # アプリケーションの起動
   python -m day_trade.cli.main
   ```

## 開発ワークフロー

### GitHub Flow

このプロジェクトはGitHub Flowを採用しています：

1. **mainブランチから新しいブランチを作成**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **開発作業とコミット**
   ```bash
   # 変更をステージング
   git add .

   # コミット（pre-commitフックが自動実行されます）
   git commit -m "feat: 新機能の説明"
   ```

3. **プッシュとプルリクエスト作成**
   ```bash
   git push origin feature/your-feature-name

   # GitHub CLIを使用してプルリクエスト作成（推奨）
   gh pr create --title "feat: 新機能の説明" --body "詳細な説明"
   ```

4. **レビューとマージ**
   - コードレビューを受ける
   - CI/CDパイプラインの通過を確認
   - 承認後、mainブランチにマージ

### ブランチ命名規則

- `feature/feature-name`: 新機能
- `fix/bug-description`: バグ修正
- `docs/documentation-update`: ドキュメント更新
- `refactor/code-improvement`: リファクタリング
- `test/test-addition`: テスト追加

## コーディング規約

### Python コーディングスタイル

- **PEP 8準拠**: Pythonの標準コーディング規約に従う
- **行の長さ**: 最大88文字（Black設定）
- **インポート順序**: isortの規則に従う

### コード品質ツール

以下のツールが自動的に実行されます：

1. **Ruff**: リンターとフォーマッター
   ```bash
   ruff check . --fix
   ruff format .
   ```

2. **Black**: コードフォーマッター
   ```bash
   black .
   ```

3. **MyPy**: 型チェック
   ```bash
   mypy src/
   ```

4. **Bandit**: セキュリティチェック
   ```bash
   bandit -r src/
   ```

### コメントとドキュメント

- **Docstring**: すべての関数、クラス、モジュールにdocstringを記述
- **型ヒント**: すべての関数に型ヒントを追加
- **コメント**: 複雑なロジックには適切なコメントを追加

```python
def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """RSI（Relative Strength Index）を計算します。

    Args:
        prices: 価格データのリスト
        period: 計算期間（デフォルト: 14）

    Returns:
        RSI値のリスト

    Raises:
        ValueError: period が 1 未満または prices が空の場合
    """
    if period < 1:
        raise ValueError("period は 1 以上である必要があります")
    # 実装...
```

## テストガイドライン

### テスト構造

```
tests/
├── __init__.py
├── test_core/
│   ├── test_portfolio.py
│   └── test_trade_manager.py
├── test_analysis/
│   ├── test_indicators.py
│   └── test_signals.py
└── integration/
    └── test_full_workflow.py
```

### テストの種類

1. **ユニットテスト**: 個別の関数・クラスをテスト
2. **統合テスト**: 複数のコンポーネント間の連携をテスト
3. **エンドツーエンドテスト**: アプリケーション全体の動作をテスト

### テストの実行

```bash
# 全テスト実行
pytest

# 詳細出力
pytest -v

# カバレッジ付き実行
pytest --cov=src/day_trade --cov-report=html

# 特定のテストファイル実行
pytest tests/test_portfolio.py

# 特定のテスト関数実行
pytest tests/test_portfolio.py::test_add_position
```

### テストのベストプラクティス

- **AAA パターン**: Arrange, Act, Assert の順序で記述
- **テスト名**: `test_機能_条件_期待結果` の形式
- **モック**: 外部依存をモックで置き換え
- **データ駆動テスト**: pytest.mark.parametrize を活用

```python
import pytest
from unittest.mock import Mock

class TestPortfolio:
    def test_add_position_valid_data_success(self):
        # Arrange
        portfolio = Portfolio()
        position_data = {"symbol": "7203", "quantity": 100, "price": 1500}

        # Act
        result = portfolio.add_position(position_data)

        # Assert
        assert result.success is True
        assert portfolio.get_position("7203").quantity == 100
```

## プルリクエストのガイドライン

### プルリクエストの作成前チェックリスト

- [ ] すべてのテストが通過
- [ ] コードカバレッジが80%以上
- [ ] Pre-commitフックが通過
- [ ] ドキュメントが更新済み
- [ ] CHANGELOG.md が更新済み（重要な変更の場合）

### プルリクエストのタイトル

Conventional Commits形式を使用：

- `feat: 新機能の追加`
- `fix: バグの修正`
- `docs: ドキュメントの更新`
- `style: フォーマットの修正`
- `refactor: リファクタリング`
- `test: テストの追加・修正`
- `chore: その他の変更`

### プルリクエストの説明

```markdown
## 概要
この変更の概要を記述

## 変更内容
- 変更点1
- 変更点2

## テスト方法
1. 手順1
2. 手順2

## 関連Issue
Closes #123
```

### レビュープロセス

1. **自己レビュー**: プルリクエスト作成前に自分でコードをレビュー
2. **ピアレビュー**: 他の開発者からのレビューを受ける
3. **CI/CDチェック**: 自動化されたチェックが通過することを確認
4. **承認**: レビュアーの承認を得る
5. **マージ**: mainブランチにマージ

## イシューの報告

### バグ報告

```markdown
## バグの概要
簡潔な説明

## 再現手順
1. 手順1
2. 手順2
3. エラーが発生

## 期待される動作
正常な場合の動作

## 実際の動作
実際に発生した動作

## 環境
- OS: Windows 10
- Python: 3.11
- day_trade: v1.0.0
```

### 機能要求

```markdown
## 機能の概要
実装したい機能の説明

## 背景・理由
なぜこの機能が必要なのか

## 提案する解決策
具体的な実装アイデア

## 代替案
他に考えられる解決策
```

## CI/CD パイプライン

### 自動化されたチェック

1. **コード品質チェック**
   - Ruff (リンター・フォーマッター)
   - MyPy (型チェック)
   - Bandit (セキュリティ)

2. **テスト実行**
   - ユニットテスト
   - 統合テスト
   - カバレッジ計測

3. **ビルドテスト**
   - パッケージビルド
   - 依存関係チェック

### CIのトラブルシューティング

CIが失敗した場合：

1. **ローカルでの確認**
   ```bash
   # Pre-commitフックの実行
   pre-commit run --all-files

   # テストの実行
   pytest
   ```

2. **ログの確認**
   - GitHub Actionsのログでエラーメッセージを確認
   - 必要に応じて修正してプッシュ

## 質問・サポート

- **GitHub Issues**: バグ報告や機能要求
- **GitHub Discussions**: 一般的な質問や議論
- **Pull Requests**: コードレビューや実装に関する議論

## 参考リンク

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

このプロジェクトに貢献していただき、ありがとうございます！何か質問があれば、遠慮なくIssueを作成してください。

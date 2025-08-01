# プルリクエストコンフリクト検知システム

このドキュメントでは、CI上でプルリクエストのマージコンフリクトを自動検知するシステムについて説明します。

## 概要

GitHub Actionsを使用して、プルリクエストが作成・更新された際に自動的にmainブランチとのマージコンフリクトを検知し、開発者に通知するシステムです。

## 機能

### 🔍 自動コンフリクト検知
- プルリクエスト作成時・更新時に自動実行
- mainブランチとの試行マージによるコンフリクト検知
- コンフリクトファイルの特定と報告

### 📝 自動コメント機能
- コンフリクト検知時に詳細なコメントを自動投稿
- 解決手順の具体的な説明
- 次のステップのチェックリスト

### 🎯 ステータスチェック
- GitHubのステータスチェック機能との連携
- コンフリクト有無によるPRステータス更新
- CI/CDパイプラインとの統合

## ワークフローの動作

### トリガー条件
```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
  workflow_dispatch:
```

- プルリクエストの作成 (`opened`)
- プルリクエストの更新 (`synchronize`)
- プルリクエストの再オープン (`reopened`)
- 手動実行 (`workflow_dispatch`)

### 検知プロセス

1. **リポジトリのクローン**
   ```bash
   git checkout@v4 with fetch-depth: 0
   ```

2. **mainブランチの最新化**
   ```bash
   git fetch origin main:main
   ```

3. **試行マージ**
   ```bash
   git checkout -b temp-merge-test
   git merge main --no-commit --no-ff
   ```

4. **結果の判定**
   - 成功: コンフリクトなし ✅
   - 失敗: コンフリクト検知 ⚠️

### コンフリクト解決ガイド

コンフリクトが検知された場合、以下の手順で解決してください：

#### 1. ローカル環境でmainブランチを更新
```bash
git checkout main
git pull origin main
```

#### 2. フィーチャーブランチにmainをマージ
```bash
git checkout <your-feature-branch>
git merge main
```

#### 3. コンフリクトを手動で解決
```bash
# コンフリクトマーカーを確認・編集
# <<<<<<< HEAD
# あなたの変更
# =======
# mainブランチの変更
# >>>>>>> main

# 解決後にステージング
git add .
git commit -m "Resolve merge conflicts with main"
```

#### 4. プルリクエストを更新
```bash
git push origin <your-feature-branch>
```

## 自動生成されるコメント例

```markdown
## ⚠️ Merge Conflicts Detected

このプルリクエストは現在mainブランチとの間でコンフリクトが発生しています。

### 📁 コンフリクトが発生しているファイル:
- `src/example.py`
- `tests/test_example.py`

### 🔧 解決方法:
[詳細な解決手順...]

### 📊 次のステップ:
- [ ] コンフリクトを解決
- [ ] テストが通ることを確認
- [ ] プルリクエストを更新
```

## 設定のカスタマイズ

### ワークフローの編集
`.github/workflows/conflict-detection.yml` を編集することで以下をカスタマイズできます：

- 対象ブランチの変更 (デフォルト: `main`)
- トリガー条件の調整
- コメントメッセージの変更
- 通知方法の追加

### 除外ファイル設定
特定のファイルをコンフリクト検知から除外したい場合：

```yaml
- name: Check for merge conflicts
  run: |
    # 除外ファイルの例
    git checkout -b temp-merge-test
    git merge main --no-commit --no-ff

    # 特定のファイルのコンフリクトを無視
    git reset HEAD -- docs/auto-generated.md
    git checkout -- docs/auto-generated.md
```

## トラブルシューティング

### よくある問題

#### 1. ワークフローが実行されない
- GitHub Actionsが有効になっているか確認
- ブランチ保護ルールの設定を確認
- 権限設定を確認

#### 2. 誤検知が発生する
- `.gitignore` の設定を確認
- バイナリファイルの除外設定
- 自動生成ファイルの処理

#### 3. コメントが投稿されない
- `GITHUB_TOKEN` の権限確認
- リポジトリの権限設定
- Actions の実行権限

## ベストプラクティス

### 開発フロー
1. 定期的にmainブランチから最新変更を取り込む
2. 小さな単位での頻繁なコミット
3. プルリクエスト作成前のローカルでのコンフリクト確認

### チーム運用
1. コンフリクト解決の責任者明確化
2. マージ戦略の統一 (merge/squash/rebase)
3. 定期的なワークフローのレビューと改善

## 関連リンク

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Git Merge Conflicts](https://docs.github.com/en/github/collaborating-with-pull-requests/addressing-merge-conflicts)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests)

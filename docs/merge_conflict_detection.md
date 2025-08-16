# マージコンフリクト検知システム

## 概要

Issue #883対応として、CIでマージコンフリクトを事前検知するシステムを実装しました。
このシステムにより、PRのCIチェックが通っていてもマージ時にコンフリクトが発生する問題を解決します。

## 🎯 目的

- **事前検知**: PRがマージ可能かどうかをCI段階で判定
- **詳細分析**: コンフリクトの種類と解決方法を自動分析
- **開発効率向上**: マージ失敗による時間ロスを削減
- **品質向上**: コンフリクト解決の品質と一貫性を改善

## 🛠️ システム構成

### 1. GitHub Actions ワークフロー

**ファイル**: `.github/workflows/merge-conflict-check.yml`

**トリガー**:
- Pull Request の作成・更新時
- 対象ブランチ: `main`, `develop`

**主要機能**:
- テストマージの実行
- コンフリクトファイルの特定
- 詳細なコンフリクト分析
- PR へのコメント自動投稿
- GitHub Check Run の作成

### 2. コンフリクト分析スクリプト

**ファイル**: `scripts/conflict_analyzer.py`

**機能**:
- ファイル別コンフリクト分析
- 解決難易度の評価
- 解決時間の見積もり
- 解決方法の提案
- 分析結果のレポート生成

## 🚀 使用方法

### 自動実行（推奨）

PRを作成または更新すると、自動的にコンフリクト検知が実行されます：

1. **PRを作成/更新**
2. **CI実行**: マージコンフリクトチェックが自動実行
3. **結果確認**:
   - ✅ コンフリクトなし → 自動マージ可能
   - ❌ コンフリクト検知 → 詳細なコメントがPRに投稿

### 手動実行

ローカルでコンフリクト分析を実行：

```bash
# 基本実行（現在のブランチ vs main）
python scripts/conflict_analyzer.py

# 特定のブランチを指定
python scripts/conflict_analyzer.py --target main --source feature/my-branch

# JSON形式で出力
python scripts/conflict_analyzer.py --format json --output analysis.json

# Markdownレポート生成
python scripts/conflict_analyzer.py --format markdown --output report.md
```

## 📊 分析内容

### コンフリクト分類

1. **Python関連**:
   - `python_import`: インポート文の競合
   - `python_function`: 関数定義の競合
   - `python_class`: クラス定義の競合
   - `python_general`: その他のPythonコード競合

2. **設定ファイル**:
   - `yaml_config`: YAML設定ファイルの競合
   - `json_config`: JSON設定ファイルの競合
   - `dependencies`: 依存関係ファイルの競合

3. **ドキュメント**:
   - `documentation`: MarkdownやREADMEの競合
   - `text`: テキストファイルの競合

### 難易度評価

- **Easy**: 1-2箇所の単純な競合、高い類似度
- **Medium**: 3-5箇所の競合、中程度の類似度
- **Hard**: 多数の競合、低い類似度、複雑な変更

### 解決時間見積もり

- **5-10分**: 簡単な競合
- **15-30分**: 中程度の競合
- **30-60分**: 困難な競合
- **1時間以上**: 非常に複雑な競合

## 🔧 コンフリクト解決ガイド

### 基本的な解決手順

1. **最新状態の取得**:
   ```bash
   git fetch origin main
   ```

2. **PRブランチで作業**:
   ```bash
   git checkout your-feature-branch
   git merge origin/main
   ```

3. **コンフリクト解決**:
   - エディタでコンフリクトマーカーを探す
   - `<<<<<<<`, `=======`, `>>>>>>>` を適切に解決
   - 機能とロジックの整合性を確認

4. **解決確認**:
   ```bash
   git add .
   git commit -m "Resolve merge conflicts with main"
   git push
   ```

### ファイルタイプ別の解決方法

#### Pythonファイル
- **インポート競合**: 重複削除、順序統一
- **関数競合**: 機能の重複確認、統合または分離
- **クラス競合**: 継承関係、メソッドの整合性確認
- **構文確認**: 解決後に `python -m py_compile file.py` で確認

#### YAML/JSON設定
- **構文確認**: YAML/JSONパーサーで検証
- **設定値**: ビジネスロジックに基づく優先度決定
- **キー重複**: 統合または名前変更

#### ドキュメント
- **内容統合**: 情報の重複排除
- **形式統一**: マークダウン記法の一貫性
- **リンク確認**: 内部リンクの整合性

## 📈 システム統合

### 既存CIとの連携

本システムは既存のCI要素と連携します：

1. **Pre-commit Checks** (`pre-commit.yml`)
   - コード品質チェック後にコンフリクト検知
   - 修正内容の整合性確認

2. **Personal CI** (`personal-ci.yml`)
   - 基本テスト後のマージ検証
   - システム動作確認

### 通知システム

- **GitHub Check Run**: PR画面での状態表示
- **PR コメント**: 詳細な分析結果と解決手順
- **アーティファクト**: 分析レポートの保存

## 🎛️ 設定とカスタマイズ

### ワークフロー設定

`.github/workflows/merge-conflict-check.yml` の主要設定：

```yaml
# 対象ブランチの変更
on:
  pull_request:
    branches: [ main, develop, staging ]  # 追加可能

# タイムアウト設定
jobs:
  conflict-detection:
    timeout-minutes: 10  # 調整可能
```

### 分析スクリプト設定

`scripts/conflict_analyzer.py` のカスタマイズポイント：

- **難易度評価ロジック**: `_assess_file_difficulty()` メソッド
- **解決時間見積もり**: `_estimate_resolution_time()` メソッド
- **ファイルタイプ判定**: `_determine_conflict_type()` メソッド

## 🚨 トラブルシューティング

### よくある問題

1. **ワークフローが実行されない**
   - ブランチ保護設定の確認
   - 権限設定（`contents: read`, `pull-requests: write`）の確認

2. **Git操作エラー**
   - リポジトリのクローン状態確認
   - ブランチの存在確認

3. **分析結果が不正確**
   - ファイルエンコーディングの確認
   - バイナリファイルの除外

### デバッグ方法

```bash
# ローカルでの詳細ログ
python scripts/conflict_analyzer.py --verbose

# Git状態の確認
git status
git branch -v
git log --oneline -10
```

## 📋 今後の改善予定

### Phase 2 機能

1. **予防機能**:
   - 定期的なブランチ状態チェック
   - コンフリクト発生予測
   - 自動リベース提案

2. **自動解決**:
   - 単純なコンフリクトの自動解決
   - 安全な自動マージ
   - 設定ファイルの自動統合

3. **分析向上**:
   - 機械学習による解決難易度予測
   - 過去のコンフリクトパターン学習
   - 開発者別の解決パターン分析

### 統計とメトリクス

- コンフリクト発生頻度の追跡
- 解決時間の測定
- 再発パターンの分析
- 開発効率への影響測定

## 🤝 貢献ガイド

### 機能追加

1. **新しいファイルタイプ対応**:
   - `_determine_conflict_type()` に追加
   - 対応する解決方法を `_suggest_resolution()` に追加

2. **分析ロジック改善**:
   - 難易度評価アルゴリズムの改善
   - 解決時間見積もりの精度向上

3. **通知機能拡張**:
   - Slack/Discord通知
   - メール通知
   - カスタムWebhook

### テスト

```bash
# 分析スクリプトのテスト
python -m pytest tests/test_conflict_analyzer.py

# ワークフローのテスト
# 実際のPRを作成してテスト
```

## 📚 関連ドキュメント

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Git Merge Conflict Resolution](https://git-scm.com/docs/git-merge)
- [Pre-commit Hooks Setup](docs/pre_commit_setup.md)

---

**最終更新**: 2025-08-16
**担当**: Claude Code AI Assistant
**Issue**: #883 - CIでコンフリクト検知する
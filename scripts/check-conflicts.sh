#!/bin/bash

# プルリクエストのコンフリクトをローカルで事前チェックするスクリプト
# Usage: ./scripts/check-conflicts.sh [target-branch]

set -e

# デフォルト設定
TARGET_BRANCH=${1:-main}
CURRENT_BRANCH=$(git branch --show-current)

# カラー設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 プルリクエスト コンフリクト事前チェック${NC}"
echo "======================================"
echo "現在のブランチ: $CURRENT_BRANCH"
echo "対象ブランチ: $TARGET_BRANCH"
echo ""

# 前提条件チェック
check_prerequisites() {
    echo -e "${BLUE}📋 前提条件をチェック中...${NC}"

    # Gitリポジトリかチェック
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}❌ エラー: Gitリポジトリではありません${NC}"
        exit 1
    fi

    # 現在のブランチが対象ブランチでないかチェック
    if [ "$CURRENT_BRANCH" = "$TARGET_BRANCH" ]; then
        echo -e "${RED}❌ エラー: 現在のブランチが対象ブランチと同じです${NC}"
        echo "別のブランチに切り替えてから実行してください"
        exit 1
    fi

    # 未コミットの変更がないかチェック
    if ! git diff-index --quiet HEAD --; then
        echo -e "${YELLOW}⚠️  警告: 未コミットの変更があります${NC}"
        echo "コミットまたはスタッシュしてから実行することを推奨します"
        read -p "続行しますか? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    echo -e "${GREEN}✅ 前提条件OK${NC}"
    echo ""
}

# リモートの最新を取得
fetch_latest() {
    echo -e "${BLUE}🔄 リモートの最新情報を取得中...${NC}"

    if git fetch origin "$TARGET_BRANCH":"$TARGET_BRANCH" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 最新情報を取得しました${NC}"
    else
        echo -e "${YELLOW}⚠️  リモートブランチが見つかりません。ローカルブランチで継続します${NC}"
    fi
    echo ""
}

# コンフリクトチェック
check_conflicts() {
    echo -e "${BLUE}🧪 コンフリクトをテスト中...${NC}"

    # テンポラリブランチ作成
    TEMP_BRANCH="temp-conflict-check-$(date +%s)"
    git checkout -b "$TEMP_BRANCH" > /dev/null 2>&1

    # マージテスト
    if git merge "$TARGET_BRANCH" --no-commit --no-ff > /dev/null 2>&1; then
        echo -e "${GREEN}✅ コンフリクトは検出されませんでした${NC}"
        echo -e "${GREEN}   プルリクエストは問題なくマージできます${NC}"
        CONFLICTS_FOUND=false

        # マージを中止
        git merge --abort > /dev/null 2>&1
    else
        echo -e "${RED}⚠️  マージコンフリクトが検出されました${NC}"
        CONFLICTS_FOUND=true

        # コンフリクトファイルを特定
        CONFLICTED_FILES=$(git diff --name-only --diff-filter=U)

        if [ -n "$CONFLICTED_FILES" ]; then
            echo -e "${RED}📁 コンフリクトが発生しているファイル:${NC}"
            echo "$CONFLICTED_FILES" | while read -r file; do
                echo "   - $file"
            done
        fi

        # マージを中止
        git merge --abort > /dev/null 2>&1
    fi

    # クリーンアップ
    git checkout "$CURRENT_BRANCH" > /dev/null 2>&1
    git branch -D "$TEMP_BRANCH" > /dev/null 2>&1

    echo ""
}

# 解決方法の提示
show_resolution_guide() {
    if [ "$CONFLICTS_FOUND" = true ]; then
        echo -e "${YELLOW}🔧 コンフリクト解決ガイド:${NC}"
        echo "1. 対象ブランチをローカルに更新:"
        echo "   git checkout $TARGET_BRANCH"
        echo "   git pull origin $TARGET_BRANCH"
        echo ""
        echo "2. フィーチャーブランチに戻ってマージ:"
        echo "   git checkout $CURRENT_BRANCH"
        echo "   git merge $TARGET_BRANCH"
        echo ""
        echo "3. コンフリクトを手動で解決:"
        echo "   # エディタでコンフリクトマーカーを編集"
        echo "   git add ."
        echo "   git commit -m \"Resolve merge conflicts with $TARGET_BRANCH\""
        echo ""
        echo "4. プルリクエストを更新:"
        echo "   git push origin $CURRENT_BRANCH"
        echo ""
    else
        echo -e "${GREEN}🎉 あなたのブランチはクリーンです！${NC}"
        echo -e "${GREEN}   安心してプルリクエストを作成できます${NC}"
        echo ""
    fi
}

# 統計情報の表示
show_statistics() {
    echo -e "${BLUE}📊 ブランチ統計:${NC}"

    # コミット数の差分
    AHEAD=$(git rev-list --count "$TARGET_BRANCH".."$CURRENT_BRANCH" 2>/dev/null || echo "0")
    BEHIND=$(git rev-list --count "$CURRENT_BRANCH".."$TARGET_BRANCH" 2>/dev/null || echo "0")

    echo "   あなたのブランチは $TARGET_BRANCH より $AHEAD コミット進んでいます"
    echo "   あなたのブランチは $TARGET_BRANCH より $BEHIND コミット遅れています"

    # ファイル変更統計
    CHANGED_FILES=$(git diff --name-only "$TARGET_BRANCH"..."$CURRENT_BRANCH" | wc -l)
    echo "   変更されたファイル数: $CHANGED_FILES"

    echo ""
}

# メイン実行
main() {
    check_prerequisites
    fetch_latest
    check_conflicts
    show_statistics
    show_resolution_guide

    # 終了コード
    if [ "$CONFLICTS_FOUND" = true ]; then
        echo -e "${RED}❌ コンフリクトが検出されました (終了コード: 1)${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ チェック完了 - コンフリクトなし (終了コード: 0)${NC}"
        exit 0
    fi
}

# スクリプト実行
main "$@"

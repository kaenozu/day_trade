#!/bin/bash
# Day Trade Personal - クイックセットアップスクリプト
# 🚀 3分で93%精度AI株式分析システムを起動

set -e

echo "🚀 Day Trade Personal - クイックセットアップ開始"
echo "========================================"

# システム情報確認
OS=$(uname -s)
ARCH=$(uname -m)
echo "📋 検出されたOS: $OS ($ARCH)"

# Python確認
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "❌ Pythonが見つかりません。Python 3.8以上をインストールしてください。"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "🐍 検出されたPython: $PYTHON_VERSION"

# Python バージョン確認
python_version_check() {
    $PYTHON_CMD -c "
import sys
if sys.version_info < (3, 8):
    print('❌ Python 3.8以上が必要です。現在のバージョン:', sys.version)
    sys.exit(1)
else:
    print('✅ Pythonバージョン確認完了')
"
}

python_version_check

# 仮想環境作成
echo "📦 仮想環境を作成中..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✅ 仮想環境作成完了"
else
    echo "📁 既存の仮想環境を使用"
fi

# 仮想環境アクティベート
echo "🔌 仮想環境をアクティベート中..."
if [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "MSYS"* ]]; then
    # Windows (Git Bash)
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi
echo "✅ 仮想環境アクティベート完了"

# pip アップグレード
echo "⬆️  pipをアップグレード中..."
$PIP_CMD install --upgrade pip --quiet

# 依存関係インストール
echo "📥 依存関係をインストール中..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt --quiet
    echo "✅ 依存関係インストール完了"
else
    echo "⚠️  requirements.txt が見つかりません。手動でインストールしてください。"
fi

# 設定ファイル準備
echo "⚙️  設定ファイルを準備中..."
if [ ! -f "config/settings.json" ] && [ -f "config/settings.json.example" ]; then
    cp config/settings.json.example config/settings.json
    echo "✅ 設定ファイル作成完了"
fi

# データディレクトリ作成
echo "📁 データディレクトリを作成中..."
mkdir -p data logs cache
echo "✅ ディレクトリ作成完了"

# 権限設定（Linux/macOS）
if [[ "$OS" == "Linux" ]] || [[ "$OS" == "Darwin" ]]; then
    chmod +x daytrade.py
    echo "✅ 実行権限設定完了"
fi

echo ""
echo "🎉 セットアップ完了！"
echo "========================================"
echo "🚀 システムを起動するには："
echo ""
if [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "MSYS"* ]]; then
    echo "   python daytrade.py"
else
    echo "   python3 daytrade.py"
fi
echo ""
echo "📱 ブラウザで以下にアクセス："
echo "   http://localhost:5000"
echo ""
echo "🎯 93%精度AI分析をお楽しみください！"
echo ""
echo "📚 その他のコマンド："
echo "   停止: Ctrl+C"
echo "   ヘルプ: $PYTHON_CMD daytrade.py --help"
echo "   トラブルシューティング: cat TROUBLESHOOTING_GUIDE.md"
echo ""

# 自動起動オプション
read -p "🤖 今すぐシステムを起動しますか？ (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 システム起動中..."
    echo "📱 ブラウザで http://localhost:5000 にアクセスしてください"
    echo "🛑 停止するには Ctrl+C を押してください"
    echo ""
    $PYTHON_CMD daytrade.py
fi
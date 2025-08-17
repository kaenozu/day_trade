#!/usr/bin/env python3
"""
実践投入準備状況評価（シンプル版）

明日からの実践投入可能性を評価
"""

import json
from datetime import datetime
from pathlib import Path

print("実践投入準備状況評価")
print("=" * 40)

base_dir = Path(__file__).parent
assessment = {
    'timestamp': datetime.now().isoformat(),
    'scores': {},
    'issues': [],
    'warnings': [],
    'recommendations': []
}

print("1. 基本システム確認")
basic_score = 0

# メインスクリプト確認
if (base_dir / "main.py").exists() or (base_dir / "daytrade.py").exists():
    basic_score += 20
    print("  OK メインスクリプト存在")
else:
    assessment['issues'].append("メインスクリプトなし")
    print("  NG メインスクリプトなし")

# 設定ファイル確認
config_dir = base_dir / "config"
if config_dir.exists() and len(list(config_dir.glob("*.json"))) > 0:
    basic_score += 20
    print("  OK 設定ファイル存在")
else:
    assessment['warnings'].append("設定ファイル不足")
    print("  WARN 設定ファイル要確認")

# 依存関係確認
if (base_dir / "requirements.txt").exists():
    basic_score += 20
    print("  OK requirements.txt存在")
else:
    assessment['warnings'].append("requirements.txt不足")
    print("  WARN requirements.txt不足")

# データベース確認
db_files = list(base_dir.glob("**/*.db"))
if len(db_files) > 0:
    basic_score += 20
    print(f"  OK データベース {len(db_files)}個存在")
else:
    assessment['warnings'].append("データベース要初期化")
    print("  WARN データベース要初期化")

# README確認
if (base_dir / "README.md").exists():
    basic_score += 20
    print("  OK README.md存在")
else:
    assessment['warnings'].append("README.md不足")
    print("  WARN README.md不足")

assessment['scores']['basic'] = basic_score

print("\n2. 機能システム確認")
function_score = 0

# 株価取得機能
stock_files = list(base_dir.glob("**/stock_fetcher*.py"))
if len(stock_files) > 0:
    function_score += 25
    print("  OK 株価取得機能存在")
else:
    assessment['issues'].append("株価取得機能なし")
    print("  NG 株価取得機能なし")

# 分析機能
analysis_dir = base_dir / "src" / "day_trade" / "analysis"
if analysis_dir.exists() and len(list(analysis_dir.glob("*.py"))) > 3:
    function_score += 25
    print("  OK 分析機能充実")
else:
    assessment['warnings'].append("分析機能限定的")
    print("  WARN 分析機能限定的")

# ML機能
ml_dir = base_dir / "src" / "day_trade" / "ml"
if ml_dir.exists() and len(list(ml_dir.glob("*.py"))) > 3:
    function_score += 25
    print("  OK ML機能存在")
else:
    assessment['warnings'].append("ML機能限定的")
    print("  WARN ML機能限定的")

# Webダッシュボード
dashboard_files = list(base_dir.glob("**/dashboard*.py"))
if len(dashboard_files) > 0:
    function_score += 25
    print("  OK Webダッシュボード存在")
else:
    assessment['warnings'].append("Webダッシュボード不足")
    print("  WARN Webダッシュボード不足")

assessment['scores']['function'] = function_score

print("\n3. 安全性確認")
safety_score = 0

# リスク管理
risk_dir = base_dir / "src" / "day_trade" / "risk"
if risk_dir.exists():
    safety_score += 25
    print("  OK リスク管理機能存在")
else:
    assessment['issues'].append("リスク管理機能不足")
    print("  NG リスク管理機能不足")

# エラーハンドリング
error_files = list(base_dir.glob("**/*error*.py"))
if len(error_files) > 0:
    safety_score += 25
    print("  OK エラーハンドリング存在")
else:
    assessment['warnings'].append("エラーハンドリング要改善")
    print("  WARN エラーハンドリング要改善")

# バックアップ
backup_dirs = list(base_dir.glob("**/backup*"))
if len(backup_dirs) > 0:
    safety_score += 25
    print("  OK バックアップ機能存在")
else:
    assessment['warnings'].append("バックアップ機能不足")
    print("  WARN バックアップ機能不足")

# 個人利用制限確認
readme_file = base_dir / "README.md"
personal_use_ok = False
if readme_file.exists():
    content = readme_file.read_text(encoding='utf-8')
    if "個人利用専用" in content or "Personal Use" in content:
        personal_use_ok = True

if personal_use_ok:
    safety_score += 25
    print("  OK 個人利用制限明記")
else:
    assessment['warnings'].append("個人利用制限要明記")
    print("  WARN 個人利用制限要明記")

assessment['scores']['safety'] = safety_score

print("\n4. 運用準備確認")
operation_score = 0

# 監視機能
monitoring_dir = base_dir / "src" / "day_trade" / "monitoring"
if monitoring_dir.exists():
    operation_score += 25
    print("  OK 監視機能存在")
else:
    assessment['warnings'].append("監視機能限定的")
    print("  WARN 監視機能限定的")

# ログ機能
logs_dir = base_dir / "logs"
log_files = list(base_dir.glob("**/*.log"))
if logs_dir.exists() or len(log_files) > 0:
    operation_score += 25
    print("  OK ログ機能存在")
else:
    assessment['warnings'].append("ログ機能要設定")
    print("  WARN ログ機能要設定")

# ドキュメント
doc_files = ['OPERATION_GUIDE.md', 'USER_MANUAL.md']
existing_docs = [doc for doc in doc_files if (base_dir / doc).exists()]
if len(existing_docs) > 0:
    operation_score += 25
    print(f"  OK 運用ドキュメント {len(existing_docs)}個存在")
else:
    assessment['warnings'].append("運用ドキュメント不足")
    print("  WARN 運用ドキュメント不足")

# 自動化機能
automation_dir = base_dir / "src" / "day_trade" / "automation"
if automation_dir.exists():
    operation_score += 25
    print("  OK 自動化機能存在")
else:
    assessment['warnings'].append("自動化機能限定的")
    print("  WARN 自動化機能限定的")

assessment['scores']['operation'] = operation_score

# 総合評価
total_score = sum(assessment['scores'].values()) / 4
assessment['total_score'] = total_score

print("\n" + "=" * 40)
print("評価結果")
print("=" * 40)
print(f"基本システム: {basic_score}/100")
print(f"機能システム: {function_score}/100")
print(f"安全性: {safety_score}/100")
print(f"運用準備: {operation_score}/100")
print(f"総合スコア: {total_score:.0f}/100")

# 投入可否判定
critical_issues = len(assessment['issues'])
warnings = len(assessment['warnings'])

print("\n投入可否判定:")
if critical_issues > 0:
    status = "投入不可"
    assessment['go_live_status'] = 'not_ready'
    print("NG 重要な問題があります")
elif total_score >= 80:
    status = "投入可能"
    assessment['go_live_status'] = 'ready'
    print("OK 実践投入可能です")
elif total_score >= 60:
    status = "条件付き投入可能"
    assessment['go_live_status'] = 'conditional'
    print("CONDITIONAL 条件付きで投入可能")
else:
    status = "投入不可"
    assessment['go_live_status'] = 'not_ready'
    print("NG スコアが不足しています")

# 推奨事項
print("\n推奨事項:")
if assessment['go_live_status'] == 'ready':
    assessment['recommendations'] = [
        "少額でのテスト運用から開始",
        "初回は手動での結果確認を徹底",
        "継続的な精度監視を実施",
        "投資は自己責任で実施"
    ]
elif assessment['go_live_status'] == 'conditional':
    assessment['recommendations'] = [
        "警告事項の対応完了後に開始",
        "非常に少額でのテスト運用必須",
        "手動監視体制の強化必要",
        "投資は自己責任で実施"
    ]
else:
    assessment['recommendations'] = [
        "重要な問題の解決が必要",
        "デモ環境での十分なテストを実施",
        "システムの理解を深めてから再評価",
        "現時点での実践投入は推奨しません"
    ]

for rec in assessment['recommendations']:
    print(f"  - {rec}")

if assessment['issues']:
    print(f"\n重要な問題 ({len(assessment['issues'])}件):")
    for issue in assessment['issues']:
        print(f"  ! {issue}")

if assessment['warnings']:
    print(f"\n警告事項 ({len(assessment['warnings'])}件):")
    for warning in assessment['warnings']:
        print(f"  ? {warning}")

print("\n" + "=" * 40)
print("明日からの実践投入について:")

if assessment['go_live_status'] == 'ready':
    print("OK 準備完了 - 慎重に少額から開始可能")
elif assessment['go_live_status'] == 'conditional':
    print("CONDITIONAL 条件付き可能 - 警告事項対応後")
else:
    print("NG 準備不足 - 問題解決後に再評価")

print("=" * 40)

# 結果保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = base_dir / f"production_assessment_{timestamp}.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(assessment, f, ensure_ascii=False, indent=2)

print(f"詳細結果: {results_file}")
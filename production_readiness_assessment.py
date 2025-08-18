#!/usr/bin/env python3
"""
実践投入準備状況評価 - Production Readiness Assessment

明日からの実践投入可能性を詳細評価
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# システムパス追加
sys.path.append(str(Path(__file__).parent / "src"))

print("🔍 実践投入準備状況評価")
print("=" * 50)


class ProductionReadinessAssessment:
    """実践投入準備状況評価"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.assessment_results = {
            'timestamp': datetime.now().isoformat(),
            'readiness_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'go_live_status': 'not_ready'
        }

    def assess_production_readiness(self) -> Dict:
        """実践投入準備状況評価"""
        try:
            print("1. 基本システム機能評価")
            basic_score = self._assess_basic_functionality()

            print("2. データ取得・分析機能評価")
            data_score = self._assess_data_functionality()

            print("3. AI予測システム評価")
            ai_score = self._assess_ai_functionality()

            print("4. 安全性・リスク管理評価")
            safety_score = self._assess_safety_features()

            print("5. 実運用対応評価")
            operation_score = self._assess_operation_readiness()

            # 総合評価
            scores = [basic_score, data_score, ai_score, safety_score, operation_score]
            self.assessment_results['readiness_score'] = sum(scores) / len(scores)

            self._determine_go_live_status()
            self._generate_recommendations()

            return self.assessment_results

        except Exception as e:
            print(f"評価エラー: {e}")
            self.assessment_results['critical_issues'].append(f"評価システムエラー: {e}")
            return self.assessment_results

    def _assess_basic_functionality(self) -> float:
        """基本システム機能評価"""
        score = 0.0
        max_score = 5.0

        # 1. メインスクリプト存在確認
        main_files = ['main.py', 'daytrade.py']
        main_exists = any((self.base_dir / f).exists() for f in main_files)
        if main_exists:
            score += 1.0
            print("  ✅ メインスクリプト: 存在")
        else:
            self.assessment_results['critical_issues'].append("メインスクリプトが見つかりません")
            print("  ❌ メインスクリプト: 不在")

        # 2. 設定ファイル確認
        config_dir = self.base_dir / "config"
        if config_dir.exists() and len(list(config_dir.glob("*.json"))) > 0:
            score += 1.0
            print("  ✅ 設定ファイル: 適切")
        else:
            self.assessment_results['warnings'].append("設定ファイルの不備")
            print("  ⚠️ 設定ファイル: 要確認")

        # 3. 依存関係ファイル確認
        requirements_file = self.base_dir / "requirements.txt"
        if requirements_file.exists():
            score += 1.0
            print("  ✅ 依存関係: 定義済み")
        else:
            self.assessment_results['warnings'].append("requirements.txtが不在")
            print("  ⚠️ 依存関係: 要確認")

        # 4. データベース初期化確認
        db_files = list(self.base_dir.glob("**/*.db"))
        if len(db_files) > 0:
            score += 1.0
            print(f"  ✅ データベース: {len(db_files)}個存在")
        else:
            self.assessment_results['warnings'].append("データベースファイルが見つかりません")
            print("  ⚠️ データベース: 要初期化")

        # 5. ログシステム確認
        logs_dir = self.base_dir / "logs"
        if logs_dir.exists() or len(list(self.base_dir.glob("**/*.log"))) > 0:
            score += 1.0
            print("  ✅ ログシステム: 準備済み")
        else:
            self.assessment_results['warnings'].append("ログシステムの準備不足")
            print("  ⚠️ ログシステム: 要設定")

        return score / max_score

    def _assess_data_functionality(self) -> float:
        """データ取得・分析機能評価"""
        score = 0.0
        max_score = 4.0

        # 1. 株価データ取得機能
        stock_fetcher_files = list(self.base_dir.glob("**/stock_fetcher*.py"))
        if len(stock_fetcher_files) > 0:
            score += 1.0
            print("  ✅ 株価データ取得: 実装済み")
        else:
            self.assessment_results['critical_issues'].append("株価データ取得機能が見つかりません")
            print("  ❌ 株価データ取得: 未実装")

        # 2. キャッシュシステム
        cache_dirs = [d for d in self.base_dir.glob("*cache*") if d.is_dir()]
        cache_files = list(self.base_dir.glob("**/cache*.py"))
        if len(cache_dirs) > 0 or len(cache_files) > 0:
            score += 1.0
            print("  ✅ キャッシュシステム: 実装済み")
        else:
            self.assessment_results['warnings'].append("キャッシュシステムが不完全")
            print("  ⚠️ キャッシュシステム: 要改善")

        # 3. データ分析機能
        analysis_dir = self.base_dir / "src" / "day_trade" / "analysis"
        if analysis_dir.exists() and len(list(analysis_dir.glob("*.py"))) > 5:
            score += 1.0
            print("  ✅ データ分析: 豊富な機能")
        else:
            self.assessment_results['warnings'].append("データ分析機能が限定的")
            print("  ⚠️ データ分析: 基本機能のみ")

        # 4. リアルタイムデータ対応
        realtime_files = list(self.base_dir.glob("**/realtime*.py"))
        if len(realtime_files) > 0:
            score += 1.0
            print("  ✅ リアルタイム対応: 実装済み")
        else:
            self.assessment_results['warnings'].append("リアルタイムデータ対応が限定的")
            print("  ⚠️ リアルタイム対応: 限定的")

        return score / max_score

    def _assess_ai_functionality(self) -> float:
        """AI予測システム評価"""
        score = 0.0
        max_score = 4.0

        # 1. 機械学習モデル
        ml_dir = self.base_dir / "src" / "day_trade" / "ml"
        if ml_dir.exists() and len(list(ml_dir.glob("*.py"))) > 5:
            score += 1.0
            print("  ✅ ML機能: 包括的実装")
        else:
            self.assessment_results['warnings'].append("機械学習機能が限定的")
            print("  ⚠️ ML機能: 基本実装")

        # 2. アンサンブル学習
        ensemble_files = list(self.base_dir.glob("**/ensemble*.py"))
        if len(ensemble_files) > 0:
            score += 1.0
            print("  ✅ アンサンブル学習: 実装済み")
        else:
            self.assessment_results['warnings'].append("アンサンブル学習が未実装")
            print("  ⚠️ アンサンブル学習: 要実装")

        # 3. モデルキャッシュ・最適化
        model_cache_dirs = [d for d in self.base_dir.glob("*model*cache*") if d.is_dir()]
        if len(model_cache_dirs) > 0:
            score += 1.0
            print("  ✅ モデル最適化: 実装済み")
        else:
            self.assessment_results['warnings'].append("モデル最適化機能が不足")
            print("  ⚠️ モデル最適化: 要改善")

        # 4. 予測精度検証
        validation_files = list(self.base_dir.glob("**/*validation*.py"))
        backtest_files = list(self.base_dir.glob("**/*backtest*.py"))
        if len(validation_files) > 0 or len(backtest_files) > 0:
            score += 1.0
            print("  ✅ 精度検証: 実装済み")
        else:
            self.assessment_results['warnings'].append("予測精度検証機能が不足")
            print("  ⚠️ 精度検証: 要実装")

        return score / max_score

    def _assess_safety_features(self) -> float:
        """安全性・リスク管理評価"""
        score = 0.0
        max_score = 5.0

        # 1. リスク管理システム
        risk_dir = self.base_dir / "src" / "day_trade" / "risk"
        if risk_dir.exists() and len(list(risk_dir.glob("*.py"))) > 2:
            score += 1.0
            print("  ✅ リスク管理: 実装済み")
        else:
            self.assessment_results['critical_issues'].append("リスク管理システムが不十分")
            print("  ❌ リスク管理: 要強化")

        # 2. セキュリティ機能
        security_dir = self.base_dir / "src" / "day_trade" / "security"
        if security_dir.exists():
            score += 1.0
            print("  ✅ セキュリティ: 実装済み")
        else:
            self.assessment_results['warnings'].append("セキュリティ機能が限定的")
            print("  ⚠️ セキュリティ: 要強化")

        # 3. エラーハンドリング
        error_handler_files = list(self.base_dir.glob("**/*error*.py"))
        if len(error_handler_files) > 0:
            score += 1.0
            print("  ✅ エラーハンドリング: 実装済み")
        else:
            self.assessment_results['warnings'].append("エラーハンドリングが不完全")
            print("  ⚠️ エラーハンドリング: 要改善")

        # 4. バックアップ機能
        backup_dirs = list(self.base_dir.glob("**/backup*"))
        if len(backup_dirs) > 0:
            score += 1.0
            print("  ✅ バックアップ: 実装済み")
        else:
            self.assessment_results['warnings'].append("バックアップ機能が不足")
            print("  ⚠️ バックアップ: 要実装")

        # 5. 個人利用制限
        readme_content = ""
        readme_file = self.base_dir / "README.md"
        if readme_file.exists():
            readme_content = readme_file.read_text(encoding='utf-8')

        if "個人利用専用" in readme_content or "Personal Use" in readme_content:
            score += 1.0
            print("  ✅ 利用制限: 明記済み")
        else:
            self.assessment_results['warnings'].append("個人利用制限の明記が不足")
            print("  ⚠️ 利用制限: 要明記")

        return score / max_score

    def _assess_operation_readiness(self) -> float:
        """実運用対応評価"""
        score = 0.0
        max_score = 4.0

        # 1. 監視・アラート機能
        monitoring_dir = self.base_dir / "src" / "day_trade" / "monitoring"
        alert_files = list(self.base_dir.glob("**/*alert*.py"))
        if monitoring_dir.exists() or len(alert_files) > 0:
            score += 1.0
            print("  ✅ 監視・アラート: 実装済み")
        else:
            self.assessment_results['warnings'].append("監視・アラート機能が不足")
            print("  ⚠️ 監視・アラート: 要実装")

        # 2. Webダッシュボード
        dashboard_files = list(self.base_dir.glob("**/dashboard*.py"))
        web_files = list(self.base_dir.glob("**/web*.py"))
        if len(dashboard_files) > 0 or len(web_files) > 0:
            score += 1.0
            print("  ✅ Webダッシュボード: 実装済み")
        else:
            self.assessment_results['warnings'].append("Webダッシュボードが不足")
            print("  ⚠️ Webダッシュボード: 要実装")

        # 3. 運用ドキュメント
        operation_docs = ['OPERATION_GUIDE.md', 'USER_MANUAL.md', 'TROUBLESHOOTING.md']
        existing_docs = [doc for doc in operation_docs if (self.base_dir / doc).exists()]
        if len(existing_docs) > 0:
            score += 1.0
            print(f"  ✅ 運用ドキュメント: {len(existing_docs)}個存在")
        else:
            self.assessment_results['warnings'].append("運用ドキュメントが不足")
            print("  ⚠️ 運用ドキュメント: 要作成")

        # 4. 自動化機能
        automation_dir = self.base_dir / "src" / "day_trade" / "automation"
        if automation_dir.exists() and len(list(automation_dir.glob("*.py"))) > 2:
            score += 1.0
            print("  ✅ 自動化機能: 実装済み")
        else:
            self.assessment_results['warnings'].append("自動化機能が限定的")
            print("  ⚠️ 自動化機能: 要拡張")

        return score / max_score

    def _determine_go_live_status(self):
        """実践投入可否判定"""
        score = self.assessment_results['readiness_score']
        critical_count = len(self.assessment_results['critical_issues'])

        if critical_count > 0:
            self.assessment_results['go_live_status'] = 'not_ready'
        elif score >= 0.8:
            self.assessment_results['go_live_status'] = 'ready'
        elif score >= 0.6:
            self.assessment_results['go_live_status'] = 'conditional_ready'
        else:
            self.assessment_results['go_live_status'] = 'not_ready'

    def _generate_recommendations(self):
        """推奨事項生成"""
        score = self.assessment_results['readiness_score']

        if score >= 0.8:
            self.assessment_results['recommendations'].extend([
                "✅ システムは実践投入準備ができています",
                "📊 少額でのテスト運用から開始することを推奨",
                "⚠️ 初回は手動での結果確認を徹底",
                "📈 継続的な精度監視を実施"
            ])
        elif score >= 0.6:
            self.assessment_results['recommendations'].extend([
                "⚠️ 条件付きで実践投入可能",
                "🔧 警告項目の対応完了後に開始",
                "💰 非常に少額でのテスト運用必須",
                "👁️ 手動監視体制の強化必要"
            ])
        else:
            self.assessment_results['recommendations'].extend([
                "❌ 現時点では実践投入は推奨しません",
                "🛠️ 重要な問題の解決が必要",
                "🧪 デモ環境での十分なテストを実施",
                "📚 システムの理解を深めてから再評価"
            ])

        # 共通推奨事項
        self.assessment_results['recommendations'].extend([
            "💡 投資は自己責任で実施",
            "📖 金融商品取引法の遵守",
            "🔒 個人情報の適切な管理",
            "📋 運用ログの定期的な確認"
        ])


def main():
    """メイン実行"""
    assessor = ProductionReadinessAssessment()
    results = assessor.assess_production_readiness()

    print("\n" + "=" * 50)
    print("🎯 実践投入準備状況評価結果")
    print("=" * 50)

    score = results['readiness_score']
    status = results['go_live_status']

    print(f"📊 総合スコア: {score:.1%}")
    print(f"🚦 投入可否: {status}")

    if results['critical_issues']:
        print(f"\n❌ 重要な問題 ({len(results['critical_issues'])}件):")
        for issue in results['critical_issues']:
            print(f"  • {issue}")

    if results['warnings']:
        print(f"\n⚠️ 警告事項 ({len(results['warnings'])}件):")
        for warning in results['warnings']:
            print(f"  • {warning}")

    print(f"\n💡 推奨事項:")
    for rec in results['recommendations']:
        print(f"  {rec}")

    print("\n" + "=" * 50)

    # 投入可否の最終判定
    if status == 'ready':
        print("✅ 明日からの実践投入が可能です")
        print("💰 少額でのテスト運用から開始してください")
    elif status == 'conditional_ready':
        print("⚠️ 条件付きで実践投入可能です")
        print("🔧 警告事項の対応後に慎重に開始してください")
    else:
        print("❌ 現時点での実践投入は推奨しません")
        print("🛠️ 重要な問題の解決後に再評価してください")

    print("=" * 50)

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent / f"production_readiness_assessment_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"📄 詳細結果: {results_file}")


if __name__ == "__main__":
    main()
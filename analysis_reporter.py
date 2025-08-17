import logging
from typing import List, Dict, Any

class AnalysisReporter:
    """分析結果の表示を担当するクラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def print_basic_result(self, symbol: str, result: Dict[str, Any]):
        """基本分析結果表示"""
        print(f"\n📊 {symbol} 基本分析結果")
        print(f"   価格: {result.get('price', 'N/A')} 円")
        print(f"   変動: {result.get('change', 'N/A')} %")
        print(f"   シグナル: {result.get('signal', 'HOLD')}")
        print(f"   信頼度: {result.get('confidence', 0.7):.1%}")

    def print_detailed_result(self, symbol: str, result: Dict[str, Any]):
        """詳細分析結果表示"""
        self.print_basic_result(symbol, result) # 内部呼び出しも修正

        if 'ml_prediction' in result:
            ml = result['ml_prediction']
            print(f"   ML予測: {ml.get('prediction', 'N/A')}")
            print(f"   ML信頼度: {ml.get('confidence', 0):.1%}")

    def print_daytrading_result(self, symbol: str, result: Dict[str, Any]):
        """デイトレード分析結果表示"""
        self.print_detailed_result(symbol, result) # 内部呼び出しも修正

        print(f"   デイトレードスコア: {result.get('daytrading_score', 0):.1f}")
        print(f"   推奨アクション: {result.get('recommended_action', 'N/A')}")
        print(f"   リスクレベル: {result.get('risk_level', 'N/A')}")

    def print_validation_result(self, symbol: str, result: Dict[str, Any]):
        """検証結果表示"""
        print(f"\n🔍 {symbol} 予測精度検証")
        print(f"   精度: {result.get('accuracy', 0):.1%}")
        print(f"   予測数: {result.get('total_predictions', 0)}")
        print(f"   的中数: {result.get('correct_predictions', 0)}")

    def print_daytrading_summary(self, recommendations: List[Dict[str, Any]]):
        """デイトレード総合推奨表示"""
        print("\n🎯 デイトレード総合推奨")

        strong_buys = [r for r in recommendations if r.get('recommended_action') == '強い買い']
        buys = [r for r in recommendations if r.get('recommended_action') == '買い']

        if strong_buys:
            print("   🔥 強い買い推奨:")
            for rec in strong_buys:
                print(f"      {rec['symbol']} (スコア: {rec.get('daytrading_score', 0):.1f})")

        if buys:
            print("   📈 買い推奨:")
            for rec in buys:
                print(f"      {rec['symbol']} (スコア: {rec.get('daytrading_score', 0):.1f})")

    def print_validation_summary(self, results: List[Dict[str, Any]]):
        """検証サマリー表示"""
        print("\n📈 予測精度サマリー")

        valid_results = [r for r in results if 'accuracy' in r]
        if valid_results:
            avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
            print(f"   平均精度: {avg_accuracy:.1%}")
            print(f"   検証銘柄数: {len(valid_results)}")

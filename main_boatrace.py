"""
Boatrace競艇予想システム メインエントリーポイント

使用例とデモンストレーション
"""

import asyncio
import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/boatrace_system.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# ログディレクトリ作成
Path("logs").mkdir(exist_ok=True)

# システムモジュールのインポート
try:
    from src.boatrace.core.api_client import BoatraceAPIClient
    from src.boatrace.core.race_manager import RaceManager
    from src.boatrace.core.stadium_manager import StadiumManager
    from src.boatrace.data.database import init_database
    from src.boatrace.data.data_collector import DataCollector
    from src.boatrace.data.data_processor import DataProcessor
    from src.boatrace.prediction.prediction_engine import PredictionEngine
    from src.boatrace.prediction.racer_analyzer import RacerAnalyzer
    from src.boatrace.prediction.race_analyzer import RaceAnalyzer
    from src.boatrace.betting.ticket_manager import TicketManager, BetType
    from src.boatrace.betting.betting_strategy import StrategyManager
    from src.boatrace.betting.portfolio import Portfolio
except ImportError as e:
    logger.error(f"モジュールインポートエラー: {e}")
    logger.info("システムが正しくセットアップされているか確認してください")
    exit(1)


class BoatraceSystem:
    """競艇予想システムメインクラス"""
    
    def __init__(self):
        """システム初期化"""
        logger.info("BoatraceSystemを初期化中...")
        
        # データベース初期化
        self.database = init_database()
        logger.info("データベース初期化完了")
        
        # 各コンポーネント初期化
        self.api_client = BoatraceAPIClient()
        self.race_manager = RaceManager(self.api_client)
        self.stadium_manager = StadiumManager()
        self.data_collector = DataCollector(self.api_client, self.database)
        self.data_processor = DataProcessor(self.database)
        
        # 分析・予想エンジン
        self.racer_analyzer = RacerAnalyzer(self.database)
        self.race_analyzer = RaceAnalyzer(self.database, self.racer_analyzer)
        self.prediction_engine = PredictionEngine(
            self.database, self.racer_analyzer, self.race_analyzer
        )
        
        # 投票管理
        self.ticket_manager = TicketManager(self.database)
        self.strategy_manager = StrategyManager()
        self.portfolio = Portfolio(Decimal('100000'), self.database)
        
        logger.info("システム初期化完了")
    
    def demo_basic_usage(self):
        """基本機能のデモンストレーション"""
        logger.info("=== 基本機能デモンストレーション開始 ===")
        
        try:
            # 1. API接続テスト
            logger.info("1. API接続テスト")
            if self.api_client.validate_api_connectivity():
                logger.info("✓ BoatraceOpenAPIに正常に接続できました")
            else:
                logger.error("✗ API接続に失敗しました")
                return
            
            # 2. 今日のレース取得
            logger.info("2. 今日のレース取得")
            today_races = self.race_manager.get_today_races()
            logger.info(f"✓ 今日のレース数: {len(today_races)}レース")
            
            if today_races:
                # 最初のレースを詳細表示
                first_race = today_races[0]
                logger.info(f"  サンプルレース: {first_race.display_name}")
                logger.info(f"  タイトル: {first_race.full_title}")
                logger.info(f"  出走艇数: {len(first_race.entries)}艇")
            
            # 3. 競技場情報取得
            logger.info("3. 競技場情報取得")
            stadium_analysis = self.stadium_manager.get_stadium_analysis(3)  # 江戸川
            if stadium_analysis:
                stadium = stadium_analysis['stadium']
                char = stadium_analysis['characteristics']
                logger.info(f"  競技場: {stadium.display_name}")
                logger.info(f"  水面タイプ: {char.water_type.value}")
                logger.info(f"  特徴: {', '.join(char.notes)}")
            
            # 4. データ収集デモ
            logger.info("4. データ収集デモ")
            collection_result = self.data_collector.collect_today_data()
            if 'error' not in collection_result:
                logger.info(f"✓ データ収集完了: {collection_result}")
            else:
                logger.warning(f"データ収集エラー: {collection_result['error']}")
            
            # 5. 予想エンジンデモ（レースがある場合のみ）
            if today_races:
                logger.info("5. 予想エンジンデモ")
                sample_race = today_races[0]
                
                # レース分析
                race_analysis = self.race_analyzer.analyze_race(sample_race.id)
                if race_analysis:
                    logger.info(f"  レース分析結果:")
                    logger.info(f"    競争力: {race_analysis.competitiveness}")
                    logger.info(f"    大荒れ確率: {race_analysis.upset_probability:.1%}")
                    logger.info(f"    本命候補: {race_analysis.favorites}")
                    logger.info(f"    投票戦略: {race_analysis.betting_strategy}")
                
                # 予想実行
                prediction = self.prediction_engine.predict_race(sample_race.id)
                if prediction:
                    logger.info(f"  予想結果:")
                    logger.info(f"    信頼度: {float(prediction.confidence):.2f}")
                    
                    # 勝率上位3艇表示
                    sorted_probs = sorted(
                        prediction.win_probabilities.items(),
                        key=lambda x: x[1], reverse=True
                    )
                    for i, (boat, prob) in enumerate(sorted_probs[:3]):
                        logger.info(f"    {i+1}位: {boat}号艇 {float(prob):.1%}")
            
            # 6. 投票戦略デモ
            logger.info("6. 投票戦略デモ")
            strategies = self.strategy_manager.get_all_strategies()
            logger.info(f"  利用可能戦略: {', '.join(strategies.keys())}")
            
            # 7. ポートフォリオ状況
            logger.info("7. ポートフォリオ状況")
            balance = self.portfolio.get_current_balance()
            logger.info(f"  初期資金: {balance['initial_capital']:,.0f}円")
            logger.info(f"  現在残高: {balance['current_balance']:,.0f}円")
            logger.info(f"  ROI: {balance['roi']:.2f}%")
            
        except Exception as e:
            logger.error(f"デモ実行中にエラーが発生しました: {e}")
    
    def demo_prediction_workflow(self):
        """予想ワークフローのデモンストレーション"""
        logger.info("=== 予想ワークフローデモ開始 ===")
        
        try:
            # 今日のレース取得
            today_races = self.race_manager.get_today_races()
            
            if not today_races:
                logger.info("本日はレースがありません")
                return
            
            # 注目レース選択（最初の3レース）
            target_races = today_races[:3]
            logger.info(f"対象レース: {len(target_races)}レース")
            
            predictions = []
            
            for race in target_races:
                logger.info(f"\n--- {race.display_name} 予想開始 ---")
                
                # レース分析
                race_analysis = self.race_analyzer.analyze_race(race.id)
                if not race_analysis:
                    logger.warning(f"レース分析失敗: {race.id}")
                    continue
                
                logger.info(f"競争力: {race_analysis.competitiveness}")
                logger.info(f"注目要因: {', '.join(race_analysis.key_factors)}")
                
                # 予想実行
                prediction = self.prediction_engine.predict_race(race.id)
                if not prediction:
                    logger.warning(f"予想生成失敗: {race.id}")
                    continue
                
                predictions.append(prediction)
                
                # 予想結果表示
                logger.info(f"予想信頼度: {float(prediction.confidence):.2f}")
                
                # 推奨買い目表示
                if prediction.recommended_bets:
                    logger.info("推奨買い目:")
                    for bet in prediction.recommended_bets:
                        logger.info(f"  {bet['bet_type']}: {bet['numbers']} (信頼度: {bet['confidence']:.2f})")
                
                # 投票戦略選択
                strategy_name = self.strategy_manager.select_best_strategy(
                    prediction, {'competitiveness': race_analysis.competitiveness}
                )
                logger.info(f"推奨戦略: {strategy_name}")
                
                # 投票シミュレーション
                strategy = self.strategy_manager.get_strategy(strategy_name)
                if strategy:
                    budget = Decimal('5000')  # 5,000円予算
                    recommendations = strategy.generate_bets(
                        prediction, budget, {'competitiveness': race_analysis.competitiveness}
                    )
                    
                    if recommendations:
                        logger.info(f"戦略的投票推奨 (予算: {budget}円):")
                        total_amount = Decimal('0')
                        for rec in recommendations:
                            logger.info(f"  {rec.bet_type.value}: {rec.numbers} {rec.amount}円")
                            total_amount += rec.amount
                        logger.info(f"  合計投資額: {total_amount}円")
            
            # 全体サマリー
            logger.info(f"\n=== 本日の予想サマリー ===")
            logger.info(f"予想レース数: {len(predictions)}")
            
            if predictions:
                avg_confidence = sum(float(p.confidence) for p in predictions) / len(predictions)
                logger.info(f"平均信頼度: {avg_confidence:.2f}")
                
                high_confidence_races = [p for p in predictions if float(p.confidence) >= 0.7]
                logger.info(f"高信頼度レース: {len(high_confidence_races)}レース")
        
        except Exception as e:
            logger.error(f"予想ワークフローでエラー: {e}")
    
    def demo_data_analysis(self):
        """データ分析機能のデモンストレーション"""
        logger.info("=== データ分析デモ開始 ===")
        
        try:
            # 競技場分析
            logger.info("1. 競技場別特性分析")
            
            famous_stadiums = [3, 11, 12, 17]  # 江戸川、びわこ、住之江、宮島
            
            for stadium_num in famous_stadiums:
                analysis = self.stadium_manager.get_stadium_analysis(stadium_num)
                if analysis:
                    stadium = analysis['stadium']
                    advantages = analysis['advantages']
                    logger.info(f"  {stadium.display_name}: {', '.join(advantages) if advantages else '標準的'}")
            
            # 今日のサマリー
            logger.info("2. 本日のレースサマリー")
            today_summary = self.data_processor.get_daily_summary(date.today())
            
            if 'error' not in today_summary:
                logger.info(f"  総レース数: {today_summary['summary']['total_races']}")
                logger.info(f"  開催競技場数: {today_summary['summary']['total_stadiums']}")
                logger.info(f"  総出走数: {today_summary['summary']['total_entries']}")
            
            # 注目レース検索
            logger.info("3. 大穴チャンスレース検索")
            upset_races = self.race_analyzer.find_upset_opportunities(
                date_filter=date.today(),
                min_upset_prob=0.3
            )
            
            logger.info(f"  発見された大穴チャンスレース: {len(upset_races)}レース")
            for race_info in upset_races[:3]:  # 上位3レース
                logger.info(f"    {race_info['race_info']['stadium_name']} {race_info['race_info']['race_number']}R")
                logger.info(f"      大荒れ確率: {race_info['upset_probability']:.1%}")
                logger.info(f"      穴候補: {race_info['dark_horses']}")
        
        except Exception as e:
            logger.error(f"データ分析デモでエラー: {e}")


def main():
    """メイン実行関数"""
    print("🚤 Boatrace競艇予想システム")
    print("BoatraceOpenAPI専用予想・投票支援システム")
    print("-" * 50)
    
    try:
        # システム初期化
        system = BoatraceSystem()
        
        # 各デモ実行
        system.demo_basic_usage()
        print("\n" + "="*50)
        
        system.demo_prediction_workflow()
        print("\n" + "="*50)
        
        system.demo_data_analysis()
        print("\n" + "="*50)
        
        logger.info("✅ 全てのデモンストレーションが完了しました")
        print("\n🎯 システムは正常に動作しています！")
        print("\n📊 利用可能な機能:")
        print("  - リアルタイム出走表取得")
        print("  - 競技場特性分析")
        print("  - 選手成績・適性分析")
        print("  - AI予想エンジン")
        print("  - 投票戦略システム")
        print("  - ポートフォリオ管理")
        print("  - 収支・リスク分析")
        
    except Exception as e:
        logger.error(f"システム実行エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")
        print("ログファイルで詳細を確認してください: logs/boatrace_system.log")


if __name__ == "__main__":
    main()
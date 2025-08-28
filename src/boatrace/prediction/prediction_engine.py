"""
Boatrace予想エンジン

統計分析・機械学習を組み合わせた総合予想システム
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
import json

from ..core.data_models import PredictionResult
from ..data.database import Database, get_database, Prediction
from .racer_analyzer import RacerAnalyzer
from .race_analyzer import RaceAnalyzer

logger = logging.getLogger(__name__)


class PredictionEngine:
    """予想エンジン"""
    
    ENGINE_NAME = "BoatracePredictor"
    ENGINE_VERSION = "1.0.0"
    
    def __init__(self, 
                 database: Optional[Database] = None,
                 racer_analyzer: Optional[RacerAnalyzer] = None,
                 race_analyzer: Optional[RaceAnalyzer] = None):
        """
        初期化
        
        Args:
            database: データベース
            racer_analyzer: 選手分析器
            race_analyzer: レース分析器
        """
        self.database = database or get_database()
        self.racer_analyzer = racer_analyzer or RacerAnalyzer(database)
        self.race_analyzer = race_analyzer or RaceAnalyzer(database, self.racer_analyzer)
    
    def predict_race(self, race_id: str) -> Optional[PredictionResult]:
        """
        レース予想を実行
        
        Args:
            race_id: レースID
            
        Returns:
            予想結果
        """
        logger.info(f"レース予想開始: {race_id}")
        
        # レース分析実行
        race_analysis = self.race_analyzer.analyze_race(race_id)
        if not race_analysis:
            logger.error(f"レース分析失敗: {race_id}")
            return None
        
        # 各艇の予想計算
        win_probabilities = self._calculate_win_probabilities(race_analysis)
        place_probabilities = self._calculate_place_probabilities(race_analysis, win_probabilities)
        
        # 買い目推奨生成
        recommended_bets = self._generate_recommended_bets(
            race_analysis, win_probabilities, place_probabilities
        )
        
        # 予想信頼度計算
        confidence = self._calculate_prediction_confidence(race_analysis)
        
        # 予想結果作成
        prediction_result = PredictionResult(
            race_id=race_id,
            predicted_at=datetime.now(),
            win_probabilities={
                boat_num: Decimal(str(prob)) 
                for boat_num, prob in win_probabilities.items()
            },
            place_probabilities={
                boat_num: Decimal(str(prob)) 
                for boat_num, prob in place_probabilities.items()
            },
            recommended_bets=recommended_bets,
            confidence=Decimal(str(confidence))
        )
        
        # データベースに保存
        self._save_prediction(prediction_result)
        
        logger.info(f"レース予想完了: {race_id} (信頼度: {confidence:.2f})")
        return prediction_result
    
    def predict_multiple_races(self, race_ids: List[str]) -> List[PredictionResult]:
        """
        複数レース予想を実行
        
        Args:
            race_ids: レースIDリスト
            
        Returns:
            予想結果リスト
        """
        results = []
        
        for race_id in race_ids:
            try:
                result = self.predict_race(race_id)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"レース予想エラー {race_id}: {e}")
                continue
        
        return results
    
    def get_daily_predictions(self, target_date: date) -> List[PredictionResult]:
        """
        指定日の予想結果を取得
        
        Args:
            target_date: 対象日
            
        Returns:
            予想結果リスト
        """
        with self.database.get_session() as session:
            predictions = session.query(Prediction).join(
                self.database.Race
            ).filter(
                self.database.Race.date == target_date,
                Prediction.engine_name == self.ENGINE_NAME
            ).all()
            
            results = []
            for pred in predictions:
                result = self._convert_db_prediction_to_result(pred)
                if result:
                    results.append(result)
            
            return results
    
    def get_prediction_accuracy(self, days_back: int = 30) -> Dict[str, Any]:
        """
        予想精度を分析
        
        Args:
            days_back: 分析対象日数
            
        Returns:
            精度分析結果
        """
        # 実際の結果データと比較が必要
        # 現在は簡易的な統計のみ
        
        cutoff_date = date.today() - timedelta(days=days_back)
        
        with self.database.get_session() as session:
            predictions = session.query(Prediction).join(
                self.database.Race
            ).filter(
                self.database.Race.date >= cutoff_date,
                Prediction.engine_name == self.ENGINE_NAME
            ).all()
            
            if not predictions:
                return {'error': '分析対象の予想データがありません'}
            
            # 統計計算
            confidence_scores = [float(p.confidence) for p in predictions]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            return {
                'total_predictions': len(predictions),
                'average_confidence': round(avg_confidence, 3),
                'analysis_period': f'{days_back}日間',
                'engine_name': self.ENGINE_NAME,
                'engine_version': self.ENGINE_VERSION
            }
    
    def _calculate_win_probabilities(self, race_analysis) -> Dict[int, float]:
        """1着確率を計算"""
        if not race_analysis:
            return {}
        
        # 基本確率（均等）
        base_probability = 1.0 / 6.0  # 6艇均等
        probabilities = {i: base_probability for i in range(1, 7)}
        
        # 本命補正
        for boat_num in race_analysis.favorites:
            if boat_num in probabilities:
                probabilities[boat_num] *= 1.8  # 本命は1.8倍
        
        # 穴補正
        for boat_num in race_analysis.dark_horses:
            if boat_num in probabilities:
                probabilities[boat_num] *= 1.3  # 穴は1.3倍
        
        # 競争力による調整
        if race_analysis.competitiveness == "激戦":
            # 均等化
            for boat_num in probabilities:
                probabilities[boat_num] = base_probability * 0.95 + probabilities[boat_num] * 0.05
        elif race_analysis.competitiveness == "実力差":
            # 本命集中
            for boat_num in race_analysis.favorites:
                if boat_num in probabilities:
                    probabilities[boat_num] *= 1.5
        
        # 正規化
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        
        return probabilities
    
    def _calculate_place_probabilities(self, race_analysis, win_probabilities: Dict[int, float]) -> Dict[int, float]:
        """3着以内確率を計算"""
        # 1着確率から3着以内確率を推定
        place_probabilities = {}
        
        for boat_num, win_prob in win_probabilities.items():
            # 3着以内は1着確率の約2.5-3倍程度
            place_prob = min(win_prob * 2.8, 0.85)  # 上限85%
            place_probabilities[boat_num] = place_prob
        
        return place_probabilities
    
    def _generate_recommended_bets(self, 
                                 race_analysis, 
                                 win_probabilities: Dict[int, float], 
                                 place_probabilities: Dict[int, float]) -> List[Dict[str, Any]]:
        """推奨買い目を生成"""
        recommendations = []
        
        # 勝率順ソート
        sorted_win = sorted(win_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_2 = [boat for boat, prob in sorted_win[:2]]
        top_3 = [boat for boat, prob in sorted_win[:3]]
        
        # 戦略に応じた買い目生成
        strategy = race_analysis.betting_strategy
        
        if "手堅い" in strategy:
            # 単勝・2連単を中心
            recommendations.append({
                'bet_type': 'win',
                'numbers': str(top_2[0]),
                'confidence': 0.8,
                'reason': '本命単勝'
            })
            
            recommendations.append({
                'bet_type': 'exacta',
                'numbers': f"{top_2[0]}-{top_2[1]}",
                'confidence': 0.7,
                'reason': '本命サイド2連単'
            })
        
        elif "多点買い" in strategy:
            # 3連単多点
            for i, first in enumerate(top_3):
                for j, second in enumerate(top_3):
                    if i != j:
                        for k, third in enumerate(top_3):
                            if k != i and k != j:
                                recommendations.append({
                                    'bet_type': 'trifecta',
                                    'numbers': f"{first}-{second}-{third}",
                                    'confidence': 0.4,
                                    'reason': '多点勝負'
                                })
        
        elif "穴狙い" in strategy:
            # 穴馬絡みの高配当
            for dark_horse in race_analysis.dark_horses:
                recommendations.append({
                    'bet_type': 'exacta',
                    'numbers': f"{dark_horse}-{top_2[0]}",
                    'confidence': 0.5,
                    'reason': '穴馬からの高配当狙い'
                })
        
        else:
            # バランス型
            recommendations.append({
                'bet_type': 'quinella',
                'numbers': f"{top_2[0]}-{top_2[1]}",
                'confidence': 0.6,
                'reason': 'バランス重視'
            })
        
        # 上位5つまで
        return recommendations[:5]
    
    def _calculate_prediction_confidence(self, race_analysis) -> float:
        """予想信頼度を計算"""
        confidence = race_analysis.confidence
        
        # 競争力による調整
        if race_analysis.competitiveness == "実力差":
            confidence *= 1.1  # 実力差があれば予想しやすい
        elif race_analysis.competitiveness == "激戦":
            confidence *= 0.9  # 激戦は予想困難
        
        # 要因の多さによる調整
        if len(race_analysis.key_factors) >= 3:
            confidence *= 0.95  # 要因が多いと複雑
        
        return min(max(confidence, 0.0), 1.0)
    
    def _save_prediction(self, prediction_result: PredictionResult) -> None:
        """予想結果をデータベースに保存"""
        try:
            with self.database.get_session() as session:
                # 既存の予想をチェック
                existing = session.query(Prediction).filter(
                    Prediction.race_id == prediction_result.race_id,
                    Prediction.engine_name == self.ENGINE_NAME
                ).first()
                
                if existing:
                    # 更新
                    existing.predicted_at = prediction_result.predicted_at
                    existing.confidence = prediction_result.confidence
                    existing.win_prob_1 = prediction_result.win_probabilities.get(1, Decimal('0'))
                    existing.win_prob_2 = prediction_result.win_probabilities.get(2, Decimal('0'))
                    existing.win_prob_3 = prediction_result.win_probabilities.get(3, Decimal('0'))
                    existing.win_prob_4 = prediction_result.win_probabilities.get(4, Decimal('0'))
                    existing.win_prob_5 = prediction_result.win_probabilities.get(5, Decimal('0'))
                    existing.win_prob_6 = prediction_result.win_probabilities.get(6, Decimal('0'))
                    existing.place_prob_1 = prediction_result.place_probabilities.get(1, Decimal('0'))
                    existing.place_prob_2 = prediction_result.place_probabilities.get(2, Decimal('0'))
                    existing.place_prob_3 = prediction_result.place_probabilities.get(3, Decimal('0'))
                    existing.place_prob_4 = prediction_result.place_probabilities.get(4, Decimal('0'))
                    existing.place_prob_5 = prediction_result.place_probabilities.get(5, Decimal('0'))
                    existing.place_prob_6 = prediction_result.place_probabilities.get(6, Decimal('0'))
                    existing.recommended_bets = json.dumps(prediction_result.recommended_bets, ensure_ascii=False)
                    
                else:
                    # 新規作成
                    prediction = Prediction(
                        race_id=prediction_result.race_id,
                        predicted_at=prediction_result.predicted_at,
                        engine_name=self.ENGINE_NAME,
                        engine_version=self.ENGINE_VERSION,
                        confidence=prediction_result.confidence,
                        win_prob_1=prediction_result.win_probabilities.get(1, Decimal('0')),
                        win_prob_2=prediction_result.win_probabilities.get(2, Decimal('0')),
                        win_prob_3=prediction_result.win_probabilities.get(3, Decimal('0')),
                        win_prob_4=prediction_result.win_probabilities.get(4, Decimal('0')),
                        win_prob_5=prediction_result.win_probabilities.get(5, Decimal('0')),
                        win_prob_6=prediction_result.win_probabilities.get(6, Decimal('0')),
                        place_prob_1=prediction_result.place_probabilities.get(1, Decimal('0')),
                        place_prob_2=prediction_result.place_probabilities.get(2, Decimal('0')),
                        place_prob_3=prediction_result.place_probabilities.get(3, Decimal('0')),
                        place_prob_4=prediction_result.place_probabilities.get(4, Decimal('0')),
                        place_prob_5=prediction_result.place_probabilities.get(5, Decimal('0')),
                        place_prob_6=prediction_result.place_probabilities.get(6, Decimal('0')),
                        recommended_bets=json.dumps(prediction_result.recommended_bets, ensure_ascii=False)
                    )
                    session.add(prediction)
                    
        except Exception as e:
            logger.error(f"予想結果保存エラー: {e}")
    
    def _convert_db_prediction_to_result(self, prediction: Prediction) -> Optional[PredictionResult]:
        """データベースの予想を結果オブジェクトに変換"""
        try:
            win_probabilities = {
                1: prediction.win_prob_1,
                2: prediction.win_prob_2,
                3: prediction.win_prob_3,
                4: prediction.win_prob_4,
                5: prediction.win_prob_5,
                6: prediction.win_prob_6
            }
            
            place_probabilities = {
                1: prediction.place_prob_1,
                2: prediction.place_prob_2,
                3: prediction.place_prob_3,
                4: prediction.place_prob_4,
                5: prediction.place_prob_5,
                6: prediction.place_prob_6
            }
            
            recommended_bets = []
            if prediction.recommended_bets:
                try:
                    recommended_bets = json.loads(prediction.recommended_bets)
                except json.JSONDecodeError:
                    pass
            
            return PredictionResult(
                race_id=prediction.race_id,
                predicted_at=prediction.predicted_at,
                win_probabilities=win_probabilities,
                place_probabilities=place_probabilities,
                recommended_bets=recommended_bets,
                confidence=prediction.confidence
            )
            
        except Exception as e:
            logger.error(f"予想結果変換エラー: {e}")
            return None
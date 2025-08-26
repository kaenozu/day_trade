"""
教育システムAPIルーティング

用語解説、投資戦略ガイド、事例分析、学習モジュール、クイズ生成API
"""

from fastapi import APIRouter, HTTPException

from ...utils.logging_config import get_context_logger
from .app_config import get_educational_system

logger = get_context_logger(__name__)

router = APIRouter(prefix="/api/education", tags=["education"])


@router.get("/glossary/{term}")
async def get_term_explanation(term: str):
    """用語解説取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        explanation = educational_system.get_term_explanation(term)
        if explanation:
            return {"success": True, "explanation": explanation}
        else:
            return {"success": False, "message": "該当する用語が見つかりませんでした"}

    except Exception as e:
        logger.error(f"用語解説取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"用語解説取得エラー: {str(e)}"
        ) from e


@router.get("/glossary")
async def search_glossary(query: str = "", category: str = None):
    """用語集検索"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        if query:
            results = educational_system.search_glossary(query, category)
        else:
            # 全カテゴリ取得
            categories = educational_system.get_categories()
            return {"success": True, "categories": categories}

        return {"success": True, "results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"用語集検索エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"用語集検索エラー: {str(e)}"
        ) from e


@router.get("/strategies/{strategy}")
async def get_strategy_guide(strategy: str):
    """投資戦略ガイド取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        guide = educational_system.get_strategy_guide(strategy)
        if guide:
            return {"success": True, "guide": guide}
        else:
            return {
                "success": False,
                "message": "該当する戦略ガイドが見つかりませんでした",
            }

    except Exception as e:
        logger.error(f"戦略ガイド取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"戦略ガイド取得エラー: {str(e)}"
        ) from e


@router.get("/case-studies/{case}")
async def get_case_study(case: str):
    """過去事例分析取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        study = educational_system.get_case_study(case)
        if study:
            return {"success": True, "case_study": study}
        else:
            return {
                "success": False,
                "message": "該当する事例分析が見つかりませんでした",
            }

    except Exception as e:
        logger.error(f"事例分析取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"事例分析取得エラー: {str(e)}"
        ) from e


@router.get("/learning/{module}")
async def get_learning_module(module: str):
    """学習モジュール取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        learning_module = educational_system.get_learning_module(module)
        if learning_module:
            return {"success": True, "module": learning_module}
        else:
            return {
                "success": False,
                "message": "該当する学習モジュールが見つかりませんでした",
            }

    except Exception as e:
        logger.error(f"学習モジュール取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"学習モジュール取得エラー: {str(e)}"
        ) from e


@router.post("/quiz")
async def generate_quiz(request: dict):
    """クイズ問題生成"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        topic = request.get("topic")
        difficulty = request.get("difficulty")
        count = request.get("count", 5)

        if count > 20:
            raise HTTPException(
                status_code=400, detail="問題数は20問以下にしてください"
            )

        questions = educational_system.get_quiz_questions(
            topic=topic, difficulty=difficulty, count=count
        )

        return {
            "success": True,
            "questions": questions,
            "count": len(questions),
            "settings": {"topic": topic, "difficulty": difficulty, "count": count},
        }

    except Exception as e:
        logger.error(f"クイズ生成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"クイズ生成エラー: {str(e)}"
        ) from e


@router.get("/progress/{user_id}")
async def get_learning_progress(user_id: str):
    """学習進捗取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        progress = educational_system.get_learning_progress(user_id)
        recommendations = educational_system.get_recommendations(user_id)

        return {
            "success": True,
            "progress": progress,
            "recommendations": recommendations,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"学習進捗取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"学習進捗取得エラー: {str(e)}"
        ) from e


@router.post("/progress/{user_id}")
async def update_learning_progress(user_id: str, request: dict):
    """学習進捗更新"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        module = request.get("module")
        quiz_score = request.get("quiz_score")
        study_time = request.get("study_time")

        progress = educational_system.update_learning_progress(
            user_id=user_id,
            module=module,
            quiz_score=quiz_score,
            study_time=study_time,
        )

        return {"success": True, "progress": progress, "user_id": user_id}

    except Exception as e:
        logger.error(f"学習進捗更新エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"学習進捗更新エラー: {str(e)}"
        ) from e


@router.get("/topics")
async def get_learning_topics():
    """利用可能な学習トピック一覧取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        # モック学習トピック
        topics = [
            {
                "id": "basic_trading",
                "name": "基本的な取引知識",
                "description": "株式取引の基礎知識と用語",
                "difficulty": "beginner",
                "modules": 5
            },
            {
                "id": "technical_analysis",
                "name": "テクニカル分析",
                "description": "チャート分析とテクニカル指標",
                "difficulty": "intermediate",
                "modules": 8
            },
            {
                "id": "risk_management",
                "name": "リスク管理",
                "description": "ポートフォリオ管理とリスク制御",
                "difficulty": "advanced",
                "modules": 6
            },
            {
                "id": "market_psychology",
                "name": "市場心理学",
                "description": "投資家心理と市場動向",
                "difficulty": "intermediate",
                "modules": 4
            }
        ]

        return {
            "success": True,
            "topics": topics,
            "count": len(topics)
        }

    except Exception as e:
        logger.error(f"学習トピック取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"トピック取得エラー: {str(e)}"
        ) from e


@router.get("/achievements/{user_id}")
async def get_user_achievements(user_id: str):
    """ユーザーの達成状況取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        # モック達成データ
        achievements = [
            {
                "id": "first_quiz",
                "name": "初回クイズ完了",
                "description": "最初のクイズに挑戦しました",
                "earned_at": "2024-11-15T10:30:00",
                "badge_url": "/static/badges/first_quiz.png"
            },
            {
                "id": "technical_master",
                "name": "テクニカル分析マスター",
                "description": "テクニカル分析モジュールを完了",
                "earned_at": "2024-11-20T14:15:00",
                "badge_url": "/static/badges/technical_master.png"
            }
        ]

        stats = {
            "total_study_time": "24時間30分",
            "completed_modules": 12,
            "quiz_average": 85.5,
            "current_streak": 7,
            "level": "中級",
            "next_level_progress": 65
        }

        return {
            "success": True,
            "user_id": user_id,
            "achievements": achievements,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"達成状況取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"達成状況取得エラー: {str(e)}"
        ) from e


@router.post("/bookmark")
async def bookmark_content(request: dict):
    """コンテンツブックマーク"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        user_id = request.get("user_id")
        content_type = request.get("content_type")  # glossary, strategy, case_study
        content_id = request.get("content_id")
        
        if not all([user_id, content_type, content_id]):
            raise HTTPException(
                status_code=400, detail="必要なパラメータが不足しています"
            )

        # モックブックマーク処理
        success = True  # 実際の実装ではDBに保存
        
        if success:
            return {
                "success": True,
                "message": "コンテンツがブックマークされました",
                "bookmark_id": f"bookmark_{user_id}_{content_type}_{content_id}"
            }

    except Exception as e:
        logger.error(f"ブックマーク処理エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ブックマーク処理エラー: {str(e)}"
        ) from e


@router.get("/bookmarks/{user_id}")
async def get_user_bookmarks(user_id: str):
    """ユーザーのブックマーク一覧取得"""
    educational_system = get_educational_system()

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        # モックブックマークデータ
        bookmarks = [
            {
                "id": "bookmark_001",
                "content_type": "glossary",
                "content_id": "rsi",
                "title": "RSI（相対力指数）",
                "bookmarked_at": "2024-11-25T09:15:00"
            },
            {
                "id": "bookmark_002", 
                "content_type": "strategy",
                "content_id": "momentum",
                "title": "モメンタム戦略",
                "bookmarked_at": "2024-11-24T16:20:00"
            }
        ]

        return {
            "success": True,
            "user_id": user_id,
            "bookmarks": bookmarks,
            "count": len(bookmarks)
        }

    except Exception as e:
        logger.error(f"ブックマーク取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ブックマーク取得エラー: {str(e)}"
        ) from e
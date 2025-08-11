        except Exception as e:
            logger.error(f"アンサンブル作成エラー: {e}")
            return {
                "ensemble_volatility": current_realized * 100,
                "ensemble_confidence": 0.3,
                "error": str(e),
            }
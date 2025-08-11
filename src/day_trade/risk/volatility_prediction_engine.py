logger.info(
                f"ML訓練完了: {symbol} - 最良モデル: {best_model_name} "
                f"(R²={results[best_model_name]['r2']:.3f})"
            )
                results["gradient_boosting"] = {
                    "mse": float(gb_mse),
                    "rmse": float(np.sqrt(gb_mse)),
                    "r2": float(gb_r2),
                    "feature_importance": gb_model.feature_importances_.tolist(),
                }

            # 最良モデル選択
            best_model_name = min(results.keys(), key=lambda x: results[x]["mse"])

            # モデルとスケーラーを保存
            self.ml_models[symbol] = {
                "models": models,
                "scaler": scaler,
                "feature_names": X.columns.tolist(),
                "best_model_name": best_model_name,
                "target_horizon": target_horizon,
            }

            training_result = {
                "symbol": symbol,
                "target_horizon": target_horizon,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(X.columns),
                "models": results,
                "best_model": best_model_name,
                "best_performance": results[best_model_name],
                "feature_names": X.columns.tolist(),
                "training_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"ML訓練完了: {symbol} - 最良モデル: {best_model_name} "
                f"(R²={results[best_model_name]['r2']:.3f})"
            )
            return training_result
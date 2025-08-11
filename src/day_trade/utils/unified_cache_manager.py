if memory_percent > 85:  # 85%以上ならアグレッシブにクリア
                logger.warning(
                    f"高メモリ使用率検出: {memory_percent}% - L1キャッシュクリア"
                )
                self.l1_cache.clear()

                if memory_percent > 90:  # 90%以上ならL2も部分クリア
                    logger.warning("極度の高メモリ使用率 - L2キャッシュ部分クリア")
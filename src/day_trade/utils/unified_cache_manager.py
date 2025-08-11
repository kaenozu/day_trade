if memory_percent > 90:  # 90%以上ならL2も部分クリア
                    logger.warning(
                        "極度の高メモリ使用率 - L2キャッシュ部分クリア"
                    )
                    # L2の半分をクリア
                    l2_entries = len(self.l2_cache.cache)
                    clear_count = l2_entries // 2
                    keys_to_clear = list(self.l2_cache.cache.keys())[:clear_count]
                    for key in keys_to_clear:
                        self.l2_cache.delete(key)
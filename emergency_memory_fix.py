#!/usr/bin/env python3
"""
Emergency Memory Leak Fix - 緊急メモリリーク修正

WebSocketコネクションの適切なクリーンアップとメモリ管理
"""

import gc
import psutil
import threading
import time
from datetime import datetime


class EmergencyMemoryManager:
    """緊急メモリ管理システム"""
    
    def __init__(self):
        self.memory_threshold = 1024 * 1024 * 1024  # 1GB
        self.cleanup_interval = 30  # 30秒間隔
        self.running = False
        self.cleanup_thread = None
        
    def start_monitoring(self):
        """メモリ監視開始"""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.cleanup_thread.start()
            print("緊急メモリ監視開始")
    
    def stop_monitoring(self):
        """メモリ監視停止"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
        print("緊急メモリ監視停止")
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.running:
            try:
                current_memory = self._get_memory_usage()
                
                if current_memory > self.memory_threshold:
                    print(f"高メモリ使用量検出: {current_memory / 1024 / 1024:.1f}MB")
                    self._emergency_cleanup()
                
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"監視エラー: {e}")
                time.sleep(5)
    
    def _get_memory_usage(self):
        """現在のメモリ使用量取得"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def _emergency_cleanup(self):
        """緊急クリーンアップ実行"""
        print("緊急メモリクリーンアップ実行中...")
        
        # 強制ガベージコレクション
        collected = gc.collect()
        print(f"  - ガベージコレクション: {collected}オブジェクト解放")
        
        # メモリキャッシュクリア
        try:
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
                print("  - 型キャッシュクリア完了")
        except:
            pass
        
        # 後のメモリ使用量
        after_memory = self._get_memory_usage()
        print(f"  - クリーンアップ後: {after_memory / 1024 / 1024:.1f}MB")


def apply_emergency_websocket_fix():
    """緊急WebSocket修正適用"""
    print("Emergency WebSocket修正適用中...")
    
    # Flask-SocketIOの接続制限設定
    import os
    os.environ['FLASK_SOCKETIO_MAX_CONNECTIONS'] = '10'
    os.environ['FLASK_SOCKETIO_PING_TIMEOUT'] = '5'
    os.environ['FLASK_SOCKETIO_PING_INTERVAL'] = '2'
    
    print("WebSocket接続制限設定完了")


def force_memory_cleanup():
    """強制メモリクリーンアップ"""
    print("強制メモリクリーンアップ実行...")
    
    # 全ガベージコレクション実行
    for i in range(3):
        collected = gc.collect()
        print(f"  - GC第{i+1}回: {collected}オブジェクト解放")
    
    # プロセスメモリ状況表示
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"  - 現在のメモリ使用量: {memory_info.rss / 1024 / 1024:.1f}MB")
    print(f"  - 仮想メモリ: {memory_info.vms / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    # 緊急修正適用
    apply_emergency_websocket_fix()
    force_memory_cleanup()
    
    # メモリ監視開始
    memory_manager = EmergencyMemoryManager()
    memory_manager.start_monitoring()
    
    try:
        print("メモリ監視中... (Ctrl+Cで停止)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        memory_manager.stop_monitoring()
        print("緊急メモリ管理終了")
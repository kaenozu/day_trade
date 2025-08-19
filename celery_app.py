#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery Application Factory - Issue #939対応
"""

from celery import Celery

def make_celery(app_name=__name__):
    """Celeryアプリケーションインスタンスを作成"""
    # Redisをブローカーおよびバックエンドとして設定
    redis_url = 'redis://localhost:6379/0'
    return Celery(
        app_name,
        backend=redis_url,
        broker=redis_url,
        include=['celery_tasks']  # タスクが定義されているモジュール
    )

celery_app = make_celery()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gunicorn configuration file.
Issue #939対応: 本番用WSGIサーバー設定
"""

import multiprocessing

# --- Server Mechanics ---

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'  # or 'gevent' or 'eventlet' for async workers

# --- Server Socket ---

bind = '0.0.0.0:8000'

# --- Logging ---

accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'

# --- Process Naming ---

proc_name = 'daytrade_web_gunicorn'

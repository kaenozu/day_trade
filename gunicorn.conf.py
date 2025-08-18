# Gunicorn Configuration for Day Trade Personal
# Issue #901 Phase 2: プロダクション用WSGI設定

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "day_trade_web"

# Daemon mode
daemon = False
pidfile = "logs/gunicorn.pid"

# User/group (for Linux/Unix production)
# user = "daytrader"
# group = "daytrader"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
preload_app = True
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

# SSL (uncomment for HTTPS)
# keyfile = "/path/to/ssl.key"
# certfile = "/path/to/ssl.crt"

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    server.log.info("Worker pre-fork")

def when_ready(server):
    server.log.info("Day Trade Personal Web Server ready on %s", bind)

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def on_exit(server):
    server.log.info("Day Trade Personal Web Server shutting down")
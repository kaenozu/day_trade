web: gunicorn --config gunicorn.conf.py daytrade_web:app
worker: celery -A celery_app.celery_app worker --loglevel=info
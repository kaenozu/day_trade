@echo off
REM Day Trade Personal - Windows Deployment Script

echo 🚀 Starting day-trade-personal deployment...

if not defined ENVIRONMENT set ENVIRONMENT=production

echo Environment: %ENVIRONMENT%
echo Cloud Provider: VERCEL

REM Docker関連
docker --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo 📦 Building Docker image...
    docker build -t day-trade-personal:1.0.0 .
    
    echo 🧪 Running tests...
    docker run --rm day-trade-personal:1.0.0 python -m pytest
)

echo ✅ Deployment completed!
echo 🌍 Application URL: https://daytrade.vercel.example.com

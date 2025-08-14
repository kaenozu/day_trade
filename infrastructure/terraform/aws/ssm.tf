# Issue #800 Phase 3: AWS SSM Parameter Store設定
# アプリケーション設定・シークレット管理

# Market Data API Key
resource "aws_ssm_parameter" "market_data_api_key" {
  name  = "/day-trade/${var.environment}/market-data-api-key"
  type  = "SecureString"
  value = "PLACEHOLDER_API_KEY_TO_BE_UPDATED"

  tags = {
    Name = "day-trade-${var.environment}-market-data-api-key"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

# Slack Webhook URL
resource "aws_ssm_parameter" "slack_webhook_url" {
  name  = "/day-trade/${var.environment}/slack-webhook-url"
  type  = "SecureString"
  value = "PLACEHOLDER_SLACK_WEBHOOK_TO_BE_UPDATED"

  tags = {
    Name = "day-trade-${var.environment}-slack-webhook"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

# JWT Secret Key
resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "aws_ssm_parameter" "jwt_secret_key" {
  name  = "/day-trade/${var.environment}/jwt-secret-key"
  type  = "SecureString"
  value = random_password.jwt_secret.result

  tags = {
    Name = "day-trade-${var.environment}-jwt-secret"
  }
}

# API Key Encryption Key
resource "random_password" "api_encryption_key" {
  length  = 32
  special = false
}

resource "aws_ssm_parameter" "api_encryption_key" {
  name  = "/day-trade/${var.environment}/api-encryption-key"
  type  = "SecureString"
  value = random_password.api_encryption_key.result

  tags = {
    Name = "day-trade-${var.environment}-api-encryption"
  }
}

# EnsembleSystem Configuration
resource "aws_ssm_parameter" "ensemble_config_xgboost" {
  name  = "/day-trade/${var.environment}/ensemble-config-use-xgboost"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-ensemble-xgboost"
  }
}

resource "aws_ssm_parameter" "ensemble_config_catboost" {
  name  = "/day-trade/${var.environment}/ensemble-config-use-catboost"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-ensemble-catboost"
  }
}

resource "aws_ssm_parameter" "ensemble_config_random_forest" {
  name  = "/day-trade/${var.environment}/ensemble-config-use-random-forest"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-ensemble-random-forest"
  }
}

resource "aws_ssm_parameter" "ensemble_config_lstm" {
  name  = "/day-trade/${var.environment}/ensemble-config-use-lstm-transformer"
  type  = "String"
  value = "false"

  tags = {
    Name = "day-trade-${var.environment}-ensemble-lstm"
  }
}

# Scheduler Configuration
resource "aws_ssm_parameter" "trading_hours_start" {
  name  = "/day-trade/${var.environment}/trading-hours-start"
  type  = "String"
  value = "09:00"

  tags = {
    Name = "day-trade-${var.environment}-trading-hours-start"
  }
}

resource "aws_ssm_parameter" "trading_hours_end" {
  name  = "/day-trade/${var.environment}/trading-hours-end"
  type  = "String"
  value = "15:00"

  tags = {
    Name = "day-trade-${var.environment}-trading-hours-end"
  }
}

resource "aws_ssm_parameter" "auto_trading_enabled" {
  name  = "/day-trade/${var.environment}/auto-trading-enabled"
  type  = "String"
  value = var.environment == "production" ? "false" : "false"

  tags = {
    Name = "day-trade-${var.environment}-auto-trading-enabled"
  }
}

# Performance Configuration
resource "aws_ssm_parameter" "max_parallel_predictions" {
  name  = "/day-trade/${var.environment}/max-parallel-predictions"
  type  = "String"
  value = "10"

  tags = {
    Name = "day-trade-${var.environment}-max-parallel-predictions"
  }
}

resource "aws_ssm_parameter" "cache_ttl_seconds" {
  name  = "/day-trade/${var.environment}/cache-ttl-seconds"
  type  = "String"
  value = "300"

  tags = {
    Name = "day-trade-${var.environment}-cache-ttl"
  }
}

resource "aws_ssm_parameter" "model_update_interval_hours" {
  name  = "/day-trade/${var.environment}/model-update-interval-hours"
  type  = "String"
  value = "24"

  tags = {
    Name = "day-trade-${var.environment}-model-update-interval"
  }
}

# Monitoring Configuration
resource "aws_ssm_parameter" "log_level" {
  name  = "/day-trade/${var.environment}/log-level"
  type  = "String"
  value = var.environment == "production" ? "WARNING" : "INFO"

  tags = {
    Name = "day-trade-${var.environment}-log-level"
  }
}

resource "aws_ssm_parameter" "enable_monitoring" {
  name  = "/day-trade/${var.environment}/enable-monitoring"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-enable-monitoring"
  }
}

# Email Configuration (for notifications)
resource "aws_ssm_parameter" "email_smtp_server" {
  name  = "/day-trade/${var.environment}/email-smtp-server"
  type  = "String"
  value = "smtp.gmail.com"

  tags = {
    Name = "day-trade-${var.environment}-email-smtp-server"
  }
}

resource "aws_ssm_parameter" "email_smtp_port" {
  name  = "/day-trade/${var.environment}/email-smtp-port"
  type  = "String"
  value = "587"

  tags = {
    Name = "day-trade-${var.environment}-email-smtp-port"
  }
}

resource "aws_ssm_parameter" "email_username" {
  name  = "/day-trade/${var.environment}/email-username"
  type  = "SecureString"
  value = "PLACEHOLDER_EMAIL_TO_BE_UPDATED"

  tags = {
    Name = "day-trade-${var.environment}-email-username"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "email_password" {
  name  = "/day-trade/${var.environment}/email-password"
  type  = "SecureString"
  value = "PLACEHOLDER_EMAIL_PASSWORD_TO_BE_UPDATED"

  tags = {
    Name = "day-trade-${var.environment}-email-password"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

# Feature Flags
resource "aws_ssm_parameter" "feature_flag_advanced_analytics" {
  name  = "/day-trade/${var.environment}/feature-flag-advanced-analytics"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-feature-advanced-analytics"
  }
}

resource "aws_ssm_parameter" "feature_flag_real_time_data" {
  name  = "/day-trade/${var.environment}/feature-flag-real-time-data"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-feature-real-time-data"
  }
}

resource "aws_ssm_parameter" "feature_flag_smart_symbol_selection" {
  name  = "/day-trade/${var.environment}/feature-flag-smart-symbol-selection"
  type  = "String"
  value = "true"

  tags = {
    Name = "day-trade-${var.environment}-feature-smart-symbol-selection"
  }
}

# Application Version
resource "aws_ssm_parameter" "application_version" {
  name  = "/day-trade/${var.environment}/application-version"
  type  = "String"
  value = "1.0.0"

  tags = {
    Name = "day-trade-${var.environment}-application-version"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

# Build Information
resource "aws_ssm_parameter" "build_commit_sha" {
  name  = "/day-trade/${var.environment}/build-commit-sha"
  type  = "String"
  value = "PLACEHOLDER_COMMIT_SHA"

  tags = {
    Name = "day-trade-${var.environment}-build-commit-sha"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "build_timestamp" {
  name  = "/day-trade/${var.environment}/build-timestamp"
  type  = "String"
  value = timestamp()

  tags = {
    Name = "day-trade-${var.environment}-build-timestamp"
  }

  lifecycle {
    ignore_changes = [value]
  }
}
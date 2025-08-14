# Issue #800 Phase 3: AWS ElastiCache Redis設定
# 高可用性・レプリケーション・暗号化対応

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "day-trade-${var.environment}-cache-subnet"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "day-trade-${var.environment}-cache-subnet"
  }
}

# ElastiCache Parameter Group
resource "aws_elasticache_parameter_group" "main" {
  family = "redis7.x"
  name   = "day-trade-${var.environment}-redis7"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  tags = {
    Name = "day-trade-${var.environment}-redis7"
  }
}

# ElastiCache Replication Group (Redis Cluster)
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "day-trade-${var.environment}-redis"
  description                = "Redis cluster for day-trade ${var.environment}"

  # Engine settings
  engine               = "redis"
  engine_version       = "7.0"
  node_type           = var.redis_node_type
  port                = var.redis_port
  parameter_group_name = aws_elasticache_parameter_group.main.name

  # Cluster settings
  num_cache_clusters = var.environment == "production" ? 2 : 1

  # Network settings
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.elasticache.id]

  # High Availability (Multi-AZ)
  multi_az_enabled           = var.environment == "production" ? true : false
  automatic_failover_enabled = var.environment == "production" ? true : false

  # Backup settings
  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window         = "03:00-05:00"

  # Maintenance
  maintenance_window = "sun:05:00-sun:06:00"

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = random_password.redis_auth_token.result

  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.elasticache_slow.name
    destination_type = "cloudwatch-logs"
    log_format      = "text"
    log_type        = "slow-log"
  }

  tags = {
    Name = "day-trade-${var.environment}-redis"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Random password for Redis AUTH
resource "random_password" "redis_auth_token" {
  length  = 32
  special = true
}

# CloudWatch Log Group for ElastiCache
resource "aws_cloudwatch_log_group" "elasticache_slow" {
  name              = "/aws/elasticache/${var.environment}/slow-log"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = "day-trade-${var.environment}-elasticache-slow-log"
  }
}

# Redis URL for SSM Parameter Store
resource "aws_ssm_parameter" "redis_url" {
  name  = "/day-trade/${var.environment}/redis-url"
  type  = "SecureString"
  value = "redis://:${random_password.redis_auth_token.result}@${aws_elasticache_replication_group.main.configuration_endpoint_address}:${var.redis_port}"

  tags = {
    Name = "day-trade-${var.environment}-redis-url"
  }
}

# Redis Auth Token for SSM Parameter Store
resource "aws_ssm_parameter" "redis_auth_token" {
  name  = "/day-trade/${var.environment}/redis-auth-token"
  type  = "SecureString"
  value = random_password.redis_auth_token.result

  tags = {
    Name = "day-trade-${var.environment}-redis-auth-token"
  }
}

# ElastiCache User (Redis 6.0+)
resource "aws_elasticache_user" "app_user" {
  user_id       = "day-trade-app-user"
  user_name     = "day-trade-app"
  access_string = "on ~* &* +@all"
  engine        = "REDIS"
  passwords     = [random_password.redis_auth_token.result]

  tags = {
    Name = "day-trade-${var.environment}-app-user"
  }
}

# ElastiCache User Group
resource "aws_elasticache_user_group" "main" {
  engine        = "REDIS"
  user_group_id = "day-trade-${var.environment}-user-group"
  user_ids      = ["default", aws_elasticache_user.app_user.user_id]

  tags = {
    Name = "day-trade-${var.environment}-user-group"
  }

  lifecycle {
    ignore_changes = [user_ids]
  }
}

# CloudWatch Alarms for ElastiCache
resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  alarm_name          = "day-trade-${var.environment}-redis-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = var.alarm_cpu_threshold
  alarm_description   = "This metric monitors redis cpu utilization"

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.id
  }

  tags = {
    Name = "day-trade-${var.environment}-redis-cpu-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  alarm_name          = "day-trade-${var.environment}-redis-memory-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = var.alarm_memory_threshold
  alarm_description   = "This metric monitors redis memory utilization"

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.id
  }

  tags = {
    Name = "day-trade-${var.environment}-redis-memory-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "redis_connections" {
  alarm_name          = "day-trade-${var.environment}-redis-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CurrConnections"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "1000"
  alarm_description   = "This metric monitors redis current connections"

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.id
  }

  tags = {
    Name = "day-trade-${var.environment}-redis-connections-alarm"
  }
}
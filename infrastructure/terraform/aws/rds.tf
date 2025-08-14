# Issue #800 Phase 3: AWS RDS PostgreSQL設定
# 高可用性・自動バックアップ・暗号化対応

# Database Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "day-trade-${var.environment}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "day-trade-${var.environment}-db-subnet-group"
  }
}

# RDS Parameter Group
resource "aws_db_parameter_group" "main" {
  family = "postgres15"
  name   = "day-trade-${var.environment}-postgres15"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "log_checkpoints"
    value = "1"
  }

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_disconnections"
    value = "1"
  }

  parameter {
    name  = "log_lock_waits"
    value = "1"
  }

  tags = {
    Name = "day-trade-${var.environment}-postgres15"
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "day-trade-${var.environment}-postgres"

  # Engine settings
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class

  # Storage settings
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  # Database settings
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  port     = 5432

  # Network settings
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Backup settings
  backup_retention_period = var.backup_retention_period
  backup_window          = var.backup_window
  maintenance_window     = var.maintenance_window
  copy_tags_to_snapshot  = true

  # High Availability
  multi_az = var.environment == "production" ? true : false

  # Parameter and option groups
  parameter_group_name = aws_db_parameter_group.main.name

  # Monitoring
  monitoring_interval = var.enable_monitoring ? 60 : 0
  monitoring_role_arn = var.enable_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  enabled_cloudwatch_logs_exports = [
    "postgresql"
  ]

  # Performance Insights
  performance_insights_enabled          = var.environment == "production" ? true : false
  performance_insights_retention_period = var.environment == "production" ? 7 : null

  # Security
  deletion_protection = var.enable_deletion_protection
  skip_final_snapshot = var.environment == "production" ? false : true
  final_snapshot_identifier = var.environment == "production" ? "day-trade-${var.environment}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null

  # Auto minor version upgrade
  auto_minor_version_upgrade = true

  tags = {
    Name = "day-trade-${var.environment}-postgres"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# RDS Enhanced Monitoring Role
resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.enable_monitoring ? 1 : 0

  name = "day-trade-${var.environment}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "day-trade-${var.environment}-rds-monitoring-role"
  }
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count = var.enable_monitoring ? 1 : 0

  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# RDS Subnet Group for Read Replica (if needed)
resource "aws_db_subnet_group" "replica" {
  count = var.environment == "production" ? 1 : 0

  name       = "day-trade-${var.environment}-replica-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "day-trade-${var.environment}-replica-db-subnet-group"
  }
}

# Read Replica for Production
resource "aws_db_instance" "replica" {
  count = var.environment == "production" ? 1 : 0

  identifier = "day-trade-${var.environment}-postgres-replica"

  # Replica settings
  replicate_source_db = aws_db_instance.main.identifier

  # Instance settings
  instance_class = var.db_instance_class

  # Network settings
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Monitoring
  monitoring_interval = var.enable_monitoring ? 60 : 0
  monitoring_role_arn = var.enable_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  # Performance Insights
  performance_insights_enabled = true

  # Auto minor version upgrade
  auto_minor_version_upgrade = true

  tags = {
    Name = "day-trade-${var.environment}-postgres-replica"
  }
}

# Database URL for SSM Parameter Store
resource "aws_ssm_parameter" "database_url" {
  name  = "/day-trade/${var.environment}/database-url"
  type  = "SecureString"
  value = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/${var.db_name}"

  tags = {
    Name = "day-trade-${var.environment}-database-url"
  }
}

# Read Replica URL for SSM Parameter Store
resource "aws_ssm_parameter" "database_replica_url" {
  count = var.environment == "production" ? 1 : 0

  name  = "/day-trade/${var.environment}/database-replica-url"
  type  = "SecureString"
  value = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.replica[0].endpoint}:${aws_db_instance.replica[0].port}/${var.db_name}"

  tags = {
    Name = "day-trade-${var.environment}-database-replica-url"
  }
}
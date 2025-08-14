# Issue #800 Phase 3: AWS Terraform出力値

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

# Security Group Outputs
output "alb_security_group_id" {
  description = "ALB security group ID"
  value       = aws_security_group.alb.id
}

output "ecs_security_group_id" {
  description = "ECS security group ID"
  value       = aws_security_group.ecs.id
}

output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}

output "elasticache_security_group_id" {
  description = "ElastiCache security group ID"
  value       = aws_security_group.elasticache.id
}

# ALB Outputs
output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB zone ID"
  value       = aws_lb.main.zone_id
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.main.arn
}

# Target Group Outputs
output "ml_service_target_group_arn" {
  description = "ML Service target group ARN"
  value       = aws_lb_target_group.ml_service.arn
}

output "data_service_target_group_arn" {
  description = "Data Service target group ARN"
  value       = aws_lb_target_group.data_service.arn
}

output "scheduler_service_target_group_arn" {
  description = "Scheduler Service target group ARN"
  value       = aws_lb_target_group.scheduler_service.arn
}

# ECS Outputs
output "ecs_cluster_id" {
  description = "ECS cluster ID"
  value       = aws_ecs_cluster.main.id
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_task_execution_role_arn" {
  description = "ECS task execution role ARN"
  value       = aws_iam_role.ecs_task_execution_role.arn
}

output "ecs_task_role_arn" {
  description = "ECS task role ARN"
  value       = aws_iam_role.ecs_task_role.arn
}

# Task Definition Outputs
output "ml_service_task_definition_arn" {
  description = "ML Service task definition ARN"
  value       = aws_ecs_task_definition.ml_service.arn
}

output "data_service_task_definition_arn" {
  description = "Data Service task definition ARN"
  value       = aws_ecs_task_definition.data_service.arn
}

output "scheduler_service_task_definition_arn" {
  description = "Scheduler Service task definition ARN"
  value       = aws_ecs_task_definition.scheduler_service.arn
}

# Database Outputs
output "database_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "database_port" {
  description = "Database port"
  value       = aws_db_instance.main.port
}

output "database_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "database_replica_endpoint" {
  description = "Database replica endpoint"
  value       = var.environment == "production" ? aws_db_instance.replica[0].endpoint : null
  sensitive   = true
}

# ElastiCache Outputs
output "redis_configuration_endpoint" {
  description = "Redis configuration endpoint"
  value       = aws_elasticache_replication_group.main.configuration_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.main.port
}

output "redis_auth_token_arn" {
  description = "Redis auth token SSM parameter ARN"
  value       = aws_ssm_parameter.redis_auth_token.arn
  sensitive   = true
}

# S3 Outputs
output "ml_models_bucket_name" {
  description = "ML models S3 bucket name"
  value       = aws_s3_bucket.ml_models.bucket
}

output "ml_models_bucket_arn" {
  description = "ML models S3 bucket ARN"
  value       = aws_s3_bucket.ml_models.arn
}

output "data_cache_bucket_name" {
  description = "Data cache S3 bucket name"
  value       = aws_s3_bucket.data_cache.bucket
}

output "data_cache_bucket_arn" {
  description = "Data cache S3 bucket ARN"
  value       = aws_s3_bucket.data_cache.arn
}

output "backups_bucket_name" {
  description = "Backups S3 bucket name"
  value       = aws_s3_bucket.backups.bucket
}

output "backups_bucket_arn" {
  description = "Backups S3 bucket ARN"
  value       = aws_s3_bucket.backups.arn
}

output "logs_bucket_name" {
  description = "Logs S3 bucket name"
  value       = aws_s3_bucket.logs.bucket
}

output "logs_bucket_arn" {
  description = "Logs S3 bucket ARN"
  value       = aws_s3_bucket.logs.arn
}

# CloudWatch Outputs
output "ml_service_log_group_name" {
  description = "ML Service CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ml_service.name
}

output "data_service_log_group_name" {
  description = "Data Service CloudWatch log group name"
  value       = aws_cloudwatch_log_group.data_service.name
}

output "scheduler_service_log_group_name" {
  description = "Scheduler Service CloudWatch log group name"
  value       = aws_cloudwatch_log_group.scheduler_service.name
}

# SNS Outputs
output "ml_model_updates_topic_arn" {
  description = "ML model updates SNS topic ARN"
  value       = aws_sns_topic.ml_model_updates.arn
}

# KMS Outputs
output "s3_kms_key_id" {
  description = "S3 KMS key ID"
  value       = aws_kms_key.s3.key_id
}

output "s3_kms_key_arn" {
  description = "S3 KMS key ARN"
  value       = aws_kms_key.s3.arn
}

# SSM Parameter Outputs
output "database_url_parameter_arn" {
  description = "Database URL SSM parameter ARN"
  value       = aws_ssm_parameter.database_url.arn
  sensitive   = true
}

output "redis_url_parameter_arn" {
  description = "Redis URL SSM parameter ARN"
  value       = aws_ssm_parameter.redis_url.arn
  sensitive   = true
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}

# Application URLs
output "application_urls" {
  description = "Application service URLs"
  value = {
    ml_service        = "https://${var.domain_name}/ml"
    data_service      = "https://${var.domain_name}/data"
    scheduler_service = "https://${var.domain_name}/scheduler"
  }
}

# Monitoring URLs
output "monitoring_urls" {
  description = "Monitoring service URLs"
  value = {
    cloudwatch = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}"
    rds        = "https://console.aws.amazon.com/rds/home?region=${var.aws_region}"
    elasticache = "https://console.aws.amazon.com/elasticache/home?region=${var.aws_region}"
    ecs        = "https://console.aws.amazon.com/ecs/home?region=${var.aws_region}#/clusters/${aws_ecs_cluster.main.name}"
  }
}
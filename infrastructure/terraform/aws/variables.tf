# Issue #800 Phase 3: AWS Terraform変数定義

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.100.0/24", "10.0.200.0/24"]
}

# ECS設定
variable "ecs_cluster_name" {
  description = "ECS cluster name"
  type        = string
  default     = "day-trade"
}

variable "ml_service_cpu" {
  description = "CPU units for ML service"
  type        = number
  default     = 2048
}

variable "ml_service_memory" {
  description = "Memory for ML service in MiB"
  type        = number
  default     = 4096
}

variable "data_service_cpu" {
  description = "CPU units for Data service"
  type        = number
  default     = 1024
}

variable "data_service_memory" {
  description = "Memory for Data service in MiB"
  type        = number
  default     = 2048
}

variable "scheduler_service_cpu" {
  description = "CPU units for Scheduler service"
  type        = number
  default     = 512
}

variable "scheduler_service_memory" {
  description = "Memory for Scheduler service in MiB"
  type        = number
  default     = 1024
}

# RDS設定
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "day_trade"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "day_trade_user"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# ElastiCache設定
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

variable "redis_port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

# S3設定
variable "s3_bucket_prefix" {
  description = "S3 bucket prefix"
  type        = string
  default     = "day-trade"
}

# CloudWatch設定
variable "log_retention_in_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# ALB設定
variable "certificate_arn" {
  description = "SSL certificate ARN for ALB"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "day-trade.company.com"
}

# Auto Scaling設定
variable "ml_service_min_capacity" {
  description = "Minimum capacity for ML service"
  type        = number
  default     = 2
}

variable "ml_service_max_capacity" {
  description = "Maximum capacity for ML service"
  type        = number
  default     = 10
}

variable "data_service_min_capacity" {
  description = "Minimum capacity for Data service"
  type        = number
  default     = 1
}

variable "data_service_max_capacity" {
  description = "Maximum capacity for Data service"
  type        = number
  default     = 5
}

variable "scheduler_service_min_capacity" {
  description = "Minimum capacity for Scheduler service"
  type        = number
  default     = 1
}

variable "scheduler_service_max_capacity" {
  description = "Maximum capacity for Scheduler service"
  type        = number
  default     = 3
}

# タグ
variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

# セキュリティ設定
variable "enable_deletion_protection" {
  description = "Enable deletion protection for production resources"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 7
}

variable "backup_window" {
  description = "Database backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "maintenance_window" {
  description = "Database maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

# 監視設定
variable "enable_monitoring" {
  description = "Enable detailed monitoring"
  type        = bool
  default     = true
}

variable "alarm_cpu_threshold" {
  description = "CPU utilization threshold for alarms"
  type        = number
  default     = 80
}

variable "alarm_memory_threshold" {
  description = "Memory utilization threshold for alarms"
  type        = number
  default     = 80
}
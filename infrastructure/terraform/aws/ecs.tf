# Issue #800 Phase 3: AWS ECS Fargate設定
# ML Service + Data Service + Scheduler Service

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.ecs_cluster_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"

      log_configuration {
        cloud_watch_log_group_name = aws_cloudwatch_log_group.ecs_exec.name
      }
    }
  }

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}"
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "ecs_exec" {
  name              = "/aws/ecs/${var.ecs_cluster_name}-${var.environment}/exec"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}-exec-logs"
  }
}

resource "aws_cloudwatch_log_group" "ml_service" {
  name              = "/aws/ecs/${var.ecs_cluster_name}-${var.environment}/ml-service"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}-ml-service-logs"
  }
}

resource "aws_cloudwatch_log_group" "data_service" {
  name              = "/aws/ecs/${var.ecs_cluster_name}-${var.environment}/data-service"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}-data-service-logs"
  }
}

resource "aws_cloudwatch_log_group" "scheduler_service" {
  name              = "/aws/ecs/${var.ecs_cluster_name}-${var.environment}/scheduler-service"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}-scheduler-service-logs"
  }
}

# ECS Task Execution Role
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.ecs_cluster_name}-${var.environment}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Role (for application permissions)
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.ecs_cluster_name}-${var.environment}-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.ecs_cluster_name}-${var.environment}-task-role"
  }
}

# Task Role Policy for S3 and other AWS services
resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.ecs_cluster_name}-${var.environment}-task-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*",
          aws_s3_bucket.data_cache.arn,
          "${aws_s3_bucket.data_cache.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# ML Service Task Definition
resource "aws_ecs_task_definition" "ml_service" {
  family                   = "day-trade-${var.environment}-ml-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ml_service_cpu
  memory                   = var.ml_service_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "ml-service"
      image = "ghcr.io/${data.aws_caller_identity.current.account_id}/day-trade/ml-service:latest"

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "ML_MODEL_PATH"
          value = "/app/models"
        },
        {
          name  = "LOG_LEVEL"
          value = "WARNING"
        }
      ]

      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = aws_ssm_parameter.database_url.arn
        },
        {
          name      = "REDIS_URL"
          valueFrom = aws_ssm_parameter.redis_url.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_service.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      essential = true
    }
  ])

  tags = {
    Name = "day-trade-${var.environment}-ml-service"
  }
}

# Data Service Task Definition
resource "aws_ecs_task_definition" "data_service" {
  family                   = "day-trade-${var.environment}-data-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.data_service_cpu
  memory                   = var.data_service_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "data-service"
      image = "ghcr.io/${data.aws_caller_identity.current.account_id}/day-trade/data-service:latest"

      portMappings = [
        {
          containerPort = 8001
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "DATA_CACHE_PATH"
          value = "/app/data"
        },
        {
          name  = "LOG_LEVEL"
          value = "WARNING"
        }
      ]

      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = aws_ssm_parameter.database_url.arn
        },
        {
          name      = "REDIS_URL"
          valueFrom = aws_ssm_parameter.redis_url.arn
        },
        {
          name      = "MARKET_DATA_API_KEY"
          valueFrom = aws_ssm_parameter.market_data_api_key.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.data_service.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      essential = true
    }
  ])

  tags = {
    Name = "day-trade-${var.environment}-data-service"
  }
}

# Scheduler Service Task Definition
resource "aws_ecs_task_definition" "scheduler_service" {
  family                   = "day-trade-${var.environment}-scheduler-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.scheduler_service_cpu
  memory                   = var.scheduler_service_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "scheduler-service"
      image = "ghcr.io/${data.aws_caller_identity.current.account_id}/day-trade/scheduler-service:latest"

      portMappings = [
        {
          containerPort = 8002
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "SCHEDULER_CONFIG_PATH"
          value = "/app/schedules"
        },
        {
          name  = "LOG_LEVEL"
          value = "WARNING"
        },
        {
          name  = "MARKET_HOURS_TIMEZONE"
          value = "Asia/Tokyo"
        }
      ]

      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = aws_ssm_parameter.database_url.arn
        },
        {
          name      = "REDIS_URL"
          valueFrom = aws_ssm_parameter.redis_url.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.scheduler_service.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8002/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      essential = true
    }
  ])

  tags = {
    Name = "day-trade-${var.environment}-scheduler-service"
  }
}
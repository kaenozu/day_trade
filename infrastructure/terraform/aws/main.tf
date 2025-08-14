# Issue #800 Phase 3: AWS環境 Terraform設定
# ECS Fargate + RDS + ElastiCache + ALB構成

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "day-trade-terraform-state"
    key    = "production/terraform.tfstate"
    region = "ap-northeast-1"

    dynamodb_table = "day-trade-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "day-trade"
      Environment = var.environment
      Owner       = "development-team"
      ManagedBy   = "terraform"
    }
  }
}

# データソース
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC設定
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "day-trade-${var.environment}-vpc"
  }
}

# インターネットゲートウェイ
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "day-trade-${var.environment}-igw"
  }
}

# パブリックサブネット
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "day-trade-${var.environment}-public-${count.index + 1}"
    Type = "public"
  }
}

# プライベートサブネット
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "day-trade-${var.environment}-private-${count.index + 1}"
    Type = "private"
  }
}

# NAT Gateway用EIP
resource "aws_eip" "nat" {
  count = 2

  domain = "vpc"

  tags = {
    Name = "day-trade-${var.environment}-nat-eip-${count.index + 1}"
  }
}

# NAT Gateway
resource "aws_nat_gateway" "main" {
  count = 2

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "day-trade-${var.environment}-nat-${count.index + 1}"
  }
}

# パブリックルートテーブル
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "day-trade-${var.environment}-public-rt"
  }
}

# プライベートルートテーブル
resource "aws_route_table" "private" {
  count = 2

  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = {
    Name = "day-trade-${var.environment}-private-rt-${count.index + 1}"
  }
}

# ルートテーブル関連付け
resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# セキュリティグループ: ALB
resource "aws_security_group" "alb" {
  name_prefix = "day-trade-${var.environment}-alb-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "day-trade-${var.environment}-alb-sg"
  }
}

# セキュリティグループ: ECS
resource "aws_security_group" "ecs" {
  name_prefix = "day-trade-${var.environment}-ecs-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8002
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "day-trade-${var.environment}-ecs-sg"
  }
}

# セキュリティグループ: RDS
resource "aws_security_group" "rds" {
  name_prefix = "day-trade-${var.environment}-rds-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = {
    Name = "day-trade-${var.environment}-rds-sg"
  }
}

# セキュリティグループ: ElastiCache
resource "aws_security_group" "elasticache" {
  name_prefix = "day-trade-${var.environment}-elasticache-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = {
    Name = "day-trade-${var.environment}-elasticache-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "day-trade-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.environment == "production" ? true : false

  tags = {
    Name = "day-trade-${var.environment}-alb"
  }
}

# ALB Target Groups
resource "aws_lb_target_group" "ml_service" {
  name     = "day-trade-${var.environment}-ml"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "day-trade-${var.environment}-ml-tg"
  }
}

resource "aws_lb_target_group" "data_service" {
  name     = "day-trade-${var.environment}-data"
  port     = 8001
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "day-trade-${var.environment}-data-tg"
  }
}

resource "aws_lb_target_group" "scheduler_service" {
  name     = "day-trade-${var.environment}-scheduler"
  port     = 8002
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "day-trade-${var.environment}-scheduler-tg"
  }
}
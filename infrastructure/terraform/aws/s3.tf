# Issue #800 Phase 3: AWS S3設定
# MLモデル・データ保存・バックアップ

# S3 Bucket for ML Models
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.s3_bucket_prefix}-${var.environment}-ml-models"

  tags = {
    Name = "${var.s3_bucket_prefix}-${var.environment}-ml-models"
  }
}

resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    id     = "ml_models_lifecycle"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# S3 Bucket for Data Cache
resource "aws_s3_bucket" "data_cache" {
  bucket = "${var.s3_bucket_prefix}-${var.environment}-data-cache"

  tags = {
    Name = "${var.s3_bucket_prefix}-${var.environment}-data-cache"
  }
}

resource "aws_s3_bucket_versioning" "data_cache" {
  bucket = aws_s3_bucket.data_cache.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "data_cache" {
  bucket = aws_s3_bucket.data_cache.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_cache" {
  bucket = aws_s3_bucket.data_cache.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data_cache" {
  bucket = aws_s3_bucket.data_cache.id

  rule {
    id     = "data_cache_lifecycle"
    status = "Enabled"

    expiration {
      days = 30
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.s3_bucket_prefix}-${var.environment}-backups"

  tags = {
    Name = "${var.s3_bucket_prefix}-${var.environment}-backups"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "backups" {
  bucket = aws_s3_bucket.backups.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.s3.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backups_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 2555  # 7 years
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# S3 Bucket for Logs
resource "aws_s3_bucket" "logs" {
  bucket = "${var.s3_bucket_prefix}-${var.environment}-logs"

  tags = {
    Name = "${var.s3_bucket_prefix}-${var.environment}-logs"
  }
}

resource "aws_s3_bucket_encryption" "logs" {
  bucket = aws_s3_bucket.logs.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "logs_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# KMS Key for S3 Encryption
resource "aws_kms_key" "s3" {
  description             = "KMS key for S3 encryption in day-trade ${var.environment}"
  deletion_window_in_days = var.environment == "production" ? 30 : 7

  tags = {
    Name = "day-trade-${var.environment}-s3-kms"
  }
}

resource "aws_kms_alias" "s3" {
  name          = "alias/day-trade-${var.environment}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# SSM Parameters for S3 Buckets
resource "aws_ssm_parameter" "ml_models_bucket" {
  name  = "/day-trade/${var.environment}/s3-ml-models-bucket"
  type  = "String"
  value = aws_s3_bucket.ml_models.bucket

  tags = {
    Name = "day-trade-${var.environment}-ml-models-bucket"
  }
}

resource "aws_ssm_parameter" "data_cache_bucket" {
  name  = "/day-trade/${var.environment}/s3-data-cache-bucket"
  type  = "String"
  value = aws_s3_bucket.data_cache.bucket

  tags = {
    Name = "day-trade-${var.environment}-data-cache-bucket"
  }
}

resource "aws_ssm_parameter" "backups_bucket" {
  name  = "/day-trade/${var.environment}/s3-backups-bucket"
  type  = "String"
  value = aws_s3_bucket.backups.bucket

  tags = {
    Name = "day-trade-${var.environment}-backups-bucket"
  }
}

# S3 Bucket Notification for ML Model Updates
resource "aws_s3_bucket_notification" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  topic {
    topic_arn = aws_sns_topic.ml_model_updates.arn
    events    = ["s3:ObjectCreated:*"]

    filter_prefix = "models/"
    filter_suffix = ".pkl"
  }
}

# SNS Topic for ML Model Updates
resource "aws_sns_topic" "ml_model_updates" {
  name = "day-trade-${var.environment}-ml-model-updates"

  tags = {
    Name = "day-trade-${var.environment}-ml-model-updates"
  }
}

resource "aws_sns_topic_policy" "ml_model_updates" {
  arn = aws_sns_topic.ml_model_updates.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action   = "SNS:Publish"
        Resource = aws_sns_topic.ml_model_updates.arn
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}
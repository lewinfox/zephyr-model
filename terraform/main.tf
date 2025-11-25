terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data source to get the latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security group for SSH access
resource "aws_security_group" "zephyr_training" {
  name        = "zephyr-training-sg"
  description = "Security group for Zephyr model training instance"

  # SSH access
  ingress {
    description = "SSH from allowed IPs"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Optional: Jupyter notebook access
  ingress {
    description = "Jupyter notebook (optional)"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = var.enable_jupyter ? var.allowed_ssh_cidrs : []
  }

  # Outbound internet access
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "zephyr-training-sg"
    Project     = "zephyr-model"
    Environment = var.environment
  }
}

# SSH key pair
resource "aws_key_pair" "zephyr_training" {
  key_name   = var.key_pair_name
  public_key = file(var.public_key_path)

  tags = {
    Name        = "zephyr-training-key"
    Project     = "zephyr-model"
    Environment = var.environment
  }
}

# IAM role for EC2 instance (for S3 access)
resource "aws_iam_role" "zephyr_training" {
  name = "zephyr-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "zephyr-training-role"
    Project     = "zephyr-model"
    Environment = var.environment
  }
}

# IAM policy for S3 access
resource "aws_iam_role_policy" "zephyr_s3_access" {
  name = "zephyr-s3-access"
  role = aws_iam_role.zephyr_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      }
    ]
  })
}

# IAM instance profile
resource "aws_iam_instance_profile" "zephyr_training" {
  name = "zephyr-training-profile"
  role = aws_iam_role.zephyr_training.name

  tags = {
    Name        = "zephyr-training-profile"
    Project     = "zephyr-model"
    Environment = var.environment
  }
}

# User data script to set up instance
data "template_file" "user_data" {
  template = file("${path.module}/user_data.sh")

  vars = {
    git_repo_url       = var.git_repo_url
    s3_bucket          = var.s3_bucket
    s3_data_key        = var.s3_data_key
    auto_start_training = var.auto_start_training
  }
}

# EC2 instance for training
resource "aws_instance" "zephyr_training" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  # Use spot instance if requested
  instance_market_options {
    market_type = var.use_spot_instance ? "spot" : null

    dynamic "spot_options" {
      for_each = var.use_spot_instance ? [1] : []
      content {
        max_price          = var.spot_max_price
        spot_instance_type = "one-time"
      }
    }
  }

  key_name               = aws_key_pair.zephyr_training.key_name
  vpc_security_group_ids = [aws_security_group.zephyr_training.id]
  iam_instance_profile   = aws_iam_instance_profile.zephyr_training.name

  # EBS root volume
  root_block_device {
    volume_size           = var.ebs_volume_size
    volume_type           = "gp3"
    delete_on_termination = true

    tags = {
      Name        = "zephyr-training-volume"
      Project     = "zephyr-model"
      Environment = var.environment
    }
  }

  # User data for automated setup
  user_data = var.enable_user_data ? data.template_file.user_data.rendered : null

  tags = {
    Name        = "zephyr-training"
    Project     = "zephyr-model"
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  # Ensure instance is recreated if user data changes
  user_data_replace_on_change = true
}

# Elastic IP (optional, for persistent IP address)
resource "aws_eip" "zephyr_training" {
  count    = var.allocate_elastic_ip ? 1 : 0
  instance = aws_instance.zephyr_training.id
  domain   = "vpc"

  tags = {
    Name        = "zephyr-training-eip"
    Project     = "zephyr-model"
    Environment = var.environment
  }
}

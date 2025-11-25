variable "aws_region" {
  description = "AWS region to launch the instance"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type (GPU-enabled)"
  type        = string
  default     = "g4dn.xlarge"

  validation {
    condition     = can(regex("^(g4dn|g5|p3|p4)", var.instance_type))
    error_message = "Instance type must be a GPU-enabled instance (g4dn, g5, p3, or p4 series)."
  }
}

variable "use_spot_instance" {
  description = "Use spot instance for cost savings (can be interrupted)"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum price for spot instance (empty string for on-demand price)"
  type        = string
  default     = "0.30" # ~2x spot price for g4dn.xlarge for safety
}

variable "ebs_volume_size" {
  description = "Size of EBS root volume in GB"
  type        = number
  default     = 100
}

variable "key_pair_name" {
  description = "Name for the SSH key pair"
  type        = string
  default     = "zephyr-training-key"
}

variable "public_key_path" {
  description = "Path to SSH public key file"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH to the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"] # WARNING: Open to all. Change to your IP for security.
}

variable "enable_jupyter" {
  description = "Enable Jupyter notebook access on port 8888"
  type        = bool
  default     = false
}

variable "s3_bucket" {
  description = "S3 bucket name for data and model storage"
  type        = string
  default     = ""
}

variable "s3_data_key" {
  description = "S3 key path for training data (e.g., 'zephyr-training/data.csv')"
  type        = string
  default     = "zephyr-training/data.csv"
}

variable "git_repo_url" {
  description = "Git repository URL to clone (leave empty to skip)"
  type        = string
  default     = ""
}

variable "enable_user_data" {
  description = "Enable user data script for automated setup"
  type        = bool
  default     = true
}

variable "auto_start_training" {
  description = "Automatically start training on instance launch"
  type        = bool
  default     = false
}

variable "allocate_elastic_ip" {
  description = "Allocate an Elastic IP for persistent IP address"
  type        = bool
  default     = false
}

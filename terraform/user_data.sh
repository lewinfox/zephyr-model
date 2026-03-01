#!/bin/bash
set -e

# User data script for Zephyr training instance
# This script runs on first boot to set up the environment

echo "=== Zephyr Training Instance Setup ==="
echo "Starting at $(date)"

# Update system
echo "Updating system packages..."
apt-get update -qq

# Ensure Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Verify NVIDIA drivers and Docker GPU support
echo "Verifying NVIDIA drivers..."
nvidia-smi || echo "WARNING: nvidia-smi failed"

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    apt-get install -y docker-compose-plugin
fi

# Configure instance for training
echo "Configuring instance..."

# Set up working directory
WORK_DIR="/home/ubuntu/zephyr-model"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository if URL provided
if [ -n "${git_repo_url}" ]; then
    echo "Cloning repository from ${git_repo_url}..."
    if [ -d "$WORK_DIR/.git" ]; then
        echo "Repository already exists, pulling latest..."
        git -C "$WORK_DIR" pull
    else
        git clone "${git_repo_url}" "$WORK_DIR"
    fi
    chown -R ubuntu:ubuntu "$WORK_DIR"
fi

# Download training data from S3 if configured
if [ -n "${s3_bucket}" ] && [ -n "${s3_data_key}" ]; then
    echo "Downloading training data from S3..."
    mkdir -p "$WORK_DIR/modelling"
    aws s3 cp "s3://${s3_bucket}/${s3_data_key}" "$WORK_DIR/modelling/data.csv" || echo "WARNING: Failed to download data from S3"
    chown -R ubuntu:ubuntu "$WORK_DIR/modelling"
fi

# Create necessary directories
mkdir -p "$WORK_DIR/models" "$WORK_DIR/logs"
chown -R ubuntu:ubuntu "$WORK_DIR"

# Build Docker image if Dockerfile exists
if [ -f "$WORK_DIR/Dockerfile" ]; then
    echo "Building Docker image..."
    cd "$WORK_DIR"
    docker build -t zephyr-model:latest . || echo "WARNING: Docker build failed"
fi

# Auto-start training if enabled
if [ "${auto_start_training}" = "true" ]; then
    echo "Auto-starting training..."
    cd "$WORK_DIR"

    # Check if data exists
    if [ -f "$WORK_DIR/modelling/data.csv" ]; then
        # Run training in detached mode
        docker run -d \
            --name zephyr-training \
            --gpus all \
            -v "$WORK_DIR/modelling/data.csv:/workspace/modelling/data.csv:ro" \
            -v "$WORK_DIR/models:/workspace/models" \
            -v "$WORK_DIR/logs:/workspace/logs" \
            --shm-size 8g \
            zephyr-model:latest \
            python modelling/train.py

        echo "Training started. Check logs with: docker logs -f zephyr-training"
    else
        echo "WARNING: Training data not found at $WORK_DIR/modelling/data.csv"
        echo "Skipping auto-start training."
    fi
fi

# Create a helpful status file
cat > /home/ubuntu/INSTANCE_STATUS.txt <<EOF
Zephyr Training Instance
========================

Setup completed at: $(date)

Instance Information:
- Instance ID: $(ec2-metadata --instance-id | cut -d' ' -f2)
- Instance Type: $(ec2-metadata --instance-type | cut -d' ' -f2)
- Public IP: $(ec2-metadata --public-ipv4 | cut -d' ' -f2)
- Availability Zone: $(ec2-metadata --availability-zone | cut -d' ' -f2)

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)

Working Directory: $WORK_DIR

Useful Commands:
----------------
# Check if training is running
docker ps | grep zephyr-training

# View training logs
docker logs -f zephyr-training

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check trained models
ls -lh $WORK_DIR/models/

# Upload model to S3 (if configured)
aws s3 cp $WORK_DIR/models/weather_forecast_multi_station.pkl s3://${s3_bucket}/models/

EOF

chown ubuntu:ubuntu /home/ubuntu/INSTANCE_STATUS.txt

echo "=== Setup Complete ==="
echo "Instance ready for training at $(date)"
echo "See /home/ubuntu/INSTANCE_STATUS.txt for details"

# Log completion to CloudWatch Logs (if configured)
echo "User data script completed successfully" | logger -t zephyr-setup

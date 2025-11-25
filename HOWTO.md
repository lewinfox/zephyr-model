# How to Train Zephyr Model on AWS EC2 with GPU

This guide walks you through launching a GPU-enabled EC2 instance, deploying the Zephyr training container, and monitoring progress from your local machine.

---

## Quick Start with Terraform (Recommended)

**The easiest way** to launch and manage training instances is with Terraform:

```bash
cd terraform

# Configure (first time only)
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars  # Edit your settings

# Launch instance
terraform init
terraform apply

# Get SSH command
terraform output ssh_connection_command

# Destroy when done
terraform destroy
```

**See [terraform/README.md](terraform/README.md) for detailed Terraform documentation.**

---

## Table of Contents

### Terraform Method (Recommended)
- [Quick Start with Terraform](#quick-start-with-terraform-recommended)
- [Terraform Detailed Guide](terraform/README.md)

### Manual Method
1. [Prerequisites](#prerequisites)
2. [Step 1: Launch GPU EC2 Instance](#step-1-launch-gpu-ec2-instance)
3. [Step 2: Connect to Instance](#step-2-connect-to-instance)
4. [Step 3: Install Dependencies](#step-3-install-dependencies)
5. [Step 4: Transfer Training Data](#step-4-transfer-training-data)
6. [Step 5: Build and Run Container](#step-5-build-and-run-container)
7. [Step 6: Monitor Training Progress](#step-6-monitor-training-progress)
8. [Step 7: Retrieve Trained Model](#step-7-retrieve-trained-model)
9. [Step 8: Clean Up](#step-8-clean-up)
10. [Troubleshooting](#troubleshooting)
11. [Cost Optimization Tips](#cost-optimization-tips)

---

# Terraform Method (Recommended)

## Why Use Terraform?

- **Simple**: `terraform apply` to launch, `terraform destroy` to stop
- **Automated**: Auto-setup, auto-training options
- **Reproducible**: Same configuration every time
- **Cost-effective**: Easy to use spot instances
- **Version-controlled**: Infrastructure as code

## Basic Workflow

### 1. First-Time Setup

```bash
cd terraform

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings (IMPORTANT: set your IP address)
nano terraform.tfvars
```

**Key settings to configure**:
```hcl
# Change this to your IP for security
allowed_ssh_cidrs = ["YOUR_IP/32"]

# Optional: S3 bucket for data storage
s3_bucket = "your-bucket-name"

# Optional: Git repository
git_repo_url = "https://github.com/yourusername/zephyr-model.git"
```

### 2. Initialize Terraform (One-Time)

```bash
terraform init
```

### 3. Launch Training Instance

```bash
# Review what will be created
terraform plan

# Launch instance
terraform apply
# Type 'yes' when prompted
```

**Wait 2-3 minutes** for instance to launch and configure.

### 4. Connect and Monitor

```bash
# Get connection command
terraform output ssh_connection_command

# Connect to instance
ssh -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw instance_public_ip)

# Or monitor remotely
terraform output monitoring_commands
```

### 5. Clean Up (Stop Paying)

```bash
# Destroy instance and all resources
terraform destroy
# Type 'yes' when prompted
```

**Cost**: ~$0.09 per training run (30 minutes with spot instance)

## Advanced Terraform Usage

### Automated Training

Configure automatic training on launch:

```hcl
# In terraform.tfvars
auto_start_training = true
git_repo_url = "https://github.com/yourusername/zephyr-model.git"
s3_bucket = "your-bucket"
s3_data_key = "zephyr-training/data.csv"
```

Then simply:
```bash
terraform apply    # Launch, train, wait
# ... wait 20-30 minutes ...
terraform destroy  # Clean up
```

### Multiple Environments

```bash
# Development
terraform apply -var-file="terraform.tfvars.dev"

# Production
terraform apply -var-file="terraform.tfvars.prod"
```

### Cost Optimization

```hcl
# Budget: Use spot instances (default)
use_spot_instance = true
spot_max_price = "0.20"

# Reliability: Use on-demand
use_spot_instance = false
```

**For complete Terraform documentation, see [terraform/README.md](terraform/README.md)**

---

# Manual Method

The sections below describe the manual process. **Use this if you need more control or want to understand the underlying steps.**

---

## Prerequisites

### On Your Local Machine

1. **AWS CLI** installed and configured:
   ```bash
   # Install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install

   # Configure with your credentials
   aws configure
   # Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output format (json)
   ```

2. **SSH key pair** for EC2 access:
   ```bash
   # Create a new key pair
   aws ec2 create-key-pair \
     --key-name zephyr-training-key \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/zephyr-training-key.pem

   # Set correct permissions
   chmod 400 ~/.ssh/zephyr-training-key.pem
   ```

3. **Training data** prepared:
   ```bash
   # Ensure you have modelling/data.csv ready
   ls -lh modelling/data.csv
   ```

### AWS Account Requirements

- AWS account with permissions to launch EC2 instances
- GPU instance quota (g4dn.xlarge or similar)
- VPC with internet gateway (default VPC is fine)

---

## Step 1: Launch GPU EC2 Instance

### Option A: Using AWS CLI (Recommended)

```bash
# 1. Find the latest Deep Learning AMI with Docker + NVIDIA drivers
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name,CreationDate]' \
  --output table

# Save the ImageId from above (e.g., ami-0abcdef1234567890)
export AMI_ID="ami-XXXXXXXXXXXXXXXXX"

# 2. Create security group
aws ec2 create-security-group \
  --group-name zephyr-training-sg \
  --description "Security group for Zephyr model training"

# Get the security group ID
export SG_ID=$(aws ec2 describe-security-groups \
  --group-names zephyr-training-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# 3. Add SSH access (replace YOUR_IP with your IP address)
export YOUR_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${YOUR_IP}/32

# 4. Launch instance
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g4dn.xlarge \
  --key-name zephyr-training-key \
  --security-group-ids $SG_ID \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=zephyr-training}]' \
  --count 1

# 5. Get instance ID and public IP
export INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=zephyr-training" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

export INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Public IP: $INSTANCE_IP"
```

### Option B: Using AWS Console

1. Go to **EC2 Dashboard** → **Launch Instance**
2. **Name**: `zephyr-training`
3. **AMI**: Search for "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
4. **Instance Type**: `g4dn.xlarge` (1 GPU, 16 GB RAM, 4 vCPUs)
   - **Cost**: ~$0.526/hour on-demand (~$0.158/hour spot)
5. **Key Pair**: Select `zephyr-training-key` (or create new)
6. **Network Settings**:
   - Create security group: `zephyr-training-sg`
   - Allow SSH (port 22) from your IP
7. **Storage**: 100 GB gp3 EBS volume
8. **Advanced Details** → **Spot instances** (optional, for cost savings)
9. Click **Launch Instance**
10. Note the **Public IPv4 address**

### Recommended Instance Types

| Instance Type | GPU | vCPUs | RAM | Cost/hour (on-demand) | Cost/hour (spot) |
|---------------|-----|-------|-----|----------------------|------------------|
| g4dn.xlarge | T4 (16GB) | 4 | 16 GB | $0.526 | ~$0.158 |
| g4dn.2xlarge | T4 (16GB) | 8 | 32 GB | $0.752 | ~$0.226 |
| g5.xlarge | A10G (24GB) | 4 | 16 GB | $1.006 | ~$0.302 |
| g5.2xlarge | A10G (24GB) | 8 | 32 GB | $1.212 | ~$0.364 |

**Recommendation**: Start with `g4dn.xlarge` spot instance for best value.

---

## Step 2: Connect to Instance

### Wait for Instance to be Ready

```bash
# Wait for instance to pass status checks (takes 2-3 minutes)
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID

# Or check manually
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID
```

### SSH into Instance

```bash
# Connect via SSH
ssh -i ~/.ssh/zephyr-training-key.pem ubuntu@$INSTANCE_IP

# Or if you didn't export INSTANCE_IP:
ssh -i ~/.ssh/zephyr-training-key.pem ubuntu@<YOUR_INSTANCE_PUBLIC_IP>
```

**Note**: If you see "Connection refused", wait another minute for the instance to fully boot.

---

## Step 3: Install Dependencies

Once connected to the EC2 instance:

```bash
# 1. Verify GPU is available
nvidia-smi
# Should show NVIDIA T4 GPU

# 2. Install Docker (if not pre-installed)
# The Deep Learning AMI usually has Docker pre-installed
docker --version

# If Docker is not installed:
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
# Log out and back in for group changes to take effect

# 3. Install NVIDIA Container Toolkit (if not pre-installed)
# Check if already installed:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If above fails, install nvidia-docker:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Should show GPU info

# 5. Install Docker Compose
sudo apt-get install -y docker-compose-plugin
docker compose version
```

---

## Step 4: Transfer Training Data

### Option A: Transfer from Local Machine (Small Data)

From your **local machine**:

```bash
# Create a tarball of necessary files
tar -czf zephyr-data.tar.gz modelling/data.csv .env

# Transfer to EC2 instance
scp -i ~/.ssh/zephyr-training-key.pem \
  zephyr-data.tar.gz \
  ubuntu@$INSTANCE_IP:/home/ubuntu/
```

On the **EC2 instance**:

```bash
# Extract data
mkdir -p ~/zephyr-model
cd ~/zephyr-model
tar -xzf ~/zephyr-data.tar.gz
```

### Option B: Download from S3 (Recommended for Large Data)

First, upload data to S3 from your **local machine**:

```bash
# Upload to S3
aws s3 cp modelling/data.csv s3://your-bucket-name/zephyr-training/data.csv
aws s3 cp .env s3://your-bucket-name/zephyr-training/.env
```

On the **EC2 instance**:

```bash
# Download from S3
mkdir -p ~/zephyr-model/modelling
cd ~/zephyr-model

aws s3 cp s3://your-bucket-name/zephyr-training/data.csv modelling/data.csv
aws s3 cp s3://your-bucket-name/zephyr-training/.env .env

# Verify data
ls -lh modelling/data.csv
```

### Option C: Clone from Git (If Data in Repo)

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/zephyr-model.git
cd zephyr-model

# If data not in repo, transfer separately
```

---

## Step 5: Build and Run Container

On the **EC2 instance**:

```bash
cd ~/zephyr-model

# 1. Clone repository (if not already done)
# git clone <your-repo-url>
# cd zephyr-model

# 2. Ensure training data is in place
ls -lh modelling/data.csv

# 3. Create directories for outputs
mkdir -p models logs

# 4. Build Docker image
docker build -t zephyr-model:latest .

# 5. Verify GPU access
docker run --rm --gpus all zephyr-model:latest \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Should output:
# CUDA available: True
# GPU: Tesla T4

# 6. Run training in background
docker compose up -d train

# Or run directly with docker:
docker run -d \
  --name zephyr-training \
  --gpus all \
  -v $(pwd)/modelling/data.csv:/workspace/modelling/data.csv:ro \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  --shm-size 8g \
  zephyr-model:latest \
  python modelling/train.py

# 7. Verify container is running
docker ps
```

---

## Step 6: Monitor Training Progress

### Real-time Logs

```bash
# Follow logs in real-time
docker compose logs -f train

# Or with docker directly:
docker logs -f zephyr-training

# To exit log view: Ctrl+C (container keeps running)
```

### Check GPU Usage

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

### Check Container Status

```bash
# List running containers
docker ps

# Check if training container exited
docker ps -a | grep zephyr

# If container exited, check exit code
docker inspect zephyr-training --format='{{.State.ExitCode}}'
# 0 = success, non-zero = error
```

### Access Interactive Shell (for Debugging)

```bash
# Execute shell in running container
docker exec -it zephyr-training bash

# Inside container, you can:
# - Check files: ls -l /workspace/models/
# - Monitor process: ps aux | grep python
# - Check GPU: nvidia-smi
# - View logs: tail -f /workspace/logs/*.log

# Exit: type 'exit' or Ctrl+D
```

### Monitor from Local Machine (without SSH)

Set up SSH tunnel for easier monitoring:

```bash
# From local machine, open SSH connection with port forwarding
ssh -i ~/.ssh/zephyr-training-key.pem \
  -L 8888:localhost:8888 \
  ubuntu@$INSTANCE_IP

# In another terminal, watch logs remotely
ssh -i ~/.ssh/zephyr-training-key.pem ubuntu@$INSTANCE_IP \
  "docker logs -f zephyr-training"
```

### Estimated Training Time

For 600k samples, 19 stations, 20 epochs:
- **g4dn.xlarge (T4)**: ~20-30 minutes
- **g5.xlarge (A10G)**: ~10-15 minutes

---

## Step 7: Retrieve Trained Model

### Check Training Completed

```bash
# On EC2 instance
docker ps -a | grep zephyr-training

# If Status shows "Exited (0)", training completed successfully

# List generated models
ls -lh models/
```

### Download Model to Local Machine

From your **local machine**:

```bash
# Download trained model
scp -i ~/.ssh/zephyr-training-key.pem \
  ubuntu@$INSTANCE_IP:/home/ubuntu/zephyr-model/models/*.pkl \
  ./models/

# Download logs (optional)
scp -i ~/.ssh/zephyr-training-key.pem \
  ubuntu@$INSTANCE_IP:/home/ubuntu/zephyr-model/logs/* \
  ./logs/
```

### Or Upload to S3 (Recommended)

On the **EC2 instance**:

```bash
# Upload model to S3
aws s3 cp models/weather_forecast_multi_station.pkl \
  s3://your-bucket-name/zephyr-models/$(date +%Y%m%d_%H%M%S)_weather_forecast.pkl

# Upload logs
aws s3 sync logs/ s3://your-bucket-name/zephyr-logs/
```

From your **local machine**:

```bash
# Download from S3
aws s3 cp s3://your-bucket-name/zephyr-models/20251126_143022_weather_forecast.pkl \
  ./models/weather_forecast_multi_station.pkl
```

---

## Step 8: Clean Up

### Stop and Remove Container

On the **EC2 instance**:

```bash
# Stop container
docker compose down

# Or with docker directly:
docker stop zephyr-training
docker rm zephyr-training

# Remove image (optional, to free space)
docker rmi zephyr-model:latest
```

### Terminate EC2 Instance

From your **local machine**:

```bash
# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Verify termination
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].State.Name'
```

Or via **AWS Console**:
1. Go to EC2 Dashboard
2. Select instance `zephyr-training`
3. **Instance State** → **Terminate instance**

### Delete Security Group (optional)

```bash
# Delete security group
aws ec2 delete-security-group --group-id $SG_ID
```

### Remove Key Pair (optional)

```bash
# Delete key pair from AWS
aws ec2 delete-key-pair --key-name zephyr-training-key

# Delete local key file
rm ~/.ssh/zephyr-training-key.pem
```

---

## Troubleshooting

### Problem: "Connection refused" when SSHing

**Solution**: Instance may still be booting. Wait 2-3 minutes and try again.

```bash
# Check instance status
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID
```

### Problem: "Permission denied (publickey)"

**Solution**: Check key file permissions and path.

```bash
chmod 400 ~/.ssh/zephyr-training-key.pem
ssh -i ~/.ssh/zephyr-training-key.pem ubuntu@$INSTANCE_IP
```

### Problem: "nvidia-smi not found" or GPU not detected

**Solution**: Ensure you launched a GPU instance (g4dn, g5, p3, etc.).

```bash
# Check instance type
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].InstanceType'

# Should return g4dn.xlarge or similar
```

### Problem: Docker can't access GPU

**Solution**: Install/restart nvidia-container-toolkit.

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Problem: "No space left on device"

**Solution**: EBS volume too small. Resize or use larger volume.

```bash
# Check disk usage
df -h

# Resize EBS volume via AWS Console, then:
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```

### Problem: Training crashes with "CUDA out of memory"

**Solution**: Reduce batch size in training config.

```bash
# Edit train.py to use smaller batch size
# Or pass as environment variable
docker run -e BATCH_SIZE=64 ... zephyr-model:latest
```

### Problem: Can't download model files (large size)

**Solution**: Use S3 transfer or compress before downloading.

```bash
# On EC2 instance, compress model
cd ~/zephyr-model
tar -czf model_output.tar.gz models/ logs/

# Download compressed archive
scp -i ~/.ssh/zephyr-training-key.pem \
  ubuntu@$INSTANCE_IP:/home/ubuntu/zephyr-model/model_output.tar.gz \
  ./
```

---

## Cost Optimization Tips

### 1. Use Spot Instances

Spot instances are **~70% cheaper** than on-demand:

```bash
# Launch spot instance (CLI)
aws ec2 run-instances \
  --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.20","SpotInstanceType":"one-time"}}' \
  ... # other parameters same as before
```

**Caveat**: Spot instances can be interrupted if capacity is needed.

### 2. Stop Instance Instead of Terminating

If you plan to train again soon:

```bash
# Stop instance (keeps EBS volume, small storage cost)
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Restart later
aws ec2 start-instances --instance-ids $INSTANCE_ID
```

**Cost**: Only pay for EBS storage (~$0.08/GB/month) when stopped.

### 3. Use Smaller Instance for Setup

Launch a small instance (t3.micro) for setup/debugging, then switch to GPU instance only for training.

### 4. Automate with Scripts

Create a launch script that:
1. Launches spot instance
2. Runs training automatically
3. Uploads model to S3
4. Terminates instance

See `scripts/train_on_ec2.sh` (to be created) for automated workflow.

### 5. Monitor Costs

```bash
# Check current month's EC2 costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d "$(date +%Y-%m-01)" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=SERVICE,Key=SERVICE \
  --filter file://<(echo '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}')
```

### Example Cost Calculation

Training run on **g4dn.xlarge spot instance**:
- Instance cost: $0.158/hour
- Training time: 30 minutes = 0.5 hours
- Storage: 100 GB × 1 hour = $0.008
- Data transfer: ~100 MB = negligible

**Total**: ~$0.09 per training run

---

## Quick Reference Commands

### From Local Machine

```bash
# Launch instance
aws ec2 run-instances --image-id $AMI_ID --instance-type g4dn.xlarge ...

# Get instance IP
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress'

# Connect
ssh -i ~/.ssh/zephyr-training-key.pem ubuntu@$INSTANCE_IP

# Transfer data
scp -i ~/.ssh/zephyr-training-key.pem modelling/data.csv ubuntu@$INSTANCE_IP:~/

# Download model
scp -i ~/.ssh/zephyr-training-key.pem ubuntu@$INSTANCE_IP:~/zephyr-model/models/*.pkl ./models/

# Terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

### On EC2 Instance

```bash
# Verify GPU
nvidia-smi

# Build container
docker build -t zephyr-model:latest .

# Run training
docker compose up -d train

# Monitor logs
docker logs -f zephyr-training

# Check GPU usage
watch -n 1 nvidia-smi

# Stop training
docker compose down
```

---

## Next Steps

Once you have a trained model:
1. Evaluate performance on held-out test set
2. Deploy model for inference (see deployment docs)
3. Set up continuous retraining pipeline
4. Integrate with Zephyr API for real-time predictions

For more details on model deployment and inference, see `DEPLOYMENT.md` (coming soon).

---

## Additional Resources

- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
- [AWS CLI Reference](https://docs.aws.amazon.com/cli/latest/reference/ec2/)

---

**Last Updated**: 2025-11-26

For questions or issues, refer to the main project documentation in `CLAUDE.md` or `README.md`.

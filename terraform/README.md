# Terraform Configuration for Zephyr Model Training

This directory contains Terraform configuration to automate the provisioning of GPU-enabled EC2 instances for model training.

## Quick Start

### 1. Prerequisites

- **Terraform** installed (v1.0+)
  ```bash
  # Install Terraform (macOS)
  brew install terraform

  # Or Linux
  wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
  unzip terraform_1.6.0_linux_amd64.zip
  sudo mv terraform /usr/local/bin/
  ```

- **AWS CLI** configured with credentials
  ```bash
  aws configure
  ```

- **SSH key pair** for instance access
  ```bash
  # Generate SSH key if you don't have one
  ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
  ```

### 2. Configure Terraform

```bash
cd terraform

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings
nano terraform.tfvars  # or vim, code, etc.
```

**Important settings to change**:
- `allowed_ssh_cidrs`: Change from `["0.0.0.0/0"]` to your IP address
- `s3_bucket`: Your S3 bucket name (if using S3 for data)
- `git_repo_url`: Your repository URL (if auto-cloning)

### 3. Initialize Terraform

```bash
terraform init
```

This downloads the AWS provider and sets up the backend.

### 4. Review the Execution Plan

```bash
terraform plan
```

This shows what resources will be created.

### 5. Launch Training Instance

```bash
terraform apply
```

Type `yes` when prompted. The instance will be launched in ~2-3 minutes.

### 6. Get Instance Details

```bash
# View all outputs
terraform output

# Get SSH command
terraform output ssh_connection_command

# Get instance IP
terraform output instance_public_ip
```

### 7. Connect to Instance

```bash
# Use the SSH command from output
ssh -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw instance_public_ip)

# Or add to ~/.ssh/config
terraform output -raw ssh_config_entry >> ~/.ssh/config
ssh zephyr-training
```

### 8. Monitor Training

```bash
# From local machine, monitor logs
ssh -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw instance_public_ip) "docker logs -f zephyr-training"

# Check GPU usage
ssh -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw instance_public_ip) "nvidia-smi"
```

### 9. Destroy Instance (Stop Paying)

```bash
terraform destroy
```

Type `yes` when prompted. The instance and all resources will be terminated.

---

## Configuration Options

### Instance Types

| Instance | GPU | vCPUs | RAM | Cost/hour (spot) | Use Case |
|----------|-----|-------|-----|------------------|----------|
| g4dn.xlarge | T4 (16GB) | 4 | 16 GB | ~$0.16 | Budget |
| g4dn.2xlarge | T4 (16GB) | 8 | 32 GB | ~$0.23 | Balanced |
| g5.xlarge | A10G (24GB) | 4 | 16 GB | ~$0.30 | Performance |
| g5.2xlarge | A10G (24GB) | 8 | 32 GB | ~$0.36 | High Performance |

### Cost Optimization

**Use Spot Instances** (recommended):
```hcl
use_spot_instance = true
spot_max_price = "0.30"
```

**On-Demand Instances** (guaranteed availability):
```hcl
use_spot_instance = false
```

### Automated Training

Set `auto_start_training = true` to automatically start training when instance launches:

```hcl
auto_start_training = true
git_repo_url = "https://github.com/yourusername/zephyr-model.git"
s3_bucket = "your-bucket"
s3_data_key = "zephyr-training/data.csv"
```

**Workflow**:
1. `terraform apply` → Instance launches
2. User data script runs:
   - Clones repository
   - Downloads data from S3
   - Builds Docker image
   - Starts training automatically
3. Training completes → Upload model to S3
4. `terraform destroy` → Instance terminates

**Total time**: Setup (2 min) + Training (20 min) + Upload (1 min) = ~23 minutes

**Total cost**: 0.4 hours × $0.16 = **~$0.06**

---

## Advanced Usage

### Using S3 for Data Storage

1. **Upload data to S3**:
   ```bash
   aws s3 mb s3://your-zephyr-bucket
   aws s3 cp modelling/data.csv s3://your-zephyr-bucket/zephyr-training/data.csv
   ```

2. **Configure Terraform**:
   ```hcl
   s3_bucket = "your-zephyr-bucket"
   s3_data_key = "zephyr-training/data.csv"
   ```

3. **Instance will auto-download** data on launch

### Multiple Environments

Create separate `.tfvars` files for different environments:

```bash
# Development
terraform apply -var-file="terraform.tfvars.dev"

# Production
terraform apply -var-file="terraform.tfvars.prod"
```

### Remote State (Team Collaboration)

Store Terraform state in S3 for team access:

```hcl
# Add to main.tf
terraform {
  backend "s3" {
    bucket = "your-terraform-state-bucket"
    key    = "zephyr-training/terraform.tfstate"
    region = "us-east-1"
  }
}
```

### Custom User Data Script

Modify `user_data.sh` to customize instance setup:
- Install additional tools
- Configure monitoring
- Set up experiment tracking
- Custom training parameters

---

## Workflow Examples

### Example 1: Quick Training Run

```bash
# Launch instance
terraform apply -auto-approve

# Wait for training to complete (~20 min)
# Monitor with: terraform output monitoring_commands

# Download model
scp -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw instance_public_ip):~/zephyr-model/models/*.pkl ./models/

# Destroy instance
terraform destroy -auto-approve
```

### Example 2: Interactive Development

```bash
# Launch with Jupyter enabled
terraform apply -var="enable_jupyter=true"

# Access Jupyter
terraform output jupyter_url

# Develop interactively in browser

# When done
terraform destroy
```

### Example 3: Automated Pipeline

```bash
# Create automated training script
cat > train_pipeline.sh <<'EOF'
#!/bin/bash
set -e

echo "Starting automated training pipeline..."

# Apply Terraform
terraform apply -auto-approve

# Get instance IP
INSTANCE_IP=$(terraform output -raw instance_public_ip)

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
sleep 120

# Monitor training
echo "Monitoring training..."
ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP "docker logs -f zephyr-training" &

# Wait for training to complete
while ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP "docker ps | grep -q zephyr-training"; do
    echo "Training still running..."
    sleep 60
done

echo "Training completed!"

# Download model
echo "Downloading model..."
scp -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP:~/zephyr-model/models/*.pkl ./models/

# Cleanup
echo "Cleaning up..."
terraform destroy -auto-approve

echo "Pipeline complete!"
EOF

chmod +x train_pipeline.sh
./train_pipeline.sh
```

---

## Troubleshooting

### Issue: "Error launching instance"

**Check**: GPU instance quota
```bash
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-DB2E81BA
```

**Solution**: Request quota increase in AWS Console

### Issue: "Spot instance interrupted"

**Solution**: Use on-demand instance or increase `spot_max_price`

### Issue: "SSH connection refused"

**Wait**: Instance may still be booting (takes 2-3 minutes)

### Issue: "Permission denied (publickey)"

**Check**: Key path in `terraform.tfvars`
```bash
ls -la ~/.ssh/id_rsa.pub
```

### Issue: Terraform state locked

**Solution**: Force unlock (use carefully)
```bash
terraform force-unlock <LOCK_ID>
```

---

## File Structure

```
terraform/
├── main.tf                    # Main Terraform configuration
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output definitions
├── user_data.sh              # Instance setup script
├── terraform.tfvars.example  # Example configuration
├── terraform.tfvars          # Your configuration (gitignored)
└── README.md                 # This file
```

---

## Security Best Practices

1. **Restrict SSH access**:
   ```hcl
   allowed_ssh_cidrs = ["YOUR_IP/32"]
   ```

2. **Use S3 bucket policies** to restrict access

3. **Don't commit** `terraform.tfvars` or `*.tfstate` files

4. **Use IAM roles** instead of hardcoded credentials

5. **Enable CloudTrail** for audit logging

6. **Rotate SSH keys** regularly

---

## Cost Monitoring

### Check Current Costs

```bash
# Current month EC2 costs
aws ce get-cost-and-usage \
    --time-period Start=$(date -d "$(date +%Y-%m-01)" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=SERVICE \
    --filter file://<(echo '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}')
```

### Set Cost Alerts

Use AWS Budgets to alert when costs exceed threshold:
```bash
aws budgets create-budget \
    --account-id <ACCOUNT_ID> \
    --budget file://budget.json \
    --notifications-with-subscribers file://notifications.json
```

---

## Additional Resources

- [Terraform AWS Provider Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/)
- [AWS Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)

---

## Quick Reference

```bash
# Initialize
terraform init

# Plan changes
terraform plan

# Apply changes
terraform apply

# Show outputs
terraform output

# Destroy resources
terraform destroy

# Format code
terraform fmt

# Validate configuration
terraform validate

# Show current state
terraform show
```

---

**Last Updated**: 2025-11-26

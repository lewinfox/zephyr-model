output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.zephyr_training.id
}

output "instance_type" {
  description = "Instance type"
  value       = aws_instance.zephyr_training.instance_type
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the instance"
  value       = aws_instance.zephyr_training.public_dns
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.zephyr_training.id
}

output "ssh_connection_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${replace(var.public_key_path, ".pub", "")} ubuntu@${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip}"
}

output "ssh_config_entry" {
  description = "SSH config entry for easy connection"
  value       = <<-EOT
    Host zephyr-training
      HostName ${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip}
      User ubuntu
      IdentityFile ${replace(var.public_key_path, ".pub", "")}
      StrictHostKeyChecking no
      UserKnownHostsFile /dev/null
  EOT
}

output "jupyter_url" {
  description = "Jupyter notebook URL (if enabled)"
  value       = var.enable_jupyter ? "http://${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip}:8888" : "Jupyter not enabled"
}

output "monitoring_commands" {
  description = "Useful commands for monitoring training"
  value       = <<-EOT
    # SSH to instance
    ${replace(var.public_key_path, ".pub", "") != "" ? "ssh -i ${replace(var.public_key_path, ".pub", "")} ubuntu@${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip}" : "ssh ubuntu@${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip}"}

    # Monitor logs remotely
    ssh -i ${replace(var.public_key_path, ".pub", "")} ubuntu@${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip} "docker logs -f zephyr-training"

    # Check GPU usage remotely
    ssh -i ${replace(var.public_key_path, ".pub", "")} ubuntu@${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip} "nvidia-smi"

    # Download trained model
    scp -i ${replace(var.public_key_path, ".pub", "")} ubuntu@${var.allocate_elastic_ip ? aws_eip.zephyr_training[0].public_ip : aws_instance.zephyr_training.public_ip}:~/zephyr-model/models/*.pkl ./models/
  EOT
}

output "estimated_cost" {
  description = "Estimated hourly cost (approximate)"
  value       = var.use_spot_instance ? "~$0.16/hour (spot) - actual cost depends on spot pricing" : "~$0.53/hour (on-demand for g4dn.xlarge)"
}

output "instance_state" {
  description = "Current state of the instance"
  value       = aws_instance.zephyr_training.instance_state
}

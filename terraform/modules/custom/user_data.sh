#!/bin/bash

# MLflow EC2 Setup Script
# This script installs and configures MLflow on Amazon Linux 2023

set -e

# Update system
dnf update -y

# Install required packages including nmap-ncat for connectivity testing
dnf install -y python3 python3-pip git docker awscli nmap-ncat cronie

# Install CloudWatch agent
dnf install -y amazon-cloudwatch-agent

# Start and enable cron service
systemctl enable crond
systemctl start crond

# Install Python packages
# Note: Install packages individually to avoid system conflicts
pip3 install --ignore-installed PyMySQL cryptography mysql-connector-python
pip3 install --ignore-installed boto3
pip3 install --ignore-installed mlflow==2.16.2

# Create mlflow user
useradd -m -s /bin/bash mlflow
usermod -aG docker mlflow

# Create directories
mkdir -p /home/mlflow/logs
mkdir -p /opt/mlflow
chown -R mlflow:mlflow /home/mlflow /opt/mlflow

# URL encode the database password to handle special characters
DB_PASSWORD_ENCODED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${db_password}', safe=''))")

# Configure MLflow environment
cat > /opt/mlflow/mlflow.env << EOF
MLFLOW_BACKEND_STORE_URI=mysql+pymysql://${db_username}:$DB_PASSWORD_ENCODED@${db_endpoint}/${db_name}
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://${mlflow_bucket_name}/mlflow-artifacts
AWS_DEFAULT_REGION=${aws_region}
EOF

# Create MLflow systemd service
cat > /etc/systemd/system/mlflow.service << 'EOF'
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=mlflow
Group=mlflow
WorkingDirectory=/opt/mlflow
EnvironmentFile=/opt/mlflow/mlflow.env
ExecStart=/usr/local/bin/mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create CloudWatch agent configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/home/mlflow/logs/mlflow.log",
                        "log_group_name": "/aws/ec2/mlflow/${project_name}",
                        "log_stream_name": "{instance_id}/mlflow.log",
                        "timezone": "UTC"
                    },
                    {
                        "file_path": "/var/log/messages",
                        "log_group_name": "/aws/ec2/mlflow/${project_name}",
                        "log_stream_name": "{instance_id}/system.log",
                        "timezone": "UTC"
                    }
                ]
            }
        }
    }
}
EOF

# Wait for RDS to be ready with timeout
echo "Waiting for database to be ready..."
db_host=$(echo ${db_endpoint} | cut -d: -f1)
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if nc -z $db_host ${db_port}; then
        echo "Database is ready!"
        break
    else
        echo "Database not ready yet, waiting... (attempt $attempt/$max_attempts)"
        if [ $attempt -eq $max_attempts ]; then
            echo "ERROR: Database did not become ready after $max_attempts attempts"
            exit 1
        fi
        sleep 10
        ((attempt++))
    fi
done

# Test database connection and create tables
python3 << EOF
import mysql.connector
import time
import sys

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='${db_endpoint}'.split(':')[0],
            port=${db_port},
            user='${db_username}',
            password='${db_password}',
            database='${db_name}'
        )
        
        if connection.is_connected():
            print("Successfully connected to MySQL database")
            connection.close()
            break
            
    except Exception as e:
        print(f"Attempt {retry_count + 1}: Database connection failed: {e}")
        retry_count += 1
        if retry_count < max_retries:
            time.sleep(10)
        else:
            print("Failed to connect to database after all retries")
            sys.exit(1)
EOF

# Start and enable services
systemctl daemon-reload
systemctl enable mlflow
systemctl start mlflow

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Enable Docker service
systemctl enable docker
systemctl start docker

# Create a simple health check script
cat > /opt/mlflow/health_check.sh << 'EOF'
#!/bin/bash
response=$(curl -s -o /dev/null -w "%%{http_code}" http://localhost:5000/health)
if [ $response -eq 200 ]; then
    echo "MLflow is healthy"
    exit 0
else
    echo "MLflow is not responding"
    exit 1
fi
EOF

chmod +x /opt/mlflow/health_check.sh

# Setup cron job for health monitoring
echo "*/5 * * * * /opt/mlflow/health_check.sh >> /home/mlflow/logs/health.log 2>&1" | crontab -u mlflow -

# Create startup script for additional configuration
cat > /opt/mlflow/startup.sh << 'EOF'
#!/bin/bash
echo "MLflow server startup completed at $(date)"
echo "MLflow UI accessible at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
echo "Backend store: ${MLFLOW_BACKEND_STORE_URI}"
echo "Artifact store: ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
EOF

chmod +x /opt/mlflow/startup.sh
sudo -u mlflow /opt/mlflow/startup.sh

echo "MLflow setup completed successfully!"

# Final status check
sleep 30
if systemctl is-active --quiet mlflow; then
    echo "MLflow service is running successfully"
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
    echo "MLflow UI is available at: http://$PUBLIC_IP:5000"
else
    echo "MLflow service failed to start"
    systemctl status mlflow
    journalctl -u mlflow -n 20
    exit 1
fi 
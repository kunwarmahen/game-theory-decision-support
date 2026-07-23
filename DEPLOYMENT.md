# Game Theory Decision Analyzer - Deployment Guide

This guide covers deploying the Game Theory Decision Analyzer on a traditional Linux server.

## Prerequisites

- Ubuntu/Debian Linux server (20.04 LTS or newer recommended)
- Python 3.8 or higher
- Ollama installed and running (see [Ollama Installation](#ollama-installation))
- sudo/root access
- At least 8GB RAM recommended for a mid-size Gemma model (more for larger models)

> The decision tree is rendered in the browser (vis-network), so no Graphviz or other
> system graphics libraries are required.

## Quick Start

### 1. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and required system packages
sudo apt install -y python3 python3-pip python3-venv

# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Set Up Application Directory

```bash
# Create application directory
sudo mkdir -p /opt/game-theory-analyzer
sudo chown $USER:$USER /opt/game-theory-analyzer

# Navigate to your project directory and copy files
cd /home/mahen/Documents/ai/game_theory/decision_support
cp -r . /opt/game-theory-analyzer/

# Navigate to deployment directory
cd /opt/game-theory-analyzer
```

### 3. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables (use nano or your preferred editor)
nano .env
```

Update the `.env` file with your settings:

```ini
# Application Settings
APP_NAME="Game Theory Decision Analyzer"
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false
DEBUG=false

# Provider: "ollama" (local Gemma) or "openai"
DEFAULT_PROVIDER=ollama

# Ollama (local models such as Gemma)
DEFAULT_OLLAMA_URL=http://localhost:11434
DEFAULT_MODEL_NAME=gemma4:12b

# OpenAI (optional — leave blank to disable). Read from the server env; never sent
# from the browser.
OPENAI_API_KEY=
DEFAULT_OPENAI_MODEL=gpt-4o

# Generation
TEMPERATURE=0.2

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/game-theory-analyzer/app.log

# CORS Settings (comma-separated origins, or * for all)
CORS_ORIGINS=*
```

### 5. Create Log Directory

```bash
sudo mkdir -p /var/log/game-theory-analyzer
sudo chown www-data:www-data /var/log/game-theory-analyzer
```

### 6. Test the Application

```bash
# Make sure Ollama is running
ollama serve &

# Pull a model if you haven't already
ollama pull gemma4:12b

# Test the application
source venv/bin/activate
python app.py
```

Visit `http://your-server-ip:8000` in your browser to verify it's working.

Press Ctrl+C to stop the test server.

## Production Deployment with systemd

### 1. Install systemd Service

```bash
# Copy service file to systemd directory
sudo cp game-theory-analyzer.service /etc/systemd/system/

# Update the service file if your paths are different
sudo nano /etc/systemd/system/game-theory-analyzer.service

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable game-theory-analyzer

# Start the service
sudo systemctl start game-theory-analyzer

# Check service status
sudo systemctl status game-theory-analyzer
```

### 2. Manage the Service

```bash
# Start service
sudo systemctl start game-theory-analyzer

# Stop service
sudo systemctl stop game-theory-analyzer

# Restart service
sudo systemctl restart game-theory-analyzer

# View logs
sudo journalctl -u game-theory-analyzer -f

# Or view application logs
tail -f /var/log/game-theory-analyzer/app.log
tail -f /var/log/game-theory-analyzer/error.log
```

## Nginx Reverse Proxy (Optional but Recommended)

For production, it's recommended to use Nginx as a reverse proxy:

### 1. Install Nginx

```bash
sudo apt install -y nginx
```

### 2. Configure Nginx

Create a new Nginx configuration file:

```bash
sudo nano /etc/nginx/sites-available/game-theory-analyzer
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or server IP

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed in future)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 3. Enable the Site

```bash
# Create symbolic link to enable the site
sudo ln -s /etc/nginx/sites-available/game-theory-analyzer /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### 4. SSL/HTTPS with Let's Encrypt (Recommended)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Certbot will automatically configure Nginx for HTTPS
```

## Ollama Installation

### Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama

# Enable Ollama to start on boot
sudo systemctl enable ollama

# Pull your preferred model
ollama pull gemma4:12b
```

### Available Models

```bash
# List pulled models (these populate the model dropdown in the UI)
ollama list

# Pull other chat-capable models
ollama pull gemma4:e4b     # smaller/faster Gemma variant
ollama pull qwen2.5:14b
ollama pull llama3
```

## Firewall Configuration

```bash
# Allow HTTP and HTTPS through firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# If not using Nginx, allow the app port directly
sudo ufw allow 8000/tcp

# Enable firewall
sudo ufw enable
```

## Monitoring and Maintenance

### View Logs

```bash
# Application logs
tail -f /var/log/game-theory-analyzer/app.log

# Error logs
tail -f /var/log/game-theory-analyzer/error.log

# Systemd service logs
sudo journalctl -u game-theory-analyzer -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Log Rotation

Create a logrotate configuration:

```bash
sudo nano /etc/logrotate.d/game-theory-analyzer
```

Add the following:

```
/var/log/game-theory-analyzer/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload game-theory-analyzer > /dev/null 2>&1 || true
    endscript
}
```

### Updates and Maintenance

```bash
# Stop the service
sudo systemctl stop game-theory-analyzer

# Navigate to app directory
cd /opt/game-theory-analyzer

# Pull latest code (if using git)
git pull

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart game-theory-analyzer

# Check status
sudo systemctl status game-theory-analyzer
```

## Performance Tuning

### Increase Worker Processes

For better performance under load, you can configure uvicorn to use multiple workers.

Edit `app.py` main function or create a separate production startup script:

```python
uvicorn.run(
    "app:app",
    host=settings.app_host,
    port=settings.app_port,
    workers=4,  # Adjust based on CPU cores
    log_level=settings.log_level.lower()
)
```

### Resource Limits

Edit the systemd service file to set resource limits:

```bash
sudo nano /etc/systemd/system/game-theory-analyzer.service
```

Add under `[Service]`:

```ini
# Memory limits
MemoryLimit=2G
MemoryMax=4G

# CPU limits
CPUQuota=200%  # 2 CPU cores max
```

Reload systemd after changes:

```bash
sudo systemctl daemon-reload
sudo systemctl restart game-theory-analyzer
```

## Troubleshooting

### Service Won't Start

```bash
# Check service status
sudo systemctl status game-theory-analyzer

# View detailed logs
sudo journalctl -u game-theory-analyzer -n 100

# Check if port is already in use
sudo netstat -tulpn | grep 8000
```

### Ollama Connection Issues

```bash
# Check if Ollama is running
sudo systemctl status ollama

# Test Ollama connection
curl http://localhost:11434/api/tags

# Check Ollama logs
sudo journalctl -u ollama -f
```

### Permission Issues

```bash
# Fix ownership
sudo chown -R www-data:www-data /opt/game-theory-analyzer
sudo chown -R www-data:www-data /var/log/game-theory-analyzer

# Fix permissions
sudo chmod -R 755 /opt/game-theory-analyzer
```

### Empty or Truncated Model Responses

Some local models (e.g. Gemma) support a "thinking" mode that can consume the token
budget and return empty content; the app disables it automatically. If analyses still
come back empty or truncated, try a different or smaller model from the dropdown, or
shorten the query.

## Security Best Practices

1. **Use HTTPS**: Always use SSL/TLS in production (Let's Encrypt recommended)
2. **Firewall**: Configure UFW or iptables to restrict access
3. **Regular Updates**: Keep the system and dependencies updated
4. **Secure .env**: Ensure `.env` file has restrictive permissions:
   ```bash
   chmod 600 /opt/game-theory-analyzer/.env
   ```
5. **CORS Configuration**: Update `CORS_ORIGINS` in `.env` to restrict origins in production
6. **Monitoring**: Set up monitoring and alerting for the service

## Backup

```bash
# Backup application directory
sudo tar -czf game-theory-analyzer-backup-$(date +%Y%m%d).tar.gz /opt/game-theory-analyzer

# Backup logs
sudo tar -czf game-theory-logs-backup-$(date +%Y%m%d).tar.gz /var/log/game-theory-analyzer
```

## Support

For issues and questions:
- Check the logs: `/var/log/game-theory-analyzer/`
- Review Ollama documentation: https://ollama.com/docs
- FastAPI documentation: https://fastapi.tiangolo.com/

## Quick Commands Reference

```bash
# Start/Stop/Restart Service
sudo systemctl start game-theory-analyzer
sudo systemctl stop game-theory-analyzer
sudo systemctl restart game-theory-analyzer

# View Logs
sudo journalctl -u game-theory-analyzer -f
tail -f /var/log/game-theory-analyzer/app.log

# Check Service Status
sudo systemctl status game-theory-analyzer

# Test Application
curl http://localhost:8000

# Pull Ollama Models
ollama pull gemma4:12b
ollama list
```

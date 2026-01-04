#!/bin/bash
# EC2 Setup Script for Crypto Pulse Tracker
# Run as root or with sudo on Ubuntu 22.04/24.04

set -e

echo "=== Crypto Pulse Tracker - EC2 Setup ==="

# Update system
echo "Updating system..."
apt update && apt upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Install Docker Compose
echo "Installing Docker Compose..."
apt install -y docker-compose-plugin

# Install Nginx
echo "Installing Nginx..."
apt install -y nginx

# Install Git
apt install -y git

# Create app directory
echo "Creating app directory..."
mkdir -p /opt/crypto-pulse
cd /opt/crypto-pulse

# Clone repository (replace with your repo URL)
echo "Clone your repository:"
echo "git clone https://github.com/YOUR_USERNAME/crypto-pulse.git ."
echo ""
read -p "Press Enter after cloning your repo..."

# Create .env file
echo "Creating .env file..."
cat > .env << 'EOF'
# Telegram Bot
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHANNEL_ID=your_channel_id

# Tracked Coin
TRACKED_COIN=MEME

# CORS - Add your Vercel frontend domain
CORS_ORIGINS=https://your-app.vercel.app,https://yourdomain.com

# Optional
COINGECKO_API_KEY=
DATA_PERSISTENCE=memory

# Server
PIPELINE_PORT=8000
WEBHOOK_PORT=8080
EOF

echo "Edit .env with your actual values:"
echo "nano /opt/crypto-pulse/.env"
read -p "Press Enter after editing .env..."

# Setup Nginx
echo "Setting up Nginx..."
cp deploy/nginx.conf /etc/nginx/sites-available/crypto-pulse

# Update domain in nginx config
read -p "Enter your API domain (e.g., api.yourdomain.com): " API_DOMAIN
sed -i "s/api.yourdomain.com/$API_DOMAIN/g" /etc/nginx/sites-available/crypto-pulse

# Enable site
ln -sf /etc/nginx/sites-available/crypto-pulse /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Restart nginx
systemctl restart nginx
systemctl enable nginx

# Create systemd service for the app
echo "Creating systemd service..."
cat > /etc/systemd/system/crypto-pulse.service << 'EOF'
[Unit]
Description=Crypto Pulse Tracker
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/crypto-pulse
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable crypto-pulse
systemctl start crypto-pulse

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Point your domain DNS to this server's IP"
echo "2. In Cloudflare: Add A record for $API_DOMAIN → $(curl -s ifconfig.me)"
echo "3. In Cloudflare: SSL/TLS → Full (strict) or Flexible"
echo "4. Test: curl http://$API_DOMAIN/health"
echo ""
echo "Useful commands:"
echo "  View logs:     journalctl -u crypto-pulse -f"
echo "  Restart:       systemctl restart crypto-pulse"
echo "  Docker logs:   cd /opt/crypto-pulse && docker compose logs -f"

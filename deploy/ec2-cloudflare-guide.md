# EC2 + Cloudflare + Nginx Deployment Guide

## Architecture

```
User → Cloudflare (SSL/CDN) → Nginx (reverse proxy) → Docker (main.py)
                                                    ↓
Frontend (Vercel) ←──────────────────────────────────
```

---

## Step 1: Launch EC2 Instance

### AWS Console

1. Go to EC2 → Launch Instance
2. **AMI:** Ubuntu 22.04 LTS
3. **Instance Type:** t3.small (2GB RAM) or t3.medium (4GB RAM)
4. **Storage:** 20GB gp3 (EBS)
5. **Security Group:** Allow:
   - SSH (22) from your IP
   - HTTP (80) from anywhere
   - HTTPS (443) from anywhere

### Get Your Instance IP

After launch, note the **Public IPv4 address** (e.g., `54.123.45.67`)

---

## Step 2: Setup Cloudflare

### Add Your Domain

1. Go to [cloudflare.com](https://cloudflare.com) → Add Site
2. Enter your domain
3. Select Free plan
4. Update nameservers at your registrar

### Add DNS Records

Go to DNS → Records → Add Record:

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | api | 54.123.45.67 (your EC2 IP) | ✅ Proxied |
| A | @ | (your Vercel IP or CNAME) | ✅ Proxied |

### SSL Settings

Go to SSL/TLS:

- **Mode:** Full (strict) if you have cert, or **Flexible** for simplest setup
- **Edge Certificates:** On
- **Always Use HTTPS:** On

### Security Settings (Optional but Recommended)

- **Security → WAF:** Enable basic rules
- **Security → Bots:** Enable bot fight mode

---

## Step 3: Setup EC2 Server

### SSH into your instance

```bash
ssh -i your-key.pem ubuntu@54.123.45.67
```

### Run the setup script

```bash
# Download and run setup script
curl -O https://raw.githubusercontent.com/YOUR_REPO/main/deploy/ec2-setup.sh
chmod +x ec2-setup.sh
sudo ./ec2-setup.sh
```

### Or manual setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu

# Install Nginx
sudo apt install -y nginx git

# Clone your repo
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/crypto-pulse.git
cd crypto-pulse

# Create .env
sudo nano .env
# Add your environment variables

# Copy nginx config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/crypto-pulse
sudo nano /etc/nginx/sites-available/crypto-pulse
# Update server_name to your domain

# Enable nginx site
sudo ln -s /etc/nginx/sites-available/crypto-pulse /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Start the app
sudo docker compose up -d
```

---

## Step 4: Configure Frontend (Vercel)

### Deploy Frontend to Vercel

```bash
cd frontend
vercel
```

### Set Environment Variables in Vercel

Go to Vercel Dashboard → Your Project → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com
```

### Custom Domain on Vercel

1. Settings → Domains → Add
2. Enter: `yourdomain.com` or `app.yourdomain.com`
3. Add CNAME in Cloudflare pointing to Vercel

---

## Step 5: Test Everything

```bash
# Test backend health
curl https://api.yourdomain.com/health

# Test metrics
curl https://api.yourdomain.com/api/metrics

# Test WebSocket (using wscat)
npx wscat -c wss://api.yourdomain.com/ws

# Open frontend
open https://yourdomain.com
```

---

## Useful Commands

### On EC2 Server

```bash
# View app logs
cd /opt/crypto-pulse && docker compose logs -f

# Restart app
docker compose restart

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Check nginx status
sudo systemctl status nginx

# Reload nginx config
sudo nginx -t && sudo systemctl reload nginx
```

### Update Deployment

```bash
cd /opt/crypto-pulse
git pull
docker compose down
docker compose up -d --build
```

---

## Troubleshooting

### 502 Bad Gateway

- Check if Docker container is running: `docker compose ps`
- Check container logs: `docker compose logs`
- Verify port 8000 is listening: `netstat -tlnp | grep 8000`

### WebSocket not connecting

- Ensure Cloudflare WebSocket is enabled (Network → WebSockets)
- Check nginx WebSocket config has proper upgrade headers

### CORS errors

- Verify CORS_ORIGINS in .env includes your frontend domain
- Include both http and https versions if needed

### SSL issues

- If using Cloudflare Flexible: nginx only needs port 80
- If using Cloudflare Full: need SSL cert on nginx (use Let's Encrypt)

---

## Cost Estimate

| Service | Cost |
|---------|------|
| EC2 t3.small | ~$15/month |
| EBS 20GB | ~$2/month |
| Cloudflare | Free |
| Vercel | Free |
| **Total** | **~$17/month** |

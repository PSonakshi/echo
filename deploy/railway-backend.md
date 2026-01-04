# Railway Backend Deployment (main.py)

## Railway Environment Variables

Set these in Railway Dashboard → Your Service → Variables:

```env
# Required
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=your_channel_id
TRACKED_COIN=MEME

# CORS - Add your Vercel frontend URL
CORS_ORIGINS=https://your-app.vercel.app,https://yourdomain.com

# Optional
COINGECKO_API_KEY=CG-xxx
DATA_PERSISTENCE=memory
```

## Deploy Steps

1. Push repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo (it will use the root `railway.json`)
4. Add environment variables above
5. Deploy!

## Get Your Railway URL

After deploy:

- Settings → Networking → Generate Domain
- You'll get something like: `crypto-pulse-production.up.railway.app`

## Update Frontend

In Vercel, set:

```
NEXT_PUBLIC_API_URL=https://crypto-pulse-production.up.railway.app
NEXT_PUBLIC_WS_URL=wss://crypto-pulse-production.up.railway.app
```

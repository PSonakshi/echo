# Dockerfile for Railway deployment (API server only)
# For Pathway pipeline, use: docker-compose up (uses Dockerfile.pathway)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (no Pathway needed for API)
COPY requirements-railway.txt ./requirements-railway.txt
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application code
COPY . .

# Railway sets PORT env variable
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

# Run the full API server (reads PORT from env)
CMD ["python", "main.py"]

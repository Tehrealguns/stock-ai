FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create data directory for persistent SQLite DB
RUN mkdir -p /data

# Railway sets PORT dynamically
ENV PORT=8888

# Start the server — uses $PORT from env
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT

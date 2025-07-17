FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Collect static files
RUN python deepmine_sentinel_ai/manage.py collectstatic --noinput

EXPOSE 8000

CMD ["python", "deepmine_sentinel_ai/manage.py", "runserver", "0.0.0.0:8000"]

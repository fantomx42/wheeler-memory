FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mount point for the wheeler memory library (read-only)
# We expect it to be mounted to /app/wheeler_memory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy viewer app
COPY . /app/viewer

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "viewer.app:app", "--host", "0.0.0.0", "--port", "8000"]

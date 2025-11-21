# Use modern Python base image instead of deprecated transformers image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements_inference.txt .
RUN pip install --no-cache-dir "dvc[gdrive]" && \
    pip install --no-cache-dir -r requirements_inference.txt

# Copy application code
COPY ./ .

# Initialise dvc and configure remote server
RUN dvc init --no-scm && \
    dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954 && \
    dvc remote modify storage gdrive_use_service_account true && \
    dvc remote modify storage gdrive_service_account_json_file_path creds.json && \
    cat .dvc/config && \
    dvc pull dvcfiles/trained_model.dvc

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

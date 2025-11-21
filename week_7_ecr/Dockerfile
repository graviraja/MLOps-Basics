# Use modern Python base image instead of deprecated transformers image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set environment variables (experimental - consider using secrets management in production)
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Copy requirements first for better layer caching
COPY requirements_inference.txt .
RUN pip install --no-cache-dir "dvc[s3]" && \
    pip install --no-cache-dir -r requirements_inference.txt

# Copy application code
COPY ./ .

# Initialise dvc and configure remote server
RUN dvc init --no-scm && \
    dvc remote add -d model-store s3://models-dvc/trained_models/ && \
    cat .dvc/config && \
    dvc pull dvcfiles/trained_model.dvc

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

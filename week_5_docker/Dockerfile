# Use modern Python base image instead of deprecated transformers image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

# Copy application code
COPY ./ .

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

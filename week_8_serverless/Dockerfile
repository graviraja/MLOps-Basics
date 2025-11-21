# Use specific version of AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir -p $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH="${PYTHONPATH}:./" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Install system dependencies
RUN yum install -y git gcc-c++ && yum clean all

# Copy and install requirements first for better layer caching
COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt && \
    pip install --no-cache-dir "dvc[s3]"

# Copy application code
COPY ./ .

# Configure DVC and pull model
RUN dvc init --no-scm && \
    dvc remote add -d model-store s3://models-dvc/trained_models/ && \
    dvc pull dvcfiles/trained_model.dvc && \
    python lambda_handler.py && \
    chmod -R 0755 $MODEL_DIR && \
    ls

CMD [ "lambda_handler.lambda_handler" ]
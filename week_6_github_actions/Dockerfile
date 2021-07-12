FROM huggingface/transformers-pytorch-cpu:latest
ARG GDRIVE_SERVICE_ACCOUNT
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements_inference.txt
RUN pip install "dvc[gdrive]"
RUN dvc init --no-scm
RUN dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
RUN printf '%s\n' "$GDRIVE_SERVICE_ACCOUNT" > creds.json
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage --local gdrive_service_account_json_file_path creds.json
RUN cd dvcfiles && dvc pull trained_model.dvc
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

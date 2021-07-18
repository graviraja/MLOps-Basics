FROM huggingface/transformers-pytorch-cpu:latest

ARG GDRIVE_SERVICE_ACCOUNT
RUN echo $GDRIVE_SERVICE_ACCOUNT


COPY ./ /app
WORKDIR /app

# RUN echo $GDRIVE_SERVICE_ACCOUNT  > creds.json
RUN printf '%s' $GDRIVE_SERVICE_ACCOUNT > creds.json
RUN cat creds.json
RUN python3 parse_json.py

# # install requirements
# RUN pip install "dvc[gdrive]"
# RUN pip install -r requirements_inference.txt

# # initialise dvc
# RUN dvc init --no-scm
# # configuring remote server in dvc
# RUN dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
# RUN echo $GDRIVE_SERVICE_ACCOUNT > creds.json
# RUN python3 parse_json.py
# RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify storage gdrive_service_account_json_file_path test.json

# # pulling the trained model
# RUN dvc pull dvcfiles/trained_model.dvc

# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# # running the application
# EXPOSE 8000
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

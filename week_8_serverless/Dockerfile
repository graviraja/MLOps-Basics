FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY


#this envs are experimental
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


# install requirements
RUN pip install "dvc[s3]"
RUN pip install -r requirements_inference.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d model-store s3://models-dvc/trained_models/

# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH "${PYTHONPATH}:./"

RUN ls

# running the application
CMD [ "lambda_handler.lambda_handler" ]

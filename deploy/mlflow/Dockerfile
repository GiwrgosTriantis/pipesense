# MLflow tracking server for the prod-like profile: the official image plus the Postgres driver
# and boto3 for the S3 artifact store. Clients never need S3 credentials: artifacts are proxied.
FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.35.81

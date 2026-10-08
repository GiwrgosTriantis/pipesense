FROM continuumio/miniconda3:latest

COPY environment.yml .
RUN conda env create -f environment.yml

RUN groupadd -g 1313 pipesense && usermod -a -G pipesense root

ENV PIPESENSE_ENV=production
ENV AWS_DEFAULT_REGION=us-east-1
ENV WORK_DIR=/shared/output
ENV DATA_DIR=/app/data

COPY pipeline/common.py /app/pipeline/common.py
COPY pipeline/prepare_data.py /app/pipeline/prepare_data.py
COPY pipeline/train_model.py /app/pipeline/train_model.py
COPY pipeline/predict.py /app/pipeline/predict.py
COPY run_all.sh /app/run_all.sh
COPY data/pipes.csv /app/data/pipes.csv

WORKDIR /app

SHELL ["conda", "run", "-n", "pipesense", "/bin/bash", "-c"]

CMD ["conda", "run", "-n", "pipesense", "bash", "run_all.sh"]

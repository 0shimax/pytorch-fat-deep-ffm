FROM python:3.9.1-slim-buster

RUN apt update \
    && apt install -y curl wget gosu jq unzip libgomp1 python-dev default-libmysqlclient-dev python3-dev

RUN pip install -U pip \
    && pip install --pre -U scikit-learn \
    && pip install \
    numpy \
    pandas \
    ipython \
    jupyter \
    matplotlib \
    seaborn \
    flake8 \
    black \
    torch \
    torchvision

RUN pip install mlflow \
    xfeat
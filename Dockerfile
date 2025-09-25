FROM python:3.11-slim-bookworm

ENV HOME=/home/fbf \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR $HOME

# System deps + Azure CLI (bookworm)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release apt-transport-https \
      build-essential git wget libxml2-utils gir1.2-secret-1 \
      gdal-bin libgdal-dev libproj-dev libgeos-dev libspatialindex-dev \
      libudunits2-dev libcairo2-dev libgirepository1.0-dev gfortran \
      libeccodes0 python3-eccodes \
    && curl -sL https://packages.microsoft.com/keys/microsoft.asc \
         | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.asc.gpg \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ bookworm main" \
         > /etc/apt/sources.list.d/azure-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends azure-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Minimal Python tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Python deps
COPY requirements.txt $HOME/requirements.txt
RUN pip install -r $HOME/requirements.txt

# App code
COPY IBF-Typhoon-model $HOME/IBF-Typhoon-model
WORKDIR $HOME/IBF-Typhoon-model
RUN pip install .
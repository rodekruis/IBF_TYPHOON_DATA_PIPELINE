FROM mambaorg/micromamba:latest

# Set the environment variable for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Switch to root user to install packages
USER root

# Set up main directory
RUN mkdir -p /home/fbf/forecast
ENV HOME /home/fbf
WORKDIR $HOME

# Install Ubuntu packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        python3-pip \
        python3-eccodes \
        ca-certificates \
        wget \
        libxml2-utils \
        gir1.2-secret-1 \
        pkg-config \
        libturbojpeg0-dev \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Azure CLI
RUN apt-get update && apt-get -y upgrade && \
    apt-get -f -y install ca-certificates curl apt-transport-https lsb-release gnupg && \
    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.asc.gpg && \
    CLI_REPO=$(lsb_release -cs) && \
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ ${CLI_REPO} main" \
    > /etc/apt/sources.list.d/azure-cli.list && \
    apt-get update && \
    apt-get install -y azure-cli && \
    rm -rf /var/lib/apt/lists/* 
	

# Update pip and install initial dependencies
#RUN python3 -m pip install --no-cache-dir --upgrade  pip setuptools wheel numpy

#copy yml file to install climada and other required libraries 
COPY env_climada.yml /home/fbf/

# Copy the ibf data pipeline code rest of the code and install the package
ADD IBF-Typhoon-model .

RUN micromamba create -y -f env_climada.yml -n climada_env && \
 	micromamba clean --all --yes  

# Set the default environment
ENV CONDA_DEFAULT_ENV=climada_env
ENV PATH=/opt/conda/envs/climada_env/bin:$PATH

# Activate the environment
RUN echo "source activate climada_env" > ~/.bashrc

# Clone CLIMADA repository
RUN git clone https://github.com/CLIMADA-project/climada_python.git /home/fbf/climada_python && \
    cd /home/fbf/climada_python && \
    git checkout tags/v4.0.0 && \
	mkdir -p /home/fbf/src/climada && cp -r /home/fbf/climada_python/climada/* /home/fbf/src/climada/

WORKDIR /home/fbf/climada_python

# Install CLIMADA in editable mode
#RUN python3 -m pip install -e .
	
#RUN micromamba activate climada_env
RUN python3 -m pip install --no-cache -e .

# copy data folder needed by climada, a quick fix 
RUN mkdir -p /opt/conda/envs/climada_env/lib/python3.9/site-packages/data && cp -r /home/fbf/climada_python/data/* /opt/conda/envs/climada_env/lib/python3.9/site-packages/data/

# Return to home directory
WORKDIR $HOME

#RUN pip install --no-cache-dir -r requirements.txt 
RUN python3 -m pip install --no-cache-dir pybufrkit==0.2.22 setuptools wheel numpy

# install ibf data pipeline 
#ADD IBF-Typhoon-model .
RUN pip install .

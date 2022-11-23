FROM rocker/r-ubuntu:20.04

# Set up main directory
RUN mkdir --parents /home/fbf/forecast
ENV HOME /home/fbf
WORKDIR $HOME

# Install required system packages
# python3-eccodes is required for python cfgrib
# libproj-dev & libgeos-dev required for python cartopy
# libspatialindex-dev required for python Rtree (which is required for geopandas sjoin)
# libudunits2-dev required for R units (which is required for R tmap)
# libssl-dev required for R s2 (which is required for tmap)
# libgdal-dev required for R sf


RUN apt-get update && apt-get install -y \
    software-properties-common \
    nano \
    vim \
    python3-pip \
	python3-eccodes \
    git \
    wget \
    libxml2-utils \
    gir1.2-secret-1\
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge --auto-remove \
    && apt-get clean

#gdal-bin libgdal-dev \
	
RUN add-apt-repository ppa:ubuntugis/ppa \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
    python-numpy \
    postgresql-client \
	libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    libudunits2-dev \
    libssl-dev \
    libgnutls28-dev \
    libeccodes0 \
	libcairo2-dev\
	libgirepository1.0-dev\
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge --auto-remove \
    && apt-get clean	
	
RUN apt-get update && apt-get install -y curl
 
RUN apt-get update && apt-get -y upgrade && \
    apt-get -f -y install curl apt-transport-https lsb-release gnupg && \
    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.asc.gpg && \
    CLI_REPO=$(lsb_release -cs) && \
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ ${CLI_REPO} main" \
    > /etc/apt/sources.list.d/azure-cli.list && \
    apt-get update && \
    apt-get install -y azure-cli && \
    rm -rf /var/lib/apt/lists/*
	

# update pip
RUN python3 -m pip install --no-cache-dir \
    pip \
    setuptools \
    wheel \
    --upgrade \
    && python3 -m pip install --no-cache-dir numpy


# Install Python dependencies
COPY requirements.txt /home/fbf/
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and install
ADD IBF-Typhoon-model .
RUN pip install .



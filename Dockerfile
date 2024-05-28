FROM ghcr.io/osgeo/gdal:ubuntu-small-latest
# gdal/ogr binaries are located at /usr/bin

ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev


RUN apt-get update && apt-get install -y software-properties-common gcc && \
  add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
  python3.12 python3-distutils python3-pip python3-apt libpq-dev \
  build-essential python3-dev python3.12-venv 

# Install OpenGL libraries
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libgl1-mesa-dri

# Create a virtual environment using venv from the Python standard library
RUN python3.12 -m venv /env

# Activate the virtual environment and install the historef package
COPY dist/historef-latest-py3-none-any.whl /tmp/
RUN /env/bin/pip install --upgrade pip && \
    /env/bin/pip install /tmp/historef-latest-py3-none-any.whl

ENV PATH="/env/bin:$PATH"

# Set the entrypoint to run historef
ENTRYPOINT ["/env/bin/historef"]


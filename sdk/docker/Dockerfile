FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Main apt packages
RUN apt-get update \
 && apt-get install -y python3.8 \
 && apt-get install -y python3-pip \
 && apt-get install -y virtualenv \
 && apt-get install -y git
 
# Install some global dependencies
RUN apt-get install -y libglib2.0-0 \
 && apt-get install -y libsm6 \
 && apt-get install -y libxext6 \
 && apt-get install -y libxrender-dev

# Upgrade to current pip version
RUN pip3 install --upgrade pip

# Install basic CV and ML tools
RUN pip3 install setuptools \
 && pip3 install numpy \
 && pip3 install scikit-image \
 && pip3 install Pillow \
 && pip3 install torch \
 && pip3 install torchvision
 
# Install special python packages for ML
RUN pip3 install segmentation_models_pytorch \
 && pip3 install albumentations

RUN echo '#!/bin/bash\npython3 "$@"' > /usr/bin/python \
 && chmod +x /usr/bin/python

RUN echo '#!/bin/bash\nvirtualenv -p python3 "$@"' > /usr/bin/venv \
 && chmod +x /usr/bin/venv

RUN echo '#!/bin/bash\nbash "$@"' > /usr/bin/sh \
 && chmod +x /usr/bin/sh

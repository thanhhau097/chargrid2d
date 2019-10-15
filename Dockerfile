FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG py_version=3
# Validate that arguments are specified
RUN test $py_version || exit 1

# Install python and nginx
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        jq \
        libsm6 \
        libxext6 \
        libxrender-dev \
        nginx && \
    if [ $py_version -eq 3 ]; \
       then apt-get install -y --no-install-recommends python3.6-dev \
           && ln -s -f /usr/bin/python3.6 /usr/bin/python; \
       else apt-get install -y --no-install-recommends python-dev; fi && \
    rm -rf /var/lib/apt/lists/*
    
# Install pip
RUN cd /tmp && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && rm get-pip.py

RUN pip install --upgrade pip
RUN pip install sagemaker
RUN pip install torch
RUN pip install torchvision
RUN pip install scikit-learn
RUN pip install albumentations

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install git 
RUN apt-get update \
    && apt-get install -y git

# Copy the code
RUN rm -rf /opt/ml/code/
COPY . /opt/ml/code/

ENV PATH="/opt/ml/code:${PATH}"

# Clean up
RUN rm -rf /root/.cache
RUN rm -rf /var/lib/apt/lists/* ~/.cache/pip
RUN apt-get autoremove && apt-get clean

# Set VNT for training instance
RUN unlink /etc/localtime
RUN ln -s /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime

# Specify Working dir
WORKDIR /opt/ml/code/

ENV export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
# Starts framework
ENTRYPOINT ["python", "main.py"]

# , "--config", "configs/sagemaker_config.json"
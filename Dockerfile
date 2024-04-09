# Start from the NVIDIA CUDA base image with CUDA and cuDNN
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-pip \
        software-properties-common \
        unzip \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install TensorFlow GPU
RUN pip install tensorflow-gpu==2.4.1

# Install additional Python dependencies for deep learning
RUN pip install numpy scipy matplotlib ipython jupyter pandas sympy nose

# Set the working directory
WORKDIR /workspace

# need to put models' code and any additional files required into the Docker image here
# depend on project's structure
# COPY ./model_directory /workspace/model_directory

# Command to run when starting the container
CMD ["bash"]


# build image
#docker build -t tensorflow-gpu-models .

# run containers
#docker run --gpus all -it tensorflow-gpu-models

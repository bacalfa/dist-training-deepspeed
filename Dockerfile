FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Basic deps
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y git curl build-essential python3 python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create app dir
ENV WORKSPACE_DIR=/workspace
WORKDIR ${WORKSPACE_DIR}
RUN python3 -m venv ${WORKSPACE_DIR}/venv
COPY requirements.txt ${WORKSPACE_DIR}/
RUN ${WORKSPACE_DIR}/venv/bin/pip install --upgrade pip
RUN ${WORKSPACE_DIR}/venv/bin/pip install -r requirements.txt

# Copy code
COPY . ${WORKSPACE_DIR}/

ENV PATH="${WORKSPACE_DIR}/venv/bin:${PATH}"
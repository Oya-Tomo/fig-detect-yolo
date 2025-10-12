FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    curl \
    wget \
    git \
    libssl-dev \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Sync the project
WORKDIR /workspace
RUN git config --global --add safe.directory /workspace
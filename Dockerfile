# Multi-stage build for GOP project
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt requirements-gpu.txt requirements-all.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY cli.py main.py setup.py pyproject.toml ./
COPY config/ ./config/
COPY README.md LICENSE ./

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash gop && \
    chown -R gop:gop /app
USER gop

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/logs

# Expose port for potential web interface
EXPOSE 8000

# Set default command
CMD ["gop", "--help"]

# Development stage
FROM base as development

# Switch back to root for additional development tools
USER root

# Install additional development dependencies
RUN pip install --no-cache-dir jupyter notebook

# Switch back to gop user
USER gop

# Set development command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "gop" ]; then\n\
    exec gop "${@:2}"\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command for production
CMD ["gop", "--help"]

# GPU stage (optional)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    python3.10-dev \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt requirements-gpu.txt requirements-all.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-gpu.txt

# Copy source code
COPY src/ ./src/
COPY cli.py main.py setup.py pyproject.toml ./
COPY config/ ./config/
COPY README.md LICENSE ./

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash gop && \
    chown -R gop:gop /app
USER gop

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/logs

# Set default command
CMD ["gop", "--help"]
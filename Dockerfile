# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF, OCR, and ML packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set Python path and environment variables
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1

# Set optimal memory settings
ENV OMP_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=1

# Run the main application
CMD ["python", "src/main.py"]

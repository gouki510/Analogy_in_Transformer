# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (including fonts for matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    fonts-inter \
    fonts-dejavu-core \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create results directory
RUN mkdir -p results

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]

# Default arguments (can be overridden)
CMD ["--sample", "samples/sample_1.json", "--device", "cuda", "--plot"]

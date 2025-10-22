# Use NVIDIA PyTorch base (for GPU + torch)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (including bentofile.yaml, service.py, and movielens_model.pt)
COPY . .

# Build the Bento (this creates a bento in the BentoML store)
RUN bentoml build

# Expose BentoML default port
EXPOSE 3000

# Serve using the service path from bentofile.yaml
CMD ["bentoml", "serve", "service:MovieLensEncoder", "--host", "0.0.0.0", "--port", "3000"]
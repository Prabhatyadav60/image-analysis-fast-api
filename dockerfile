
# Use a minimal base image
FROM python:3.10-slim

# Install system dependencies needed for Tesseract OCR and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Pre-download YOLOv8 model to cache (improves startup performance)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# Copy application code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 1. Base Image: Use slim Python 3.10
# Note: If you eventually use GPU, you must switch this to a CUDA-enabled base image.
# For now (CPU inference), this is the most efficient choice.
FROM python:3.11.13-slim

# 2. System Dependencies (Crucial Step)
# - libgl1 & libglib2.0: Required by OpenCV/image processing libraries often pulled in by scikit-image
# - build-essential: Sometimes needed to compile Python extensions (like numpy/scipy) if wheels aren't found
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy Requirements first (Docker Cache Layering)
# This prevents re-installing all heavy libs (TF, Numpy) every time you change 1 line of code.
COPY requirements.txt .

# 5. Install Python Libraries
# --no-cache-dir reduces image size by removing pip cache
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Environment Variables
# - TF_CPP_MIN_LOG_LEVEL=2: Hides messy TensorFlow info messages, shows only errors
# - PORT: Cloud Run default
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PORT=8080

# 8. Run Command
CMD ["python", "app.py"]
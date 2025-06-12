# Use a base image with Python 3.10 and CUDA 11.8 support
FROM runpod/pytorch:2.2.2-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg     git     && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 &&     pip install --no-cache-dir -r requirements.txt

# Download pre-trained models
RUN python3 -m pip install "huggingface_hub[cli]" &&     huggingface-cli download EEEELY/DICE-Talk --local-dir checkpoints --exclude "*/.git*" "*/.gitattributes" &&     huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints/stable-video-diffusion-img2vid-xt --exclude "*/.git*" "*/.gitattributes" &&     huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny --exclude "*/.git*" "*/.gitattributes"

# Expose port for RunPod handler (if necessary, though typically not for serverless)
# EXPOSE 8000

# Command to run the RunPod handler (will be defined in runpod_handler.py)
CMD ["python", "-m", "runpod_handler"]

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY wheels/ /wheels/

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --find-links=/wheels --no-index -r requirements.txt

COPY . .

CMD ["python3", "text_to_speech_route.py"]

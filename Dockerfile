# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
CMD ["python", "-m", "scripts.smoke", "--device", "cuda"]

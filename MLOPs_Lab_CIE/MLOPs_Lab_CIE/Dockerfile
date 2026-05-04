FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY data/   data/
COPY models/ models/
COPY src/    src/

# Default entrypoint: run the CLI predictor
ENTRYPOINT ["python", "src/predict_cli.py"]